# NER Redaction - Инструмент для автоматического распознавания и анонимизации персональных данных в русскоязычных текстах


Реализует гибридный подход: правила на основе библиотеки Yargy и регулярных выражений для форматных типов сущностей и дообученная нейросетевая модель Slovnet для контекстно-зависимых типов.


## Поддерживаемые типы сущностей

| Класс | Описание | Метод |
|---|---|---|
| `PERSON` | ФИО, фамилии, имена, инициалы | Slovnet (дообученный) |
| `PHONE` | Телефонные номера (все форматы) | Yargy + regex |
| `EMAIL` | Адреса электронной почты | regex |
| `ADDRESS` | Почтовые и фактические адреса | Slovnet (дообученный) |
| `ID` | ИНН, СНИЛС, номера документов и договоров | Yargy + regex |

## Быстрый старт

### Требования

- Python >= 3.10
- Docker + Docker Compose (для запуска контейнера)


### Загрузка файлов модели

Файлы модели не хранятся в репозитории GitHub и автоматически скачиваются из GitHub Releases и внешних источников.

Перед запуском необходимо скачать веса:

```bash
python scripts/download_assets.py
```

После выполнения в директории `models/final_model/` должны появиться:

```text
models/final_model/
├── navec_news_v1_1B_250K_300d_100q.tar
└── slovnet_ner_pii_ru_hard_no_pd.tar
```

### Запуск через Docker

```bash
docker compose up --build
```

После запуска сервис будет доступен:

- REST API: `http://localhost:8080`
- Swagger UI: `http://localhost:8080/docs`
- Web UI: `http://localhost:8080/ui`

### Локальный запуск (без Docker)

Установка зависимостей:

```bash
pip install -r requirements.txt
python scripts/download_assets.py
```

Запуск сервиса:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```


## Интерфейсы

### CLI

```bash
# Анонимизация текста
python -m src.ner_redaction.cli text "..." --mode replace

# Анонимизация файла
python -m src.ner_redaction.cli file input.txt \
  --mode replace \
  --output-dir outputs/

Результат:
outputs/input.anonymized.txt   - анонимизированный текст
outputs/input.report.json      - отчёт с найденными сущностями
```


Режимы анонимизации (`--mode`):
- `replace` - замена на метку класса: `[PERSON]`, `[PHONE]` и т.д.
- `mask` - маскирование символами: `****`
- `pseudonymize` - замена случайными псевдонимами того же типа

### REST API

После запуска сервиса интерактивная документация Swagger доступна по адресу:

`http://localhost:8080/docs`


Примеры использования API:

```bash
# Анонимизация текста
curl -X POST http://localhost:8080/deidentify-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Петрова Анна, email: anna@mail.ru", "mode": "replace"}'

# Анонимизация файла
curl -X POST http://localhost:8080/deidentify-file \
  -F "file=@document.txt" \
  -F "mode=replace"

# Анонимизация архива
curl -X POST http://localhost:8080/deidentify-archive \
  -F "file=@documents.zip" \
  -F "mode=replace" \
```

### Веб-интерфейс

Помимо REST API сервис предоставляет веб-интерфейс для интерактивной работы.

В веб-интерфейсе доступны:

- ввод и анонимизация текста;
- загрузка отдельных файлов;
- загрузка архивов с пакетной обработкой;
- выбор режима анонимизации и сущностей для анонимизации;
- просмотр результата обработки.


## Архитектура

```
Входные данные (текст / .txt / ZIP)
        │
        ▼
   RedactionPipeline
   ├── chunking (блоки по 1500 символов, перекрытие 200)
   │
   ├── HybridPIIExtractor
   │   ├── RuleBasedPIIExtractor  →  PHONE, EMAIL, ID
   │   └── SlovnetPIIExtractor    →  PERSON, ADDRESS
   │
   ├── разрешение пересечений
   ├── фильтрация allowlist / denylist
   │
   └── TextAnonymizer (replace / mask / pseudonymize)
        │
        ▼
   Анонимизированный текст + отчёт о найденных сущностях 
```

## Структура репозитория

```
ner-redaction/
├── configs/          # Конфигурации сущностей и модели
├── app/              # FastAPI приложение (REST API + веб-интерфейс)
├── src/ner_redaction/ # Основной пакет: pipeline, детекторы, анонимизатор
├── scripts/          # Скрипты подготовки данных, правил, обучения
├── notebooks/        # Jupyter-ноутбуки с экспериментами
├── data/             # Выборки данных и документация корпуса
└── models/           # Веса финальной модели
```

## Данные

Корпус (~1900 текстов, ~4000 размеченных сущностей) сформирован из открытых источников (Nerus, NEREL, FactRuEval-2016, WiNER, Wikipedia, Mokoron Russian Twitter) и синтетически сгенерированных примеров. Подробнее - в [`data/README.md`](data/README.md).

## Ноутбуки с экспериментами

Все этапы с проведенными экспериментами содержатся в Jupyter-ноутбуках в [`notebooks/`](notebooks/README.md): от сбора данных до дообучения модели и оценки гибридного подхода.


