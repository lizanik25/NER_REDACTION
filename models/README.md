# Модели

## Структура

```
models/
├── README.md
└── final_model/
    ├── slovnet_ner_pii_ru_hard_no_pd.tar
    └── navec_news_v1_1B_250K_300d_100q.tar
```

## Финальная модель

Гибридная модель состоит из двух компонентов:

**1. Дообученный Slovnet NER** - обрабатывает контекстно-зависимые классы PERSON и ADDRESS.

- Базовая модель: [Slovnet](https://github.com/natasha/slovnet) от Natasha
- Эмбеддинги: [Navec](https://github.com/natasha/navec) (`navec_news_v1_1B_250K_300d_100q`)


**2. Rule-based детекторы** - обрабатывают форматные классы PHONE, EMAIL, ID.

- Код находится в `src/ner_redaction/rule_based.py`

## Загрузка весов

```bash
python scripts/download_assets.py
```

Скрипт загружает:

- `slovnet_ner_pii_ru_hard_no_pd.tar` — дообученные веса Slovnet NER из GitHub Releases;
- `navec_news_v1_1B_250K_300d_100q.tar` — эмбеддинги Navec.

Файлы автоматически сохраняются в:

```text
models/final_model/
```

Веса модели не входят в Git-репозиторий (добавлены в `.gitignore`). В репозитории хранится только структура каталогов (`.gitkeep`).


