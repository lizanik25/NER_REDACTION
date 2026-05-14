# Модели

## Структура

```
models/
├── README.md       
└── final_model/     - веса финальной гибридной модели
    ├── slovnet_ner.bin                              - дообученные веса Slovnet NER
    └── navec_hudlit_v1_12B_500K_250d_100q.tar      - эмбеддинги Navec
```

## Финальная модель

Гибридная модель состоит из двух компонентов:

**1. Дообученный Slovnet NER** - обрабатывает контекстно-зависимые классы PERSON и ADDRESS.

- Базовая модель: [Slovnet](https://github.com/natasha/slovnet) от Natasha
- Эмбеддинги: [Navec](https://github.com/natasha/navec) (`navec_hudlit_v1_12B_500K_250d_100q`)


**2. Rule-based детекторы** - обрабатывают форматные классы PHONE, EMAIL, ID.

- Код находится в `src/ner_redaction/rule_based.py`

## Загрузка весов

```bash
python scripts/download_assets.py
```

Скрипт загружает:
- `slovnet_ner.bin` - дообученные веса (из релизов репозитория)
- `navec_hudlit_v1_12B_500K_250d_100q.tar` - эмбеддинги Navec (~1.5 GB)

Веса модели не входят в репозиторий Git (добавлены в `.gitignore`). Каталог `final_model/` содержит `.gitkeep`.


