import random
import re
from collections import Counter
from dataclasses import dataclass
from faker import Faker
import pandas as pd

random.seed(42)
fake = Faker("ru_RU")
Faker.seed(42)

TOTAL_TEXTS = 300

TEXT_TYPE_DISTRIBUTION = {
    1: 90,
    2: 100,
    3: 80,
    4: 30,
}

REPEAT_SAME_TYPE_TARGET = (62, 74)
REPEAT_SAME_VALUE_TARGET = (20, 28)

TARGET_ENTITY_COUNTS = {
    "PHONE": 235,
    "EMAIL": 195,
    "ADDRESS": 175,
    "ID": 142,
}
TARGET_TOTAL_ENTITIES = sum(TARGET_ENTITY_COUNTS.values())

BASE_TYPE_PRIORITY = {
    "PHONE": 1.45,
    "EMAIL": 1.18,
    "ADDRESS": 0.88,
    "ID": 0.72,
}

@dataclass
class Entity:
    type: str
    value: str
    subtype: str
    repeated_same_value: bool = False


POPULAR_MALE_NAMES_EN = [
    "Mikhail", "Petr", "Valery", "Oleg", "Anatoly", "Matvey", "Daniil",
    "Denis", "Roman", "Semyon", "Gennady", "Boris", "Lev", "Anton",
    "Nikolay", "Yaroslav", "Vyacheslav", "Maxim", "Stepan", "Egor",
    "Artem", "Evgeny", "Alexey", "Alexander", "Igor", "Vasily", "Vladimir"
]

POPULAR_FEMALE_NAMES_EN = [
    "Maria", "Anna", "Ekaterina", "Tatiana", "Elizaveta", "Alexandra",
    "Victoria", "Yulia", "Polina", "Valentina", "Angelina", "Karina",
    "Marina", "Varvara", "Alina", "Irina", "Valeria", "Ulyana", "Mila",
    "Alevtina", "Milana", "Svetlana"
]

POPULAR_LAST_NAMES_EN = [
    "Ivanov", "Petrov", "Smirnov", "Kuznetsov", "Popov", "Vasilyev",
    "Sokolov", "Mikhailov", "Novikov", "Fedorov", "Morozov", "Volkov",
    "Alekseev", "Lebedev", "Semenov", "Egorov", "Pavlov", "Kozlov",
    "Stepanov", "Nikolaev", "Orlov", "Andreev", "Makarov", "Zakharov"
]

EMAIL_DOMAINS = [
    "mail.ru", "gmail.com", "yandex.ru", "company.ru", "site.org",
    "bank.ru", "corp.ru", "service.ru", "domain.com"
]

ROLE_EMAILS = [
    "support@company.ru", "hr@company.ru", "admin@site.org",
    "info@bank.ru", "contact@service.ru", "office@company.ru",
    "helpdesk@corp.ru"
]

CITIES = [
    "Москва", "Санкт-Петербург", "Казань", "Самара", "Екатеринбург",
    "Сочи", "Краснодар", "Балашиха", "Воронеж", "Владимир",
    "Уфа", "Ярославль", "Калуга", "Псков", "Богородицк",
    "Тула", "Тверь", "Рязань"
]

CITY_TO_REGION = {
    "Москва": "Московская область",
    "Санкт-Петербург": None,
    "Казань": "респ. Татарстан",
    "Самара": None,
    "Екатеринбург": None,
    "Сочи": "Краснодарский край",
    "Краснодар": "Краснодарский край",
    "Балашиха": "Московская область",
    "Воронеж": None,
    "Владимир": None,
    "Уфа": "респ. Башкортостан",
    "Ярославль": None,
    "Калуга": None,
    "Псков": None,
    "Богородицк": None,
    "Тула": None,
    "Тверь": None,
    "Рязань": None
}

VILLAGES = [
    "д. Митяево",
    "дер. Ивановка",
    "пос. Лазаревское",
    "село Отрадное",
    "СНТ Товарное",
    "д. Березовка",
    "пос. Зеленый",
    "село Никольское"
]

UL_NAMES = [
    "Ленина", "Тверская", "Мясницкая", "Калараш", "Центральная",
    "Советская", "Молодёжная", "Неделина", "Спортивная", "Беговая",
    "Зелёная", "Парковая", "Полевая", "Лесная"
]

PER_NAMES = [
    "Ленина", "Тверская", "Мясницкая", "Садовая", "Школьная", "Октябрьская"
]

NAB_NAMES = ["Набережная", "Речная", "Озерная"]
BLVD_NAMES = ["Ленина", "Тверская", "Победы", "Южный"]
SH_NAMES = ["Сибирское", "Пригородное", "Южное"]
PROSPECT_NAMES = ["Ленина", "Победы", "Центральный", "Московский"]

FULL_STREETS = [
    "Невский проспект",
    "Приморский проспект",
    "Болотный тракт",
    "Кутузовский проспект",
    "Ленинградский проспект"
]

ENTITY_TYPES = ["PHONE", "EMAIL", "ADDRESS", "ID"]

TEXT_CONTEXTS = [
    {
        "single_prefixes": [
            "В заявлении указано:",
            "В анкете клиента указано:",
            "В договоре зафиксировано:",
            "В служебной записке приведено:",
            "В обращении пользователя указано:",
            "В регистрационной форме указано:",
            "В карточке клиента указано:",
            "В материалах обращения приведено:",
            "В реквизитах стороны указано:",
            "Для обратной связи оставлено:",
            "В представленных документах содержится:",
            "В базе данных зафиксировано:",
            "В учетной записи указано:",
        ],
        "plural_prefixes": [
            "В заявлении указаны следующие данные:",
            "В анкете клиента содержатся сведения:",
            "В договоре зафиксированы следующие реквизиты:",
            "В служебной записке приведены данные:",
            "В обращении пользователя указаны следующие сведения:",
            "В регистрационной форме были указаны:",
            "В карточке клиента указаны:",
            "В материалах обращения приведены сведения:",
            "В реквизитах стороны указаны:",
            "Для обратной связи были оставлены данные:",
            "В представленных документах содержится информация:",
            "В базе данных зафиксированы сведения:",
            "В учетной записи указаны данные:",
        ],
        "single_endings": [
            ".",
            ", указанное при регистрации.",
            ", указанное в заявке.",
            ", указанное в профиле пользователя.",
            ", зафиксированное в системе.",
            ", предоставленное клиентом ранее.",
            ", актуальное на момент обращения.",
            ", используемое для связи.",
            ", указанное при оформлении договора.",
        ],
        "plural_endings": [
            ".",
            ", указанные при регистрации.",
            ", указанные в заявке.",
            ", указанные в профиле пользователя.",
            ", зафиксированные в системе.",
            ", предоставленные клиентом ранее.",
            ", актуальные на момент обращения.",
            ", используемые для связи.",
            ", указанные при оформлении договора.",
        ],
    },
    {
        "single_prefixes": [
            "Согласно предоставленной информации указано:",
            "Согласно анкете указано:",
            "Согласно договорным данным указано:",
            "По результатам рассмотрения указано:",
            "По данным, предоставленным клиентом, указано:",
            "В ходе обработки обращения получено:",
            "В рамках проверки выявлено:",
            "В профиле клиента указано:",
            "В карточке пользователя указано:",
            "В регистрационных данных клиента указано:",
        ],
        "plural_prefixes": [
            "Согласно предоставленной информации указаны:",
            "Согласно анкете указаны следующие сведения:",
            "Согласно договорным данным указаны:",
            "По результатам рассмотрения указаны:",
            "По данным, предоставленным клиентом, указаны:",
            "В ходе обработки обращения были получены данные:",
            "В рамках проверки были выявлены следующие данные:",
            "В профиле клиента указаны данные:",
            "В карточке пользователя указаны:",
            "В регистрационных данных клиента указаны:",
        ],
        "single_endings": [
            ".",
            ", требующее дополнительной проверки.",
            ", подтвержденное пользователем.",
            ", сохраненное в учетной записи.",
            ", предоставленное для связи.",
            ", внесенное в систему ранее.",
        ],
        "plural_endings": [
            ".",
            ", требующие дополнительной проверки.",
            ", подтвержденные пользователем.",
            ", сохраненные в учетной записи.",
            ", предоставленные для связи.",
            ", внесенные в систему ранее.",
        ],
    },
]

MIDDLE_TEMPLATES = [
    ", а также ",
    ", вместе с ",
    ", кроме того ",
    ", дополнительно ",
    ", в том числе ",
    ", среди прочего ",
    ", наряду с этим ",
    ", а именно ",
]

TRANSLIT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya"
}

def digits(n: int) -> str:
    return "".join(random.choices("0123456789", k=n))


def translit_ru_to_lat(text: str) -> str:
    text = text.strip().lower().replace("-", " ")
    result = []
    for ch in text:
        if ch in TRANSLIT_MAP:
            result.append(TRANSLIT_MAP[ch])
        elif ch.isalnum():
            result.append(ch)
        elif ch in {" ", "_"}:
            result.append("_")
    value = "".join(result)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_fake_postcode() -> str:
    raw = str(fake.postcode())
    digits_only = re.sub(r"\D", "", raw)
    if len(digits_only) >= 6:
        return digits_only[:6]
    if len(digits_only) > 0:
        return digits_only.zfill(6)
    return digits(6)


def safe_fake_login() -> str:
    login = fake.user_name().lower()
    login = re.sub(r"[^a-z0-9._-]", "", login)
    login = re.sub(r"[._-]{2,}", "_", login).strip("._-")
    if len(login) < 3:
        login = f"user{random.randint(100, 999)}"
    return login


def safe_uuid_chunk(length: int = 8) -> str:
    chunk = fake.uuid4().split("-")[0].lower()
    chunk = re.sub(r"[^a-f0-9]", "", chunk)
    if len(chunk) < length:
        chunk += "".join(random.choices("abcdef0123456789", k=length - len(chunk)))
    return chunk[:length]


def random_person_for_email():
    use_faker = random.random() < 0.45
    if use_faker:
        first = fake.first_name()
        last = fake.last_name()
        first_lat = translit_ru_to_lat(first)
        last_lat = translit_ru_to_lat(last)
        if first_lat and last_lat:
            return first_lat, last_lat

    first = random.choice(POPULAR_MALE_NAMES_EN + POPULAR_FEMALE_NAMES_EN).lower()
    last = random.choice(POPULAR_LAST_NAMES_EN).lower()
    return first, last


def get_region_for_city(city: str):
    return CITY_TO_REGION.get(city)


def choose_street():
    choice = random.random()
    if choice < 0.34:
        return f"ул. {random.choice(UL_NAMES)}"
    elif choice < 0.46:
        return f"пер. {random.choice(PER_NAMES)}"
    elif choice < 0.54:
        return f"наб. {random.choice(NAB_NAMES)}"
    elif choice < 0.62:
        return f"б-р {random.choice(BLVD_NAMES)}"
    elif choice < 0.70:
        return f"ш. {random.choice(SH_NAMES)}"
    elif choice < 0.80:
        return f"пр-кт {random.choice(PROSPECT_NAMES)}"
    else:
        return random.choice(FULL_STREETS)


def choose_address_core(city=None):
    city = city or random.choice(CITIES)
    street = choose_street()
    index = safe_fake_postcode() if random.random() < 0.45 else digits(6)

    house = random.randint(1, 120)
    flat = random.randint(1, 200)
    building = random.randint(1, 5)
    office = random.randint(1, 40)

    region = get_region_for_city(city)
    return {
        "city": city,
        "region": region,
        "street": street,
        "house": house,
        "flat": flat,
        "building": building,
        "office": office,
        "index": index,
    }


def clean_compact_street(street: str) -> str:
    return street.replace(" ", "")


def normalize_sentence_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+:", ":", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+;", ";", text)
    return text.strip()


def deduplicate_entities_preserve_order(entities):
    seen = set()
    result = []
    for e in entities:
        key = (e.type, e.value)
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def join_entities_varied(entities):
    parts = [e.value for e in entities]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} и {parts[1]}"

    result = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        if i == len(parts) - 1:
            connector = " и " if random.random() < 0.6 else random.choice(MIDDLE_TEMPLATES)
        else:
            connector = random.choice(MIDDLE_TEMPLATES)
        result += connector + part
    return result


def render_text(entities):
    entities = deduplicate_entities_preserve_order(entities)
    body = join_entities_varied(entities)

    n = len(entities)
    is_single = n == 1

    context = random.choice(TEXT_CONTEXTS)
    prefix = random.choice(context["single_prefixes"] if is_single else context["plural_prefixes"])
    ending = random.choice(context["single_endings"] if is_single else context["plural_endings"])

    text = f"{prefix} {body}"
    if ending and not text.endswith("."):
        text += ending
    return normalize_sentence_spacing(text)


def get_counts_from_rows(rows):
    counts = Counter()
    for row in rows:
        counts["PHONE"] += row["phones"]
        counts["EMAIL"] += row["emails"]
        counts["ADDRESS"] += row["addresses"]
        counts["ID"] += row["ids"]
    return counts


def deficit_for_type(entity_type, current_counts):
    return TARGET_ENTITY_COUNTS[entity_type] - current_counts.get(entity_type, 0)


def type_weight(entity_type, current_counts):
    deficit = deficit_for_type(entity_type, current_counts)
    base = BASE_TYPE_PRIORITY[entity_type]
    if deficit > 0:
        return base * (1.0 + deficit / max(TARGET_ENTITY_COUNTS[entity_type], 1) * 3.0)
    return base * 0.12


def weighted_sample_without_replacement(items, weights, k):
    chosen = []
    pool = list(zip(items, weights))
    while pool and len(chosen) < k:
        names = [x[0] for x in pool]
        ws = [max(0.0001, x[1]) for x in pool]
        pick = random.choices(names, weights=ws, k=1)[0]
        chosen.append(pick)
        pool = [x for x in pool if x[0] != pick]
    return chosen


def gen_phone(force_context=False):
    subtype_groups = [
        ("A1", 0.19), ("A2", 0.20), ("A3", 0.018), ("A4", 0.20),
        ("A5", 0.09), ("A6", 0.075), ("A7", 0.05), ("A8", 0.055),
        ("B1", 0.09), ("B2", 0.05), ("C1", 0.01), ("C2", 0.002),
        ("C3", 0.001), ("D", 0.105), ("E", 0.085), ("F", 0.006),
        ("G", 0.01), ("H", 0.003), ("I", 0.001)
    ]

    if force_context:
        subtype = random.choice(["E", "D", "G"])
    else:
        subtypes, weights = zip(*subtype_groups)
        subtype = random.choices(subtypes, weights=weights, k=1)[0]

    area = random.choice(["495", "499", "812", "926", "903", "916"])
    p2, p3, p4 = digits(3), digits(2), digits(2)

    if subtype == "A1":
        value = random.choice([f"+7{area}{p2}{p3}{p4}", f"8{area}{p2}{p3}{p4}", f"7{area}{p2}{p3}{p4}"])
    elif subtype == "A2":
        value = random.choice([f"+7 {area} {p2} {p3} {p4}", f"8 {area} {p2} {p3} {p4}", f"{area} {p2} {p3} {p4}"])
    elif subtype == "A3":
        value = random.choice([f"+7 {area} {digits(2)} {digits(2)} {digits(3)}", f"8 {area} {digits(4)} {digits(3)}"])
    elif subtype == "A4":
        value = random.choice([f"+7 ({area}) {p2}-{p3}-{p4}", f"8 ({area}) {p2}-{p3}-{p4}", f"({area}) {p2}-{p3}-{p4}"])
    elif subtype == "A5":
        value = random.choice([f"+7-{area}-{p2}-{p3}-{p4}", f"8-{area}-{p2}-{p3}-{p4}"])
    elif subtype == "A6":
        value = random.choice([f"+7.{area}.{p2}.{p3}.{p4}", f"8.{area}.{p2}.{p3}.{p4}"])
    elif subtype == "A7":
        value = random.choice([f"+7 ({area}) {p2} {p3}-{p4}", f"+7-{area} {p2}-{p3} {p4}"])
    elif subtype == "A8":
        value = random.choice([f"8({area}){p2}{p3}{p4}", f"+7({area}){p2}{p3}{p4}"])
    elif subtype == "B1":
        value = random.choice([f"{area}{p2}{p3}{p4}", f"{area} {p2} {p3} {p4}"])
    elif subtype == "B2":
        value = f"({area}) {p2}-{p3}-{p4}"
    elif subtype == "C1":
        value = random.choice([f"{p2}-{p3}-{p4}", f"{p2} {p3} {p4}"])
    elif subtype == "C2":
        value = random.choice([f"тел.: {digits(2)}-{digits(2)}-{digits(2)}", f"внутренний {digits(2)}-{digits(2)}-{digits(2)}"])
    elif subtype == "C3":
        value = random.choice([f"тел.: {digits(1)}-{digits(2)}-{digits(2)}", f"внутренний {digits(1)}-{digits(2)}-{digits(2)}"])
    elif subtype == "D":
        base = random.choice([
            f"+7 (495) {p2}-{p3}-{p4}",
            f"8-495-{p2}-{p3}-{p4}",
            f"+7495{p2}{p3}{p4}",
            f"8 (495) {p2}{p3}{p4}"
        ])
        ext = random.choice([f"доб. {digits(3)}", f"(вн. {digits(3)})", f"ext {digits(2)}", f"#{digits(3)}"])
        value = f"{base} {ext}"
    elif subtype == "E":
        prefix = random.choice(["тел.", "телефон:", "контактный номер:", "моб.:", "звонить по номеру", "наберите", "связаться по"])
        num = random.choice([f"+7{area}{p2}{p3}{p4}", f"8 ({area}) {p2}-{p3}-{p4}", f"8-{area}-{p2}-{p3}-{p4}"])
        value = f"{prefix} {num}"
    elif subtype == "F":
        num = f"+7{area}{p2}{p3}{p4}"
        value = random.choice([f"тел.{num}", f"Ivanov{num}"])
    elif subtype == "G":
        value = random.choice([
            f"тел. 8(495){p2}-{p3}-{p4}, {digits(2)}, {digits(2)}",
            f"тел.: {p2}-{p3}-{p4} / {digits(2)}",
            f"тел.: {p2}-{p3}-{p4} и {p2}-{p3}-{digits(2)}"
        ])
    elif subtype == "H":
        value = random.choice([
            "восемь девятьсот двадцать шесть сто двадцать три сорок пять шестьдесят семь",
            "8 (девятьсот двадцать шесть) 123 сорок пять 67"
        ])
    else:
        value = random.choice([f"8{area}-{p2}{p3}{p4}", f"+7 {area}{p2}{p3}{p4}"])

    return Entity("PHONE", value, subtype)


def gen_email():
    subtype_groups = [
        ("A", 0.55), ("B", 0.25), ("C", 0.14), ("H", 0.02),
        ("I", 0.015), ("K", 0.005), ("OTHER", 0.015)
    ]
    subtypes, weights = zip(*subtype_groups)
    subtype = random.choices(subtypes, weights=weights, k=1)[0]

    first_lat, last_lat = random_person_for_email()
    domain = random.choice(EMAIL_DOMAINS)
    login = safe_fake_login()

    if subtype == "A":
        value = random.choice([
            f"{last_lat}@{domain}",
            f"{first_lat}.{last_lat}@yandex.ru",
            f"{last_lat}_{first_lat}_{random.randint(1980, 2026)}@mail.ru",
            f"{login}@gmail.com",
            f"super-man-{random.randint(1, 999)}@mail.ru",
        ])
    elif subtype == "B":
        value = random.choice([
            f"{last_lat}.{first_lat}@company.ru",
            f"{first_lat[0]}.{last_lat}@domain.com",
            f"{last_lat}_{first_lat[0]}@mail.ru",
            f"{first_lat}_{last_lat}@mail.ru",
        ])
    elif subtype == "C":
        value = random.choice(ROLE_EMAILS)
    elif subtype == "H":
        core = random.choice([f"{last_lat}@mail.ru", f"{login}@gmail.com"])
        value = random.choice([f"({core})", f"[{core}]", f"{core}.", f"{core},"])
    elif subtype == "I":
        value = random.choice([f"{last_lat} @ mail . ru", f"{last_lat}@ mail.ru", f"{last_lat} @mail.ru"])
    elif subtype == "K":
        core = f"{last_lat}@mail.ru"
        value = random.choice([f"email:{core}", f"{core}тел"])
    else:
        value = random.choice([
            f"{last_lat}+urgent@gmail.com",
            f"{last_lat}+test1@company.ru",
            "ivan@pochta.rf",
            "director@ivanov.moscow",
            f"{last_lat}@localhost",
            "user@192.168.1.1",
            "noreply@domain.ru",
            "mailer-daemon@google.com",
            f"{last_lat}@sub.domain.com",
            f"{last_lat}@very-long-company-name.ru",
            "IVANOV@MAIL.RU",
            f"{last_lat}@gmal.com",
            f"{last_lat}@mail,ru",
        ])

    return Entity("EMAIL", value, subtype)


def gen_address():
    subtype_groups = [
        ("A", 0.44), ("B", 0.18), ("K", 0.15), ("L", 0.08),
        ("E", 0.035), ("OTHER_FULL", 0.115)
    ]
    subtypes, weights = zip(*subtype_groups)
    subtype = random.choices(subtypes, weights=weights, k=1)[0]

    core = choose_address_core()
    city = core["city"]
    region = core["region"]
    street = core["street"]
    house = core["house"]
    flat = core["flat"]
    building = core["building"]
    office = core["office"]
    index = core["index"]

    region_prefix = f"Россия, {region}, " if region else ""

    if subtype == "A":
        value = random.choice([
            f"{index}, г. {city}, {street}, д. {house}, стр. {building}, кв. {flat}",
            f"{region_prefix}г. {city}, {street}, д. {house}, кв. {flat}",
            f"г. {city}, {street}, д. {house}, оф. {office}",
        ])
    elif subtype == "B":
        value = random.choice([
            f"{city} {street} {house} квартира {flat}",
            f"{street} {house}-{flat}",
            f"г. {city}",
            f"{street}",
        ])
    elif subtype == "K":
        value = random.choice([
            f"{city} {street} д {house} кв {flat}",
            f"г {city} {street} {house}",
        ])
    elif subtype == "L":
        compact_street = clean_compact_street(street)
        value = random.choice([
            f"адрес:г.{city},{compact_street},д.{house}",
            f"{city},{compact_street},д.{house}кв.{flat}"
        ])
    elif subtype == "E":
        value = random.choice([
            'г. Сочи, пос. Лазаревское, ул. Калараш, гостевой дом "Светлана"',
            "МО, дер. Ивановка, 3-й дом от магазина",
            'рядом с ТЦ "Глобус"',
            "напротив школы №5"
        ])
    else:
        value = random.choice([
            f"{region if region else 'Московская область'}, Боровский р-н, {random.choice(VILLAGES)}, д. {house}",
            f"км {random.randint(1, 999)} трассы М-4 Дон, владение {random.randint(1, 10)}",
            f"г. Краснодар, Прикубанский округ, ГПЗ-24, уч. {random.randint(1, 50)}",
            f"в/ч {random.randint(10000, 99999)}, общежитие №{random.randint(1, 9)}",
            f"123 Main Street, Apt 4B, New York, NY, 10001, USA",
            f"ul. Lenina, d. {house}, kv. {flat}, Moscow, Russia",
            f"{index}, г. {city}, д. {house}А",
            f"{street}, корп. {random.randint(1, 5)}",
            f"{street}, стр. {random.randint(1, 5)}",
            f"{street}, вл. {random.randint(1, 20)}"
        ])

    return Entity("ADDRESS", value, subtype)


def is_repeatable_address(ent: Entity) -> bool:
    if ent.subtype not in {"A", "K", "L", "OTHER_FULL"}:
        return False
    if len(ent.value) < 18:
        return False
    if ent.value in {'рядом с ТЦ "Глобус"', "напротив школы №5"}:
        return False
    if re.fullmatch(r"(г\.\s*\w+|ул\.\s*.+|пер\.\s*.+|наб\.\s*.+|б-р\s*.+|ш\.\s*.+|пр-кт\s*.+)", ent.value.strip()):
        return False
    return True


def gen_repeatable_address():
    for _ in range(50):
        ent = gen_address()
        if is_repeatable_address(ent):
            return ent
    core = choose_address_core()
    return Entity("ADDRESS", f"г. {core['city']}, {core['street']}, д. {core['house']}, кв. {core['flat']}", "A")


def gen_id():
    subtype_groups = [
        ("A", 0.31), ("B", 0.43), ("C", 0.17), ("D", 0.09)
    ]
    subtypes, weights = zip(*subtype_groups)
    subtype = random.choices(subtypes, weights=weights, k=1)[0]

    uuid_num = str(int(safe_uuid_chunk(6), 16))[:6]

    if subtype == "A":
        value = random.choice([
            f"ID: {random.randint(10000, 99999)}",
            f"id={random.randint(10000, 99999)}",
            f"№{random.randint(10000, 99999)}",
            f"user_id:{random.randint(10000, 99999)}",
            f"id={uuid_num}",
        ])
    elif subtype == "B":
        value = random.choice([
            f"договор №АБ-{random.randint(2024, 2027)}/{random.randint(1, 12):02d}-{random.randint(1, 99):02d}",
            f"контракт {random.randint(10, 99)}-{random.randint(100, 999)}-В",
            f"№ {random.randint(100, 999)}-ФЗ",
            f"заявка №{random.randint(2024, 2027)}-{random.randint(1000, 9999)}"
        ])
    elif subtype == "C":
        value = random.choice([
            f"ИНН {''.join(random.choices('0123456789', k=12))}",
            f"ИНН:{''.join(random.choices('0123456789', k=12))}",
            f"{digits(3)}-{digits(3)}-{digits(3)} {digits(2)}"
        ])
    else:
        value = random.choice([
            f"р/с {''.join(random.choices('0123456789', k=20))}",
            f"номер счета {''.join(random.choices('0123456789', k=20))}"
        ])

    return Entity("ID", value, subtype)


def gen_repeat_id_for_label(label: str):
    if label == "договор":
        return Entity("ID", f"номер договора №АБ-{random.randint(2024, 2027)}/{random.randint(1, 12):02d}-{random.randint(1, 99):02d}", "repeat_id")
    if label == "заявка":
        return Entity("ID", f"номер заявки №{random.randint(2024, 2027)}-{random.randint(1000, 9999)}", "repeat_id")
    if label == "счет":
        return Entity("ID", f"номер счета {''.join(random.choices('0123456789', k=20))}", "repeat_id")
    if label == "инн":
        return Entity("ID", f"ИНН:{''.join(random.choices('0123456789', k=12))}", "repeat_id")
    return Entity("ID", f"ID клиента {random.randint(10000, 99999)}", "repeat_id")


GEN_MAP = {
    "PHONE": gen_phone,
    "EMAIL": gen_email,
    "ADDRESS": gen_address,
    "ID": gen_id,
}


def choose_entity_types(n_types: int, current_counts):
    items = ENTITY_TYPES[:]
    weights = [type_weight(t, current_counts) for t in items]
    return weighted_sample_without_replacement(items, weights, min(n_types, 4))


def instantiate_entities(type_list):
    return [GEN_MAP[t]() for t in type_list]


def count_by_type(entities):
    return Counter(e.type for e in entities)


def pick_type_for_repeat(candidate_types, current_counts):
    weights = [type_weight(t, current_counts) for t in candidate_types]
    return random.choices(candidate_types, weights=weights, k=1)[0]


def add_same_type_repeat(entities, current_counts):
    type_counts = count_by_type(entities)
    candidate_types = [t for t, c in type_counts.items() if c < 2]
    if not candidate_types:
        return entities

    chosen_type = pick_type_for_repeat(candidate_types, current_counts)

    if chosen_type == "PHONE":
        entities.append(gen_phone(force_context=False))
    elif chosen_type == "EMAIL":
        entities.append(gen_email())
    elif chosen_type == "ADDRESS":
        if deficit_for_type("ADDRESS", current_counts) > 0:
            entities.append(gen_repeatable_address())
    else:
        if deficit_for_type("ID", current_counts) > 0:
            entities.append(gen_id())

    return entities


def add_same_value_repeat(entities, current_counts):
    type_counts = count_by_type(entities)
    candidates = [e for e in entities if type_counts[e.type] < 2]
    if not candidates:
        return entities

    scored = []
    for e in candidates:
        if e.type == "ADDRESS" and len(e.value) < 18:
            continue
        if e.type == "EMAIL" and e.subtype in {"K", "I", "OTHER"}:
            continue
        if e.type == "PHONE" and e.subtype in {"C2", "C3", "H", "I"}:
            continue
        scored.append((e, type_weight(e.type, current_counts)))

    if not scored:
        return entities

    chosen = random.choices([x[0] for x in scored], weights=[x[1] for x in scored], k=1)[0]
    entities.append(Entity(chosen.type, chosen.value, chosen.subtype, repeated_same_value=True))
    return entities


def add_mandatory_repeat_case(entities, current_counts):
    existing = {e.type for e in entities}
    type_counts = count_by_type(entities)

    phone_need = deficit_for_type("PHONE", current_counts)
    email_need = deficit_for_type("EMAIL", current_counts)
    address_need = deficit_for_type("ADDRESS", current_counts)
    id_need = deficit_for_type("ID", current_counts)

    if "PHONE" in existing and type_counts["PHONE"] < 2 and phone_need > 0 and random.random() < 0.26:
        entities.append(gen_phone(force_context=True))
        return entities

    if "EMAIL" in existing and type_counts["EMAIL"] < 2 and email_need > 0 and random.random() < 0.16:
        entities.append(gen_email())
        return entities

    if "ADDRESS" in existing and type_counts["ADDRESS"] < 2 and address_need > 12 and random.random() < 0.08:
        entities.append(Entity("ADDRESS", f"адрес регистрации: {gen_repeatable_address().value}", "repeat_addr"))
        return entities

    if "ID" in existing and type_counts["ID"] < 2 and id_need > 8 and random.random() < 0.06:
        pair = random.choice(["договор", "заявка", "счет", "инн"])
        entities.append(gen_repeat_id_for_label(pair))
        return entities

    return entities


BAD_ADDRESS_PATTERNS = [
    r"пр-кт\s+Приморский проспект",
    r"пр-кт\s+Невский проспект",
    r"ул\.\s+Невский проспект",
    r"ул\.\s+Сибирский тракт",
    r"наб\.\s+Набережная",
    r"ш\.\s+Сибирское",
    r"ш\.\s+Пригородное",
]

BAD_REPEAT_ID_PATTERNS = [
    r"номер договора\s+user_id",
    r"номер заявки\s+договор",
    r"номер заявки\s+ИНН",
    r"номер договора\s+договор",
    r"номер заявки\s+заявка",
    r"номер счета\s+р/с",
]

BAD_EMAIL_PATTERNS = [
    r"@mail\.ruтел\b",
    r"@mail\.\s+ruтел\b",
]


def is_bad_example(text: str) -> bool:
    for p in BAD_ADDRESS_PATTERNS + BAD_REPEAT_ID_PATTERNS + BAD_EMAIL_PATTERNS:
        if re.search(p, text, flags=re.IGNORECASE):
            return True

    if text.count("адрес регистрации:") + text.count("адрес проживания:") > 2:
        return True

    if len(text) < 25 and text.count(",") > 3:
        return True

    if re.search(r"[.,]{2,}", text):
        return True

    phoneish = len(re.findall(r"(\+7|8\(|8-|тел\.|телефон:|моб\.:|звонить по номеру|связаться по)", text))
    idish = len(re.findall(r"(ID:|id=|user_id:|ИНН|№\s?\d|договор|контракт|заявка|р/с|номер счета)", text))
    if phoneish > 3 or idish > 4:
        return True

    if re.search(r"\b(договор №[^\s,]+).*\b\1\b", text):
        return True

    return False


def build_structure_plan():
    plan = []
    for n_types, count in TEXT_TYPE_DISTRIBUTION.items():
        plan.extend([n_types] * count)
    random.shuffle(plan)
    return plan


def trim_entities(entities, n_types, current_counts):
    result = []
    cnt = Counter()

    max_entities = 5 if n_types >= 3 else 4

    for e in entities:
        if len(result) >= max_entities:
            break

        max_per_type = 2
        if e.type == "ADDRESS" and deficit_for_type("ADDRESS", current_counts) <= 0:
            max_per_type = 1
        if e.type == "ID" and deficit_for_type("ID", current_counts) <= 0:
            max_per_type = 1

        if cnt[e.type] >= max_per_type:
            continue

        result.append(e)
        cnt[e.type] += 1

    return result


def try_generate_one(n_types, current_counts, repeat_same_type_budget, repeat_same_value_budget):
    entity_types = choose_entity_types(n_types, current_counts)
    entities = instantiate_entities(entity_types)

    if n_types >= 2 and repeat_same_type_budget > 0 and random.random() < 0.22:
        entities = add_same_type_repeat(entities, current_counts)
        repeat_same_type_budget -= 1

    if n_types >= 3 and repeat_same_value_budget > 0 and random.random() < 0.08:
        entities = add_same_value_repeat(entities, current_counts)
        repeat_same_value_budget -= 1

    if n_types >= 2:
        entities = add_mandatory_repeat_case(entities, current_counts)

    entities = trim_entities(entities, n_types, current_counts)
    entities = deduplicate_entities_preserve_order(entities)
    text = render_text(entities)

    return entities, text, repeat_same_type_budget, repeat_same_value_budget


def generate_dataset():
    plan = build_structure_plan()
    rows = []

    repeat_same_type_budget = random.randint(*REPEAT_SAME_TYPE_TARGET)
    repeat_same_value_budget = random.randint(*REPEAT_SAME_VALUE_TARGET)

    for n_types in plan:
        success = False
        current_counts = get_counts_from_rows(rows)

        for _attempt in range(10):
            entities, text, repeat_same_type_budget, repeat_same_value_budget = try_generate_one(
                n_types, current_counts, repeat_same_type_budget, repeat_same_value_budget
            )
            if not is_bad_example(text):
                success = True
                break

        if not success:
            continue

        rows.append({
            "text": text,
            "n_entity_types": len(set(e.type for e in entities)),
            "n_entities_total": len(entities),
            "phones": sum(e.type == "PHONE" for e in entities),
            "emails": sum(e.type == "EMAIL" for e in entities),
            "addresses": sum(e.type == "ADDRESS" for e in entities),
            "ids": sum(e.type == "ID" for e in entities),
            "entity_values": [e.value for e in entities],
            "entity_types": [e.type for e in entities],
            "entity_subtypes": [e.subtype for e in entities],
        })

    df = pd.DataFrame(rows)

    while len(df) < TOTAL_TEXTS:
        current_counts = get_counts_from_rows(df.to_dict("records"))

        n_types = random.choices(
            population=list(TEXT_TYPE_DISTRIBUTION.keys()),
            weights=[0.14, 0.28, 0.34, 0.24],
            k=1
        )[0]

        success = False
        for _attempt in range(10):
            entities, text, repeat_same_type_budget, repeat_same_value_budget = try_generate_one(
                n_types, current_counts, repeat_same_type_budget, repeat_same_value_budget
            )
            if not is_bad_example(text):
                success = True
                break

        if not success:
            continue

        new_row = {
            "text": text,
            "n_entity_types": len(set(e.type for e in entities)),
            "n_entities_total": len(entities),
            "phones": sum(e.type == "PHONE" for e in entities),
            "emails": sum(e.type == "EMAIL" for e in entities),
            "addresses": sum(e.type == "ADDRESS" for e in entities),
            "ids": sum(e.type == "ID" for e in entities),
            "entity_values": [e.value for e in entities],
            "entity_types": [e.type for e in entities],
            "entity_subtypes": [e.subtype for e in entities],
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df.head(TOTAL_TEXTS)


def validate_dataset(df):
    df = df.copy()

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[~df["text"].apply(is_bad_example)].reset_index(drop=True)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    print(df["n_entity_types"].value_counts().sort_index())

    total_phone = df["phones"].sum()
    total_email = df["emails"].sum()
    total_address = df["addresses"].sum()
    total_id = df["ids"].sum()
    total = total_phone + total_email + total_address + total_id

    if total > 0:
        print("PHONE:", total_phone, round(total_phone / total, 3))
        print("EMAIL:", total_email, round(total_email / total, 3))
        print("ADDRESS:", total_address, round(total_address / total, 3))
        print("ID:", total_id, round(total_id / total, 3))

    subtype_counter = Counter()
    for row in df["entity_subtypes"]:
        subtype_counter.update(row)

    for k, v in subtype_counter.most_common(20):
        print(f"{k}: {v}")

    for t in ENTITY_TYPES:
        print(f"{t}: target={TARGET_ENTITY_COUNTS[t]}, actual={df[t.lower() + 's' if t != 'ADDRESS' else 'addresses'].sum() if t != 'ID' else df['ids'].sum()}")

    return df


df = generate_dataset()
df = validate_dataset(df)
df.to_csv("synthetic_300_entities_v8_balanced_counts.csv", index=False, encoding="utf-8")
