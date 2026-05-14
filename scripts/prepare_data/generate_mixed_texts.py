import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
Faker.seed(SEED)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ensure_final_punct(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def cleanup_sentence(text: str) -> str:
    text = normalize_spaces(str(text))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace("« ", "«").replace(" »", "»")
    return ensure_final_punct(text)



def gen_phone() -> str:
    code = random.choice(["495", "499", "812", "916", "925", "926", "903", "985"])
    a, b, c = random.randint(100, 999), random.randint(10, 99), random.randint(10, 99)
    return random.choice([
        f"+7 {code} {a:03d}-{b:02d}-{c:02d}",
        f"8 ({code}) {a:03d}-{b:02d}-{c:02d}",
        f"+7{code}{a:03d}{b:02d}{c:02d}",
        f"8-{code}-{a:03d}-{b:02d}-{c:02d}",
    ])


def gen_email() -> str:
    domains = ["mail.ru", "gmail.com", "yandex.ru", "bk.ru", "inbox.ru"]
    first = random.choice(["ivan", "petr", "anna", "olga", "sergey", "alexey", "maria"])
    last  = random.choice(["ivanov", "petrov", "sidorov", "smirnov", "volkov", "popov"])
    num   = random.randint(1, 2026)
    local = random.choice([f"{first}.{last}", f"{last}{num}", f"{first}_{last}", first])
    return f"{local}@{random.choice(domains)}"


def gen_address() -> str:
    cities  = ["Москва", "Санкт-Петербург", "Казань", "Самара", "Краснодар", "Екатеринбург"]
    streets = ["Тверская", "Ленина", "Мясницкая", "Невский проспект", "Ново-Садовая"]
    city, street = random.choice(cities), random.choice(streets)
    house, flat, postal = random.randint(1, 120), random.randint(1, 250), random.randint(101000, 199999)
    return random.choice([
        f"{postal}, г. {city}, ул. {street}, д. {house}, кв. {flat}",
        f"г. {city}, ул. {street}, дом {house}, квартира {flat}",
        f"г. {city}, {street}, д. {house}",
    ])


def gen_id() -> str:
    letters = "".join(random.choices("АБВГДЕЖЗИКЛМНОПРСТУФХ", k=2))
    inn   = "".join(str(random.randint(0, 9)) for _ in range(12))
    snils = (f"{random.randint(100,999)}-{random.randint(100,999)}-"
             f"{random.randint(100,999)} {random.randint(10,99)}")
    return random.choice([
        f"ИНН {inn}", snils,
        f"договор №{letters}-{random.randint(2024,2026)}/{random.randint(1,12):02d}-{random.randint(1,99):02d}",
        f"заявка №{random.randint(2024,2026)}-{random.randint(1,99999):05d}",
    ])


GENERATORS = {"PHONE": gen_phone, "EMAIL": gen_email, "ADDRESS": gen_address, "ID": gen_id}

ENTITY_LABELS = {
    "PHONE":   ["Телефон", "Контактный телефон", "Тел."],
    "EMAIL":   ["Электронная почта", "Email", "Адрес электронной почты"],
    "ADDRESS": ["Адрес регистрации", "Адрес проживания", "Фактический адрес"],
    "ID":      ["Идентификатор", "Номер документа", "Регистрационный номер"],
}

SECTION_HEADERS = [
    "Контактные данные:", "Дополнительные сведения:",
    "Сведения для обратной связи:", "Персональные данные:",
]

DOC_TEMPLATES = [
    "Заявление\n\n{body_block}\n\n{fields_block}",
    "Обращение\n\n{body_block}\n\n{fields_block}",
    "Анкета\n\n{body_block}\n\n{fields_block}",
    "Карточка обращения\n\n{body_block}\n\n{fields_block}",
    "Пояснение\n\n{body_block}\n\n{fields_block}",
]

BODY_INTROS = [
    "В заявлении указано следующее: {body}",
    "В тексте обращения указано: {body}",
    "Заявитель сообщает следующее: {body}",
    "В разделе «Дополнительные сведения» указано: {body}",
    "Согласно представленным материалам: {body}",
]


def generate_fields_block(entities: list) -> str:
    lines = [random.choice(SECTION_HEADERS)]
    for ent in entities:
        label = random.choice(ENTITY_LABELS[ent["type"]])
        lines.append(f"{label}: {ent['value']}")
    return "\n".join(lines)


def render_mixed_text(real_text: str, entity_types: tuple) -> tuple:
    entities = [{"type": t, "value": GENERATORS[t]()} for t in entity_types]
    body   = random.choice(BODY_INTROS).format(body=cleanup_sentence(real_text))
    fields = generate_fields_block(entities)
    text   = random.choice(DOC_TEMPLATES).format(body_block=body, fields_block=fields)
    text   = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, entities



def make_combo_pool(n: int) -> list:
    w1, w2, w3 = int(n * 0.55), int(n * 0.30), int(n * 0.12)
    w4 = n - w1 - w2 - w3
    pool = (
        [("PHONE",)]   * (w1 // 4) +
        [("EMAIL",)]   * (w1 // 4) +
        [("ADDRESS",)] * (w1 // 4) +
        [("ID",)]      * (w1 - 3 * (w1 // 4)) +
        [("PHONE", "EMAIL")]   * (w2 // 3) +
        [("PHONE", "ID")]      * (w2 // 3) +
        [("ADDRESS", "EMAIL")] * (w2 - 2 * (w2 // 3)) +
        [("PHONE", "EMAIL", "ID")]      * (w3 // 2) +
        [("PHONE", "EMAIL", "ADDRESS")] * (w3 - w3 // 2) +
        [("PHONE", "EMAIL", "ADDRESS", "ID")] * w4
    )
    random.shuffle(pool)
    return pool


def build_mixed_dataset(person_df: pd.DataFrame, n_mixed: int) -> pd.DataFrame:
    person_df = (person_df
                 .dropna(subset=["text"])
                 .pipe(lambda df: df[df["text"].str.strip() != ""])
                 .reset_index(drop=True))

    if len(person_df) < n_mixed:
        person_df = person_df.sample(n=n_mixed, replace=True, random_state=SEED)
    else:
        person_df = person_df.sample(n=n_mixed, random_state=SEED)

    person_df = person_df.reset_index(drop=True)
    combos = make_combo_pool(n_mixed)[:n_mixed]

    rows = []
    for i, combo in enumerate(combos):
        real_text = person_df.loc[i, "text"]
        source    = str(person_df.loc[i, "source"]) if "source" in person_df.columns else "unknown"
        mixed_text, entities = render_mixed_text(real_text, combo)
        rows.append({
            "source":              f"{source} + synthetic",
            "text":                mixed_text,
            "added_types":         ",".join(sorted(combo)),
            "n_added_types":       len(combo),
            "added_entities_json": json.dumps(entities, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate mixed texts (PERSON + synthetic PII)")
    parser.add_argument("--person_data", default="data/interim/person_texts.csv")
    parser.add_argument("--synth_data",  default=None)
    parser.add_argument("--output",  default="data/interim/mixed_texts.csv")
    parser.add_argument("--n_mixed", type=int, default=950)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mixed_df = build_mixed_dataset(person_df, args.n_mixed)

    if args.synth_data:
        synth_df = pd.read_csv(args.synth_data)
        if "source" not in synth_df.columns:
            synth_df["source"] = "synthetic"
        final_df = pd.concat(
            [mixed_df[["source", "text"]], synth_df[["source", "text"]]],
            ignore_index=True
        )
        final_df = final_df[final_df["text"].str.strip() != ""].reset_index(drop=True)
    else:
        final_df = mixed_df


if __name__ == "__main__":
    main()
