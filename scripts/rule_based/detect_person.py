import argparse
import ast
import re
import time
from pathlib import Path

import pandas as pd
from yargy import Parser, or_, rule
from yargy.predicates import caseless, custom, eq

ENTITY_CLASSES = ["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"]


def resolve_overlaps(entities: list) -> list:
    entities = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"]))
    )
    selected = []
    for ent in entities:
        has_overlap = False
        for old in selected[:]:
            if not (ent["end"] <= old["start"] or ent["start"] >= old["end"]):
                has_overlap = True
                if (ent["end"] - ent["start"]) > (old["end"] - old["start"]):
                    selected.remove(old)
                    selected.append(ent)
                break
        if not has_overlap:
            selected.append(ent)
    return sorted(selected, key=lambda e: e["start"])


class YargyPersonDetector:

    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def token_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        CAP_WORD = token_pred(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")
        INITIAL = token_pred(r"[А-ЯЁ]")

        PARTICLE = or_(
            rule(caseless("оглы")),
            rule(caseless("кызы")),
            rule(caseless("ибн")),
            rule(caseless("бен")),
            rule(caseless("бин")),
            rule(caseless("аль")),
            rule(caseless("де")),
            rule(caseless("ле")),
            rule(caseless("фон")),
            rule(caseless("ван")),
        )

        FULL_3 = rule(CAP_WORD, CAP_WORD, CAP_WORD)
        FULL_2 = rule(CAP_WORD, CAP_WORD)

        INITIALS_BEFORE = rule(
            INITIAL, eq("."),
            INITIAL.optional(), eq(".").optional(),
            CAP_WORD
        )

        INITIALS_AFTER = rule(
            CAP_WORD,
            INITIAL, eq("."),
            INITIAL.optional(), eq(".").optional()
        )

        PARTICLE_NAME = rule(
            CAP_WORD,
            CAP_WORD.optional(),
            PARTICLE,
            CAP_WORD.optional()
        )

        PERSON = or_(
            PARTICLE_NAME,
            INITIALS_BEFORE,
            INITIALS_AFTER,
            FULL_3,
            FULL_2,
        )

        self.parser = Parser(PERSON)

        self.person_regexes = [
            r"\b[А-ЯЁ]\s*\.\s*[А-ЯЁ]\s*\.?\s*"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ]\s*\.\s*"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ]\s*\.\s*[А-ЯЁ]\s*\.\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ]\.?\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"(?:оглы|кызы)\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"(?:ибн|бен|бин)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
            r"(?:\s+(?:Аль|аль)\s+[А-ЯЁ][а-яё]+)?"
            r"(?:\s+[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"(?:ибн|бен|бин)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"(?:аль|Аль)-[А-ЯЁ][а-яё]+\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"(?:де|ле|фон|ван|Ле|Ла|Де|Дю)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ][а-яё]+-[А-ЯЁ][а-яё]+\s+"
            r"(?:де\s+)?[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b",

            r"\b(?:президент|министр|депутат|сенатор|мэр|судья|адвокат|прокурор|"
            r"губернатор|директор|руководитель|председатель|секретарь|"
            r"бизнесмен|предприниматель|космонавт|артист|актер|актриса|"
            r"тренер|спортсмен|журналист|писатель|поэт|генерал|"
            r"генерал-майор|генерал-лейтенант|полковник|вице-адмирал|"
            r"свидетель|обвиняемый|обвиняемая|подсудимый|подсудимая|"
            r"гражданин|гражданка|заявитель|заявительница|потерпевший|потерпевшая|"
            r"докладчик|докладчица|спикер|премьер|премьер-министр|"
            r"биатлонист|фигурист|математик|врач|певица|киноактриса|"
            r"житель|жителя|жительница|жительницу|эксперт|методист|"
            r"наркоторговец|заместитель)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
            r"(?:\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?){0,2}\b",

            r"\b(?:отец|мать|сын|дочь|брат|сестра|супруг|супруга|жена|муж|"
            r"родитель|опекун|бабушка|дедушка|внучка|внук)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
            r"(?:\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?){0,2}\b",

            r"\b(?:по словам|сообщил|сообщила|заявил|заявила|отметил|отметила|"
            r"рассказал|рассказала|прокомментировал|принял участие|дала интервью|"
            r"дал интервью|написал|написала|обсудил|спросил|ответил|"
            r"назначить|уволить|сам|бывший|новый)\s+"
            r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
            r"(?:\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?){0,2}\b",
        ]

        self.prefix_strip_regex = re.compile(
            r"^(?:президент|министр|депутат|сенатор|мэр|судья|адвокат|прокурор|"
            r"губернатор|директор|руководитель|председатель|секретарь|"
            r"бизнесмен|предприниматель|космонавт|артист|актер|актриса|"
            r"тренер|спортсмен|журналист|писатель|поэт|генерал|"
            r"генерал-майор|генерал-лейтенант|полковник|вице-адмирал|"
            r"свидетель|обвиняемый|обвиняемая|подсудимый|подсудимая|"
            r"гражданин|гражданка|заявитель|заявительница|потерпевший|потерпевшая|"
            r"докладчик|докладчица|спикер|премьер|премьер-министр|"
            r"биатлонист|фигурист|математик|врач|певица|киноактриса|"
            r"житель|жителя|жительница|жительницу|эксперт|методист|"
            r"наркоторговец|заместитель|бывший|новый|сам|"
            r"назначить|уволить|"
            r"отец|мать|сын|дочь|брат|сестра|супруг|супруга|жена|муж|"
            r"родитель|опекун|бабушка|дедушка|внучка|внук|"
            r"по словам|сообщил|сообщила|заявил|заявила|отметил|отметила|"
            r"рассказал|рассказала|прокомментировал|принял участие|дала интервью|"
            r"дал интервью|написал|написала|обсудил|спросил|ответил)\s+",
            flags=re.IGNORECASE
        )

        self.bad_exact = {
            "Гран При", "Гран При Мексики", "Российская Федерация",
            "Российской Федерации", "Совет Федерации", "Совета Федерации",
            "Сбербанк Управление", "Сбербанк Управление Активами",
            "Пенсионный фонд", "РИА Новости", "Говорит Москва",
            "Golden Spin", "Fitness Balance", "Calories Tracker",
            "За права", "День судьи", "Пояснение Из", "На Кавминводах",
            "Шадринске Курганской", "Стратегия Будущего",
            "Московской Хельсинкской", "Войска Калининского",
            "Республики Бурятия", "Второй Опиумной", "Первым Чрезвычайным",
            "Верховным Главнокомандующим", "Совете Безопасности",
            "Южной Америки", "Южной Корее", "История Российская",
            "Дублина Графтон", "Эстонской Республике",
            "Академика Сахарова", "Генштаба Вооружённых",
        }

        self.bad_tokens = {
            "России", "Российской", "Федерации", "США", "РФ",
            "Минэнерго", "Минобороны", "Госдумы", "МВД", "КНДР",
            "МОК", "РАН", "АО", "ООО", "ЗАО", "ПАО",
            "Сбербанк", "Управление", "Активами",
            "Совета", "Федерации", "Эквадора", "Азербайджана",
            "Армении", "Катара", "Калининграда", "Вашингтон",
            "Москвы", "Парижа", "Сингапуре", "Тобольске",
            "Франции", "Венесуэлы", "Аргентины", "Минска",
            "Курганской", "Шадринске", "Кавминводах",
            "Республики", "Бурятия", "Саудовской", "Аравии",
            "Южной", "Америки", "Корее", "Баумана",
        }

    def trim_person_span(self, text: str, start: int, end: int):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1] in ",;:!?()[]«»\"'":
            end -= 1
        if end > start and text[end - 1] == ".":
            value = text[start:end]
            if not re.search(r"(?:[А-ЯЁ]\s*\.\s*){1,2}$", value):
                end -= 1
        changed = True
        while changed:
            changed = False
            value = text[start:end]
            m = self.prefix_strip_regex.match(value)
            if m:
                start += m.end()
                while start < end and text[start].isspace():
                    start += 1
                changed = True
        return start, end

    def strip_bad_left_tokens(self, text: str, start: int, end: int):
        value = text[start:end]
        tokens = list(re.finditer(r"\S+", value))
        while len(tokens) >= 2:
            first = tokens[0].group().strip(".,;:!?()[]«»\"'")
            if first in self.bad_tokens:
                start += tokens[0].end()
                while start < end and text[start].isspace():
                    start += 1
                value = text[start:end]
                tokens = list(re.finditer(r"\S+", value))
            else:
                break
        return start, end

    def is_bad_person_candidate(self, text: str, start: int, end: int) -> bool:
        value = text[start:end].strip()
        value_lower = value.lower()

        if len(value) < 2:
            return True
        if re.search(r"\d", value):
            return True
        if not re.search(r"[А-ЯЁ]", value):
            return True

        normalized_value = re.sub(r"\s+", " ", value)
        if normalized_value in self.bad_exact:
            return True

        bad_words = [
            "обращение", "заявление", "сведения", "анкета", "карточка",
            "адрес", "телефон", "почта", "email", "номер", "документ",
            "регистрационный", "контактные", "данные", "реквизиты",
            "улица", "проспект", "тракт", "дом", "квартира", "корпус",
            "сведения для", "дополнительные сведения", "фонд", "агентство",
            "министерство", "университет", "школа", "банк", "концерн",
            "правительство", "парламент", "суд", "область", "район",
            "войска", "республика", "республики", "история", "совет",
            "безопасности", "главнокомандующий", "главнокомандующим",
            "коллегия", "генштаб", "южной", "северной", "западной",
            "восточной",
        ]
        if any(x in value_lower for x in bad_words):
            return True

        if re.search(
            r"\b(?:ул|улица|проспект|пр-кт|пер|переулок|наб|б-р|дом|д\.|"
            r"кв|оф|корп|стр|обл|область|край|респ|район|р-н|инн|id|"
            r"ооо|ао|пао|зао|банк|фонд|университет|агентство|министерство|"
            r"совет|безопасности|войска|республика|генштаб|коллегия)\b",
            value_lower
        ):
            return True

        tokens = normalized_value.split()
        if len(tokens) > 1 and any(t in self.bad_tokens for t in tokens):
            return True

        if len(tokens) > 3 and not re.search(
            r"\b(?:оглы|кызы|ибн|бен|бин|аль|де|ле|фон|ван)\b",
            value_lower
        ):
            return True

        if len(tokens) == 1:
            left = text[max(0, start - 70):start].lower()
            right = text[end:min(len(text), end + 70)].lower()
            strong_context = re.search(
                r"(?:по словам|заявил|заявила|сообщил|сообщила|"
                r"отметил|отметила|рассказал|рассказала|"
                r"подозревается|обвинили|назначен|выступил|"
                r"выступила|уличил|пожаловался|вручил|"
                r"адвоката|жена|дочь|супруг|супруга|свидетель|"
                r"подсудимая|подсудимый|гражданин|гражданка|"
                r"россиянки|бразильца|принц|глава|лидера|имя)\s+$",
                left
            )
            strong_right = re.search(
                r"^\s*(?:уличил|пожаловался|занимала|выступил|"
                r"выступила|поддержал|создал|отрицает|"
                r"сообщил|сообщила|подтвердила|присутствовала|"
                r"вышел|на должность)",
                right
            )
            if not strong_context and not strong_right:
                return True

        return False

    def _add_entity(self, entities: list, text: str, start: int, end: int, source: str):
        start, end = self.trim_person_span(text, start, end)
        start, end = self.strip_bad_left_tokens(text, start, end)
        start, end = self.trim_person_span(text, start, end)
        if start >= end:
            return
        if self.is_bad_person_candidate(text, start, end):
            return
        entities.append({
            "start": start,
            "end": end,
            "label": "PERSON",
            "text": text[start:end],
            "source": source,
        })

    def predict_one(self, text: str) -> list:
        entities = []
        for match in self.parser.findall(text):
            self._add_entity(entities, text, match.span.start, match.span.stop, "yargy_person")
        for pattern in self.person_regexes:
            for m in re.finditer(pattern, text):
                self._add_entity(entities, text, m.start(), m.end(), "regex_person")
        return resolve_overlaps(entities)

    def predict_batch(self, texts: list) -> list:
        return [self.predict_one(t) for t in texts]


def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if "text" not in df.columns:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)
    if "label" in df.columns:
        df["label"] = df["label"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    else:
        df["label"] = [[] for _ in range(len(df))]
    return df.dropna(subset=["text"]).reset_index(drop=True)


def normalize_gold(label_list) -> list:
    if label_list is None or (isinstance(label_list, float) and pd.isna(label_list)):
        return []
    if isinstance(label_list, str):
        label_list = ast.literal_eval(label_list)
    return [{"start": int(x[0]), "end": int(x[1]), "label": x[2]} for x in label_list]


def entity_key(ent: dict) -> tuple:
    return (ent["start"], ent["end"], ent["label"])


def evaluate_person(df: pd.DataFrame, name: str = "test") -> dict:
    detector = YargyPersonDetector()
    texts = df["text"].astype(str).tolist()
    gold = [normalize_gold(x) for x in df["label"]]

    t0 = time.time()
    preds = detector.predict_batch(texts)
    elapsed = time.time() - t0

    tp = fp = fn = 0
    for g, p in zip(gold, preds):
        g_set = {entity_key(e) for e in g if e["label"] == "PERSON"}
        p_set = {entity_key(e) for e in p if e["label"] == "PERSON"}
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_no_pii(df: pd.DataFrame) -> dict:
    detector = YargyPersonDetector()
    texts = df["text"].astype(str).tolist()
    preds = detector.predict_batch(texts)
    total_fp = sum(len(p) for p in preds)
    texts_with_fp = sum(1 for p in preds if p)
    return {"texts_with_fp": texts_with_fp, "total_fp": total_fp}


def main():
    parser = argparse.ArgumentParser(description="Rule-based PERSON detector evaluation")
    parser.add_argument("--test", default="data/processed/test_sample.csv")
    parser.add_argument("--no_pii", default="data/processed/no_pii_sample.csv")
    parser.add_argument("--tricky", default="data/processed/tricky_sample.csv")
    parser.add_argument("--output", default="outputs/person_results/")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    evaluate_person(load_dataset(args.test), "test")
    evaluate_no_pii(load_dataset(args.no_pii))
    evaluate_person(load_dataset(args.tricky), "tricky")


if __name__ == "__main__":
    main()
