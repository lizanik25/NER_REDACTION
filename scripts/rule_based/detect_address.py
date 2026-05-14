import ast
import re
import time
import pandas as pd

from yargy import Parser, rule, or_
from yargy.predicates import eq, caseless, custom


ENTITY_CLASSES = ["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"]


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

    return df


def normalize_gold(label_list):
    return [
        {"start": int(x[0]), "end": int(x[1]), "label": x[2]}
        for x in label_list
    ]


def entity_key(ent):
    return (ent["start"], ent["end"], ent["label"])


def resolve_overlaps(entities):
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

class YargyAddressDetector:
    def __init__(self):
        EXTRA_PART = (
            r"(?:\s*,?\s*"
            r"(?:квартира|кв\.?|строение|стр\.?|корпус|корп\.?|офис|оф\.?|"
            r"владение|вл\.?|литер|лит\.?|к\.?)"
            r"\s*(?:\d{1,4}[А-Яа-яA-Za-z]?|[А-Яа-яA-Za-z])"
            r")*"
            r"(?:\s*,?\s*\d{1,2}\s*этаж)?"
        )

        STREET_TYPES = (
            r"ул\.?|улица|пр-кт|пр\.?|проспект|пр-т\.?|пер\.?|переулок|"
            r"наб\.?|б-р|бульвар|ш\.?|шоссе|тракт"
        )

        self.address_regexes = [
            r"(?:\d{6}\s*,\s*)?"
            r"(?:(?:Россия|МО|Московская\s+область|[А-ЯЁ][а-яё]+\s+(?:область|обл\.?|край|респ\.?|республика))\s*,\s*)?"
            r"(?:г\.?\s*)?[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,\s*"
            rf"(?:(?:{STREET_TYPES})\s*)?"
            r"[А-ЯЁA-Za-z0-9][А-ЯЁа-яёA-Za-z0-9\s\-]*?"
            r"\s*,\s*"
            r"(?:д\.?|дом)\s*\d{1,4}[А-Яа-яA-Za-z]?"
            + EXTRA_PART,

            r"(?:\d{6}\s*,\s*)?"
            r"(?:(?:Россия|МО|Московская\s+область|[А-ЯЁ][а-яё]+\s+(?:область|обл\.?|край|респ\.?|республика))\s*,\s*)?"
            r"г\.?\s*[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,\s*"
            r"[А-ЯЁ][А-ЯЁа-яёA-Za-z\-]*(?:\s+[А-ЯЁа-яёA-Za-z\-]+)?"
            r"\s*,\s*"
            r"(?:д\.?|дом)\s*\d{1,4}[А-Яа-яA-Za-z]?"
            + EXTRA_PART,

            r"г\.?\s*[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s+"
            rf"(?:(?:{STREET_TYPES})\s+)?"
            r"[А-ЯЁ][а-яёA-Za-z\-]+(?:\s+[А-ЯЁа-яёA-Za-z\-]+)?"
            r"\s+"
            r"(?:д\.?\s*)?\d{1,4}[А-Яа-яA-Za-z]?"
            r"(?:\s*(?:кв\.?|квартира|оф\.?|офис|стр\.?|строение|корп\.?|корпус|к\.?)"
            r"\s*(?:\d{1,4}[А-Яа-яA-Za-z]?|[А-Яа-яA-Za-z]))?",

            r"(?:адрес\s*:\s*)?"
            r"(?:г\.?\s*)?[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,?\s*"
            rf"(?:{STREET_TYPES})"
            r"\s*"
            r"[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,?\s*"
            r"(?:д\.?|дом)?\s*\d{1,4}[А-Яа-яA-Za-z]?"
            r"(?:\s*,?\s*(?:квартира|кв\.?|офис|оф\.?|строение|стр\.?|корпус|корп\.?|к\.?)"
            r"\s*(?:\d{1,4}[А-Яа-яA-Za-z]?|[А-Яа-яA-Za-z]))?",

            r"[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,\s*"
            rf"(?:{STREET_TYPES})"
            r"\s*[А-ЯЁ][а-яёA-Za-z\-]+"
            r"\s*,\s*д\.?\s*\d{1,4}[А-Яа-яA-Za-z]?"
            r"(?:\s*кв\.?\s*\d{1,4}[А-Яа-яA-Za-z]?)?",

            r"(?:(?:адрес|проживания|регистрации|корреспонденции|место\s+проживания)\s*:?\s*)"
            r"[А-ЯЁ][а-яёA-Za-z\-]+(?:\s+[А-ЯЁ][а-яёA-Za-z\-]+){0,3}"
            r"\s+\d{1,4}-\d{1,4}",

            r"[А-ЯЁ][а-яёA-Za-z\-]+(?:\s+[А-ЯЁа-яёA-Za-z\-]+){0,2}"
            r"\s+\d{1,4}[А-Яа-яA-Za-z]?"
            r"\s+(?:стр\.?|строение|корп\.?|корпус|к\.?|вл\.?|лит\.?|этаж)"
            r"\s*(?:\d{1,4}[А-Яа-яA-Za-z]?|[А-Яа-яA-Za-z])",

            r"[А-ЯЁ][а-яё\-]+\s+(?:обл\.?|область|край|респ\.?|республика)"
            r"\s*,\s*"
            r"[А-ЯЁ][а-яё\-]+\s+(?:р-н|район)"
            r"\s*,\s*"
            r"(?:пос\.?|пгт\.?|дер\.?|деревня|д\.?|с\.?|село)\s*[А-ЯЁ][а-яё\-]+"
            r"\s*,\s*"
            r"(?:д\.?|дом)\s*\d{1,4}[А-Яа-яA-Za-z]?",

            r"км\s*\d{1,4}\s+трассы\s+[A-ZА-ЯЁ]-?\d+\s+[А-ЯЁA-Zа-яёa-z\-]+"
            r"\s*,\s*"
            r"(?:владение|вл\.?)\s*\d{1,4}[А-Яа-яA-Za-z]?",

            r"\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?)"
            r"\s*,\s*(?:Apt|Apartment)\s*\w+"
            r"\s*,\s*[A-Z][a-z]+"
            r"\s*,\s*[A-Z]{2}"
            r"\s*,\s*\d{5}"
            r"\s*,\s*USA",

            r"ul\.?\s+[A-Z][a-z]+"
            r"\s*,\s*d\.?\s*\d+"
            r"\s*,\s*kv\.?\s*\d+"
            r"\s*,\s*[A-Z][a-z]+"
            r"\s*,\s*Russia",

            r"(?:дом\s+)?напротив\s+школы\s*№?\s*\d+",
            r"рядом\s+с\s+ТЦ\s+[\"«'][^\"»']+[\"»']",

            r"в/ч\s*\d{3,10}\s*,?\s*общежитие\s*№?\s*\d+",
        ]

        self.parser = None

    def trim_address_span(self, text: str, start: int, end: int):
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1] in ".,;:":
            end -= 1

        prefix_patterns = [
            r"^(?:фактический\s+адрес\s*:?\s*)",
            r"^(?:адрес\s+проживания\s*:?\s*)",
            r"^(?:адрес\s+регистрации\s*:?\s*)",
            r"^(?:адрес\s+для\s+корреспонденции\s*:?\s*)",
            r"^(?:адрес\s*:?\s*)",
            r"^(?:для\s+корреспонденции\s*:?\s*)",
            r"^(?:место\s+проживания\s*:?\s*)",
            r"^(?:места\s+проживания\s*:?\s*)",
            r"^(?:проживания\s*:?\s*)",
            r"^(?:регистрации\s*:?\s*)",
            r"^(?:корреспонденции\s*:?\s*)",
            r"^(?:этим\s+)",
            r"^(?:того\s+)",
        ]

        changed = True
        while changed:
            changed = False
            value = text[start:end]

            for pattern in prefix_patterns:
                m = re.match(pattern, value, flags=re.IGNORECASE)
                if m:
                    start += m.end()
                    changed = True
                    break

        return start, end

    def extend_address_left(self, text: str, start: int, end: int):
        left = text[max(0, start - 120):start]

        m = re.search(r"\d{6}\s*,\s*$", left)
        if m:
            start = start - (len(left) - m.start())

        left = text[max(0, start - 120):start]
        m = re.search(
            r"(?:Россия|МО|Московская\s+область|[А-ЯЁ][а-яё]+\s+(?:область|обл\.?|край|респ\.?|республика))\s*,\s*$",
            left,
            flags=re.IGNORECASE
        )
        if m:
            start = start - (len(left) - m.start())

        return start, end

    def extend_address_right(self, text: str, start: int, end: int):
        tail = text[end:end + 80]

        m = re.match(r"-\d{1,4}", tail)
        if m:
            end += m.end()
            tail = text[end:end + 80]

        m = re.match(
            r"(?:\s*,?\s*"
            r"(?:квартира|кв\.?|строение|стр\.?|корпус|корп\.?|офис|оф\.?|"
            r"владение|вл\.?|литер|лит\.?|к\.?)"
            r"\s*(?:\d{1,4}[А-Яа-яA-Za-z]?|[А-Яа-яA-Za-z])"
            r")+",
            tail,
            flags=re.IGNORECASE
        )

        if m:
            end += m.end()
            tail = text[end:end + 80]

        m = re.match(r"\s*,?\s*\d{1,2}\s*этаж", tail, flags=re.IGNORECASE)
        if m:
            end += m.end()
            tail = text[end:end + 80]

        value = text[start:end]
        bad_tail = re.search(
            r"(?:,\s*|\s+)(?:кр|кро|кром|кроме|корпу|корпус\s*$)$",
            value,
            flags=re.IGNORECASE
        )
        if bad_tail:
            end = start + bad_tail.start()

        value = text[start:end]
        stop = re.search(
            r"\s+(?:и|а также|наряду|вместе|дополнительно|номер|регистрационный|"
            r"идентификатор|телефон|email|почта|инн|договор|контракт|кроме|"
            r"используемые|предоставленные|подтвержденные|сохраненные|указанные)\b",
            value,
            flags=re.IGNORECASE
        )
        if stop:
            end = start + stop.start()

        while end > start and text[end - 1] in " ,.;:":
            end -= 1

        return start, end

    def is_bad_address_candidate(self, text: str, start: int, end: int) -> bool:
        value = text[start:end].strip()
        value_lower = value.lower()

        if len(value) < 5:
            return True

        if "гистрационный" in value_lower:
            return True

        if re.search(
            r"\b(?:контракт|контракта|контрактом|договор|договора|договором|заявка|"
            r"номер\s+документа|регистрационный\s+номер|id)\b",
            value_lower
        ):
            return True

        if re.search(
            r"\b(?:родился|родилась|рождения|года|лет|процент|процентов|секунды|"
            r"задержан|заместитель|министр|титул|турнире|календарь|яхту|стоимостью|"
            r"перевели|карта|карту|страны|главе|освободить|набрал)\b",
            value_lower
        ):
            return True

        left_context = text[max(0, start - 40):start].lower()
        if re.search(
            r"(?:контракт|контракта|контрактом|договор|договора|договором|"
            r"номер\s+документа|регистрационный\s+номер)\s*$",
            left_context
        ):
            return True

        if re.fullmatch(r"(?:д\.?|кв\.?|оф\.?|стр\.?|корп\.?|к\.?)\s*\d+", value_lower):
            return True

        markers = [
            "г.", "город", "ул", "улица", "проспект", "пр-кт", "пр-т", "пер", "переулок",
            "наб", "б-р", "тракт", "ш.", "шоссе", "дом", "д.", "кв", "оф",
            "стр", "строение", "корп", "к.", "обл", "край", "респ", "район", "р-н",
            "мо", "россия", "в/ч", "общежитие", "этаж", "владение", "литер",
            "street", "apt", "ul.", "kv.", "russia", "usa", "напротив", "рядом",
            "км", "трассы"
        ]

        has_marker = any(m in value_lower for m in markers)

        has_number_address = bool(
            re.search(r"[А-ЯЁA-Z][а-яёa-z\-]+\s+\d{1,4}(?:-\d{1,4})?", value)
        )

        if not has_marker and not has_number_address:
            return True

        return False

    def _add_entity(self, entities, text, start, end, source):
        start, end = self.trim_address_span(text, start, end)
        start, end = self.extend_address_left(text, start, end)
        start, end = self.extend_address_right(text, start, end)
        start, end = self.trim_address_span(text, start, end)

        if start >= end:
            return

        if self.is_bad_address_candidate(text, start, end):
            return

        entities.append({
            "start": start,
            "end": end,
            "label": "ADDRESS",
            "text": text[start:end],
            "source": source,
        })

    def predict_one(self, text: str):
        entities = []

        for pattern in self.address_regexes:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                self._add_entity(
                    entities,
                    text,
                    m.start(),
                    m.end(),
                    "regex_address",
                )

        return resolve_overlaps(entities)


def calculate_metrics(gold_all, pred_all, classes=ENTITY_CLASSES):
    per_class = {}
    total_tp = total_fp = total_fn = 0

    for cls in classes:
        tp = fp = fn = 0

        for gold, pred in zip(gold_all, pred_all):
            gold_cls = {entity_key(e) for e in gold if e["label"] == cls}
            pred_cls = {entity_key(e) for e in pred if e["label"] == cls}

            tp += len(gold_cls & pred_cls)
            fp += len(pred_cls - gold_cls)
            fn += len(gold_cls - pred_cls)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[cls] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_f1 = sum(per_class[c]["F1"] for c in classes) / len(classes)

    return {
        "per_class": per_class,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def metrics_to_table(metrics):
    rows = []

    for cls in ENTITY_CLASSES:
        m = metrics["per_class"][cls]
        rows.append({
            "Класс": cls,
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "TP": m["TP"],
            "FP": m["FP"],
            "FN": m["FN"],
        })

    return pd.DataFrame(rows)


def run_yargy_address_experiment(path: str):
    df = load_dataset(path)
    detector = YargyAddressDetector()

    texts = df["text"].astype(str).tolist()
    gold = [normalize_gold(x) for x in df["label"].tolist()]

    start_time = time.time()
    predictions = [detector.predict_one(text) for text in texts]
    elapsed = time.time() - start_time

    metrics = calculate_metrics(gold, predictions)
    class_table = metrics_to_table(metrics)

    summary = {
        "method": "Yargy-ADDRESS-only",
        "micro_precision": round(metrics["micro_precision"], 4),
        "micro_recall": round(metrics["micro_recall"], 4),
        "micro_f1": round(metrics["micro_f1"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "time_sec": round(elapsed, 4),
        "texts_per_sec": round(len(df) / elapsed, 4) if elapsed > 0 else None,
    }

    pred_rows = []

    for i, (text, preds, gold_labels) in enumerate(zip(texts, predictions, gold)):
        pred_rows.append({
            "id": i,
            "text": text,
            "gold": gold_labels,
            "predictions": preds,
        })

    predictions_df = pd.DataFrame(pred_rows)

    print(pd.DataFrame([summary]).to_string(index=False))

    print(class_table.to_string(index=False))

    return summary, class_table, predictions_df



def print_address_errors(predictions_df: pd.DataFrame):
    rows = []

    for _, row in predictions_df.iterrows():
        gold = [e for e in row["gold"] if e["label"] == "ADDRESS"]
        pred = [e for e in row["predictions"] if e["label"] == "ADDRESS"]

        gold_set = {entity_key(e) for e in gold}
        pred_set = {entity_key(e) for e in pred}

        fp = [p for p in pred if entity_key(p) not in gold_set]
        fn = [g for g in gold if entity_key(g) not in pred_set]

        if fp or fn:
            rows.append({
                "id": row["id"],
                "text": row["text"],
                "false_positive": fp,
                "false_negative": fn,
            })

    errors_df = pd.DataFrame(rows)


    for _, row in errors_df.iterrows():
        print("\n---")
        print(f"ID: {row['id']}")
        print(f"Текст: {row['text']}")

        if row["false_positive"]:
            print("FP:")
            for e in row["false_positive"]:
                print(f'  "{e["text"]}" [{e["start"]}:{e["end"]}] → ADDRESS')

        if row["false_negative"]:
            print("FN:")
            for e in row["false_negative"]:
                start, end = e["start"], e["end"]
                print(f'  "{row["text"][start:end]}" [{start}:{end}] → ADDRESS')

    return errors_df



