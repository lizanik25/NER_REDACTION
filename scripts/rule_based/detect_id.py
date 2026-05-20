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



class YargyIdDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE)
                )
            )

        DIG1_2 = digits_pred(r"\d{1,2}")
        DIG2 = digits_pred(r"\d{2}")
        DIG3 = digits_pred(r"\d{3}")
        DIG4 = digits_pred(r"\d{4}")
        DIG1_20 = digits_pred(r"\d{1,20}")
        DIG11 = digits_pred(r"\d{11}")
        DIG12 = digits_pred(r"\d{12}")

        RU_LETTERS = token_pred(r"[А-ЯЁA-Z]{1,6}")
        ALNUM = token_pred(r"[A-ZА-ЯЁ0-9]{2,20}")

        NUM_SIGN = eq("№")

        ID_MARKER = or_(
            rule(caseless("ID")),
            rule(caseless("id")),
            rule(caseless("user_id")),
        )

        ID_CONTEXT_MARKER = or_(
            rule(caseless("ID"), caseless("клиента")),
            rule(caseless("id"), caseless("клиента")),
            rule(caseless("клиента")),
            rule(caseless("регистрационный"), caseless("ID")),
            rule(caseless("регистрационный"), caseless("id")),
            rule(caseless("идентификатор")),
        )

        DOC_MARKER = or_(
            rule(caseless("договор")),
            rule(caseless("договором")),
            rule(caseless("контракт")),
            rule(caseless("контрактом")),
            rule(caseless("регистрационный"), caseless("контракт")),
            rule(caseless("заявка")),
            rule(caseless("заявки")),
            rule(caseless("заявок")),
            rule(caseless("номер"), caseless("заявки")),
            rule(caseless("номер"), caseless("заявок")),
            rule(caseless("номера"), caseless("заявок")),
            rule(caseless("регистрационные"), caseless("заявки")),
            rule(caseless("идентификаторы"), caseless("заявок")),
            rule(caseless("идентификатор"), caseless("заявки")),
            rule(caseless("документ")),
            rule(caseless("идентификатор")),
            rule(caseless("номер"), caseless("документа")),
            rule(caseless("номер"), caseless("договора")),
            rule(caseless("номер"), caseless("контракта")),
            rule(caseless("номер"), caseless("обращения")),
            rule(caseless("регистрационный"), caseless("номер")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("заявки")),
        )

        INN_MARKER = rule(caseless("ИНН"))
        SNILS_MARKER = rule(caseless("СНИЛС"))

        ACCOUNT_MARKER = or_(
            rule(caseless("счет")),
            rule(caseless("счёт")),
            rule(caseless("номер"), caseless("счета")),
            rule(caseless("номер"), caseless("счёта")),
            rule(caseless("номером"), caseless("счета")),
            rule(caseless("номером"), caseless("счёта")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счета")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счёта")),
            rule(caseless("р"), eq("/"), caseless("с")),
        )

        SIMPLE_ID = or_(
            rule(ID_MARKER, eq(":").optional(), eq("=").optional(), DIG1_20),
            rule(ID_CONTEXT_MARKER, eq(":").optional(), NUM_SIGN.optional(), DIG1_20),
            rule(NUM_SIGN, DIG1_20),
        )

        DOC_NUMBER_ALNUM = or_(
            rule(RU_LETTERS, eq("-"), DIG4, eq("/"), DIG1_2, eq("-"), DIG1_2),
            rule(DIG4, eq("-"), DIG1_20),
            rule(DIG3, eq("-"), DIG3, eq("-"), ALNUM),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, DIG2),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, eq("/"), DIG1_2),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, eq("-"), DIG2),
            rule(DIG1_2, eq("-"), DIG3, eq("-"), RU_LETTERS),
            rule(DIG1_20, eq("-"), DIG1_20),
            rule(DIG1_20, eq("-"), RU_LETTERS),
            rule(DIG1_20),
        )

        DOC_ID = or_(
            rule(DOC_MARKER, eq(":").optional(), NUM_SIGN.optional(), DOC_NUMBER_ALNUM),
            rule(NUM_SIGN, DOC_NUMBER_ALNUM),
        )

        INN_ID = rule(INN_MARKER, eq(":").optional(), DIG12)

        SNILS_FORMATTED = rule(
            SNILS_MARKER,
            eq(":").optional(),
            DIG3,
            eq("-"),
            DIG3,
            eq("-"),
            DIG3,
            DIG2,
        )

        SNILS_SOLID_WITH_MARKER = rule(
            SNILS_MARKER,
            eq(":").optional(),
            DIG11,
        )

        GOVERNMENT_ID = or_(
            INN_ID,
            SNILS_FORMATTED,
            SNILS_SOLID_WITH_MARKER,
        )

        BANK_ACCOUNT = rule(
            ACCOUNT_MARKER,
            eq(":").optional(),
            DIG1_20,
        )

        ENUM_WITH_MARKER = or_(
            rule(caseless("заявки"), eq(":").optional(), DIG1_20),
            rule(caseless("заявка"), eq(":").optional(), DIG1_20),
            rule(caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номера"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номер"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("идентификаторы"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("регистрационные"), caseless("заявки"), eq(":").optional(), DIG1_20),
        )

        ID = or_(
            GOVERNMENT_ID,
            BANK_ACCOUNT,
            DOC_ID,
            SIMPLE_ID,
            ENUM_WITH_MARKER,
        )

        self.parser = Parser(ID)

    def trim_id_span(self, text: str, start: int, end: int):
        value = text[start:end]

        patterns = [
            r"^(?:ID|id|user_id)\s*[:=]?\s*",
            r"^(?:клиента)\s*:?\s*№?\s*",
            r"^(?:ИНН|СНИЛС)\s*:?\s*",
            r"^(?:№)\s*",

            r"^(?:ом)\s*:?\s*№?\s*",
            r"^(?:а)\s+(?=\d)",

            r"^(?:документ)\s*:?\s*№?\s*",
            r"^(?:договор|договором|договора|контракт|контрактом|контракта|регистрационный\s+контракт)\s*:?\s*№?\s*",
            r"^(?:заявка|заявки|заявок|номер\s+заявки|номер\s+заявок|номера\s+заявок|регистрационные\s+заявки|идентификаторы\s+заявок|идентификатор\s+заявки|регистрационный\s+номер\s+заявки)\s*:?\s*№?\s*",
            r"^(?:номер\s+документа|номер\s+договора|номер\s+контракта|номер\s+обращения|регистрационный\s+номер)\s*:?\s*№?\s*",
            r"^(?:ID\s+клиента|id\s+клиента|регистрационный\s+ID|регистрационный\s+id|идентификатор)\s*:?\s*№?\s*",
            r"^(?:счет|счёт|счета|счёта|номером\s+счета|номером\s+счёта|номер\s+счета|номер\s+счёта|регистрационный\s+номер\s+счета|регистрационный\s+номер\s+счёта|р/с)\s*:?\s*",
        ]

        changed = True
        while changed:
            changed = False
            value = text[start:end]

            for pattern in patterns:
                m = re.match(pattern, value, flags=re.IGNORECASE)
                if m:
                    start += m.end()
                    changed = True
                    break

        return start, end

    def extend_id_span(self, text: str, start: int, end: int):
        value = text[start:end]

        if re.fullmatch(r"\d{1,4}", value):
            tail = text[end:end + 40]

            m = re.match(
                r"(?:-\d{1,6})?(?:-[A-ZА-ЯЁ0-9]{1,20})?(?:/\d{1,4})?",
                tail,
                flags=re.IGNORECASE
            )

            if m:
                end += m.end()

        return start, end

    def split_enumerated_ids(self, text: str, start: int, end: int):
        value = text[start:end]

        if re.fullmatch(r"\d{1,20}(?:\s*,\s*\d{1,20})+", value):
            return [
                (start + m.start(), start + m.end())
                for m in re.finditer(r"\d{1,20}", value)
            ]

        return [(start, end)]

    def _is_bad_id_candidate(self, text: str, start: int, end: int) -> bool:
        value = text[start:end]
        value_lower = value.lower()
        digits = re.sub(r"\D", "", value)

        left_context_80 = text[max(0, start - 80):start].lower()
        right_context_30 = text[end:end + 30]

        phone_context = re.search(
            r"(тел\.?|телефон|моб\.?|контактный номер|номер телефона|для связи)\s*[:\-]?\s*$",
            left_context_80,
            flags=re.IGNORECASE
        )

        if phone_context:
            return True

        bad_left_context = re.search(
            r"(школ[аеуыи]?|поликлиник[аеуыи]?|изолятор[аеуыи]?|общежити[ея]|"
            r"дом|кв\.?|квартира|офис|оф\.?|стр\.?|корп\.?|"
            r"рейс|шаг|этап)\s*$",
            left_context_80,
            flags=re.IGNORECASE
        )

        if bad_left_context:
            return True

        if re.fullmatch(r"\d{1,3}-\d{3}", value) and re.match(
            r"-\d{2,3}-\d{2}-\d{2}",
            right_context_30
        ):
            return True

        has_marker = re.search(
            r"(id|user_id|инн|снилс|№|договор|контракт|заявк|сч[её]т|р/с|номер|регистрационный|идентификатор|клиента)",
            value_lower + " " + left_context_80,
            flags=re.IGNORECASE
        )

        if len(digits) <= 3 and not has_marker:
            return True

        return False

    def _add_entity(self, entities, text, start, end, source):
        if self._is_bad_id_candidate(text, start, end):
            return

        start, end = self.trim_id_span(text, start, end)
        start, end = self.trim_id_span(text, start, end)
        start, end = self.extend_id_span(text, start, end)

        if start >= end:
            return

        for sub_start, sub_end in self.split_enumerated_ids(text, start, end):
            if self._is_bad_id_candidate(text, sub_start, sub_end):
                continue

            entities.append({
                "start": sub_start,
                "end": sub_end,
                "label": "ID",
                "text": text[sub_start:sub_end],
                "source": source,
            })

    def predict_one(self, text: str):
        entities = []

        for match in self.parser.findall(text):
            self._add_entity(
                entities,
                text,
                match.span.start,
                match.span.stop,
                "yargy_id",
            )

        for m in re.finditer(
            r"(?:идентификаторы|номера|номер|регистрационные)?\s*заяв(?:ок|ки)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
            text,
            flags=re.IGNORECASE
        ):
            nums_start = m.start(1)
            nums_part = m.group(1)

            for num in re.finditer(r"\d{1,20}", nums_part):
                sub_start = nums_start + num.start()
                sub_end = nums_start + num.end()

                entities.append({
                    "start": sub_start,
                    "end": sub_end,
                    "label": "ID",
                    "text": text[sub_start:sub_end],
                    "source": "yargy_id_enum",
                })

        for m in re.finditer(
            r"(?:идентификаторы|идентификатор)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
            text,
            flags=re.IGNORECASE
        ):
            nums_start = m.start(1)
            nums_part = m.group(1)

            for num in re.finditer(r"\d{1,20}", nums_part):
                sub_start = nums_start + num.start()
                sub_end = nums_start + num.end()

                entities.append({
                    "start": sub_start,
                    "end": sub_end,
                    "label": "ID",
                    "text": text[sub_start:sub_end],
                    "source": "yargy_id_enum",
                })

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



def run_yargy_id_experiment(path: str):
    df = load_dataset(path)
    detector = YargyIdDetector()

    texts = df["text"].astype(str).tolist()
    gold = [normalize_gold(x) for x in df["label"].tolist()]

    start_time = time.time()
    predictions = [detector.predict_one(text) for text in texts]
    elapsed = time.time() - start_time

    metrics = calculate_metrics(gold, predictions)
    class_table = metrics_to_table(metrics)

    summary = {
        "method": "Yargy-ID-only",
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



def print_id_errors(predictions_df: pd.DataFrame):
    rows = []

    for _, row in predictions_df.iterrows():
        gold = [e for e in row["gold"] if e["label"] == "ID"]
        pred = [e for e in row["predictions"] if e["label"] == "ID"]

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

        if row["false_positive"]:
            for e in row["false_positive"]:
                print(f'  "{e["text"]}" [{e["start"]}:{e["end"]}] → ID')

        if row["false_negative"]:
            for e in row["false_negative"]:
                start, end = e["start"], e["end"]
                print(f'  "{row["text"][start:end]}" [{start}:{end}] → ID')

    return errors_df



