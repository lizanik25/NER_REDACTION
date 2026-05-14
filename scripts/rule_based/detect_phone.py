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


def extend_phone_span(text: str, start: int, end: int):

    tail = text[end:end + 50]

    paren_ext = re.match(
        r"\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\)",
        tail,
        flags=re.IGNORECASE
    )

    ext = re.match(
        r"\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}",
        tail,
        flags=re.IGNORECASE
    )

    if paren_ext:
        end += paren_ext.end()
    elif ext:
        end += ext.end()

    tail = text[end:end + 40]

    list_tail = re.match(
        r"(?:\s*[,/]\s*\d{2}){1,4}",
        tail
    )

    if list_tail:
        end += list_tail.end()

    return start, end


def is_bad_phone_candidate(text: str, start: int, end: int, source: str = "") -> bool:
    value = text[start:end]
    digits = re.sub(r"\D", "", value)
    stripped = value.strip()

    left_context_15 = text[max(0, start - 15):start].lower()
    left_context_40 = text[max(0, start - 40):start].lower()

    phone_context = re.search(
        r"(тел\.?|телефон|моб\.?|номер телефона|контактный телефон|контактный номер|для связи)\s*[:\-]?\s*$",
        left_context_40,
        flags=re.IGNORECASE
    )

    bad_context = re.search(
        r"(номер документа|регистрационный номер|номер обращения|id клиента|user_id|id|инн|р/с|счет|номер счета|идентификатор)\s*[:=]?\s*$",
        left_context_40,
        flags=re.IGNORECASE
    )

    if "в/ч" in left_context_15:
        return True

    if bad_context and not phone_context:
        return True

    if stripped.isdigit() and len(digits) < 10:
        return True

    if stripped.isdigit() and len(digits) >= 11:
        if not re.match(r"^(?:\+?7|8)\d{10}$", stripped) and not phone_context:
            return True

    if re.fullmatch(r"\d{1,3}-\d{2}-\d{2}(?:\s*(?:[,/]\s*\d{2}){1,4})?", stripped):
        if not phone_context:
            return True

    if re.search(
        r"(код|артикул|номер договора|номер сч[её]та|номер заявки|код заказа|код операции)",
        left_context_40,
        flags=re.IGNORECASE
    ):
        if not re.search(
            r"(тел\.?|телефон|моб\.?|для связи)",
            left_context_40,
            flags=re.IGNORECASE
        ):
            return True

    return False



class YargyPhoneDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(
                        pattern,
                        token_value(t),
                        flags=re.IGNORECASE | re.VERBOSE
                    )
                )
            )

        DIG1 = digits_pred(r"\d{1}")
        DIG2 = digits_pred(r"\d{2}")
        DIG3 = digits_pred(r"\d{3}")
        DIG4 = digits_pred(r"\d{4}")
        DIG7 = digits_pred(r"\d{7}")
        DIG10 = digits_pred(r"\d{10}")
        DIG11 = digits_pred(r"\d{11}")
        DIG1_2 = digits_pred(r"\d{1,2}")
        DIG2_3 = digits_pred(r"\d{2,3}")
        DIG2_4 = digits_pred(r"\d{2,4}")
        DIG1_6 = digits_pred(r"\d{1,6}")
        DIG6 = digits_pred(r"\d{6}")

        PHONE_COMPACT_TOKEN = token_pred(
            r"""
            (?:
                \+?7[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                |
                8[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                |
                8\(\d{3}\)\d{7}
                |
                \+7\(\d{3}\)\d{7}
                |
                \d{3}-\d{3}-\d{3}\s?\d{2}
                |
                \d{4}-\d{7}
            )
            """
        )

        CODE_PAREN_TOKEN = token_pred(r"\(\d{3}\)")

        SEP = or_(rule(eq("-")), rule(eq(".")))

        PLUS7 = rule(eq("+"), eq("7"))

        PREFIX = or_(
            PLUS7,
            rule(eq("7")),
            rule(eq("8")),
        )

        CODE_PAREN = or_(
            rule(eq("("), DIG3, eq(")")),
            rule(CODE_PAREN_TOKEN),
        )

        CODE = or_(
            rule(DIG3),
            CODE_PAREN,
        )

        PHONE_SINGLE_TOKEN = rule(PHONE_COMPACT_TOKEN)

        PHONE_PLUS_SOLID = rule(eq("+"), DIG11)

        PHONE_BROKEN_PAREN = rule(
            PLUS7,
            eq("("),
            DIG10,
        )

        PHONE_DOT_FULL = rule(
            PLUS7,
            eq("."),
            DIG3,
            eq("."),
            DIG3,
            eq("."),
            DIG2,
            eq("."),
            DIG2,
        )

        PHONE_SOLID_11 = rule(DIG11)

        PHONE_8_3_4_3 = rule(
            eq("8"),
            DIG3,
            DIG4,
            DIG3,
        )

        PHONE_3_3_3_2 = rule(
            DIG3,
            eq("-"),
            DIG3,
            eq("-"),
            DIG3,
            DIG2,
        )

        PHONE_WITH_PREFIX = rule(
            PREFIX,
            SEP.optional(),
            CODE,
            SEP.optional(),
            DIG2_4,
            SEP.optional(),
            DIG1_2,
            SEP.optional(),
            DIG2_3,
        )

        PHONE_PREFIX_CODE_7DIG = rule(
            PREFIX,
            CODE,
            DIG7,
        )

        PHONE_PAREN_NO_PREFIX = rule(
            CODE_PAREN,
            DIG3,
            eq("-"),
            DIG2,
            eq("-"),
            DIG2,
        )

        PHONE_NO_PREFIX_GROUPED = rule(
            DIG3,
            DIG3,
            DIG2,
            DIG2,
        )

        PHONE_WITHOUT_PREFIX = or_(
            rule(DIG10),
            rule(
                CODE,
                SEP.optional(),
                DIG3,
                SEP.optional(),
                DIG2,
                SEP.optional(),
                DIG2,
            ),
        )

        LOCAL_PHONE = or_(
            rule(DIG3, eq("-"), DIG2, eq("-"), DIG2),
            rule(DIG2, eq("-"), DIG2, eq("-"), DIG2),
            rule(DIG1, eq("-"), DIG2, eq("-"), DIG2),
        )

        PHONE_WEIRD = or_(
            rule(DIG4, eq("-"), DIG7),
            rule(eq("8"), DIG3, SEP, DIG7),
            rule(eq("8"), DIG6, DIG2, DIG2),
            rule(PLUS7, DIG10),
        )

        LIST_TAIL = or_(
            rule(eq("/"), DIG2),
            rule(eq(","), DIG2),
            rule(eq(","), DIG2, eq(","), DIG2),
        )

        EXT_MARKER = or_(
            rule(caseless("доб")),
            rule(caseless("вн")),
            rule(caseless("ext")),
            rule(eq("#")),
        )

        EXTENSION = or_(
            rule(EXT_MARKER, eq(".").optional(), DIG1_6),
            rule(eq("("), caseless("вн"), eq(".").optional(), DIG1_6, eq(")")),
        )

        BASE_PHONE = or_(
            PHONE_SINGLE_TOKEN,
            PHONE_PLUS_SOLID,
            PHONE_BROKEN_PAREN,
            PHONE_DOT_FULL,
            PHONE_PREFIX_CODE_7DIG,
            PHONE_8_3_4_3,
            PHONE_3_3_3_2,
            PHONE_PAREN_NO_PREFIX,
            PHONE_NO_PREFIX_GROUPED,
            PHONE_WITH_PREFIX,
            PHONE_WEIRD,
            PHONE_WITHOUT_PREFIX,
            PHONE_SOLID_11,
            LOCAL_PHONE,
        )

        PHONE_WORDS = rule(
            caseless("восемь"),
            caseless("девятьсот"),
            caseless("двадцать"),
            caseless("шесть"),
            caseless("сто"),
            caseless("двадцать"),
            caseless("три"),
            caseless("сорок"),
            caseless("пять"),
            caseless("шестьдесят"),
            caseless("семь"),
        )

        PHONE = or_(
            rule(
                BASE_PHONE,
                LIST_TAIL.optional(),
                EXTENSION.optional(),
            ),
            PHONE_WORDS,
        )

        self.parser = Parser(PHONE)

        self.phone_regexes = [
            r"(?<!\d)(?:\+7|8)\.\d{3}\.\d{3}\.\d{2}\.\d{2}(?!\d)",

            r"(?<!\d)(?:\+7|8)\(\d{3}\)\d{7}(?!\d)",

            r"(?<!\d)(?:\+?7|8)\d{10}(?!\d)",

            r"(?<!\d)(?:\+7|8)[-\s]\d{3}[-\s]\d{3}[-\s]\d{2}[-\s]\d{2}(?!\d)",

            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{3}[\s-]\d{2}[\s-]\d{2}(?!\d)",

            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{7}(?!\d)",

            r"(?<!\d)(?:\+7|8|7)\s+\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",

            r"(?<!\d)\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",

            r"(?<!\d)\(\d{3}\)\s*\d{3}-\d{2}-\d{2}(?!\d)",

            r"(?<!\d)\+?7\d{10}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",

            r"(?<!\d)8\s*\(\d{3}\)\s*\d{7}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",

            r"(?<![\d-])\d{3}-\d{2}-\d{2}(?:\s*(?:[,/]\s*\d{2}){1,4})?(?![\d-])",

            r"(?<!\d)\d{1,2}-\d{2}-\d{2}(?!\d)",
        ]

    def _add_entity(self, entities, text, start, end, source):
        if is_bad_phone_candidate(text, start, end, source=source):
            return

        start, end = extend_phone_span(text, start, end)

        entities.append({
            "start": start,
            "end": end,
            "label": "PHONE",
            "text": text[start:end],
            "source": source,
        })

    def predict_one(self, text: str):
        entities = []

        for match in self.parser.findall(text):
            self._add_entity(
                entities=entities,
                text=text,
                start=match.span.start,
                end=match.span.stop,
                source="yargy",
            )

        for pattern in self.phone_regexes:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                self._add_entity(
                    entities=entities,
                    text=text,
                    start=m.start(),
                    end=m.end(),
                    source="regex_phone",
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



def run_yargy_phone_experiment(path: str):
    df = load_dataset(path)
    detector = YargyPhoneDetector()

    texts = df["text"].tolist()
    gold = [normalize_gold(x) for x in df["label"].tolist()]

    start_time = time.time()
    predictions = [detector.predict_one(text) for text in texts]
    elapsed = time.time() - start_time

    metrics = calculate_metrics(gold, predictions)
    class_table = metrics_to_table(metrics)

    summary = {
        "method": "Yargy-PHONE-plus-regex",
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



def print_phone_errors(predictions_df: pd.DataFrame):
    rows = []

    for _, row in predictions_df.iterrows():
        gold = [e for e in row["gold"] if e["label"] == "PHONE"]
        pred = [e for e in row["predictions"] if e["label"] == "PHONE"]

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
                print(f'  "{e["text"]}" [{e["start"]}:{e["end"]}] → PHONE')

        if row["false_negative"]:
            for e in row["false_negative"]:
                start, end = e["start"], e["end"]
                print(f'  "{row["text"][start:end]}" [{start}:{end}] → PHONE')

    return errors_df


