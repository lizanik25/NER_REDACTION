import ast
import re
import time
import pandas as pd

from yargy import Parser, rule, or_
from yargy.predicates import eq, custom


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



class YargyEmailDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE)
                )
            )

        LOCAL_PART = token_pred(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
        )

        DOMAIN_PART = token_pred(
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
        )

        TLD = token_pred(
            r"[A-ZА-ЯЁ0-9]{2,20}"
        )

        EMAIL_FULL_TOKEN = token_pred(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
            r"@"
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
            r"(?:\.[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*|,[A-ZА-ЯЁ]{2,15})+"
        )

        DOT_OR_COMMA = or_(rule(eq(".")), rule(eq(",")))

        EMAIL_BASIC = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL_MULTI_DOMAIN = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL_THREE_DOMAIN = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL = or_(
            rule(EMAIL_FULL_TOKEN),
            EMAIL_THREE_DOMAIN,
            EMAIL_MULTI_DOMAIN,
            EMAIL_BASIC,
        )

        self.parser = Parser(EMAIL)

        self.email_regex = re.compile(
            r"[A-ZА-ЯЁ0-9]+(?:\s*[._+\-]\s*[A-ZА-ЯЁ0-9]+)*"
            r"\s*@\s*"
            r"[A-ZА-ЯЁ0-9]+(?:\s*-\s*[A-ZА-ЯЁ0-9]+)*"
            r"(?:"
            r"\s*\.\s*[A-ZА-ЯЁ0-9]+(?:\s*-\s*[A-ZА-ЯЁ0-9]+)*"
            r"|,[A-ZА-ЯЁ]{2,15}"
            r")+",
            flags=re.IGNORECASE
        )

    def normalize_email_candidate(self, value: str) -> str:
        compact = re.sub(r"\s+", "", value)
        compact = compact.replace(",", ".")
        return compact

    def trim_email_span(self, text: str, start: int, end: int):
        value = text[start:end]

        m = re.search(r",\s+", value)
        if m:
            end = start + m.start()

        while end > start and text[end - 1] in ".,;:!?)]}":
            end -= 1

        return start, end

    def is_bad_email_candidate(self, text: str, start: int, end: int) -> bool:
        start, end = self.trim_email_span(text, start, end)
        value = text[start:end]
        compact = self.normalize_email_candidate(value)

        if start > 0:
            prev_char = text[start - 1]
            if re.match(r"[A-Za-zА-Яа-яЁё0-9._+\-]", prev_char):
                return True

        if end < len(text):
            next_char = text[end]
            if re.match(r"[A-Za-zА-Яа-яЁё0-9_\-]", next_char):
                return True

        if not re.fullmatch(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
            r"@"
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
            r"(?:\.[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*)+",
            compact,
            flags=re.IGNORECASE
        ):
            return True

        return False

    def extend_email_span(self, text: str, start: int, end: int):
        window_start = max(0, start - 100)
        window_end = min(len(text), end + 100)
        window = text[window_start:window_end]

        best = None

        for m in self.email_regex.finditer(window):
            candidate_start = window_start + m.start()
            candidate_end = window_start + m.end()
            candidate_start, candidate_end = self.trim_email_span(
                text,
                candidate_start,
                candidate_end
            )

            if candidate_start <= start and candidate_end >= end:
                if best is None or (candidate_end - candidate_start) > (best[1] - best[0]):
                    best = (candidate_start, candidate_end)

        if best:
            return best

        return self.trim_email_span(text, start, end)

    def _add_entity(self, entities, text, start, end, source):
        start, end = self.extend_email_span(text, start, end)
        start, end = self.trim_email_span(text, start, end)

        if self.is_bad_email_candidate(text, start, end):
            return

        entities.append({
            "start": start,
            "end": end,
            "label": "EMAIL",
            "text": text[start:end],
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
                "yargy_email",
            )

        for m in self.email_regex.finditer(text):
            self._add_entity(
                entities,
                text,
                m.start(),
                m.end(),
                "regex_email",
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



def run_yargy_email_experiment(path: str):
    df = load_dataset(path)
    detector = YargyEmailDetector()

    texts = df["text"].astype(str).tolist()
    gold = [normalize_gold(x) for x in df["label"].tolist()]

    start_time = time.time()
    predictions = [detector.predict_one(text) for text in texts]
    elapsed = time.time() - start_time

    metrics = calculate_metrics(gold, predictions)
    class_table = metrics_to_table(metrics)

    summary = {
        "method": "Yargy-EMAIL-only",
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


def print_email_errors(predictions_df: pd.DataFrame):
    rows = []

    for _, row in predictions_df.iterrows():
        gold = [e for e in row["gold"] if e["label"] == "EMAIL"]
        pred = [e for e in row["predictions"] if e["label"] == "EMAIL"]

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
                print(f'  "{e["text"]}" [{e["start"]}:{e["end"]}] → EMAIL')

        if row["false_negative"]:
            print("FN:")
            for e in row["false_negative"]:
                start, end = e["start"], e["end"]
                print(f'  "{row["text"][start:end]}" [{start}:{end}] → EMAIL')

    return errors_df



