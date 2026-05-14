import ast
import time
import pandas as pd


ENTITY_CLASSES = ["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"]


def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")

        if "text" not in df.columns:
            df = pd.read_csv(path)

    except Exception:
        df = pd.read_csv(path)

    if "label" not in df.columns:
        df["label"] = [[] for _ in range(len(df))]
    else:
        df["label"] = df["label"].fillna("[]")

    return df


def normalize_gold(label_list):

    if label_list is None:
        return []

    if isinstance(label_list, float):
        return []

    if isinstance(label_list, str):
        label_list = label_list.strip()

        if label_list == "" or label_list == "[]":
            return []

        try:
            label_list = ast.literal_eval(label_list)
        except Exception:
            return []

    if not isinstance(label_list, list):
        return []

    result = []

    for x in label_list:

        if not isinstance(x, (list, tuple)):
            continue

        if len(x) < 3:
            continue

        result.append({
            "start": int(x[0]),
            "end": int(x[1]),
            "label": str(x[2]),
        })

    return result


def entity_key(ent):
    return (ent["start"], ent["end"], ent["label"])


def resolve_overlaps_global(entities):

    priority = {
        "EMAIL": 5,
        "PHONE": 4,
        "ID": 3,
        "ADDRESS": 2,
        "PERSON": 1,
    }

    entities = sorted(
        entities,
        key=lambda e: (
            e["start"],
            -(e["end"] - e["start"]),
            -priority.get(e["label"], 0)
        )
    )

    selected = []

    for ent in entities:
        keep = True

        for old in selected[:]:
            overlap = not (
                ent["end"] <= old["start"] or ent["start"] >= old["end"]
            )

            if overlap:
                ent_len = ent["end"] - ent["start"]
                old_len = old["end"] - old["start"]

                ent_score = (ent_len, priority.get(ent["label"], 0))
                old_score = (old_len, priority.get(old["label"], 0))

                if ent_score > old_score:
                    selected.remove(old)
                else:
                    keep = False

                break

        if keep:
            selected.append(ent)

    return sorted(selected, key=lambda e: e["start"])


class RuleBasedExtractor:
    def __init__(self):
        self.detectors = {
            "PHONE": YargyPhoneDetector(),
            "EMAIL": YargyEmailDetector(),
            "ID": YargyIdDetector(),
            "ADDRESS": YargyAddressDetector(),
            "PERSON": YargyPersonDetector(),
        }

    def predict_one(self, text: str):
        entities = []

        for label, detector in self.detectors.items():
            preds = detector.predict_one(text)

            for ent in preds:
                ent = dict(ent)
                ent["label"] = label

                if "text" not in ent:
                    ent["text"] = text[ent["start"]:ent["end"]]

                ent["source_detector"] = label
                entities.append(ent)

        return resolve_overlaps_global(entities)



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



def build_errors_by_class(predictions_df: pd.DataFrame, cls: str):
    rows = []

    for _, row in predictions_df.iterrows():
        text = row["text"]

        gold = [e for e in row["gold"] if e["label"] == cls]
        pred = [e for e in row["predictions"] if e["label"] == cls]

        gold_set = {entity_key(e) for e in gold}
        pred_set = {entity_key(e) for e in pred}

        fp = [p for p in pred if entity_key(p) not in gold_set]
        fn = [g for g in gold if entity_key(g) not in pred_set]

        if fp or fn:
            rows.append({
                "id": row["id"],
                "text": text,
                "class": cls,
                "false_positive": fp,
                "false_negative": fn,
            })

    return pd.DataFrame(rows)


def print_errors_by_class(errors_df: pd.DataFrame, cls: str, limit=None):
    print(f"Количество текстов с ошибками: {len(errors_df)}")

    shown = errors_df if limit is None else errors_df.head(limit)

    for _, row in shown.iterrows():
        print(f"ID: {row['id']}")
        print(f"Текст: {row['text']}")

        if row["false_positive"]:
            print("FP:")
            for e in row["false_positive"]:
                print(f'  "{e.get("text", row["text"][e["start"]:e["end"]])}" [{e["start"]}:{e["end"]}] → {e["label"]}')

        if row["false_negative"]:
            print("FN:")
            for e in row["false_negative"]:
                start, end = e["start"], e["end"]
                print(f'  "{row["text"][start:end]}" [{start}:{end}] → {e["label"]}')



def run_rule_based_experiment(path: str, dataset_name: str = "dataset", print_error_limit=None):
    df = load_dataset(path)
    extractor = RuleBasedExtractor()

    texts = df["text"].astype(str).tolist()
    gold = [normalize_gold(x) for x in df["label"].tolist()]

    start_time = time.time()
    predictions = [extractor.predict_one(text) for text in texts]
    elapsed = time.time() - start_time

    metrics = calculate_metrics(gold, predictions)
    class_table = metrics_to_table(metrics)

    summary = {
        "dataset": dataset_name,
        "method": "Rule-based unified",
        "texts_total": len(df),
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

    errors_by_class = {}

    for cls in ENTITY_CLASSES:
        errors_df = build_errors_by_class(predictions_df, cls)
        errors_by_class[cls] = errors_df
        print_errors_by_class(errors_df, cls, limit=print_error_limit)

    return summary, class_table, predictions_df, errors_by_class
