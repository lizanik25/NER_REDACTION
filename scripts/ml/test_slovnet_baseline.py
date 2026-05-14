import argparse
import ast
import re
import time
from pathlib import Path

import pandas as pd
from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
from razdel import sentenize

ENTITY_CLASSES = ["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"]


def load_labeled_file(path: str) -> pd.DataFrame:
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

    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def load_no_pii_file(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if "text" not in df.columns:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    if "label" not in df.columns:
        df["label"] = [[] for _ in range(len(df))]
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df



def preprocess_text(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    if mode == "lowercase":
        return text.lower()
    if mode == "clean":
        return re.sub(r"\s+", " ", text).strip()


class NatashaSlovnetNER:

    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)

    def predict_one(self, text: str, split_mode: str = "whole_text") -> list:
        entities = []

        if split_mode == "whole_text":
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)
            for span in doc.spans:
                if span.type == "PER":
                    entities.append({
                        "start": span.start,
                        "end": span.stop,
                        "label": "PERSON",
                        "text": text[span.start:span.stop],
                    })

        elif split_mode == "sentence_split":
            for sent in sentenize(text):
                offset = sent.start
                doc = Doc(sent.text)
                doc.segment(self.segmenter)
                doc.tag_ner(self.ner_tagger)
                for span in doc.spans:
                    if span.type == "PER":
                        start = offset + span.start
                        end = offset + span.stop
                        entities.append({
                            "start": start,
                            "end": end,
                            "label": "PERSON",
                            "text": text[start:end],
                        })
        else:
            raise ValueError(f"{split_mode}")

        return entities

    def predict_batch(self, texts: list, batch_size: int = 8,
                      split_mode: str = "whole_text") -> list:
        all_preds = []
        for i in range(0, len(texts), batch_size):
            for text in texts[i:i + batch_size]:
                all_preds.append(self.predict_one(text, split_mode=split_mode))
        return all_preds



def normalize_gold(label_list) -> list:
    if label_list is None:
        return []
    if isinstance(label_list, float) and pd.isna(label_list):
        return []
    if isinstance(label_list, str):
        label_list = ast.literal_eval(label_list)
    return [{"start": int(x[0]), "end": int(x[1]), "label": x[2]} for x in label_list]


def entity_key(ent: dict) -> tuple:
    return (ent["start"], ent["end"], ent["label"])


def calculate_metrics(gold_all: list, pred_all: list,
                      classes: list = ENTITY_CLASSES) -> dict:
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[cls] = {
            "Precision": precision, "Recall": recall, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(per_class[c]["F1"] for c in classes) / len(classes)

    return {
        "per_class": per_class,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def metrics_to_df(metrics: dict) -> pd.DataFrame:
    rows = []
    for cls in ENTITY_CLASSES:
        m = metrics["per_class"][cls]
        rows.append({
            "Класс": cls,
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
        })
    return pd.DataFrame(rows)



def run_experiment(df: pd.DataFrame, preprocessing_mode: str = "raw",
                   split_mode: str = "whole_text", batch_size: int = 8):
    model = NatashaSlovnetNER()
    texts = [preprocess_text(str(t), preprocessing_mode) for t in df["text"]]
    gold = [normalize_gold(x) for x in df["label"]]

    t0 = time.time()
    predictions = model.predict_batch(texts, batch_size=batch_size, split_mode=split_mode)
    elapsed = time.time() - t0

    metrics = calculate_metrics(gold, predictions)
    return metrics, predictions, elapsed


def run_all_experiments(df: pd.DataFrame):

    preprocessing_modes = ["raw", "lowercase", "clean"]
    split_modes = ["whole_text", "sentence_split"]
    batch_sizes = [1, 8, 16]

    summary_rows = []
    best_result = None

    for prep in preprocessing_modes:
        for split in split_modes:
            for bs in batch_sizes:
                metrics, predictions, elapsed = run_experiment(df, prep, split, bs)
                row = {
                    "preprocessing": prep,
                    "split_mode": split,
                    "batch_size": bs,
                    "micro_precision": round(metrics["micro_precision"], 4),
                    "micro_recall": round(metrics["micro_recall"], 4),
                    "micro_f1": round(metrics["micro_f1"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                    "time_sec": round(elapsed, 4),
                    "texts_per_sec": round(len(df) / elapsed, 4) if elapsed > 0 else None,
                }
                summary_rows.append(row)

                if best_result is None or metrics["micro_f1"] > best_result["metrics"]["micro_f1"]:
                    best_result = {
                        "settings": row,
                        "metrics": metrics,
                        "predictions": predictions,
                    }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["micro_f1", "macro_f1", "texts_per_sec"], ascending=False
    )
    class_table = metrics_to_df(best_result["metrics"])
    return summary_df, class_table, best_result



def build_person_errors_df(df: pd.DataFrame, predictions: list) -> pd.DataFrame:
    rows = []
    gold_all = [normalize_gold(x) for x in df["label"]]

    for i, (text, gold, pred) in enumerate(
        zip(df["text"].astype(str), gold_all, predictions)
    ):
        gold_p = [e for e in gold if e["label"] == "PERSON"]
        pred_p = [e for e in pred if e["label"] == "PERSON"]
        gold_set = {entity_key(e) for e in gold_p}
        pred_set = {entity_key(e) for e in pred_p}

        fp = [e for e in pred_p if entity_key(e) not in gold_set]
        fn = [e for e in gold_p if entity_key(e) not in pred_set]

        if fp or fn:
            rows.append({
                "id": i,
                "source": df.loc[i, "source"] if "source" in df.columns else None,
                "text": text,
                "false_positives": [
                    {**e, "text": text[e["start"]:e["end"]]} for e in fp
                ],
                "false_negatives": [
                    {**e, "text": text[e["start"]:e["end"]]} for e in fn
                ],
            })
    return pd.DataFrame(rows)


def build_tricky_fp_df(df: pd.DataFrame, predictions: list) -> pd.DataFrame:
    rows = []
    total_fp = 0
    for i, row in df.iterrows():
        text = str(row["text"])
        gold = normalize_gold(row["label"])
        pred = predictions[i]
        gold_set = {entity_key(e) for e in gold}
        fps = [e for e in pred if entity_key(e) not in gold_set]
        if fps:
            total_fp += len(fps)
            rows.append({
                "id": i,
                "text": text,
                "fp_count": len(fps),
                "false_positives": "; ".join(
                    f'"{e.get("text", text[e["start"]:e["end"]])}" '
                    f'[{e["start"]}:{e["end"]}, {e["label"]}]'
                    for e in fps
                ),
            })
    fp_df = pd.DataFrame(rows)
    return fp_df


def build_no_pii_fp_df(df: pd.DataFrame, predictions: list) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row["text"])
        preds = predictions[i]
        if preds:
            rows.append({
                "id": i,
                "text": text,
                "fp_count": len(preds),
                "predictions": "; ".join(
                    f'"{e.get("text", text[e["start"]:e["end"]])}" '
                    f'[{e["start"]}:{e["end"]}, {e["label"]}]'
                    for e in preds
                ),
            })
    fp_df = pd.DataFrame(rows)
    return fp_df



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Natasha/Slovnet NER baseline"
    )
    parser.add_argument("--test", default="data/processed/test_sample.csv")
    parser.add_argument("--no_pii", default="data/processed/no_pii_sample.csv")
    parser.add_argument("--tricky", default="data/processed/tricky_sample.csv")
    parser.add_argument("--output", default="outputs/baseline_results/")
    parser.add_argument(
        "--full_search", action="store_true",
        help="Перебор всех комбинаций preprocessing/split_mode/batch_size"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = load_labeled_file(args.test)

    if args.full_search:
        summary_df, class_table, best = run_all_experiments(df_test)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(out_dir / "test_experiments_summary.csv", index=False)
    else:
        metrics, predictions, elapsed = run_experiment(df_test, "raw", "whole_text", 8)
        class_table = metrics_to_df(metrics)
        best = {"metrics": metrics, "predictions": predictions}

    print(class_table.to_string(index=False))
    m = best["metrics"]
    print(f"\nmicro-F1: {m['micro_f1']:.4f}  macro-F1: {m['macro_f1']:.4f}")

    errors_df = build_person_errors_df(df_test, best["predictions"])

    class_table.to_csv(out_dir / "test_per_class_metrics.csv", index=False)
    errors_df.to_csv(out_dir / "test_person_errors.csv", index=False)

    df_no_pii = load_no_pii_file(args.no_pii)
    _, preds_no, elapsed_no = run_experiment(df_no_pii, "raw", "whole_text", 8)
    fp_no = build_no_pii_fp_df(df_no_pii, preds_no)
    fp_no.to_csv(out_dir / "no_pii_false_positives.csv", index=False)

    df_tricky = load_labeled_file(args.tricky)
    model = NatashaSlovnetNER()
    t0 = time.time()
    preds_tricky = [model.predict_one(str(t)) for t in df_tricky["text"]]

    fp_tricky = build_tricky_fp_df(df_tricky, preds_tricky)
    fp_tricky.to_csv(out_dir / "tricky_false_positives.csv", index=False)


if __name__ == "__main__":
    main()
