import argparse
import ast
import time
from pathlib import Path

import pandas as pd

ENTITY_CLASSES = ["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"]
RULE_LABELS = {"PHONE", "EMAIL", "ID"}
ML_LABELS = {"PERSON", "ADDRESS"}


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
    return df.reset_index(drop=True)


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


def metrics_to_df(metrics: dict, method_name: str = "") -> pd.DataFrame:
    rows = []
    for cls in ENTITY_CLASSES:
        m = metrics["per_class"][cls]
        row = {
            "Класс": cls,
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
        }
        if method_name:
            row["method"] = method_name
        rows.append(row)
    return pd.DataFrame(rows)


def print_metrics(metrics: dict, name: str = ""):
    if name:
        print(f"\n{name}")
    for cls in ENTITY_CLASSES:
        m = metrics["per_class"][cls]
        print(f"  {cls:10s}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  "
              f"F1={m['F1']:.4f}  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  {'micro-F1':10s}  {metrics['micro_f1']:.4f}")
    print(f"  {'macro-F1':10s}  {metrics['macro_f1']:.4f}")


class BaselinePredictor:

    def __init__(self):
        from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
        self.segmenter = Segmenter()
        self.ner_tagger = NewsNERTagger(NewsEmbedding())
        self._Doc = Doc

    def predict_one(self, text: str) -> list:
        doc = self._Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        return [
            {"start": sp.start, "end": sp.stop, "label": "PERSON"}
            for sp in doc.spans if sp.type == "PER"
        ]

    def predict_batch(self, texts: list) -> list:
        return [self.predict_one(t) for t in texts]




class FinetunedSlovnetPredictor:

    def __init__(self, navec_path: str, weights_path: str):
        from navec import Navec
        from slovnet import NER as SlovnetNER
        navec = Navec.load(navec_path)
        self.model = SlovnetNER.load(weights_path)
        self.model.navec(navec)

    def predict_one(self, text: str, threshold: float = 0.0) -> list:
        markup = self.model(text)
        return [
            {"start": sp.start, "end": sp.stop, "label": sp.type, "score": 1.0}
            for sp in markup.spans
            if sp.score >= threshold
        ]

    def predict_batch(self, texts: list) -> list:
        return [self.predict_one(t) for t in texts]


def resolve_overlaps(entities: list) -> list:
    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    selected = []
    for ent in entities:
        if not any(
            not (ent["end"] <= old["start"] or ent["start"] >= old["end"])
            for old in selected
        ):
            selected.append(ent)
    return selected


class HybridPredictor:


    def __init__(self, navec_path: str, weights_path: str):
        self.ml = FinetunedSlovnetPredictor(navec_path, weights_path)
        try:
            import sys
            sys.path.insert(0, "scripts/rule_based")
            from combine_rules import RuleBasedExtractor
            self.rb = RuleBasedExtractor()
        except ImportError:
            self.rb = None

    def predict_one(self, text: str) -> list:
        ml_preds = [
            e for e in self.ml.predict_one(text)
            if e["label"] in ML_LABELS
        ]
        rb_preds = self.rb.predict_one(text) if self.rb else []
        rb_preds = [e for e in rb_preds if e["label"] in RULE_LABELS]

        combined = ml_preds + rb_preds
        return resolve_overlaps(combined)

    def predict_batch(self, texts: list) -> list:
        return [self.predict_one(t) for t in texts]



def run_evaluation(predictor, df: pd.DataFrame, name: str) -> dict:
    gold_all = [normalize_gold(x) for x in df["label"]]
    texts = df["text"].astype(str).tolist()

    t0 = time.time()
    pred_all = predictor.predict_batch(texts)
    elapsed = time.time() - t0

    metrics = calculate_metrics(gold_all, pred_all)
    print_metrics(metrics, name)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PII detection models")
    parser.add_argument("--method", choices=["baseline", "finetune", "hybrid"],
                        default="finetune")
    parser.add_argument("--test", default="data/processed/test_sample.csv")
    parser.add_argument("--no_pii", default="data/processed/no_pii_sample.csv")
    parser.add_argument("--tricky", default="data/processed/tricky_sample.csv")
    parser.add_argument("--navec", default="models/final_model/navec_news_v1_1B_250K_300d_100q.tar")
    parser.add_argument("--weights", default="models/final_model/slovnet_ner_pii_ru_hard_no_pd.tar")
    parser.add_argument("--output", default="outputs/eval_results/")
    parser.add_argument("--all_splits", action="store_true",
                        help="Оценить на всех трёх выборках")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Метод: {args.method}")
    if args.method == "baseline":
        predictor = BaselinePredictor()
    elif args.method == "finetune":
        predictor = FinetunedSlovnetPredictor(args.navec, args.weights)
    else:
        predictor = HybridPredictor(args.navec, args.weights)

    df_test = load_labeled_file(args.test)
    metrics = run_evaluation(predictor, df_test, f"{args.method} — test ({len(df_test)} texts)")
    table = metrics_to_df(metrics, args.method)
    table.to_csv(out_dir / f"{args.method}_test_metrics.csv", index=False)

    if args.all_splits:
        df_no = load_labeled_file(args.no_pii)
        run_evaluation(predictor, df_no, f"{args.method} — no-PII ({len(df_no)} texts)")

        df_tricky = load_labeled_file(args.tricky)
        run_evaluation(predictor, df_tricky, f"{args.method} — tricky ({len(df_tricky)} texts)")



if __name__ == "__main__":
    main()
