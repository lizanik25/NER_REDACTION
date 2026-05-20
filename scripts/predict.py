import argparse
import json
import time
from pathlib import Path

import pandas as pd


def load_pipeline():
    from ner_redaction.pipeline import RedactionPipeline
    return RedactionPipeline()


def predict_batch(pipeline, texts: list[str], mode: str, classes: list[str]) -> list[dict]:
    results = []
    for text in texts:
        result = pipeline.process(text, mode=mode, classes=classes)
        results.append({
            "anonymized_text": result.anonymized_text,
            "entities": json.dumps(result.entities, ensure_ascii=False),
            "entities_count": result.entities_count,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch PII prediction")
    parser.add_argument("--input", required=True, help="Input CSV with 'text' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--mode", default="replace", choices=["replace", "mask", "pseudonymize"])
    parser.add_argument("--classes", nargs="+", default=["PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"])
    parser.add_argument("--text_col", default="text")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()

    pipeline = load_pipeline()

    t0 = time.time()
    results = predict_batch(pipeline, texts, args.mode, args.classes)
    elapsed = time.time() - t0

    result_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
    out_df.to_csv(args.output, index=False, encoding="utf-8")

    total_entities = sum(r["entities_count"] for r in results)
    print(f"Total entities found: {total_entities}")


if __name__ == "__main__":
    main()
