import argparse
import json
from pathlib import Path

import pandas as pd


VALID_LABELS = {"PERSON", "PHONE", "EMAIL", "ADDRESS", "ID"}


def safe_load_json(value) -> list:
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value or value in ("[]", "nan"):
            return []
        try:
            return json.loads(value)
        except Exception:
            try:
                return json.loads(value.replace("'", '"'))
            except Exception:
                return []
    return []


def convert_row(row) -> dict | None:
    text = str(row["text"]).strip()
    spans = safe_load_json(row.get("label", "[]"))

    labels = []
    for s in spans:
        start = s.get("start")
        end = s.get("end")
        label = s.get("label", "")
        if label not in VALID_LABELS:
            continue
        if start is None or end is None or start >= end:
            continue
        if end > len(text):
            continue
        labels.append([start, end, label])

    return {"text": text, "label": labels}


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Doccano JSONL")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = convert_row(row)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1


if __name__ == "__main__":
    main()
