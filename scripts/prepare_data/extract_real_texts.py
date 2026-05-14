import argparse
import re
from pathlib import Path

import pandas as pd
from razdel import sentenize


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_sentence(text: str, min_len: int = 30, max_len: int = 300) -> bool:
    return min_len <= len(text) <= max_len


def extract_from_nerus(path: str, max_samples: int = 2000) -> pd.DataFrame:
    from nerus import load_nerus

    rows = []
    docs = load_nerus(path)
    for doc in docs:
        for sent in doc.sents:
            tokens = sent.tokens
            has_person = any("PER" in t.tag for t in tokens)
            if not has_person:
                continue

            text = sent.text.strip()
            if not is_valid_sentence(text):
                continue

            entities = []
            for t in tokens:
                tag = t.tag
                if "PER" in tag:
                    entities.append({"start": t.start, "end": t.stop, "label": "PERSON"})

            rows.append({"source": "nerus", "text": text, "auto_entities": str(entities)})
            if len(rows) >= max_samples:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def extract_from_hf_dataset(dataset_name: str, split: str = "train", max_samples: int = 500) -> pd.DataFrame:
    from datasets import load_dataset

    rows = []
    ds = load_dataset(dataset_name, split=split)
    for item in ds:
        text = item.get("tokens") or item.get("text", "")
        if isinstance(text, list):
            text = " ".join(text)
        text = clean_text(text)
        if not is_valid_sentence(text):
            continue
        rows.append({"source": dataset_name.split("/")[-1], "text": text, "auto_entities": "[]"})
        if len(rows) >= max_samples:
            break

    return pd.DataFrame(rows)


def extract_from_wiki(path: str, max_articles: int = 500, max_samples: int = 500) -> pd.DataFrame:
    from corus import load_wiki
    from natasha import Doc, MorphVocab, NamesExtractor, NewsNERTagger, NewsSegmenter, Segmenter

    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    ner_tagger = NewsNERTagger(NewsSegmenter())
    names_extractor = NamesExtractor(morph_vocab)

    rows = []
    records = load_wiki(path)
    for i, record in enumerate(records):
        if i >= max_articles or len(rows) >= max_samples:
            break

        text = re.sub(r"=+[^=]+=+", "", record.text)  # remove section headers
        for sent in sentenize(text):
            sent_text = clean_text(sent.text)
            if not is_valid_sentence(sent_text):
                continue

            doc = Doc(sent_text)
            doc.segment(segmenter)
            doc.tag_ner(ner_tagger)
            has_person = any(sp.type == "PER" for sp in doc.spans)
            if has_person:
                rows.append({"source": "wikipedia", "text": sent_text, "auto_entities": "[]"})
                if len(rows) >= max_samples:
                    break

    return pd.DataFrame(rows)



def main():
    parser = argparse.ArgumentParser(description="Extract real texts with PERSON entities")
    parser.add_argument("--nerus", help="Path to nerus .conllu.gz file")
    parser.add_argument("--wiki", help="Path to ruwiki .xml.bz2 file")
    parser.add_argument("--output", default="data/interim/person_texts.csv")
    parser.add_argument("--max_per_source", type=int, default=1000)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frames = []

    if args.nerus:
        df = extract_from_nerus(args.nerus, args.max_per_source)
        frames.append(df)

    if args.wiki:
        df = extract_from_wiki(args.wiki, max_samples=args.max_per_source)
        frames.append(df)


    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset="text")


if __name__ == "__main__":
    main()
