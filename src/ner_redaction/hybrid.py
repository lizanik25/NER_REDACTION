from .rule_based import RuleBasedPIIExtractor
from .ml_model import MLNERModel


def spans_overlap(a: dict, b: dict) -> bool:
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])


def safe_score(ent: dict) -> float:
    score = ent.get("score", 1.0)

    if score is None:
        return 1.0

    try:
        return float(score)
    except (TypeError, ValueError):
        return 1.0


def post_filter_ml_entities(entities: list[dict]) -> list[dict]:
    filtered = []

    for ent in entities:
        label = ent["label"]

        if label not in {"PERSON", "ADDRESS"}:
            continue

        filtered.append(ent)

    return filtered


def resolve_hybrid_overlaps(entities: list[dict]) -> list[dict]:
    priority = {
        "EMAIL": 5,
        "PHONE": 4,
        "ID": 3,
        "ADDRESS": 2,
        "PERSON": 1,
    }

    def score_entity(e: dict):
        return (
            priority.get(e["label"], 0),
            e["end"] - e["start"],
            safe_score(e),
        )

    entities = sorted(
        entities,
        key=lambda e: (
            e["start"],
            -(e["end"] - e["start"]),
            -priority.get(e["label"], 0),
        ),
    )

    selected = []

    for ent in entities:
        keep = True

        for old in selected[:]:
            if spans_overlap(ent, old):
                if score_entity(ent) > score_entity(old):
                    selected.remove(old)
                else:
                    keep = False
                break

        if keep:
            selected.append(ent)

    return sorted(selected, key=lambda e: e["start"])


class HybridPIIExtractor:
    def __init__(self, model_path: str = "models/final_model"):
        self.rule_extractor = RuleBasedPIIExtractor()
        self.ml_model = MLNERModel(model_path=model_path)

    def predict_one(self, text: str) -> list[dict]:
        rule_entities = self.rule_extractor.predict_one(text)

        ml_entities = self.ml_model.predict_one(text)
        ml_entities = post_filter_ml_entities(ml_entities)

        all_entities = []

        for ent in rule_entities + ml_entities:
            ent = dict(ent)

            ent["start"] = int(ent["start"])
            ent["end"] = int(ent["end"])
            ent["text"] = ent.get("text", text[ent["start"]:ent["end"]])

            if ent.get("score") is None:
                ent["score"] = 1.0
            else:
                ent["score"] = safe_score(ent)

            if "source" not in ent:
                ent["source"] = ent.get("source_component", "unknown")

            all_entities.append(ent)

        return resolve_hybrid_overlaps(all_entities)