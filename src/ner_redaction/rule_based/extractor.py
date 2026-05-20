from .phone import YargyPhoneDetector
from .email import YargyEmailDetector
from .id import YargyIdDetector, ContextIdDetector
from .utils import resolve_rule_overlaps


class RuleBasedPIIExtractor:

    def __init__(self):
        self.detectors = {
            "ID_CONTEXT": ContextIdDetector(),
            "PHONE":      YargyPhoneDetector(),
            "EMAIL":      YargyEmailDetector(),
            "ID":         YargyIdDetector(),
        }

    def predict_one(self, text: str) -> list[dict]:

        entities = []

        for detector_name, detector in self.detectors.items():
            for ent in detector.predict_one(text):
                ent = dict(ent)

                ent["label"] = "ID" if detector_name == "ID_CONTEXT" else detector_name

                ent["score"]            = ent.get("score", 1.0)
                ent["source"]           = ent.get("source", detector_name.lower())
                ent["source_component"] = "rule"
                ent["source_detector"]  = ent.get("source_detector") or ent.get("source") or detector_name
                ent["recognizer"]       = ent.get("recognizer") or detector.__class__.__name__
                ent["text"]             = ent.get("text") or text[ent["start"]:ent["end"]]

                entities.append(ent)

        return resolve_rule_overlaps(entities)
