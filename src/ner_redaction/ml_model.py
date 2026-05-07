import re
from pathlib import Path

from navec import Navec
from slovnet import NER


ML_CLASSES = {"PERSON", "ADDRESS"}


BAD_PERSON_PREFIXES = [
    "Журналистка", "Журналист",
    "Сенатор", "Министр", "Президент",
    "Вице-премьер", "Вице-премьеру",
    "Заместитель", "Заместителем",
    "Адвокат", "Призерка", "Призёрка",
    "Лидер", "глава",
    "И. о.", "И. о", "Твиты",
]

TRAILING_CHARS = " ,.;:-—–"

BAD_TRAILING_WORDS = {
    "выступил", "выступила",
    "заявил", "заявила",
    "сообщил", "сообщила",
    "назначен", "назначена",
    "поручил", "обвинила",
}


class MLNERModel:
    def __init__(
        self,
        model_path: str = "models/final_model",
        model_name: str = "slovnet_ner_pii_ru_hard_no_pd.tar",
        navec_name: str = "navec_news_v1_1B_250K_300d_100q.tar", 
    ):
        model_dir = Path(model_path)

        self.model_path = model_dir / model_name
        self.navec_path = model_dir / navec_name

        if not self.model_path.exists():
            raise FileNotFoundError(f"Slovnet model not found: {self.model_path}")

        if not self.navec_path.exists():
            raise FileNotFoundError(f"Navec model not found: {self.navec_path}")

        self.navec = Navec.load(str(self.navec_path))

        self.model = NER.load(str(self.model_path))
        self.model.navec(self.navec)

    def predict_one(self, text: str) -> list[dict]:
        text = str(text)
        markup = self.model(text)

        spans = []

        for span in markup.spans:
            label = str(span.type)

            if label not in ML_CLASSES:
                continue

            start = int(span.start)
            end = int(span.stop)

            if label == "PERSON":
                cleaned = self._clean_person_span(text, start, end)
                if cleaned is None:
                    continue

                start, end, label = cleaned

            if start >= end:
                continue

            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": text[start:end],
                    "score": None,
                    "source": "ml",
                    "source_component": "ml",
                    "source_detector": "slovnet_finetuned",
                    "recognizer": "SlovnetNER",
                }
            )

        spans = self._merge_adjacent_persons(text, spans)

        return spans

    def _clean_person_span(self, text: str, start: int, end: int):
        while start < end and text[start].isspace():
            start += 1

        while end > start and text[end - 1].isspace():
            end -= 1

        while end > start and text[end - 1] in TRAILING_CHARS:
            end -= 1

        span_text = text[start:end].strip()

        if span_text in {",", ".", "-", "—", "–"}:
            return None

        changed = True
        while changed:
            changed = False
            span_text = text[start:end].strip()

            for prefix in BAD_PERSON_PREFIXES:
                if span_text.startswith(prefix + " "):
                    start += len(prefix) + 1
                    changed = True
                    break

        parts = text[start:end].strip().split()

        while parts and parts[-1].strip(TRAILING_CHARS).lower() in BAD_TRAILING_WORDS:
            last = parts[-1]
            end -= len(last)

            while end > start and text[end - 1].isspace():
                end -= 1

            parts = text[start:end].strip().split()

        while end > start and text[end - 1] in TRAILING_CHARS:
            end -= 1

        span_text = text[start:end].strip()

        if len(span_text) <= 1:
            return None

        return start, end, "PERSON"

    def _merge_adjacent_persons(self, text: str, spans: list[dict]) -> list[dict]:
        spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
        merged = []
        i = 0

        while i < len(spans):
            current = dict(spans[i])

            if current["label"] != "PERSON":
                merged.append(current)
                i += 1
                continue

            j = i + 1

            while j < len(spans) and spans[j]["label"] == "PERSON":
                gap = text[current["end"]:spans[j]["start"]]

                if re.fullmatch(r"\s+", gap):
                    current["end"] = spans[j]["end"]
                    current["text"] = text[current["start"]:current["end"]]
                    j += 1
                else:
                    break

            merged.append(current)
            i = j

        return merged