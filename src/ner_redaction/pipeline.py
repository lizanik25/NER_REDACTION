from .hybrid import HybridPIIExtractor, resolve_hybrid_overlaps
from .anonymizer import TextAnonymizer

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

SUPPORTED_ENTITIES = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"}
SUPPORTED_MODES = {"replace", "mask", "pseudonymize"}


class RedactionPipeline:
    def __init__(
        self,
        model_path: str = "models/final_model",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.extractor = HybridPIIExtractor(model_path=model_path)
        self.anonymizer = TextAnonymizer()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def analyze(
        self,
        text: str,
        entities: list[str] | None = None,
        threshold: float = 0.0,
    ) -> tuple[list[dict], dict]:
        """
        Извлекает сущности ПД из текста.

        Raises:
            TypeError: если text не является строкой.
            ValueError: если entities содержит неподдерживаемые типы.
            ValueError: если threshold выходит за пределы [0.0, 1.0].

        Note:
            Приоритет при конфликтах спанов: EMAIL > PHONE > ID > ADDRESS > PERSON.
            Сущности на границах чанков обрабатываются через resolve_hybrid_overlaps.
        """
        self._validate_text(text)
        self._validate_entities(entities)
        self._validate_threshold(threshold)

        if not text.strip():
            return [], {"text_length": len(text), "chunks_count": 0, "truncated": False}

        chunks = self._split_text(text)
        all_entities = []

        for chunk_start, chunk_text in chunks:
            chunk_entities = self.extractor.predict_one(chunk_text)

            for ent in chunk_entities:
                ent = dict(ent)
                ent["start"] = int(ent["start"]) + chunk_start
                ent["end"] = int(ent["end"]) + chunk_start
                ent["text"] = text[ent["start"]:ent["end"]]
                all_entities.append(ent)

        all_entities = resolve_hybrid_overlaps(all_entities)

        if threshold > 0:
            all_entities = [
                ent for ent in all_entities
                if float(ent.get("score") or 1.0) >= threshold
            ]

        if entities is not None:
            allowed = set(entities)
            all_entities = [
                ent for ent in all_entities
                if ent["label"] in allowed
            ]

        metadata = {
            "text_length": len(text),
            "chunks_count": len(chunks),
            "truncated": False,
        }

        return all_entities, metadata

    def anonymize(
        self,
        text: str,
        entities: list[dict],
        mode: str = "replace",
    ) -> tuple[str, list[dict]]:
        self._validate_text(text)
        self._validate_mode(mode)

        if not isinstance(entities, list):
            raise TypeError(f"entities must be a list, got {type(entities).__name__}")

        return self.anonymizer.anonymize(text=text, entities=entities, mode=mode)

    def deidentify(
        self,
        text: str,
        mode: str = "replace",
        entities: list[str] | None = None,
        threshold: float = 0.0,
    ) -> tuple[str, list[dict], dict]:
        found_entities, metadata = self.analyze(
            text=text,
            entities=entities,
            threshold=threshold,
        )

        anonymized_text, processed_entities = self.anonymize(
            text=text,
            entities=found_entities,
            mode=mode,
        )

        return anonymized_text, processed_entities, metadata


    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

    @staticmethod
    def _validate_entities(entities: list[str] | None) -> None:
        if entities is None:
            return

        if not isinstance(entities, list):
            raise TypeError(f"entities must be a list or None, got {type(entities).__name__}")

        unsupported = [e for e in entities if e not in SUPPORTED_ENTITIES]
        if unsupported:
            raise ValueError(
                f"Unsupported entity types: {unsupported}. "
                f"Supported: {sorted(SUPPORTED_ENTITIES)}"
            )

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

    @staticmethod
    def _validate_mode(mode: str) -> None:
    
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode: {mode!r}. "
                f"Supported modes: {sorted(SUPPORTED_MODES)}"
            )


    def _split_text(self, text: str) -> list[tuple[int, str]]:
        if len(text) <= self.chunk_size:
            return [(0, text)]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                split_at = self._find_safe_split(text, start, end)
                if split_at > start:
                    end = split_at

            chunk_text = text[start:end]
            chunks.append((start, chunk_text))

            if end >= text_len:
                break

            next_start = max(0, end - self.chunk_overlap)
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _find_safe_split(self, text: str, start: int, end: int) -> int:
        window = text[start:end]

        for separator in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]:
            idx = window.rfind(separator)
            if idx != -1 and idx > self.chunk_size * 0.5:
                return start + idx + len(separator)

        return end
