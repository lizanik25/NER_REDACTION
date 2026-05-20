import re
from fastapi import HTTPException

from app.config import MAX_TEXT_LENGTH, SUPPORTED_ENTITIES
from app.services.highlighting import build_highlight_segments
from src.ner_redaction.pipeline import RedactionPipeline


def validate_text_size(text: str) -> None:

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Text is too large",
                "text_length": len(text),
                "max_text_length": MAX_TEXT_LENGTH,
            },
        )


def validate_entities_list(entities: list[str] | None) -> list[str] | None:

    if entities is None:
        return None

    parsed = [e.strip().upper() for e in entities if e.strip()]
    unsupported = [e for e in parsed if e not in SUPPORTED_ENTITIES]

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported entity type",
                "unsupported_entities": unsupported,
                "supported_entities": SUPPORTED_ENTITIES,
            },
        )

    return parsed


def parse_entities_param(entities: str | None) -> list[str] | None:

    if not entities or entities.strip() == "":
        return None

    parsed = [item.strip().upper() for item in entities.split(",") if item.strip()]
    return validate_entities_list(parsed)


def parse_word_list_param(value: str | None) -> list[str]:
    if not value or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def count_entities_by_type(entities: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in entities:
        label = entity["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def apply_allowlist(entities: list[dict], allowlist: list[str]) -> list[dict]:

    if not allowlist:
        return entities

    allowlist_lower = {word.lower() for word in allowlist}
    return [e for e in entities if e.get("text", "").lower() not in allowlist_lower]


def apply_denylist(text: str, entities: list[dict], denylist: list[str]) -> list[dict]:

    if not denylist:
        return entities

    result = list(entities)

    for word in denylist:
        pattern = re.escape(word)
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()

            has_overlap = any(
                not (end <= e["start"] or start >= e["end"])
                for e in result
            )
            if has_overlap:
                continue

            result.append({
                "start": start,
                "end": end,
                "label": "DENYLIST",
                "score": 1.0,
                "source": "denylist",
                "source_component": "rule",
                "source_detector": "denylist",
                "text": text[start:end],
                "replacement": "[DENYLIST]",
                "anonymization_mode": None,
            })

    return sorted(result, key=lambda e: (e["start"], e["end"]))


def decode_uploaded_file(content: bytes, filename: str, max_size: int) -> str:

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "File is too large",
                "filename": filename,
                "file_size_bytes": len(content),
                "max_file_size_bytes": max_size,
            },
        )

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Only UTF-8 text files are supported",
                "filename": filename,
            },
        )


def process_text_item(
    pipeline: RedactionPipeline,
    text: str,
    mode: str,
    entities_filter: list[str] | None = None,
    threshold: float = 0.0,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    filename: str | None = None,
    index: int | None = None,
) -> dict:

    validate_text_size(text)

    detected_entities, metadata = pipeline.analyze(
        text=text,
        entities=entities_filter,
        threshold=threshold,
    )

    filtered_entities = apply_allowlist(detected_entities, allowlist or [])
    filtered_entities = apply_denylist(text, filtered_entities, denylist or [])

    anonymized_text, processed_entities = pipeline.anonymize(
        text=text,
        entities=filtered_entities,
        mode=mode,
    )

    result: dict = {
        "anonymized_text": anonymized_text,
        "highlight_segments": build_highlight_segments(text, processed_entities),
        "entities_count": len(processed_entities),
        "entities_count_by_type": count_entities_by_type(processed_entities),
        "entities": processed_entities,
        "text_length": metadata["text_length"],
        "truncated": metadata["truncated"],
        "mode": mode,
        "entities_filter": entities_filter,
        "threshold": threshold,
        "allowlist": allowlist or [],
        "denylist": denylist or [],
        "pipeline": "hybrid",
    }

    if filename is not None:
        result["filename"] = filename
    if index is not None:
        result["file_index"] = index

    return result
