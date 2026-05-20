def build_highlight_segments(text: str, entities: list[dict]) -> list[dict]:

    segments = []
    cursor = 0

    valid_entities = sorted(
        [e for e in entities if int(e.get("start", -1)) < int(e.get("end", -1))],
        key=lambda e: (int(e["start"]), int(e["end"])),
    )

    for idx, entity in enumerate(valid_entities):
        start = int(entity["start"])
        end = int(entity["end"])

        if start < cursor:
            continue

        if cursor < start:
            segments.append({"type": "text", "text": text[cursor:start]})

        segments.append({
            "type": "entity",
            "ui_id": f"entity-{idx}",
            "label": entity.get("label"),
            "text": entity.get("text", text[start:end]),
            "replacement": entity.get("replacement", f"[{entity.get('label', 'ENTITY')}]") or "",
            "start": start,
            "end": end,
            "score": entity.get("score"),
            "source": entity.get("source"),
        })

        cursor = end

    if cursor < len(text):
        segments.append({"type": "text", "text": text[cursor:]})

    return segments
