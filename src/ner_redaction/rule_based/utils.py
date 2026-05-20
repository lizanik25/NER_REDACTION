import re


def resolve_overlaps(entities: list[dict]) -> list[dict]:

    entities = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"]))
    )

    selected = []

    for ent in entities:
        has_overlap = False

        for old in selected[:]:
            if not (ent["end"] <= old["start"] or ent["start"] >= old["end"]):
                has_overlap = True

                if (ent["end"] - ent["start"]) > (old["end"] - old["start"]):
                    selected.remove(old)
                    selected.append(ent)

                break

        if not has_overlap:
            selected.append(ent)

    return sorted(selected, key=lambda e: e["start"])


def resolve_rule_overlaps(entities: list[dict]) -> list[dict]:

    priority = {
        "EMAIL": 4,
        "ID": 3,
        "PHONE": 2,
    }

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
            overlap = not (ent["end"] <= old["start"] or ent["start"] >= old["end"])

            if overlap:
                ent_score = (priority.get(ent["label"], 0), ent["end"] - ent["start"])
                old_score = (priority.get(old["label"], 0), old["end"] - old["start"])

                if ent_score > old_score:
                    selected.remove(old)
                else:
                    keep = False

                break

        if keep:
            selected.append(ent)

    return sorted(selected, key=lambda e: e["start"])


def extend_phone_span(text: str, start: int, end: int) -> tuple[int, int]:

    tail = text[end:end + 50]

    paren_ext = re.match(
        r"\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\)",
        tail,
        flags=re.IGNORECASE,
    )
    ext = re.match(
        r"\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}",
        tail,
        flags=re.IGNORECASE,
    )

    if paren_ext:
        end += paren_ext.end()
    elif ext:
        end += ext.end()

    tail = text[end:end + 40]
    list_tail = re.match(r"(?:\s*[,/]\s*\d{2}){1,4}", tail)

    if list_tail:
        end += list_tail.end()

    return start, end


def is_bad_phone_candidate(text: str, start: int, end: int, source: str = "") -> bool:
    value = text[start:end]
    digits = re.sub(r"\D", "", value)
    stripped = value.strip()

    left_context_15 = text[max(0, start - 15):start].lower()
    left_context_40 = text[max(0, start - 40):start].lower()

    phone_context = re.search(
        r"(тел\.?|телефон|моб\.?|номер телефона|контактный телефон"
        r"|контактный номер|для связи)\s*[:\-]?\s*$",
        left_context_40,
        flags=re.IGNORECASE,
    )

    bad_context = re.search(
        r"(номер документа|регистрационный номер|номер обращения"
        r"|id клиента|user_id|id|инн|р/с|счет|номер счета|идентификатор)\s*[:=]?\s*$",
        left_context_40,
        flags=re.IGNORECASE,
    )

    if "в/ч" in left_context_15:
        return True

    if bad_context and not phone_context:
        return True

    if stripped.isdigit() and len(digits) < 10:
        return True

    if stripped.isdigit() and len(digits) >= 11:
        if not re.match(r"^(?:\+?7|8)\d{10}$", stripped) and not phone_context:
            return True

    if re.fullmatch(r"\d{1,3}-\d{2}-\d{2}(?:\s*(?:[,/]\s*\d{2}){1,4})?", stripped):
        if not phone_context:
            return True

    if re.search(
        r"(код|артикул|номер договора|номер сч[её]та"
        r"|номер заявки|код заказа|код операции)",
        left_context_40,
        flags=re.IGNORECASE,
    ):
        if not re.search(
            r"(тел\.?|телефон|моб\.?|для связи)",
            left_context_40,
            flags=re.IGNORECASE,
        ):
            return True

    return False
