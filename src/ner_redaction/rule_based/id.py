import re

from yargy import Parser, rule, or_
from yargy.predicates import eq, caseless, custom

from .utils import resolve_overlaps


class YargyIdDetector:


    def __init__(self):
        self.parser = self._build_yargy_parser()

    def _build_yargy_parser(self) -> Parser:

        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE))
            )

        DIG1_2  = digits_pred(r"\d{1,2}")
        DIG2    = digits_pred(r"\d{2}")
        DIG3    = digits_pred(r"\d{3}")
        DIG4    = digits_pred(r"\d{4}")
        DIG1_20 = digits_pred(r"\d{1,20}")
        DIG11   = digits_pred(r"\d{11}")
        DIG12   = digits_pred(r"\d{12}")
        RU_LETTERS = token_pred(r"[А-ЯЁA-Z]{1,6}")
        ALNUM      = token_pred(r"[A-ZА-ЯЁ0-9]{2,20}")
        NUM_SIGN   = eq("№")

        ID_MARKER = or_(
            rule(caseless("ID")), rule(caseless("id")), rule(caseless("user_id")),
        )
        ID_CONTEXT_MARKER = or_(
            rule(caseless("ID"), caseless("клиента")),
            rule(caseless("id"), caseless("клиента")),
            rule(caseless("клиента")),
            rule(caseless("регистрационный"), caseless("ID")),
            rule(caseless("регистрационный"), caseless("id")),
            rule(caseless("идентификатор")),
        )
        DOC_MARKER = or_(
            rule(caseless("договор")), rule(caseless("договором")),
            rule(caseless("контракт")), rule(caseless("контрактом")),
            rule(caseless("регистрационный"), caseless("контракт")),
            rule(caseless("заявка")), rule(caseless("заявки")), rule(caseless("заявок")),
            rule(caseless("номер"), caseless("заявки")),
            rule(caseless("номер"), caseless("заявок")),
            rule(caseless("номера"), caseless("заявок")),
            rule(caseless("регистрационные"), caseless("заявки")),
            rule(caseless("идентификаторы"), caseless("заявок")),
            rule(caseless("идентификатор"), caseless("заявки")),
            rule(caseless("документ")), rule(caseless("идентификатор")),
            rule(caseless("номер"), caseless("документа")),
            rule(caseless("номер"), caseless("договора")),
            rule(caseless("номер"), caseless("контракта")),
            rule(caseless("номер"), caseless("обращения")),
            rule(caseless("регистрационный"), caseless("номер")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("заявки")),
        )
        INN_MARKER   = rule(caseless("ИНН"))
        SNILS_MARKER = rule(caseless("СНИЛС"))
        ACCOUNT_MARKER = or_(
            rule(caseless("счет")), rule(caseless("счёт")),
            rule(caseless("номер"), caseless("счета")),
            rule(caseless("номер"), caseless("счёта")),
            rule(caseless("номером"), caseless("счета")),
            rule(caseless("номером"), caseless("счёта")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счета")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счёта")),
            rule(caseless("р"), eq("/"), caseless("с")),
        )

        DOC_NUMBER_ALNUM = or_(
            rule(RU_LETTERS, eq("-"), DIG4, eq("/"), DIG1_2, eq("-"), DIG1_2),
            rule(DIG4, eq("-"), DIG1_20),
            rule(DIG3, eq("-"), DIG3, eq("-"), ALNUM),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, DIG2),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, eq("/"), DIG1_2),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, eq("-"), DIG2),
            rule(DIG1_2, eq("-"), DIG3, eq("-"), RU_LETTERS),
            rule(DIG1_20, eq("-"), DIG1_20),
            rule(DIG1_20, eq("-"), RU_LETTERS),
            rule(DIG1_20),
        )

        SIMPLE_ID = or_(
            rule(ID_MARKER, eq(":").optional(), eq("=").optional(), DIG1_20),
            rule(ID_CONTEXT_MARKER, eq(":").optional(), NUM_SIGN.optional(), DIG1_20),
            rule(NUM_SIGN, DIG1_20),
        )
        DOC_ID = or_(
            rule(DOC_MARKER, eq(":").optional(), NUM_SIGN.optional(), DOC_NUMBER_ALNUM),
            rule(NUM_SIGN, DOC_NUMBER_ALNUM),
        )
        INN_ID = rule(INN_MARKER, eq(":").optional(), DIG12)
        SNILS_FORMATTED = rule(
            SNILS_MARKER, eq(":").optional(),
            DIG3, eq("-"), DIG3, eq("-"), DIG3, DIG2,
        )
        SNILS_SOLID = rule(SNILS_MARKER, eq(":").optional(), DIG11)
        GOVERNMENT_ID  = or_(INN_ID, SNILS_FORMATTED, SNILS_SOLID)
        BANK_ACCOUNT   = rule(ACCOUNT_MARKER, eq(":").optional(), DIG1_20)
        ENUM_WITH_MARKER = or_(
            rule(caseless("заявки"), eq(":").optional(), DIG1_20),
            rule(caseless("заявка"), eq(":").optional(), DIG1_20),
            rule(caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номера"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номер"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("идентификаторы"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("регистрационные"), caseless("заявки"), eq(":").optional(), DIG1_20),
        )

        ID = or_(GOVERNMENT_ID, BANK_ACCOUNT, DOC_ID, SIMPLE_ID, ENUM_WITH_MARKER)
        return Parser(ID)

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:

        patterns = [
            r"^(?:ID|id|user_id)\s*[:=]?\s*",
            r"^(?:клиента)\s*:?\s*№?\s*",
            r"^(?:ИНН|СНИЛС)\s*:?\s*",
            r"^(?:№)\s*",
            r"^(?:ом)\s*:?\s*№?\s*",
            r"^(?:а)\s+(?=\d)",
            r"^(?:документ)\s*:?\s*№?\s*",
            r"^(?:договор|договором|договора|контракт|контрактом|контракта"
            r"|регистрационный\s+контракт)\s*:?\s*№?\s*",
            r"^(?:заявка|заявки|заявок|номер\s+заявки|номер\s+заявок"
            r"|номера\s+заявок|регистрационные\s+заявки"
            r"|идентификаторы\s+заявок|идентификатор\s+заявки"
            r"|регистрационный\s+номер\s+заявки)\s*:?\s*№?\s*",
            r"^(?:номер\s+документа|номер\s+договора|номер\s+контракта"
            r"|номер\s+обращения|регистрационный\s+номер)\s*:?\s*№?\s*",
            r"^(?:ID\s+клиента|id\s+клиента|регистрационный\s+ID"
            r"|регистрационный\s+id|идентификатор)\s*:?\s*№?\s*",
            r"^(?:счет|счёт|счета|счёта|номером\s+счета|номером\s+счёта"
            r"|номер\s+счета|номер\s+счёта|регистрационный\s+номер\s+счета"
            r"|регистрационный\s+номер\s+счёта|р/с)\s*:?\s*",
        ]
        changed = True
        while changed:
            changed = False
            value = text[start:end]
            for pattern in patterns:
                m = re.match(pattern, value, flags=re.IGNORECASE)
                if m:
                    start += m.end()
                    changed = True
                    break
        return start, end

    def _extend_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        value = text[start:end]
        if re.fullmatch(r"\d{1,4}", value):
            tail = text[end:end + 40]
            m = re.match(
                r"(?:-\d{1,6})?(?:-[A-ZА-ЯЁ0-9]{1,20})?(?:/\d{1,4})?",
                tail,
                flags=re.IGNORECASE,
            )
            if m:
                end += m.end()
        return start, end

    def _split_enumerated(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        value = text[start:end]
        if re.fullmatch(r"\d{1,20}(?:\s*,\s*\d{1,20})+", value):
            return [
                (start + m.start(), start + m.end())
                for m in re.finditer(r"\d{1,20}", value)
            ]
        return [(start, end)]

    def _is_bad_candidate(self, text: str, start: int, end: int) -> bool:

        value      = text[start:end]
        value_lower = value.lower()
        digits     = re.sub(r"\D", "", value)
        left_ctx   = text[max(0, start - 80):start].lower()
        right_ctx  = text[end:end + 30]

        if re.search(
            r"(тел\.?|телефон|моб\.?|контактный номер|номер телефона|для связи)\s*[:\-]?\s*$",
            left_ctx, flags=re.IGNORECASE,
        ):
            return True

        if re.search(
            r"(школ[аеуыи]?|поликлиник[аеуыи]?|изолятор[аеуыи]?|общежити[ея]"
            r"|дом|кв\.?|квартира|офис|оф\.?|стр\.?|корп\.?|рейс|шаг|этап)\s*$",
            left_ctx, flags=re.IGNORECASE,
        ):
            return True

        if re.fullmatch(r"\d{1,3}-\d{3}", value) and re.match(
            r"-\d{2,3}-\d{2}-\d{2}", right_ctx
        ):
            return True

        has_marker = re.search(
            r"(id|user_id|инн|снилс|№|договор|контракт|заявк"
            r"|сч[её]т|р/с|номер|регистрационный|идентификатор|клиента)",
            value_lower + " " + left_ctx,
            flags=re.IGNORECASE,
        )
        if len(digits) <= 3 and not has_marker:
            return True

        return False

    def _add_entity(
        self, entities: list[dict], text: str, start: int, end: int, source: str
    ) -> None:

        if self._is_bad_candidate(text, start, end):
            return

        start, end = self._trim_span(text, start, end)
        start, end = self._trim_span(text, start, end)  # второй проход для вложенных маркеров
        start, end = self._extend_span(text, start, end)

        if start >= end:
            return

        for sub_start, sub_end in self._split_enumerated(text, start, end):
            if self._is_bad_candidate(text, sub_start, sub_end):
                continue
            entities.append({
                "start": sub_start,
                "end": sub_end,
                "label": "ID",
                "text": text[sub_start:sub_end],
                "score": 1.0,
                "source": source,
                "recognizer": "YargyIdDetector",
            })

    def predict_one(self, text: str) -> list[dict]:

        if not text:
            return []

        entities: list[dict] = []

        for match in self.parser.findall(text):
            self._add_entity(entities, text, match.span.start, match.span.stop, "yargy_id")

        for pattern in [
            r"(?:идентификаторы|номера|номер|регистрационные)?\s*заяв(?:ок|ки)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
            r"(?:идентификаторы|идентификатор)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
        ]:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                nums_start = m.start(1)
                for num in re.finditer(r"\d{1,20}", m.group(1)):
                    s = nums_start + num.start()
                    e = nums_start + num.end()
                    entities.append({
                        "start": s, "end": e, "label": "ID",
                        "text": text[s:e], "score": 1.0,
                        "source": "yargy_id_enum", "recognizer": "YargyIdDetector",
                    })

        return resolve_overlaps(entities)


class ContextIdDetector:


    def __init__(self):
        strong_context = (
            r"код\s+заказа|код\s+операции|код\s+товара"
            r"|внутренний\s+код(?:\s+товара)?"
            r"|внутренний\s+идентификатор|идентификатор"
            r"|номер\s+договора|номер\s+сч[её]та|регистрационный\s+номер"
        )
        self.patterns = [
            rf"(?P<context>{strong_context})\s+(?P<value>\+?\d[\d\s\-.()–—]{{3,30}}\d)",
            rf"(?P<context>{strong_context})\s+(?P<value>[A-ZА-ЯЁ0-9]+(?:[-./][A-ZА-ЯЁ0-9]+)+)",
            r"(?P<context>указал\s+индекс)\s+(?P<value>\d{7,9})",
            r"(?P<context>индекс\s+отправления)\s+(?P<value>\d{7,9})",
            r"(?P<context>указан\s+код)\s+(?P<value>\d{3}\s+\d{3}\s+\d{2}\s+\d{2})",
            r"(?P<context>используется\s+номер)\s+(?P<value>\d{3}-\d{3}-\d{4})(?=\s+как\s+идентификатор)",
        ]

    def predict_one(self, text: str) -> list[dict]:

        if not text:
            return []

        entities: list[dict] = []

        for pattern in self.patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                start = m.start("value")
                end   = m.end("value")

                while end > start and text[end - 1] in ".,;:!?)]}":
                    end -= 1

                value  = text[start:end]
                digits = re.sub(r"\D", "", value)

                if len(digits) < 4:
                    continue

                if re.search(r"\d+\s*[–—]\s*\d+", value):
                    continue

                entities.append({
                    "start": start, "end": end,
                    "label": "ID", "text": value,
                    "score": 1.0, "source": "context_id",
                    "recognizer": "ContextIdDetector",
                    "source_component": "rule",
                    "source_detector": "context_id",
                })

        return entities
