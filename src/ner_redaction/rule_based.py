import re

from yargy import Parser, rule, or_
from yargy.predicates import eq, caseless, custom


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

    list_tail = re.match(
        r"(?:\s*[,/]\s*\d{2}){1,4}",
        tail,
    )

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
        r"(тел\.?|телефон|моб\.?|номер телефона|контактный телефон|контактный номер|для связи)\s*[:\-]?\s*$",
        left_context_40,
        flags=re.IGNORECASE,
    )

    bad_context = re.search(
        r"(номер документа|регистрационный номер|номер обращения|id клиента|user_id|id|инн|р/с|счет|номер счета|идентификатор)\s*[:=]?\s*$",
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
        r"(код|артикул|номер договора|номер сч[её]та|номер заявки|код заказа|код операции)",
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


class YargyPhoneDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(
                        pattern,
                        token_value(t),
                        flags=re.IGNORECASE | re.VERBOSE,
                    )
                )
            )

        DIG1 = digits_pred(r"\d{1}")
        DIG2 = digits_pred(r"\d{2}")
        DIG3 = digits_pred(r"\d{3}")
        DIG4 = digits_pred(r"\d{4}")
        DIG7 = digits_pred(r"\d{7}")
        DIG10 = digits_pred(r"\d{10}")
        DIG11 = digits_pred(r"\d{11}")
        DIG1_2 = digits_pred(r"\d{1,2}")
        DIG2_3 = digits_pred(r"\d{2,3}")
        DIG2_4 = digits_pred(r"\d{2,4}")
        DIG1_6 = digits_pred(r"\d{1,6}")
        DIG6 = digits_pred(r"\d{6}")

        PHONE_COMPACT_TOKEN = token_pred(
            r"""
            (?:
                \+?7[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                |
                8[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                |
                8\(\d{3}\)\d{7}
                |
                \+7\(\d{3}\)\d{7}
                |
                \d{3}-\d{3}-\d{3}\s?\d{2}
                |
                \d{4}-\d{7}
            )
            """
        )

        CODE_PAREN_TOKEN = token_pred(r"\(\d{3}\)")

        SEP = or_(rule(eq("-")), rule(eq(".")))
        PLUS7 = rule(eq("+"), eq("7"))

        PREFIX = or_(
            PLUS7,
            rule(eq("7")),
            rule(eq("8")),
        )

        CODE_PAREN = or_(
            rule(eq("("), DIG3, eq(")")),
            rule(CODE_PAREN_TOKEN),
        )

        CODE = or_(
            rule(DIG3),
            CODE_PAREN,
        )

        PHONE_SINGLE_TOKEN = rule(PHONE_COMPACT_TOKEN)
        PHONE_PLUS_SOLID = rule(eq("+"), DIG11)
        PHONE_BROKEN_PAREN = rule(PLUS7, eq("("), DIG10)

        PHONE_DOT_FULL = rule(
            PLUS7, eq("."), DIG3, eq("."), DIG3, eq("."), DIG2, eq("."), DIG2
        )

        PHONE_SOLID_11 = rule(DIG11)

        PHONE_8_3_4_3 = rule(eq("8"), DIG3, DIG4, DIG3)

        PHONE_3_3_3_2 = rule(
            DIG3, eq("-"), DIG3, eq("-"), DIG3, DIG2
        )

        PHONE_WITH_PREFIX = rule(
            PREFIX,
            SEP.optional(),
            CODE,
            SEP.optional(),
            DIG2_4,
            SEP.optional(),
            DIG1_2,
            SEP.optional(),
            DIG2_3,
        )

        PHONE_PREFIX_CODE_7DIG = rule(
            PREFIX,
            CODE,
            DIG7,
        )

        PHONE_PAREN_NO_PREFIX = rule(
            CODE_PAREN,
            DIG3,
            eq("-"),
            DIG2,
            eq("-"),
            DIG2,
        )

        PHONE_NO_PREFIX_GROUPED = rule(
            DIG3,
            DIG3,
            DIG2,
            DIG2,
        )

        PHONE_WITHOUT_PREFIX = or_(
            rule(DIG10),
            rule(
                CODE,
                SEP.optional(),
                DIG3,
                SEP.optional(),
                DIG2,
                SEP.optional(),
                DIG2,
            ),
        )

        LOCAL_PHONE = or_(
            rule(DIG3, eq("-"), DIG2, eq("-"), DIG2),
            rule(DIG2, eq("-"), DIG2, eq("-"), DIG2),
            rule(DIG1, eq("-"), DIG2, eq("-"), DIG2),
        )

        PHONE_WEIRD = or_(
            rule(DIG4, eq("-"), DIG7),
            rule(eq("8"), DIG3, SEP, DIG7),
            rule(eq("8"), DIG6, DIG2, DIG2),
            rule(PLUS7, DIG10),
        )

        LIST_TAIL = or_(
            rule(eq("/"), DIG2),
            rule(eq(","), DIG2),
            rule(eq(","), DIG2, eq(","), DIG2),
        )

        EXT_MARKER = or_(
            rule(caseless("доб")),
            rule(caseless("вн")),
            rule(caseless("ext")),
            rule(eq("#")),
        )

        EXTENSION = or_(
            rule(EXT_MARKER, eq(".").optional(), DIG1_6),
            rule(eq("("), caseless("вн"), eq(".").optional(), DIG1_6, eq(")")),
        )

        BASE_PHONE = or_(
            PHONE_SINGLE_TOKEN,
            PHONE_PLUS_SOLID,
            PHONE_BROKEN_PAREN,
            PHONE_DOT_FULL,
            PHONE_PREFIX_CODE_7DIG,
            PHONE_8_3_4_3,
            PHONE_3_3_3_2,
            PHONE_PAREN_NO_PREFIX,
            PHONE_NO_PREFIX_GROUPED,
            PHONE_WITH_PREFIX,
            PHONE_WEIRD,
            PHONE_WITHOUT_PREFIX,
            PHONE_SOLID_11,
            LOCAL_PHONE,
        )

        PHONE_WORDS = rule(
            caseless("восемь"),
            caseless("девятьсот"),
            caseless("двадцать"),
            caseless("шесть"),
            caseless("сто"),
            caseless("двадцать"),
            caseless("три"),
            caseless("сорок"),
            caseless("пять"),
            caseless("шестьдесят"),
            caseless("семь"),
        )

        PHONE = or_(
            rule(
                BASE_PHONE,
                LIST_TAIL.optional(),
                EXTENSION.optional(),
            ),
            PHONE_WORDS,
        )

        self.parser = Parser(PHONE)

        self.phone_regexes = [
            r"(?<!\d)(?:\+7|8)\.\d{3}\.\d{3}\.\d{2}\.\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\(\d{3}\)\d{7}(?!\d)",
            r"(?<!\d)(?:\+?7|8)\d{10}(?!\d)",
            r"(?<!\d)(?:\+7|8)[-\s]\d{3}[-\s]\d{3}[-\s]\d{2}[-\s]\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{3}[\s-]\d{2}[\s-]\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{7}(?!\d)",
            r"(?<!\d)(?:\+7|8|7)\s+\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",
            r"(?<!\d)\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",
            r"(?<!\d)\(\d{3}\)\s*\d{3}-\d{2}-\d{2}(?!\d)",
            r"(?<!\d)\+?7\d{10}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",
            r"(?<!\d)8\s*\(\d{3}\)\s*\d{7}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",
            r"(?<![\d-])\d{3}-\d{2}-\d{2}(?:\s*(?:[,/]\s*\d{2}){1,4})?(?![\d-])",
            r"(?<!\d)\d{1,2}-\d{2}-\d{2}(?!\d)",
        ]

    def _add_entity(self, entities: list[dict], text: str, start: int, end: int, source: str) -> None:
        if is_bad_phone_candidate(text, start, end, source=source):
            return

        start, end = extend_phone_span(text, start, end)

        entities.append(
            {
                "start": start,
                "end": end,
                "label": "PHONE",
                "text": text[start:end],
                "score": 1.0,
                "source": source,
                "recognizer": "YargyPhoneDetector",
            }
        )

    def predict_one(self, text: str) -> list[dict]:
        entities = []

        for match in self.parser.findall(text):
            self._add_entity(
                entities=entities,
                text=text,
                start=match.span.start,
                end=match.span.stop,
                source="yargy_phone",
            )

        for pattern in self.phone_regexes:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                self._add_entity(
                    entities=entities,
                    text=text,
                    start=m.start(),
                    end=m.end(),
                    source="regex_phone",
                )

        return resolve_overlaps(entities)
    

class YargyIdDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE)
                )
            )

        DIG1_2 = digits_pred(r"\d{1,2}")
        DIG2 = digits_pred(r"\d{2}")
        DIG3 = digits_pred(r"\d{3}")
        DIG4 = digits_pred(r"\d{4}")
        DIG1_20 = digits_pred(r"\d{1,20}")
        DIG11 = digits_pred(r"\d{11}")
        DIG12 = digits_pred(r"\d{12}")

        RU_LETTERS = token_pred(r"[А-ЯЁA-Z]{1,6}")
        ALNUM = token_pred(r"[A-ZА-ЯЁ0-9]{2,20}")

        NUM_SIGN = eq("№")

        ID_MARKER = or_(
            rule(caseless("ID")),
            rule(caseless("id")),
            rule(caseless("user_id")),
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
            rule(caseless("договор")),
            rule(caseless("договором")),
            rule(caseless("контракт")),
            rule(caseless("контрактом")),
            rule(caseless("регистрационный"), caseless("контракт")),
            rule(caseless("заявка")),
            rule(caseless("заявки")),
            rule(caseless("заявок")),
            rule(caseless("номер"), caseless("заявки")),
            rule(caseless("номер"), caseless("заявок")),
            rule(caseless("номера"), caseless("заявок")),
            rule(caseless("регистрационные"), caseless("заявки")),
            rule(caseless("идентификаторы"), caseless("заявок")),
            rule(caseless("идентификатор"), caseless("заявки")),
            rule(caseless("документ")),
            rule(caseless("идентификатор")),
            rule(caseless("номер"), caseless("документа")),
            rule(caseless("номер"), caseless("договора")),
            rule(caseless("номер"), caseless("контракта")),
            rule(caseless("номер"), caseless("обращения")),
            rule(caseless("регистрационный"), caseless("номер")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("заявки")),
        )

        INN_MARKER = rule(caseless("ИНН"))
        SNILS_MARKER = rule(caseless("СНИЛС"))

        ACCOUNT_MARKER = or_(
            rule(caseless("счет")),
            rule(caseless("счёт")),
            rule(caseless("номер"), caseless("счета")),
            rule(caseless("номер"), caseless("счёта")),
            rule(caseless("номером"), caseless("счета")),
            rule(caseless("номером"), caseless("счёта")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счета")),
            rule(caseless("регистрационный"), caseless("номер"), caseless("счёта")),
            rule(caseless("р"), eq("/"), caseless("с")),
        )

        SIMPLE_ID = or_(
            rule(ID_MARKER, eq(":").optional(), eq("=").optional(), DIG1_20),
            rule(ID_CONTEXT_MARKER, eq(":").optional(), NUM_SIGN.optional(), DIG1_20),
            rule(NUM_SIGN, DIG1_20),
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

        DOC_ID = or_(
            rule(DOC_MARKER, eq(":").optional(), NUM_SIGN.optional(), DOC_NUMBER_ALNUM),
            rule(NUM_SIGN, DOC_NUMBER_ALNUM),
        )

        INN_ID = rule(INN_MARKER, eq(":").optional(), DIG12)

        SNILS_FORMATTED = rule(
            SNILS_MARKER,
            eq(":").optional(),
            DIG3,
            eq("-"),
            DIG3,
            eq("-"),
            DIG3,
            DIG2,
        )

        SNILS_SOLID_WITH_MARKER = rule(
            SNILS_MARKER,
            eq(":").optional(),
            DIG11,
        )

        GOVERNMENT_ID = or_(
            INN_ID,
            SNILS_FORMATTED,
            SNILS_SOLID_WITH_MARKER,
        )

        BANK_ACCOUNT = rule(
            ACCOUNT_MARKER,
            eq(":").optional(),
            DIG1_20,
        )

        ENUM_WITH_MARKER = or_(
            rule(caseless("заявки"), eq(":").optional(), DIG1_20),
            rule(caseless("заявка"), eq(":").optional(), DIG1_20),
            rule(caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номера"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("номер"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("идентификаторы"), caseless("заявок"), eq(":").optional(), DIG1_20),
            rule(caseless("регистрационные"), caseless("заявки"), eq(":").optional(), DIG1_20),
        )

        ID = or_(
            GOVERNMENT_ID,
            BANK_ACCOUNT,
            DOC_ID,
            SIMPLE_ID,
            ENUM_WITH_MARKER,
        )

        self.parser = Parser(ID)

    def trim_id_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        patterns = [
            r"^(?:ID|id|user_id)\s*[:=]?\s*",
            r"^(?:клиента)\s*:?\s*№?\s*",
            r"^(?:ИНН|СНИЛС)\s*:?\s*",
            r"^(?:№)\s*",
            r"^(?:ом)\s*:?\s*№?\s*",
            r"^(?:а)\s+(?=\d)",
            r"^(?:документ)\s*:?\s*№?\s*",
            r"^(?:договор|договором|договора|контракт|контрактом|контракта|регистрационный\s+контракт)\s*:?\s*№?\s*",
            r"^(?:заявка|заявки|заявок|номер\s+заявки|номер\s+заявок|номера\s+заявок|регистрационные\s+заявки|идентификаторы\s+заявок|идентификатор\s+заявки|регистрационный\s+номер\s+заявки)\s*:?\s*№?\s*",
            r"^(?:номер\s+документа|номер\s+договора|номер\s+контракта|номер\s+обращения|регистрационный\s+номер)\s*:?\s*№?\s*",
            r"^(?:ID\s+клиента|id\s+клиента|регистрационный\s+ID|регистрационный\s+id|идентификатор)\s*:?\s*№?\s*",
            r"^(?:счет|счёт|счета|счёта|номером\s+счета|номером\s+счёта|номер\s+счета|номер\s+счёта|регистрационный\s+номер\s+счета|регистрационный\s+номер\s+счёта|р/с)\s*:?\s*",
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

    def extend_id_span(self, text: str, start: int, end: int) -> tuple[int, int]:
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

    def split_enumerated_ids(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        value = text[start:end]

        if re.fullmatch(r"\d{1,20}(?:\s*,\s*\d{1,20})+", value):
            return [
                (start + m.start(), start + m.end())
                for m in re.finditer(r"\d{1,20}", value)
            ]

        return [(start, end)]

    def _is_bad_id_candidate(self, text: str, start: int, end: int) -> bool:
        value = text[start:end]
        value_lower = value.lower()
        digits = re.sub(r"\D", "", value)

        left_context_80 = text[max(0, start - 80):start].lower()
        right_context_30 = text[end:end + 30]

        phone_context = re.search(
            r"(тел\.?|телефон|моб\.?|контактный номер|номер телефона|для связи)\s*[:\-]?\s*$",
            left_context_80,
            flags=re.IGNORECASE,
        )

        if phone_context:
            return True

        bad_left_context = re.search(
            r"(школ[аеуыи]?|поликлиник[аеуыи]?|изолятор[аеуыи]?|общежити[ея]|"
            r"дом|кв\.?|квартира|офис|оф\.?|стр\.?|корп\.?|"
            r"рейс|шаг|этап)\s*$",
            left_context_80,
            flags=re.IGNORECASE,
        )

        if bad_left_context:
            return True

        if re.fullmatch(r"\d{1,3}-\d{3}", value) and re.match(
            r"-\d{2,3}-\d{2}-\d{2}",
            right_context_30,
        ):
            return True

        has_marker = re.search(
            r"(id|user_id|инн|снилс|№|договор|контракт|заявк|сч[её]т|р/с|номер|регистрационный|идентификатор|клиента)",
            value_lower + " " + left_context_80,
            flags=re.IGNORECASE,
        )

        if len(digits) <= 3 and not has_marker:
            return True

        return False

    def _add_entity(self, entities: list[dict], text: str, start: int, end: int, source: str) -> None:
        if self._is_bad_id_candidate(text, start, end):
            return

        start, end = self.trim_id_span(text, start, end)
        start, end = self.trim_id_span(text, start, end)
        start, end = self.extend_id_span(text, start, end)

        if start >= end:
            return

        for sub_start, sub_end in self.split_enumerated_ids(text, start, end):
            if self._is_bad_id_candidate(text, sub_start, sub_end):
                continue

            entities.append(
                {
                    "start": sub_start,
                    "end": sub_end,
                    "label": "ID",
                    "text": text[sub_start:sub_end],
                    "score": 1.0,
                    "source": source,
                    "recognizer": "YargyIdDetector",
                }
            )

    def predict_one(self, text: str) -> list[dict]:
        entities = []

        for match in self.parser.findall(text):
            self._add_entity(
                entities,
                text,
                match.span.start,
                match.span.stop,
                "yargy_id",
            )

        for m in re.finditer(
            r"(?:идентификаторы|номера|номер|регистрационные)?\s*заяв(?:ок|ки)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
            text,
            flags=re.IGNORECASE,
        ):
            nums_start = m.start(1)
            nums_part = m.group(1)

            for num in re.finditer(r"\d{1,20}", nums_part):
                sub_start = nums_start + num.start()
                sub_end = nums_start + num.end()

                entities.append(
                    {
                        "start": sub_start,
                        "end": sub_end,
                        "label": "ID",
                        "text": text[sub_start:sub_end],
                        "score": 1.0,
                        "source": "yargy_id_enum",
                        "recognizer": "YargyIdDetector",
                    }
                )

        for m in re.finditer(
            r"(?:идентификаторы|идентификатор)\s*:\s*((?:\d{1,20}\s*,\s*)+\d{1,20})",
            text,
            flags=re.IGNORECASE,
        ):
            nums_start = m.start(1)
            nums_part = m.group(1)

            for num in re.finditer(r"\d{1,20}", nums_part):
                sub_start = nums_start + num.start()
                sub_end = nums_start + num.end()

                entities.append(
                    {
                        "start": sub_start,
                        "end": sub_end,
                        "label": "ID",
                        "text": text[sub_start:sub_end],
                        "score": 1.0,
                        "source": "yargy_id_enum",
                        "recognizer": "YargyIdDetector",
                    }
                )

        return resolve_overlaps(entities)
    
class YargyEmailDetector:
    def __init__(self):
        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE)
                )
            )

        LOCAL_PART = token_pred(r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*")
        DOMAIN_PART = token_pred(r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*")
        TLD = token_pred(r"[A-ZА-ЯЁ0-9]{2,20}")

        EMAIL_FULL_TOKEN = token_pred(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
            r"@"
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
            r"(?:\.[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*|,[A-ZА-ЯЁ]{2,15})+"
        )

        DOT_OR_COMMA = or_(rule(eq(".")), rule(eq(",")))

        EMAIL_BASIC = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL_MULTI_DOMAIN = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL_THREE_DOMAIN = rule(
            LOCAL_PART,
            eq("@"),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            eq("."),
            DOMAIN_PART,
            DOT_OR_COMMA,
            TLD,
        )

        EMAIL = or_(
            rule(EMAIL_FULL_TOKEN),
            EMAIL_THREE_DOMAIN,
            EMAIL_MULTI_DOMAIN,
            EMAIL_BASIC,
        )

        self.parser = Parser(EMAIL)

        self.email_regex = re.compile(
            r"\b[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9._+\-]*"
            r"@"
            r"[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\-]*"
            r"(?:\.[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\-]*)+",
            flags=re.IGNORECASE,
        )

    def normalize_email_candidate(self, value: str) -> str:
        compact = value.strip()
        compact = compact.replace(",", ".")
        return compact

    def trim_email_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        while end > start and text[end - 1] in ".,;:!?)]}":
            end -= 1

        return start, end

    def is_bad_email_candidate(self, text: str, start: int, end: int) -> bool:
        start, end = self.trim_email_span(text, start, end)
        value = text[start:end]
        compact = self.normalize_email_candidate(value)

        if start > 0:
            prev_char = text[start - 1]
            if re.match(r"[A-Za-zА-Яа-яЁё0-9._+\-]", prev_char):
                return True

        if end < len(text):
            next_char = text[end]
            if re.match(r"[A-Za-zА-Яа-яЁё0-9_\-]", next_char):
                return True

        if not re.fullmatch(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
            r"@"
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
            r"(?:\.[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*)+",
            compact,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    def extend_email_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        window_start = max(0, start - 100)
        window_end = min(len(text), end + 100)
        window = text[window_start:window_end]

        best = None

        for m in self.email_regex.finditer(window):
            candidate_start = window_start + m.start()
            candidate_end = window_start + m.end()

            candidate_start, candidate_end = self.trim_email_span(
                text,
                candidate_start,
                candidate_end,
            )

            if candidate_start <= start and candidate_end >= end:
                if best is None or (candidate_end - candidate_start) > (best[1] - best[0]):
                    best = (candidate_start, candidate_end)

        if best:
            return best

        return self.trim_email_span(text, start, end)

    def _add_entity(
        self,
        entities: list[dict],
        text: str,
        start: int,
        end: int,
        source: str,
    ) -> None:
        start, end = self.extend_email_span(text, start, end)
        start, end = self.trim_email_span(text, start, end)

        if start >= end:
            return

        if self.is_bad_email_candidate(text, start, end):
            return

        entities.append(
            {
                "start": start,
                "end": end,
                "label": "EMAIL",
                "text": text[start:end],
                "score": 1.0,
                "source": source,
                "recognizer": "YargyEmailDetector",
            }
        )

    def predict_one(self, text: str) -> list[dict]:
        entities = []

        for match in self.parser.findall(text):
            self._add_entity(
                entities,
                text,
                match.span.start,
                match.span.stop,
                "yargy_email",
            )

        for m in self.email_regex.finditer(text):
            self._add_entity(
                entities,
                text,
                m.start(),
                m.end(),
                "regex_email",
            )

        return resolve_overlaps(entities)
    
class ContextIdDetector:
    def __init__(self):
        strong_context = (
            r"код\s+заказа|"
            r"код\s+операции|"
            r"код\s+товара|"
            r"внутренний\s+код(?:\s+товара)?|"
            r"внутренний\s+идентификатор|"
            r"идентификатор|"
            r"номер\s+договора|"
            r"номер\s+сч[её]та|"
            r"регистрационный\s+номер"
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
        entities = []

        for pattern in self.patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                start = m.start("value")
                end = m.end("value")

                while end > start and text[end - 1] in ".,;:!?)]}":
                    end -= 1

                value = text[start:end]
                digits = re.sub(r"\D", "", value)

                if len(digits) < 4:
                    continue

                if re.search(r"\d+\s*[–—]\s*\d+", value):
                    continue

                entities.append(
                    {
                        "start": start,
                        "end": end,
                        "label": "ID",
                        "text": value,
                        "score": 1.0,
                        "source": "context_id",
                        "recognizer": "ContextIdDetector",
                        "source_component": "rule",
                        "source_detector": "context_id",
                    }
                )

        return entities
    
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
            overlap = not (
                ent["end"] <= old["start"] or ent["start"] >= old["end"]
            )

            if overlap:
                ent_len = ent["end"] - ent["start"]
                old_len = old["end"] - old["start"]

                ent_score = (priority.get(ent["label"], 0), ent_len)
                old_score = (priority.get(old["label"], 0), old_len)

                if ent_score > old_score:
                    selected.remove(old)
                else:
                    keep = False

                break

        if keep:
            selected.append(ent)

    return sorted(selected, key=lambda e: e["start"])


class RuleBasedPIIExtractor:

    def __init__(self):
        self.detectors = {
            "ID_CONTEXT": ContextIdDetector(),
            "PHONE": YargyPhoneDetector(),
            "EMAIL": YargyEmailDetector(),
            "ID": YargyIdDetector(),
        }

    def predict_one(self, text: str) -> list[dict]:
        entities = []

        for detector_name, detector in self.detectors.items():
            preds = detector.predict_one(text)

            for ent in preds:
                ent = dict(ent)

                if detector_name == "ID_CONTEXT":
                    ent["label"] = "ID"
                else:
                    ent["label"] = detector_name

                ent["score"] = ent.get("score", 1.0)
                ent["source"] = ent.get("source", detector_name.lower())
                ent["source_component"] = "rule"
                ent["source_detector"] = ent.get("source_detector", ent.get("source", detector_name))

                if "recognizer" not in ent:
                    ent["recognizer"] = detector.__class__.__name__

                if "text" not in ent:
                    ent["text"] = text[ent["start"]:ent["end"]]

                entities.append(ent)

        return resolve_rule_overlaps(entities)
    

