import re

from yargy import Parser, rule, or_
from yargy.predicates import eq, caseless, custom

from .utils import extend_phone_span, is_bad_phone_candidate, resolve_overlaps


class YargyPhoneDetector:

    def __init__(self):
        self.parser = self._build_yargy_parser()
        self.phone_regexes = self._build_regexes()

    def _build_yargy_parser(self) -> Parser:

        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def digits_pred(pattern):
            return custom(lambda t: bool(re.fullmatch(pattern, token_value(t))))

        def token_pred(pattern):
            return custom(
                lambda t: bool(
                    re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE | re.VERBOSE)
                )
            )

        DIG1  = digits_pred(r"\d{1}")
        DIG2  = digits_pred(r"\d{2}")
        DIG3  = digits_pred(r"\d{3}")
        DIG4  = digits_pred(r"\d{4}")
        DIG7  = digits_pred(r"\d{7}")
        DIG10 = digits_pred(r"\d{10}")
        DIG11 = digits_pred(r"\d{11}")
        DIG1_2 = digits_pred(r"\d{1,2}")
        DIG2_3 = digits_pred(r"\d{2,3}")
        DIG2_4 = digits_pred(r"\d{2,4}")
        DIG1_6 = digits_pred(r"\d{1,6}")
        DIG6   = digits_pred(r"\d{6}")

        PHONE_COMPACT_TOKEN = token_pred(
            r"""
            (?:
                \+?7[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                | 8[\.\-\s]?\(?\d{3}\)?[\.\-\s]?\d{3}[\.\-\s]?\d{2}[\.\-\s]?\d{2}
                | 8\(\d{3}\)\d{7}
                | \+7\(\d{3}\)\d{7}
                | \d{3}-\d{3}-\d{3}\s?\d{2}
                | \d{4}-\d{7}
            )
            """
        )

        CODE_PAREN_TOKEN = token_pred(r"\(\d{3}\)")
        SEP    = or_(rule(eq("-")), rule(eq(".")))
        PLUS7  = rule(eq("+"), eq("7"))
        PREFIX = or_(PLUS7, rule(eq("7")), rule(eq("8")))

        CODE_PAREN = or_(
            rule(eq("("), DIG3, eq(")")),
            rule(CODE_PAREN_TOKEN),
        )
        CODE = or_(rule(DIG3), CODE_PAREN)

        EXT_MARKER = or_(
            rule(caseless("доб")), rule(caseless("вн")),
            rule(caseless("ext")), rule(eq("#")),
        )
        EXTENSION = or_(
            rule(EXT_MARKER, eq(".").optional(), DIG1_6),
            rule(eq("("), caseless("вн"), eq(".").optional(), DIG1_6, eq(")")),
        )
        LIST_TAIL = or_(
            rule(eq("/"), DIG2), rule(eq(","), DIG2),
            rule(eq(","), DIG2, eq(","), DIG2),
        )

        BASE_PHONE = or_(
            rule(PHONE_COMPACT_TOKEN),
            rule(eq("+"), DIG11),
            rule(PLUS7, eq("("), DIG10),
            rule(PLUS7, eq("."), DIG3, eq("."), DIG3, eq("."), DIG2, eq("."), DIG2),
            rule(PREFIX, CODE, DIG7),
            rule(eq("8"), DIG3, DIG4, DIG3),
            rule(DIG3, eq("-"), DIG3, eq("-"), DIG3, DIG2),
            rule(CODE_PAREN, DIG3, eq("-"), DIG2, eq("-"), DIG2),
            rule(DIG3, DIG3, DIG2, DIG2),
            rule(PREFIX, SEP.optional(), CODE, SEP.optional(),
                 DIG2_4, SEP.optional(), DIG1_2, SEP.optional(), DIG2_3),
            or_(rule(DIG4, eq("-"), DIG7), rule(eq("8"), DIG3, SEP, DIG7),
                rule(eq("8"), DIG6, DIG2, DIG2), rule(PLUS7, DIG10)),
            or_(rule(DIG10),
                rule(CODE, SEP.optional(), DIG3, SEP.optional(),
                     DIG2, SEP.optional(), DIG2)),
            rule(DIG11),
            or_(rule(DIG3, eq("-"), DIG2, eq("-"), DIG2),
                rule(DIG2, eq("-"), DIG2, eq("-"), DIG2),
                rule(DIG1, eq("-"), DIG2, eq("-"), DIG2)),
        )

        PHONE_WORDS = rule(
            caseless("восемь"), caseless("девятьсот"), caseless("двадцать"),
            caseless("шесть"), caseless("сто"), caseless("двадцать"),
            caseless("три"), caseless("сорок"), caseless("пять"),
            caseless("шестьдесят"), caseless("семь"),
        )

        PHONE = or_(
            rule(BASE_PHONE, LIST_TAIL.optional(), EXTENSION.optional()),
            PHONE_WORDS,
        )

        return Parser(PHONE)

    def _build_regexes(self) -> list[str]:
        return [
            r"(?<!\d)(?:\+7|8)\.\d{3}\.\d{3}\.\d{2}\.\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\(\d{3}\)\d{7}(?!\d)",
            r"(?<!\d)(?:\+?7|8)\d{10}(?!\d)",
            r"(?<!\d)(?:\+7|8)[-\s]\d{3}[-\s]\d{3}[-\s]\d{2}[-\s]\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{3}[\s-]\d{2}[\s-]\d{2}(?!\d)",
            r"(?<!\d)(?:\+7|8)\s*\(\d{3}\)\s*\d{7}(?!\d)",
            r"(?<!\d)(?:\+7|8|7)\s+\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",
            r"(?<!\d)\d{3}\s+\d{3}\s+\d{2}\s+\d{2}(?!\d)",
            r"(?<!\d)\(\d{3}\)\s*\d{3}-\d{2}-\d{2}(?!\d)",
            r"(?<!\d)\+?7\d{10}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}"
            r"|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",
            r"(?<!\d)8\s*\(\d{3}\)\s*\d{7}(?:\s*(?:доб\.?|вн\.?|ext\.?|#)\s*\d{1,6}"
            r"|\s*\(\s*(?:вн\.?|доб\.?|ext\.?)\s*\d{1,6}\s*\))?(?!\d)",
            r"(?<![\d-])\d{3}-\d{2}-\d{2}(?:\s*(?:[,/]\s*\d{2}){1,4})?(?![\d-])",
            r"(?<!\d)\d{1,2}-\d{2}-\d{2}(?!\d)",
        ]

    def _add_entity(
        self,
        entities: list[dict],
        text: str,
        start: int,
        end: int,
        source: str,
    ) -> None:
        if is_bad_phone_candidate(text, start, end, source=source):
            return

        start, end = extend_phone_span(text, start, end)

        entities.append({
            "start": start,
            "end": end,
            "label": "PHONE",
            "text": text[start:end],
            "score": 1.0,
            "source": source,
            "recognizer": "YargyPhoneDetector",
        })

    def predict_one(self, text: str) -> list[dict]:

        if not text:
            return []

        entities: list[dict] = []

        for match in self.parser.findall(text):
            self._add_entity(entities, text, match.span.start, match.span.stop, "yargy_phone")

        for pattern in self.phone_regexes:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                self._add_entity(entities, text, m.start(), m.end(), "regex_phone")

        return resolve_overlaps(entities)
