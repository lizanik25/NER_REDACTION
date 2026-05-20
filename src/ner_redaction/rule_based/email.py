import re

from yargy import Parser, rule, or_
from yargy.predicates import eq, custom

from .utils import resolve_overlaps


class YargyEmailDetector:

    def __init__(self):
        self.parser = self._build_yargy_parser()
        self.email_regex = re.compile(
            r"\b[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9._+\-]*"
            r"@"
            r"[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\-]*"
            r"(?:\.[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\-]*)+",
            flags=re.IGNORECASE,
        )

    def _build_yargy_parser(self) -> Parser:

        def token_value(t):
            return t.value if hasattr(t, "value") else str(t)

        def token_pred(pattern):
            return custom(
                lambda t: bool(re.fullmatch(pattern, token_value(t), flags=re.IGNORECASE))
            )

        LOCAL_PART  = token_pred(r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*")
        DOMAIN_PART = token_pred(r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*")
        TLD         = token_pred(r"[A-ZА-ЯЁ0-9]{2,20}")

        EMAIL_FULL_TOKEN = token_pred(
            r"[A-ZА-ЯЁ0-9]+(?:[._+\-][A-ZА-ЯЁ0-9]+)*"
            r"@"
            r"[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*"
            r"(?:\.[A-ZА-ЯЁ0-9]+(?:-[A-ZА-ЯЁ0-9]+)*|,[A-ZА-ЯЁ]{2,15})+"
        )

        DOT_OR_COMMA = or_(rule(eq(".")), rule(eq(",")))

        EMAIL_BASIC = rule(LOCAL_PART, eq("@"), DOMAIN_PART, DOT_OR_COMMA, TLD)

        EMAIL_MULTI_DOMAIN = rule(
            LOCAL_PART, eq("@"), DOMAIN_PART, eq("."), DOMAIN_PART, DOT_OR_COMMA, TLD
        )

        EMAIL_THREE_DOMAIN = rule(
            LOCAL_PART, eq("@"), DOMAIN_PART,
            eq("."), DOMAIN_PART, eq("."), DOMAIN_PART,
            DOT_OR_COMMA, TLD,
        )

        EMAIL = or_(
            rule(EMAIL_FULL_TOKEN),
            EMAIL_THREE_DOMAIN,
            EMAIL_MULTI_DOMAIN,
            EMAIL_BASIC,
        )

        return Parser(EMAIL)

    def _normalize(self, value: str) -> str:
        return value.strip().replace(",", ".")

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        while end > start and text[end - 1] in ".,;:!?)]}":
            end -= 1
        return start, end

    def _is_bad_candidate(self, text: str, start: int, end: int) -> bool:

        start, end = self._trim_span(text, start, end)
        value   = text[start:end]
        compact = self._normalize(value)

        if start > 0 and re.match(r"[A-Za-zА-Яа-яЁё0-9._+\-]", text[start - 1]):
            return True

        if end < len(text) and re.match(r"[A-Za-zА-Яа-яЁё0-9_\-]", text[end]):
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

    def _extend_span(self, text: str, start: int, end: int) -> tuple[int, int]:

        window_start = max(0, start - 100)
        window_end   = min(len(text), end + 100)
        window       = text[window_start:window_end]
        best         = None

        for m in self.email_regex.finditer(window):
            cs = window_start + m.start()
            ce = window_start + m.end()
            cs, ce = self._trim_span(text, cs, ce)

            if cs <= start and ce >= end:
                if best is None or (ce - cs) > (best[1] - best[0]):
                    best = (cs, ce)

        return best if best else self._trim_span(text, start, end)

    def _add_entity(
        self,
        entities: list[dict],
        text: str,
        start: int,
        end: int,
        source: str,
    ) -> None:
        start, end = self._extend_span(text, start, end)
        start, end = self._trim_span(text, start, end)

        if start >= end:
            return

        if self._is_bad_candidate(text, start, end):
            return

        entities.append({
            "start": start,
            "end": end,
            "label": "EMAIL",
            "text": text[start:end],
            "score": 1.0,
            "source": source,
            "recognizer": "YargyEmailDetector",
        })

    def predict_one(self, text: str) -> list[dict]:

        if not text:
            return []

        entities: list[dict] = []

        for match in self.parser.findall(text):
            self._add_entity(entities, text, match.span.start, match.span.stop, "yargy_email")

        for m in self.email_regex.finditer(text):
            self._add_entity(entities, text, m.start(), m.end(), "regex_email")

        return resolve_overlaps(entities)
