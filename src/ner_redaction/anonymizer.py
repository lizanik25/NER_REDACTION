import re


SUPPORTED_OPERATORS = {"replace", "mask", "pseudonymize"}


class TextAnonymizer:
    def anonymize(
        self,
        text: str,
        entities: list[dict],
        mode: str = "replace",
    ) -> tuple[str, list[dict]]:
        if mode not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported anonymization mode: {mode}. "
                f"Supported modes: {sorted(SUPPORTED_OPERATORS)}"
            )

        pseudonyms = {}
        counters = {}

        prepared_entities = []

        for ent in sorted(entities, key=lambda e: e["start"]):
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            original = text[start:end]

            replacement = self._replacement(
                value=original,
                label=label,
                mode=mode,
                pseudonyms=pseudonyms,
                counters=counters,
            )

            processed_ent = dict(ent)
            processed_ent["text"] = original
            processed_ent["replacement"] = replacement
            processed_ent["anonymization_mode"] = mode

            prepared_entities.append(processed_ent)

        result = text

        for ent in sorted(prepared_entities, key=lambda e: e["start"], reverse=True):
            result = result[:ent["start"]] + ent["replacement"] + result[ent["end"]:]

        return result, prepared_entities

    def _replacement(
        self,
        value: str,
        label: str,
        mode: str,
        pseudonyms: dict,
        counters: dict,
    ) -> str:
        if mode == "mask":
            return self._mask_by_label(value, label)

        if mode == "pseudonymize":
            if value not in pseudonyms:
                counters[label] = counters.get(label, 0) + 1
                pseudonyms[value] = f"{label}_{counters[label]}"
            return pseudonyms[value]

        return f"[{label}]"

    def _mask_by_label(self, value: str, label: str) -> str:
        if label == "EMAIL":
            return self._mask_email(value)

        if label == "PHONE":
            return self._mask_phone(value)

        if label == "ID":
            return self._mask_id(value)

        if label == "PERSON":
            return self._mask_person(value)

        if label == "ADDRESS":
            return self._mask_address(value)

        return self._mask_generic(value)

    def _mask_email(self, value: str) -> str:
        compact = re.sub(r"\s+", "", value)

        if "@" not in compact:
            return self._mask_generic(value)

        local, domain = compact.split("@", 1)

        if len(local) <= 2:
            masked_local = local[0] + "*" * max(1, len(local) - 1)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def _mask_phone(self, value: str) -> str:
        chars = list(value)
        digit_positions = [i for i, ch in enumerate(chars) if ch.isdigit()]

        if len(digit_positions) <= 4:
            return self._mask_generic(value)

        keep = set(digit_positions[:1] + digit_positions[-2:])

        for i in digit_positions:
            if i not in keep:
                chars[i] = "*"

        return "".join(chars)

    def _mask_id(self, value: str) -> str:
        chars = list(value)
        alnum_positions = [i for i, ch in enumerate(chars) if ch.isalnum()]

        if len(alnum_positions) <= 4:
            return self._mask_generic(value)

        keep = set(alnum_positions[-4:])

        for i in alnum_positions:
            if i not in keep:
                chars[i] = "*"

        return "".join(chars)

    def _mask_person(self, value: str) -> str:
        parts = value.split()
        masked_parts = []

        for part in parts:
            if len(part) <= 1:
                masked_parts.append("*")
            else:
                masked_parts.append(part[0] + "*" * (len(part) - 1))

        return " ".join(masked_parts)

    def _mask_address(self, value: str) -> str:
        tokens = re.split(r"(\s+)", value)
        masked = []

        for token in tokens:
            if token.isspace():
                masked.append(token)
            elif token.isdigit():
                masked.append("*" * len(token))
            elif len(token) <= 2:
                masked.append(token)
            else:
                masked.append(token[0] + "*" * (len(token) - 1))

        return "".join(masked)

    def _mask_generic(self, value: str) -> str:
        if len(value) <= 2:
            return "*" * len(value)

        return value[0] + "*" * (len(value) - 2) + value[-1]