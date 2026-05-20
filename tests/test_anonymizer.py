import pytest
from src.ner_redaction.anonymizer import TextAnonymizer


def _ent(start: int, end: int, label: str, text: str = "") -> dict:
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": text,
        "score": 1.0,
        "source": "test",
    }


class TestReplaceMode:
    def test_single_entity_replaced(self, anonymizer):
        text = "Привет, Иван Петров!"
        entities = [_ent(8, 19, "PERSON", "Иван Петров")]
        result, _ = anonymizer.anonymize(text, entities, mode="replace")
        assert result == "Привет, [PERSON]!"

    def test_multiple_entities_replaced(self, anonymizer):
        text = "Иван Петров написал на ivan@mail.ru"
        entities = [
            _ent(0, 11, "PERSON", "Иван Петров"),
            _ent(23, 35, "EMAIL", "ivan@mail.ru"),
        ]
        result, _ = anonymizer.anonymize(text, entities, mode="replace")
        assert result == "[PERSON] написал на [EMAIL]"

    def test_all_labels_produce_correct_tag(self, anonymizer):
        for label in ("PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"):
            text = "X" * 10
            entities = [_ent(0, 10, label, "X" * 10)]
            result, _ = anonymizer.anonymize(text, entities, mode="replace")
            assert result == f"[{label}]", f"Wrong tag for label {label}"

    def test_empty_entities_returns_original_text(self, anonymizer):
        text = "Обычный текст без персональных данных"
        result, processed = anonymizer.anonymize(text, [], mode="replace")
        assert result == text
        assert processed == []


class TestMaskMode:
    def test_email_domain_preserved(self, anonymizer):
        text = "Пишите на anna@mail.ru"
        entities = [_ent(10, 22, "EMAIL", "anna@mail.ru")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert "@mail.ru" in result
        assert "anna" not in result

    def test_email_local_part_masked_with_asterisks(self, anonymizer):
        text = "user@corp.ru"
        entities = [_ent(0, 12, "EMAIL", "user@corp.ru")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert "*" in result.split("@")[0]

    def test_phone_contains_asterisks(self, anonymizer):
        text = "+7 (495) 123-45-67"
        entities = [_ent(0, 18, "PHONE", "+7 (495) 123-45-67")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert "*" in result

    def test_phone_last_digits_visible(self, anonymizer):
        text = "+7 (495) 123-45-67"
        entities = [_ent(0, 18, "PHONE", "+7 (495) 123-45-67")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert "67" in result

    def test_person_starts_with_first_char(self, anonymizer):
        text = "Петров Иван"
        entities = [_ent(0, 11, "PERSON", "Петров Иван")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert result.startswith("П")
        assert "*" in result

    def test_id_last_four_chars_visible(self, anonymizer):
        text = "7743013902"
        entities = [_ent(0, 10, "ID", "7743013902")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert "3902" in result


class TestPseudonymizeMode:
    def test_entity_replaced_with_label_and_counter(self, anonymizer):
        text = "Звонил Иван Петров"
        entities = [_ent(7, 18, "PERSON", "Иван Петров")]
        result, _ = anonymizer.anonymize(text, entities, mode="pseudonymize")
        assert "PERSON_1" in result

    def test_same_value_gets_same_pseudonym(self, anonymizer):
        text = "Иван Петров и снова Иван Петров"
        entities = [
            _ent(0, 11, "PERSON", "Иван Петров"),
            _ent(20, 31, "PERSON", "Иван Петров"),
        ]
        result, _ = anonymizer.anonymize(text, entities, mode="pseudonymize")
        left, right = result.split(" и снова ")
        assert left == right

    def test_different_values_get_different_pseudonyms(self, anonymizer):
        text = "Иван Петров и Мария Иванова"
        entities = [
            _ent(0, 11, "PERSON", "Иван Петров"),
            _ent(14, 27, "PERSON", "Мария Иванова"),
        ]
        result, _ = anonymizer.anonymize(text, entities, mode="pseudonymize")
        assert "PERSON_1" in result
        assert "PERSON_2" in result

    def test_counters_are_per_label(self, anonymizer):
        text = "anna@test.ru, ivan@test.ru, Иван Петров"
        entities = [
            _ent(0, 12, "EMAIL", "anna@test.ru"),
            _ent(14, 26, "EMAIL", "ivan@test.ru"),
            _ent(28, 39, "PERSON", "Иван Петров"),
        ]
        result, _ = anonymizer.anonymize(text, entities, mode="pseudonymize")
        assert "EMAIL_1" in result
        assert "EMAIL_2" in result
        assert "PERSON_1" in result


class TestProcessedEntities:
    def test_processed_entity_has_replacement_field(self, anonymizer):
        text = "Иван Петров"
        entities = [_ent(0, 11, "PERSON", "Иван Петров")]
        _, processed = anonymizer.anonymize(text, entities, mode="replace")
        assert processed[0]["replacement"] == "[PERSON]"

    def test_processed_entity_has_anonymization_mode(self, anonymizer):
        text = "Иван Петров"
        entities = [_ent(0, 11, "PERSON", "Иван Петров")]
        _, processed = anonymizer.anonymize(text, entities, mode="mask")
        assert processed[0]["anonymization_mode"] == "mask"

    def test_original_text_preserved_in_processed(self, anonymizer):
        text = "Пишите anna@corp.ru"
        entities = [_ent(7, 19, "EMAIL", "anna@corp.ru")]
        _, processed = anonymizer.anonymize(text, entities, mode="replace")
        assert processed[0]["text"] == "anna@corp.ru"


class TestEdgeCases:
    def test_invalid_mode_raises_value_error(self, anonymizer):
        with pytest.raises(ValueError, match="Unsupported"):
            anonymizer.anonymize("текст", [_ent(0, 5, "PERSON", "текст")], mode="delete")

    def test_overlapping_entities_resolved_without_error(self, anonymizer):
        text = "Иван Петров"
        entities = [
            _ent(0, 11, "PERSON", "Иван Петров"),
            _ent(5, 11, "PERSON", "Петров"),
        ]
        result, _ = anonymizer.anonymize(text, entities, mode="replace")
        assert "[PERSON]" in result

    def test_single_char_value_masked(self, anonymizer):
        text = "А"
        entities = [_ent(0, 1, "PERSON", "А")]
        result, _ = anonymizer.anonymize(text, entities, mode="mask")
        assert result == "*"
