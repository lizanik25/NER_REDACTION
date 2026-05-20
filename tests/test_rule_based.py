import pytest
from src.ner_redaction.rule_based.extractor import RuleBasedPIIExtractor


@pytest.fixture(scope="module")
def extractor() -> RuleBasedPIIExtractor:
    return RuleBasedPIIExtractor()


def _labels(entities: list[dict]) -> set[str]:
    return {e["label"] for e in entities}


class TestEmailDetection:
    def test_simple_email_detected(self, extractor):
        result = extractor.predict_one("Напишите нам на support@example.ru")
        assert any(e["label"] == "EMAIL" for e in result)

    def test_email_span_matches_address(self, extractor):
        text = "Почта: ivan@mail.ru, звоните"
        result = extractor.predict_one(text)
        emails = [e for e in result if e["label"] == "EMAIL"]
        assert len(emails) >= 1
        found_texts = [text[e["start"]:e["end"]] for e in emails]
        assert any("ivan@mail.ru" in t for t in found_texts)

    def test_email_with_subdomain_detected(self, extractor):
        result = extractor.predict_one("Контакт: user@corp.company.com")
        assert any(e["label"] == "EMAIL" for e in result)

    def test_plain_text_no_email(self, extractor):
        result = extractor.predict_one("Сегодня хорошая погода, без писем")
        assert not any(e["label"] == "EMAIL" for e in result)



class TestPhoneDetection:
    @pytest.mark.parametrize("text", [
        "Телефон: +7 (495) 123-45-67",
        "Звоните: 8 800 555 35 35",
        "Тел. +79991234567",
    ])
    def test_phone_detected(self, extractor, text):
        result = extractor.predict_one(text)
        assert any(e["label"] == "PHONE" for e in result), \
            f"Телефон не найден в тексте: {text!r}"

    def test_plain_number_not_phone(self, extractor):
        result = extractor.predict_one("В 2024 году вышло 15 новых продуктов")
        assert not any(e["label"] == "PHONE" for e in result)



class TestIdDetection:
    def test_inn_detected(self, extractor):
        result = extractor.predict_one("ИНН организации: 7743013902")
        assert any(e["label"] == "ID" for e in result)

    def test_snils_detected(self, extractor):
        result = extractor.predict_one("СНИЛС сотрудника: 112-233-445 95")
        assert any(e["label"] == "ID" for e in result)


class TestExtractorGeneral:
    def test_empty_text_returns_empty_list(self, extractor):
        assert extractor.predict_one("") == []

    def test_whitespace_only_returns_empty_list(self, extractor):
        assert extractor.predict_one("   \n\t  ") == []

    def test_multiple_types_in_one_text(self, extractor):
        text = "Контакт: anna@corp.ru, тел. +7 (999) 888-77-66"
        result = extractor.predict_one(text)
        found = _labels(result)
        assert "EMAIL" in found
        assert "PHONE" in found

    def test_no_overlapping_spans(self, extractor):
        text = "Данные: +7 (495) 000-00-00, ivan@test.ru, ИНН 7743013902"
        result = extractor.predict_one(text)
        sorted_result = sorted(result, key=lambda e: e["start"])
        for i in range(len(sorted_result) - 1):
            assert sorted_result[i]["end"] <= sorted_result[i + 1]["start"], (
                f"Пересекающиеся спаны: {sorted_result[i]} и {sorted_result[i + 1]}"
            )

    def test_entity_dict_has_required_keys(self, extractor):
        result = extractor.predict_one("Телефон: +7 (999) 123-45-67")
        assert len(result) > 0
        for ent in result:
            for key in ("start", "end", "label", "text"):
                assert key in ent, f"Отсутствует ключ {key!r} в сущности: {ent}"

    def test_span_text_matches_source_text(self, extractor):
        text = "Почта: hello@world.ru и телефон +7 (999) 000-11-22"
        result = extractor.predict_one(text)
        for ent in result:
            extracted = text[ent["start"]:ent["end"]]
            assert ent["text"] in extracted or extracted in ent["text"], (
                f"Текст сущности не совпадает с позицией в тексте: {ent}"
            )
