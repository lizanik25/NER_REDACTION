import pytest
from app.services.processing import (
    apply_allowlist,
    apply_denylist,
    count_entities_by_type,
    parse_entities_param,
    parse_word_list_param,
    validate_entities_list,
)
from app.services.highlighting import build_highlight_segments


class TestValidateEntitiesList:
    def test_none_returns_none(self):
        assert validate_entities_list(None) is None

    def test_valid_entities_normalized(self):
        result = validate_entities_list(["person", "email"])
        assert result == ["PERSON", "EMAIL"]

    def test_unsupported_raises_http_400(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validate_entities_list(["UNKNOWN"])
        assert exc_info.value.status_code == 400

    def test_empty_list_returns_empty(self):
        result = validate_entities_list([])
        assert result == []


class TestParseEntitiesParam:
    def test_comma_separated(self):
        result = parse_entities_param("PERSON,EMAIL")
        assert result == ["PERSON", "EMAIL"]

    def test_none_returns_none(self):
        assert parse_entities_param(None) is None

    def test_empty_string_returns_none(self):
        assert parse_entities_param("") is None

    def test_spaces_trimmed(self):
        result = parse_entities_param(" PERSON , EMAIL ")
        assert result == ["PERSON", "EMAIL"]


class TestParseWordListParam:
    def test_comma_separated(self):
        result = parse_word_list_param("Иванов,Петров")
        assert result == ["Иванов", "Петров"]

    def test_none_returns_empty(self):
        assert parse_word_list_param(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_word_list_param("") == []


class TestCountEntitiesByType:
    def test_counts_correctly(self):
        entities = [
            {"label": "PERSON"}, {"label": "PERSON"}, {"label": "EMAIL"}
        ]
        result = count_entities_by_type(entities)
        assert result == {"PERSON": 2, "EMAIL": 1}

    def test_empty_list(self):
        assert count_entities_by_type([]) == {}


class TestApplyAllowlist:
    def _make_entity(self, text, label="PERSON"):
        return {"start": 0, "end": len(text), "label": label, "text": text}

    def test_removes_allowlisted_word(self):
        entities = [self._make_entity("Иванов")]
        result = apply_allowlist(entities, ["Иванов"])
        assert result == []

    def test_case_insensitive(self):
        entities = [self._make_entity("иванов")]
        result = apply_allowlist(entities, ["Иванов"])
        assert result == []

    def test_keeps_non_allowlisted(self):
        entities = [self._make_entity("Петров")]
        result = apply_allowlist(entities, ["Иванов"])
        assert len(result) == 1

    def test_empty_allowlist_returns_all(self):
        entities = [self._make_entity("Иванов")]
        result = apply_allowlist(entities, [])
        assert len(result) == 1


class TestApplyDenylist:
    def test_adds_denylist_word(self):
        text = "Компания ПЕРВАЯ заключила договор"
        result = apply_denylist(text, [], ["ПЕРВАЯ"])
        assert len(result) == 1
        assert result[0]["label"] == "DENYLIST"
        assert result[0]["text"] == "ПЕРВАЯ"

    def test_empty_denylist_returns_unchanged(self):
        entities = [{"start": 0, "end": 5, "label": "PERSON", "text": "Иван"}]
        result = apply_denylist("Иван работает", entities, [])
        assert result == entities

    def test_no_overlap_with_existing_entity(self):
        text = "Светлана ПЕРВАЯ"
        entities = [{"start": 0, "end": 6, "label": "PERSON", "text": "Светлана"}]
        result = apply_denylist(text, entities, ["ПЕРВАЯ"])
        denylist_items = [e for e in result if e["label"] == "DENYLIST"]
        assert len(denylist_items) == 1

    def test_overlap_not_added(self):
        text = "Иванов"
        entities = [{"start": 0, "end": 6, "label": "PERSON", "text": "Иванов"}]
        result = apply_denylist(text, entities, ["Иванов"])
        # "Иванов" перекрывается с уже найденной сущностью — не добавляем
        denylist_items = [e for e in result if e["label"] == "DENYLIST"]
        assert len(denylist_items) == 0


class TestBuildHighlightSegments:
    def test_no_entities_returns_single_text_segment(self):
        text = "Привет всем"
        segments = build_highlight_segments(text, [])
        assert len(segments) == 1
        assert segments[0]["type"] == "text"
        assert segments[0]["text"] == "Привет всем"

    def test_entity_in_middle(self):
        text = "Привет Иванов всем"
        entities = [{"start": 7, "end": 13, "label": "PERSON",
                     "text": "Иванов", "replacement": "[PERSON]",
                     "score": 1.0, "source": "ml"}]
        segments = build_highlight_segments(text, entities)
        types = [s["type"] for s in segments]
        assert types == ["text", "entity", "text"]

    def test_entity_at_start(self):
        text = "Иванов работает"
        entities = [{"start": 0, "end": 6, "label": "PERSON",
                     "text": "Иванов", "replacement": "[PERSON]",
                     "score": 1.0, "source": "ml"}]
        segments = build_highlight_segments(text, entities)
        assert segments[0]["type"] == "entity"

    def test_entity_at_end(self):
        text = "Звони Иванову"
        entities = [{"start": 6, "end": 13, "label": "PERSON",
                     "text": "Иванову", "replacement": "[PERSON]",
                     "score": 1.0, "source": "ml"}]
        segments = build_highlight_segments(text, entities)
        assert segments[-1]["type"] == "entity"

    def test_invalid_entity_skipped(self):
        text = "Привет всем"
        entities = [{"start": 5, "end": 3, "label": "PERSON",
                     "text": "xx", "replacement": "[PERSON]",
                     "score": 1.0, "source": "ml"}]
        segments = build_highlight_segments(text, entities)
        assert all(s["type"] == "text" for s in segments)

    def test_ui_id_assigned(self):
        text = "Звони Иванову"
        entities = [{"start": 6, "end": 13, "label": "PERSON",
                     "text": "Иванову", "replacement": "[PERSON]",
                     "score": 1.0, "source": "ml"}]
        segments = build_highlight_segments(text, entities)
        entity_segs = [s for s in segments if s["type"] == "entity"]
        assert all("ui_id" in s for s in entity_segs)
