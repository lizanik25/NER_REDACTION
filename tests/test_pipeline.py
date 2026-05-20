import pytest
from unittest.mock import MagicMock, patch

from src.ner_redaction.pipeline import RedactionPipeline


def _make_pipeline(predict_return: list | None = None) -> RedactionPipeline:
    """Создаёт RedactionPipeline с замоканным HybridPIIExtractor."""
    with patch("src.ner_redaction.pipeline.HybridPIIExtractor") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.predict_one.return_value = predict_return or []
        mock_cls.return_value = mock_instance
        return RedactionPipeline(model_path="fake/path")


@pytest.fixture
def pipeline() -> RedactionPipeline:
    return _make_pipeline()


@pytest.fixture
def pipeline_with_entities() -> RedactionPipeline:
    sample = [
        {
            "start": 0, "end": 11, "label": "PERSON", "text": "Иван Петров",
            "score": 0.9, "source": "ml", "source_component": "ml",
        },
        {
            "start": 23, "end": 35, "label": "EMAIL", "text": "ivan@mail.ru",
            "score": 1.0, "source": "rule", "source_component": "rule",
        },
    ]
    return _make_pipeline(predict_return=sample)


class TestPipelineInit:
    def test_zero_chunk_size_raises(self):
        with patch("src.ner_redaction.pipeline.HybridPIIExtractor"):
            with pytest.raises(ValueError, match="chunk_size"):
                RedactionPipeline(model_path="x", chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with patch("src.ner_redaction.pipeline.HybridPIIExtractor"):
            with pytest.raises(ValueError, match="chunk_size"):
                RedactionPipeline(model_path="x", chunk_size=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        with patch("src.ner_redaction.pipeline.HybridPIIExtractor"):
            with pytest.raises(ValueError, match="chunk_overlap"):
                RedactionPipeline(model_path="x", chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with patch("src.ner_redaction.pipeline.HybridPIIExtractor"):
            with pytest.raises(ValueError, match="chunk_overlap"):
                RedactionPipeline(model_path="x", chunk_size=100, chunk_overlap=200)

    def test_valid_params_no_error(self):
        with patch("src.ner_redaction.pipeline.HybridPIIExtractor"):
            p = RedactionPipeline(model_path="x", chunk_size=500, chunk_overlap=50)
            assert p.chunk_size == 500
            assert p.chunk_overlap == 50


class TestValidation:
    def test_non_string_text_raises_type_error(self, pipeline):
        with pytest.raises(TypeError, match="str"):
            pipeline.analyze(12345)

    def test_none_text_raises_type_error(self, pipeline):
        with pytest.raises(TypeError):
            pipeline.analyze(None)

    def test_list_as_text_raises_type_error(self, pipeline):
        with pytest.raises(TypeError):
            pipeline.analyze(["текст"])

    def test_unsupported_entity_raises_value_error(self, pipeline):
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline.analyze("текст", entities=["UNKNOWN_TYPE"])

    def test_threshold_above_one_raises(self, pipeline):
        with pytest.raises(ValueError, match="threshold"):
            pipeline.analyze("текст", threshold=1.1)

    def test_threshold_below_zero_raises(self, pipeline):
        with pytest.raises(ValueError, match="threshold"):
            pipeline.analyze("текст", threshold=-0.1)

    def test_unsupported_mode_raises(self, pipeline):
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline.anonymize("текст", [], mode="remove")


class TestAnalyze:
    def test_empty_text_returns_empty_entities(self, pipeline):
        entities, metadata = pipeline.analyze("   ")
        assert entities == []
        assert metadata["chunks_count"] == 0

    def test_metadata_contains_required_keys(self, pipeline):
        _, metadata = pipeline.analyze("Привет, мир")
        assert "text_length" in metadata
        assert "chunks_count" in metadata
        assert "truncated" in metadata

    def test_text_length_matches(self, pipeline):
        text = "Тестовый текст"
        _, metadata = pipeline.analyze(text)
        assert metadata["text_length"] == len(text)

    def test_entity_filter_keeps_only_requested_label(self, pipeline_with_entities):
        text = "Иван Петров написал на ivan@mail.ru"
        entities, _ = pipeline_with_entities.analyze(text, entities=["EMAIL"])
        assert all(e["label"] == "EMAIL" for e in entities)
        assert not any(e["label"] == "PERSON" for e in entities)

    def test_threshold_filters_low_confidence_entities(self):
        low_score_entity = [
            {
                "start": 0, "end": 11, "label": "PERSON", "text": "Иван Петров",
                "score": 0.3, "source": "ml", "source_component": "ml",
            }
        ]
        p = _make_pipeline(predict_return=low_score_entity)
        entities, _ = p.analyze("Иван Петров", threshold=0.5)
        assert entities == []

    def test_threshold_zero_keeps_all_entities(self):
        low_score_entity = [
            {
                "start": 0, "end": 11, "label": "PERSON", "text": "Иван Петров",
                "score": 0.3, "source": "ml", "source_component": "ml",
            }
        ]
        p = _make_pipeline(predict_return=low_score_entity)
        entities, _ = p.analyze("Иван Петров", threshold=0.0)
        assert len(entities) == 1


class TestTextSplitting:
    def test_short_text_single_chunk(self, pipeline):
        text = "Короткий текст"
        chunks = pipeline._split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == (0, text)

    def test_long_text_multiple_chunks(self, pipeline):
        text = "А " * 1000
        chunks = pipeline._split_text(text)
        assert len(chunks) > 1

    def test_all_positions_covered(self, pipeline):
        text = "Слово " * 400
        chunks = pipeline._split_text(text)
        covered = set()
        for start, chunk in chunks:
            for i in range(len(chunk)):
                covered.add(start + i)
        assert covered == set(range(len(text)))

    def test_chunk_offsets_are_correct(self, pipeline):
        text = "А " * 1000
        chunks = pipeline._split_text(text)
        for start, chunk in chunks:
            assert text[start:start + len(chunk)] == chunk
