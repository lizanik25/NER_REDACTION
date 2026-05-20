import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.ner_redaction.anonymizer import TextAnonymizer


@pytest.fixture
def anonymizer() -> TextAnonymizer:
    return TextAnonymizer()


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.analyze.return_value = (
        [],
        {"text_length": 40, "chunks_count": 1, "truncated": False},
    )
    pipeline.anonymize.return_value = ("текст без ПД", [])
    return pipeline


@pytest.fixture
def test_client(mock_pipeline: MagicMock) -> TestClient:

    from app.main import app
    from app.dependencies import get_pipeline

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
