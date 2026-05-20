from functools import lru_cache
from src.ner_redaction.pipeline import RedactionPipeline
from app.config import MODEL_PATH


@lru_cache(maxsize=1)
def get_pipeline() -> RedactionPipeline:

    return RedactionPipeline(model_path=MODEL_PATH)
