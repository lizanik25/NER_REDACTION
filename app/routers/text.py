from fastapi import APIRouter, Depends
from app.models import DeidentifyTextRequest
from app.dependencies import get_pipeline
from app.services.processing import validate_entities_list, process_text_item
from src.ner_redaction.pipeline import RedactionPipeline

router = APIRouter(tags=["Text"])


@router.post("/deidentify-text", summary="Deidentify plain text")
def deidentify_text(
    request: DeidentifyTextRequest,
    pipeline: RedactionPipeline = Depends(get_pipeline),
):

    entities_filter = validate_entities_list(request.entities)

    return process_text_item(
        pipeline=pipeline,
        text=request.text,
        mode=request.mode.value,
        entities_filter=entities_filter,
        threshold=request.threshold,
        allowlist=request.allowlist,
        denylist=request.denylist,
    )


@router.post("/deidentify-ui", summary="Deidentify text (UI alias)")
def deidentify_ui(
    request: DeidentifyTextRequest,
    pipeline: RedactionPipeline = Depends(get_pipeline),
):
    return deidentify_text(request, pipeline)
