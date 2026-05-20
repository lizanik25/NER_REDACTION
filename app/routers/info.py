from fastapi import APIRouter
from app.config import SUPPORTED_ENTITIES, SUPPORTED_OPERATORS, PIPELINE_NAME

router = APIRouter(tags=["Info"])


@router.get("/health", summary="Health check")
def health():
    return {"status": "ok", "pipeline": PIPELINE_NAME, "version": "0.1.0"}


@router.get("/supported-entities", summary="Supported entity types")
def supported_entities():
    return {
        "supported_entities": SUPPORTED_ENTITIES,
        "count": len(SUPPORTED_ENTITIES),
        "pipeline": PIPELINE_NAME,
    }


@router.get("/operators", summary="Supported anonymization modes")
def operators():
    return {"operators": SUPPORTED_OPERATORS, "count": len(SUPPORTED_OPERATORS)}
