from enum import Enum
from pydantic import BaseModel, Field

from app.config import SUPPORTED_ENTITIES


class OperatorType(str, Enum):
    replace = "replace"
    mask = "mask"
    pseudonymize = "pseudonymize"


class DeidentifyTextRequest(BaseModel):
    text: str = Field(
        ...,
        examples=["Иван Петров написал на ivan@mail.ru"],
        description="Input Russian text for PII detection and anonymization",
    )
    mode: OperatorType = Field(
        default=OperatorType.replace,
        description="Anonymization mode: replace, mask, pseudonymize",
    )
    entities: list[str] | None = Field(
        default=None,
        examples=[["PERSON", "EMAIL", "PHONE"]],
        description=(
            "Entity types to detect. "
            f"Supported: {SUPPORTED_ENTITIES}. "
            "If null, all supported entities are used."
        ),
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Acceptance threshold for detected entities (0.0 = no filtering)",
    )
    allowlist: list[str] = Field(
        default_factory=list,
        description="Words that must NOT be treated as personal data",
    )
    denylist: list[str] = Field(
        default_factory=list,
        description="Words that MUST be treated as personal data even if not detected",
    )
