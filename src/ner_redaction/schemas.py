from pydantic import BaseModel, Field
from typing import Literal


EntityLabel = Literal["PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"]
AnonymizationMode = Literal["replace", "mask", "pseudonymize"]


class Entity(BaseModel):
    start: int
    end: int
    label: EntityLabel
    score: float | None = None
    source: str | None = None
    source_component: str | None = None
    source_detector: str | None = None
    recognizer: str | None = None
    text: str | None = None
    replacement: str | None = None
    anonymization_mode: AnonymizationMode | None = None


class AnalyzeRequest(BaseModel):
    text: str
    entities: list[EntityLabel] | None = None
    return_text: bool = False


class AnalyzeResponse(BaseModel):
    entities: list[Entity]
    entities_count: int
    entities_count_by_type: dict[str, int] = Field(default_factory=dict)
    text_length: int
    chunks_count: int
    truncated: bool = False
    pipeline: str = "hybrid"


class AnonymizeRequest(BaseModel):
    text: str
    entities: list[Entity]
    mode: AnonymizationMode = "replace"


class AnonymizeResponse(BaseModel):
    anonymized_text: str
    entities: list[Entity]
    entities_count: int
    entities_count_by_type: dict[str, int] = Field(default_factory=dict)
    text_length: int
    mode: AnonymizationMode
    pipeline: str = "hybrid"


class DeidentifyRequest(BaseModel):
    text: str
    mode: AnonymizationMode = "replace"
    entities: list[EntityLabel] | None = None
    return_entities: bool = True


class DeidentifyResponse(BaseModel):
    anonymized_text: str
    entities: list[Entity] = Field(default_factory=list)
    entities_count: int
    entities_count_by_type: dict[str, int] = Field(default_factory=dict)
    text_length: int
    chunks_count: int
    truncated: bool = False
    mode: AnonymizationMode
    pipeline: str = "hybrid"


class BatchDeidentifyRequest(BaseModel):
    texts: list[str]
    mode: AnonymizationMode = "replace"
    entities: list[EntityLabel] | None = None
    return_entities: bool = True


class BatchDeidentifyItemResponse(BaseModel):
    item_index: int
    anonymized_text: str
    entities: list[Entity] = Field(default_factory=list)
    entities_count: int
    entities_count_by_type: dict[str, int] = Field(default_factory=dict)
    text_length: int
    chunks_count: int
    truncated: bool = False
    mode: AnonymizationMode
    pipeline: str = "hybrid"


class BatchDeidentifyResponse(BaseModel):
    results: list[BatchDeidentifyItemResponse]
    items_count: int
    mode: AnonymizationMode
    pipeline: str = "hybrid"