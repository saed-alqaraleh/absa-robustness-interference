from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

VALID_LABELS = ("negative", "neutral", "positive")

@dataclass(frozen=True)
class AspectExample:
    domain: str
    split: str
    sentence_id: str
    sentence: str
    aspect_term: str
    aspect_from: int
    aspect_to: int
    label: str
    source_instance_id: str
    cluster_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class OpinionSpan:
    start: int
    end: int
    text: str
    provenance: str = "TOWE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class OpinionRecord:
    source_instance_id: str
    cluster_id: str
    spans: tuple[OpinionSpan, ...]
    matched_by: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spans"] = [s.to_dict() for s in self.spans]
        return d

@dataclass(frozen=True)
class GeneratedExample:
    domain: str
    source_split: str
    source_instance_id: str
    cluster_id: str
    original_sentence: str
    original_aspect: str
    original_label: str
    perturbation_family: str
    transformed_sentence: str
    transformed_aspect: str
    transformed_label: str
    modified_aspects: tuple[str, ...] = field(default_factory=tuple)
    generation_rule: str = "arts_inspired_v1"
    generation_seed: int = 0
    eligibility_reason: str = ""
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modified_aspects"] = list(self.modified_aspects)
        return d
