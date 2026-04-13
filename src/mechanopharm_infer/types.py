from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import pandas as pd


@dataclass
class LoadedTable:
    path: str
    data: pd.DataFrame
    kind: str


@dataclass
class QCReport:
    kind: str
    passed: bool
    warnings: list[str]
    metrics: dict[str, float | int | bool]


@dataclass
class DiscriminationResult:
    label: str
    evidence_flags: dict[str, bool]
    notes: list[str]
    confidence: str
    evidence_strengths: dict[str, str] | None = None
    supporting_evidence: list[str] | None = None
    counterpoints: list[str] | None = None
    warnings: list[str] | None = None


@dataclass(frozen=True)
class AssayMetadata:
    """Minimal assay-facing metadata used for preprocessing.

    Parameters are intentionally conservative at this stage so that the package
    can accept literature datasets with heterogeneous readouts while preserving
    backward compatibility with the original demo CSV files.
    """

    response_mode: str = "higher_is_stronger_effect"
    assay_family: str = "generic"
    normalization_mode: str = "raw"
    baseline_definition: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ALLOWED_RESPONSE_MODES = {
    "higher_is_stronger_effect",
    "lower_is_stronger_effect",
}

_ALLOWED_NORMALIZATION_MODES = {
    "raw",
    "vehicle_normalized",
    "control_subtracted",
    "min_max",
    "within_mechanics_min_max",
}

_ALLOWED_BASELINE_DEFINITIONS = {
    "none",
    "control_flag",
    "minimum_per_mechanics",
    "global_minimum",
}


def coerce_assay_metadata(metadata: AssayMetadata | Mapping[str, Any] | None) -> AssayMetadata:
    """Return validated assay metadata from either a dataclass or mapping."""

    if metadata is None:
        out = AssayMetadata()
    elif isinstance(metadata, AssayMetadata):
        out = metadata
    elif isinstance(metadata, Mapping):
        out = AssayMetadata(**dict(metadata))
    else:
        raise TypeError("metadata must be None, AssayMetadata, or a mapping")

    if out.response_mode not in _ALLOWED_RESPONSE_MODES:
        raise ValueError(
            "response_mode must be one of "
            f"{sorted(_ALLOWED_RESPONSE_MODES)}, got {out.response_mode!r}"
        )
    if out.normalization_mode not in _ALLOWED_NORMALIZATION_MODES:
        raise ValueError(
            "normalization_mode must be one of "
            f"{sorted(_ALLOWED_NORMALIZATION_MODES)}, got {out.normalization_mode!r}"
        )
    if out.baseline_definition not in _ALLOWED_BASELINE_DEFINITIONS:
        raise ValueError(
            "baseline_definition must be one of "
            f"{sorted(_ALLOWED_BASELINE_DEFINITIONS)}, got {out.baseline_definition!r}"
        )
    return out
