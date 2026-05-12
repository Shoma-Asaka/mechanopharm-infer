"""Core dataclasses and metadata types.

Terminology in this module follows the unified glossary of the theory paper
(Asaka, *A Thermodynamically Constrained Minimal Theory of Mechanopharmacology*).
Wherever applicable, three coarse-grained response classes are referred to as

* ``less_responsive`` (state 0)
* ``responsive`` (state 1)
* ``protected`` (state 2)

and the experimentally controlled inputs are the normalized chemical
concentration ``c`` and the reduced mechanical descriptor ``m``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    """Architecture-class inference result.

    Attributes
    ----------
    label
        One of ``two_state_supported``, ``protected_state_suggested``,
        ``inconclusive``.
    evidence_flags
        Boolean detection flags for each fingerprint.
    evidence_strengths
        ``{fingerprint: strength}`` mapping where strength is one of
        ``not_assessable``, ``none``, ``weak``, ``moderate``, ``strong``.
    fingerprint_values
        Optional structured payload exposing the numerical fingerprint values
        with bootstrap CIs (e.g. ``c_rev`` estimate, mean ``m_opt``,
        ``E_peak``, ``t_peak``, ``E_inf``).  Useful for downstream programmatic
        consumers and for JSON serialization.
    """

    label: str
    evidence_flags: dict[str, bool]
    notes: list[str]
    confidence: str
    evidence_strengths: dict[str, str] | None = None
    supporting_evidence: list[str] | None = None
    counterpoints: list[str] | None = None
    warnings: list[str] | None = None
    fingerprint_values: dict[str, Any] | None = None


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

_ALLOWED_READOUT_LEVELS = {
    # Following the theory's S vs E distinction: a proximal readout reports a
    # signaling-level quantity (e.g. channel current, calcium, phosphorylation)
    # while a phenotypic readout reports a slower downstream functional
    # outcome (e.g. viability, growth arrest, apoptosis).
    "proximal",
    "phenotypic",
    "unspecified",
}


@dataclass(frozen=True)
class AssayMetadata:
    """Minimal assay-facing metadata used for preprocessing.

    Parameters
    ----------
    response_mode
        Whether higher numerical values denote a stronger biological effect.
    assay_family
        Free-form label of the experimental setting (membrane / cell--substrate
        / flow / confined-multicellular / generic).  See the assay-mapping
        section of the theory paper.
    normalization_mode
        How the raw readout is normalized before fingerprint extraction.
    baseline_definition
        How the baseline reference is obtained when a normalization mode
        requires it.
    readout_level
        Whether the value column should be interpreted as a proximal signaling
        quantity ``S`` (the theory's primary signaling output) or a phenotypic
        outcome ``E``.  This is an annotation only -- it does not change
        numerical processing -- but it is recorded in outputs so that
        fingerprint interpretation is unambiguous downstream.
    """

    response_mode: str = "higher_is_stronger_effect"
    assay_family: str = "generic"
    normalization_mode: str = "raw"
    baseline_definition: str = "none"
    readout_level: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    if out.readout_level not in _ALLOWED_READOUT_LEVELS:
        raise ValueError(
            "readout_level must be one of "
            f"{sorted(_ALLOWED_READOUT_LEVELS)}, got {out.readout_level!r}"
        )
    return out
