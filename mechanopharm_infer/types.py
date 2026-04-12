from __future__ import annotations

from dataclasses import dataclass, field
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
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | bool] = field(default_factory=dict)


@dataclass
class DiscriminationResult:
    label: str
    evidence_flags: dict[str, bool]
    notes: list[str]
    confidence: str
