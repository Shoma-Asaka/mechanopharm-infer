from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class LoadedTable:
    path: str
    data: pd.DataFrame
    kind: str


@dataclass
class DiscriminationResult:
    label: str
    evidence_flags: dict[str, bool]
    notes: list[str]
    confidence: str
