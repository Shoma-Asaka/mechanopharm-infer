from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

ENDPOINT_REQUIRED_COLUMNS = {"c", "m", "response"}
TIMECOURSE_REQUIRED_COLUMNS = {"time", "c", "m", "value"}

_ENDPOINT_ALIASES = {
    "concentration": "c",
    "dose": "c",
    "mechanics": "m",
    "mechanical_condition": "m",
    "stiffness": "m",
    "effect": "response",
    "value": "response",
    "signal": "response",
}

_TIMECOURSE_ALIASES = {
    "concentration": "c",
    "dose": "c",
    "mechanics": "m",
    "mechanical_condition": "m",
    "stiffness": "m",
    "timepoint": "time",
    "response": "value",
    "signal": "value",
}

_ENDPOINT_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "replicate": pd.NA,
    "dataset_id": "dataset_1",
    "system": "unknown",
    "assay": "unknown",
    "condition_label": pd.NA,
    "unit_concentration": pd.NA,
    "unit_mechanics": pd.NA,
    "batch": pd.NA,
    "control_flag": False,
}

_TIMECOURSE_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "replicate": pd.NA,
    "dataset_id": "dataset_1",
    "system": "unknown",
    "assay": "unknown",
    "condition_label": pd.NA,
    "unit_concentration": pd.NA,
    "unit_mechanics": pd.NA,
    "batch": pd.NA,
    "control_flag": False,
}


CANONICAL_ENDPOINT_COLUMN_ORDER = [
    "dataset_id",
    "system",
    "assay",
    "condition_label",
    "batch",
    "replicate",
    "unit_concentration",
    "unit_mechanics",
    "control_flag",
    "c",
    "m",
    "response",
]

CANONICAL_TIMECOURSE_COLUMN_ORDER = [
    "dataset_id",
    "system",
    "assay",
    "condition_label",
    "batch",
    "replicate",
    "unit_concentration",
    "unit_mechanics",
    "control_flag",
    "time",
    "c",
    "m",
    "value",
]


def _rename_known_aliases(df: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        key = column.strip().lower()
        if key in aliases and aliases[key] not in df.columns:
            rename_map[column] = aliases[key]
    return df.rename(columns=rename_map)


def _coerce_numeric_columns(df: pd.DataFrame, columns: set[str], kind: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception as exc:
            raise ValueError(f"{kind} column '{col}' could not be converted to numeric") from exc
    return out


def _validate_required_columns(df: pd.DataFrame, required: set[str], kind: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{kind} data is missing required columns: {sorted(missing)}")


def _validate_no_missing_required_values(df: pd.DataFrame, required: set[str], kind: str) -> None:
    missing_mask = df[list(required)].isnull().any(axis=1)
    if bool(missing_mask.any()):
        n_bad = int(missing_mask.sum())
        raise ValueError(
            f"{kind} data contains missing values in required columns for {n_bad} row(s)"
        )


def _add_optional_columns(df: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def _finalize_column_order(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    front = [col for col in order if col in df.columns]
    extras = [col for col in df.columns if col not in front]
    return df[front + extras].copy()


def standardize_endpoint_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_known_aliases(df.copy(), _ENDPOINT_ALIASES)
    _validate_required_columns(out, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    out = _coerce_numeric_columns(out, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    _validate_no_missing_required_values(out, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    out = _add_optional_columns(out, _ENDPOINT_OPTIONAL_DEFAULTS)
    out["control_flag"] = out["control_flag"].fillna(False).astype(bool)
    return _finalize_column_order(out, CANONICAL_ENDPOINT_COLUMN_ORDER)


def standardize_timecourse_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_known_aliases(df.copy(), _TIMECOURSE_ALIASES)
    _validate_required_columns(out, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    out = _coerce_numeric_columns(out, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    _validate_no_missing_required_values(out, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    out = _add_optional_columns(out, _TIMECOURSE_OPTIONAL_DEFAULTS)
    out["control_flag"] = out["control_flag"].fillna(False).astype(bool)
    return _finalize_column_order(out, CANONICAL_TIMECOURSE_COLUMN_ORDER)
