from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..preprocess import prepare_endpoint_data, prepare_timecourse_data
from ..types import AssayMetadata

DataLike = str | Path | pd.DataFrame


def _load_table(data: DataLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _rename_aliases(df: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> pd.DataFrame:
    if not aliases:
        return df.copy()
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in aliases and aliases[key] not in df.columns:
            rename_map[col] = aliases[key]
    return df.rename(columns=rename_map)


def _coerce_named_mechanics(
    df: pd.DataFrame,
    *,
    mechanics_col: str = "m",
    mechanics_map: Mapping[Any, float] | None = None,
    ordered_levels: list[Any] | tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if mechanics_col not in out.columns:
        return out
    series = out[mechanics_col]
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.notna().all()):
        out[mechanics_col] = numeric.astype(float)
        return out

    if mechanics_map is not None:
        mapped = series.map(mechanics_map)
        if bool(mapped.isna().any()):
            missing = sorted({str(v) for v in series[mapped.isna()].unique()})
            raise ValueError(
                f"mechanics_map does not cover all mechanics labels; missing: {missing}"
            )
        out[mechanics_col] = mapped.astype(float)
        return out

    if ordered_levels is not None:
        rank_map = {level: float(i) for i, level in enumerate(ordered_levels)}
        mapped = series.map(rank_map)
        if bool(mapped.isna().any()):
            missing = sorted({str(v) for v in series[mapped.isna()].unique()})
            raise ValueError(
                f"ordered_levels does not cover all mechanics labels; missing: {missing}"
            )
        out[mechanics_col] = mapped.astype(float)
        return out

    unique_levels = sorted(series.dropna().astype(str).unique().tolist())
    rank_map = {level: float(i) for i, level in enumerate(unique_levels)}
    out[mechanics_col] = series.astype(str).map(rank_map).astype(float)
    return out


def _apply_dataset_metadata(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    system: str,
    assay: str,
) -> pd.DataFrame:
    out = df.copy()
    out["dataset_id"] = dataset_id
    out["system"] = system
    out["assay"] = assay
    return out


def load_and_prepare_endpoint(
    data: DataLike,
    *,
    dataset_id: str,
    system: str,
    assay: str,
    metadata: AssayMetadata | Mapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
    mechanics_map: Mapping[Any, float] | None = None,
    ordered_mechanics_levels: list[Any] | tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    df = _load_table(data)
    df = _rename_aliases(df, aliases)
    df = _coerce_named_mechanics(
        df,
        mechanics_col="m",
        mechanics_map=mechanics_map,
        ordered_levels=ordered_mechanics_levels,
    )
    df = _apply_dataset_metadata(df, dataset_id=dataset_id, system=system, assay=assay)
    return prepare_endpoint_data(df, metadata=metadata)


def load_and_prepare_timecourse(
    data: DataLike,
    *,
    dataset_id: str,
    system: str,
    assay: str,
    metadata: AssayMetadata | Mapping[str, Any] | None = None,
    aliases: Mapping[str, str] | None = None,
    mechanics_map: Mapping[Any, float] | None = None,
    ordered_mechanics_levels: list[Any] | tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    df = _load_table(data)
    df = _rename_aliases(df, aliases)
    df = _coerce_named_mechanics(
        df,
        mechanics_col="m",
        mechanics_map=mechanics_map,
        ordered_levels=ordered_mechanics_levels,
    )
    df = _apply_dataset_metadata(df, dataset_id=dataset_id, system=system, assay=assay)
    return prepare_timecourse_data(df, metadata=metadata)
