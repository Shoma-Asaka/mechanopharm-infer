from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .schema import standardize_endpoint_schema, standardize_timecourse_schema
from .types import AssayMetadata, QCReport, coerce_assay_metadata


def _baseline_series(df: pd.DataFrame, value_col: str, metadata: AssayMetadata) -> pd.Series | float:
    if metadata.baseline_definition == "none":
        return 0.0
    if metadata.baseline_definition == "global_minimum":
        return float(df[value_col].min())
    if metadata.baseline_definition == "minimum_per_mechanics":
        return df.groupby("m")[value_col].transform("min")
    if metadata.baseline_definition == "control_flag":
        if "control_flag" not in df.columns or not bool(df["control_flag"].any()):
            raise ValueError(
                "baseline_definition='control_flag' requires a truthy control_flag column"
            )
        baseline = (
            df[df["control_flag"]]
            .groupby([c for c in ["dataset_id", "system", "assay", "m"] if c in df.columns])[value_col]
            .mean()
            .rename("_baseline")
            .reset_index()
        )
        merged = df.merge(
            baseline,
            on=[c for c in ["dataset_id", "system", "assay", "m"] if c in df.columns],
            how="left",
        )
        return merged["_baseline"]
    raise ValueError(f"Unsupported baseline_definition: {metadata.baseline_definition}")


def apply_assay_metadata(
    df: pd.DataFrame,
    *,
    kind: str,
    metadata: AssayMetadata | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Apply response orientation and normalization using assay metadata."""

    md = coerce_assay_metadata(metadata)
    if kind == "endpoint":
        out = standardize_endpoint_schema(df)
        value_col = "response"
    elif kind == "timecourse":
        out = standardize_timecourse_schema(df)
        value_col = "value"
    else:
        raise ValueError("kind must be 'endpoint' or 'timecourse'")

    out = out.copy()
    out["raw_" + value_col] = out[value_col]

    if md.response_mode == "lower_is_stronger_effect":
        out[value_col] = 1.0 - out[value_col].astype(float)

    if md.normalization_mode == "raw":
        pass
    elif md.normalization_mode == "control_subtracted":
        baseline = _baseline_series(out, value_col, md)
        out[value_col] = out[value_col] - baseline
    elif md.normalization_mode == "vehicle_normalized":
        baseline = _baseline_series(out, value_col, md)
        if isinstance(baseline, pd.Series):
            denom = baseline.replace(0, np.nan)
            out[value_col] = out[value_col] / denom
        else:
            if baseline == 0:
                raise ValueError("vehicle normalization baseline is zero")
            out[value_col] = out[value_col] / float(baseline)
    elif md.normalization_mode == "min_max":
        vmin = float(out[value_col].min())
        vmax = float(out[value_col].max())
        if np.isclose(vmax, vmin):
            raise ValueError("cannot apply min_max normalization to constant response")
        out[value_col] = (out[value_col] - vmin) / (vmax - vmin)
    elif md.normalization_mode == "within_mechanics_min_max":
        grouped = out.groupby("m")[value_col]
        mins = grouped.transform("min")
        maxs = grouped.transform("max")
        denom = (maxs - mins).replace(0, np.nan)
        if bool(denom.isna().any()):
            raise ValueError(
                "cannot apply within_mechanics_min_max when any mechanics group is constant"
            )
        out[value_col] = (out[value_col] - mins) / denom
    else:
        raise ValueError(f"Unsupported normalization_mode: {md.normalization_mode}")

    out["metadata_response_mode"] = md.response_mode
    out["metadata_assay_family"] = md.assay_family
    out["metadata_normalization_mode"] = md.normalization_mode
    out["metadata_baseline_definition"] = md.baseline_definition
    return out


def prepare_endpoint_data(
    df: pd.DataFrame,
    metadata: AssayMetadata | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    return apply_assay_metadata(df, kind="endpoint", metadata=metadata)


def prepare_timecourse_data(
    df: pd.DataFrame,
    metadata: AssayMetadata | Mapping[str, object] | None = None,
) -> pd.DataFrame:
    return apply_assay_metadata(df, kind="timecourse", metadata=metadata)


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    required = {"c", "m", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"endpoint dataframe is missing required columns: {sorted(missing)}")
    grouped = df.groupby(["c", "m"], as_index=False)["response"]
    if "replicate" in df.columns and df["replicate"].notna().any():
        out = grouped.agg(response_mean="mean", response_sd="std", n="count")
    else:
        out = grouped.agg(response_mean="mean", n="count")
        out["response_sd"] = np.nan
    return out[["c", "m", "response_mean", "response_sd", "n"]].sort_values(["m", "c"]).reset_index(drop=True)


def endpoint_to_grid(summary_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"c", "m", "response_mean"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary dataframe is missing required columns: {sorted(missing)}")
    c_grid = np.sort(summary_df["c"].unique())
    m_grid = np.sort(summary_df["m"].unique())
    pivot = summary_df.pivot(index="m", columns="c", values="response_mean").reindex(index=m_grid, columns=c_grid)
    return c_grid, m_grid, pivot.to_numpy(dtype=float)


def summarize_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "c", "m", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"timecourse dataframe is missing required columns: {sorted(missing)}")
    grouped = df.groupby(["time", "c", "m"], as_index=False)["value"]
    if "replicate" in df.columns and df["replicate"].notna().any():
        out = grouped.agg(value_mean="mean", value_sd="std", n="count")
    else:
        out = grouped.agg(value_mean="mean", n="count")
        out["value_sd"] = np.nan
    return out[["time", "c", "m", "value_mean", "value_sd", "n"]].sort_values(["c", "m", "time"]).reset_index(drop=True)


def split_timecourses_by_condition(summary_df: pd.DataFrame) -> dict[tuple[float, float], pd.DataFrame]:
    required = {"time", "c", "m", "value_mean"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"timecourse summary is missing required columns: {sorted(missing)}")
    out: dict[tuple[float, float], pd.DataFrame] = {}
    for (c, m), sub in summary_df.groupby(["c", "m"]):
        out[(float(c), float(m))] = sub.sort_values("time").reset_index(drop=True).copy()
    return out


def grid_completeness(summary_df: pd.DataFrame, value_col: str) -> float:
    if summary_df.empty:
        return 0.0
    c_vals = summary_df["c"].nunique()
    m_vals = summary_df["m"].nunique()
    if c_vals == 0 or m_vals == 0:
        return 0.0
    present = summary_df[["c", "m", value_col]].dropna().drop_duplicates(["c", "m"]).shape[0]
    return float(present / (c_vals * m_vals))


def check_endpoint_qc(summary_df: pd.DataFrame, min_unique_c: int = 3, min_unique_m: int = 2, min_replicates: int = 1, min_dynamic_range: float = 0.05) -> QCReport:
    warnings: list[str] = []
    n_unique_c = int(summary_df["c"].nunique()) if not summary_df.empty else 0
    n_unique_m = int(summary_df["m"].nunique()) if not summary_df.empty else 0
    min_n = int(summary_df["n"].min()) if not summary_df.empty else 0
    median_n = float(summary_df["n"].median()) if not summary_df.empty else 0.0
    completeness = grid_completeness(summary_df, "response_mean")
    dyn = float(summary_df["response_mean"].max() - summary_df["response_mean"].min()) if not summary_df.empty else 0.0
    passed = True
    if n_unique_c < min_unique_c:
        warnings.append("Too few unique concentration levels for robust EC50 estimation.")
        passed = False
    if n_unique_m < min_unique_m:
        warnings.append("Too few unique mechanical levels for robust mechanics fingerprints.")
        passed = False
    if min_n < min_replicates:
        warnings.append("At least one endpoint condition has insufficient replicate count.")
        passed = False
    if completeness < 0.8:
        warnings.append("Endpoint grid completeness is low; optimum calls may be unstable.")
    if dyn < min_dynamic_range:
        warnings.append("Endpoint dynamic range is small; shift detection may be unreliable.")
    return QCReport(
        kind="endpoint",
        passed=passed,
        warnings=warnings,
        metrics={
            "n_unique_c": n_unique_c,
            "n_unique_m": n_unique_m,
            "min_n_per_condition": min_n,
            "median_n_per_condition": median_n,
            "grid_completeness": completeness,
            "dynamic_range": dyn,
        },
    )


def check_timecourse_qc(summary_df: pd.DataFrame, min_timepoints_per_condition: int = 3, min_duration: float = 0.0) -> QCReport:
    warnings: list[str] = []
    durations = []
    counts = []
    for _, sub in summary_df.groupby(["c", "m"]):
        counts.append(int(sub["time"].nunique()))
        durations.append(float(sub["time"].max() - sub["time"].min()))
    min_tp = min(counts) if counts else 0
    med_tp = float(np.median(counts)) if counts else 0.0
    min_dur = min(durations) if durations else 0.0
    med_dur = float(np.median(durations)) if durations else 0.0
    passed = True
    if counts and min_tp < min_timepoints_per_condition:
        warnings.append("Several conditions have too few time points for reliable peak detection.")
        passed = False
    if durations and min_dur < min_duration:
        warnings.append("Several conditions have too short an observation window.")
        passed = False
    return QCReport(
        kind="timecourse",
        passed=passed,
        warnings=warnings,
        metrics={
            "n_conditions": len(counts),
            "min_timepoints_per_condition": min_tp,
            "median_timepoints_per_condition": med_tp,
            "min_duration_observed": min_dur,
            "median_duration_observed": med_dur,
        },
    )
