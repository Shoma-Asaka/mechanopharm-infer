from __future__ import annotations

import numpy as np
import pandas as pd

from .types import QCReport


def _require_columns(df: pd.DataFrame, required: set[str], kind: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{kind} dataframe is missing required columns: {sorted(missing)}")


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    required = {"c", "m", "response"}
    _require_columns(df, required, "endpoint")

    group_cols = ["c", "m"]
    grouped = df.groupby(group_cols, as_index=False)["response"]

    if "replicate" in df.columns:
        out = grouped.agg(
            response_mean="mean",
            response_sd="std",
            n="count",
        )
    else:
        out = grouped.agg(
            response_mean="mean",
            n="count",
        )
        out["response_sd"] = np.nan

    out = out[["c", "m", "response_mean", "response_sd", "n"]]
    return out.sort_values(["m", "c"]).reset_index(drop=True)


def endpoint_to_grid(summary_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"c", "m", "response_mean"}
    _require_columns(summary_df, required, "summary")

    c_grid = np.sort(summary_df["c"].unique())
    m_grid = np.sort(summary_df["m"].unique())

    pivot = summary_df.pivot(index="m", columns="c", values="response_mean")
    pivot = pivot.reindex(index=m_grid, columns=c_grid)

    response = pivot.to_numpy(dtype=float)
    return c_grid, m_grid, response


def grid_completeness(summary_df: pd.DataFrame, value_col: str) -> float:
    required = {"c", "m", value_col}
    _require_columns(summary_df, required, "grid completeness")

    n_c = int(summary_df["c"].nunique())
    n_m = int(summary_df["m"].nunique())
    if n_c == 0 or n_m == 0:
        return 0.0

    observed = int(summary_df[["c", "m"]].drop_duplicates().shape[0])
    expected = n_c * n_m
    return float(observed / expected) if expected > 0 else 0.0


def check_endpoint_qc(
    summary_df: pd.DataFrame,
    min_unique_c: int = 3,
    min_unique_m: int = 2,
    min_replicates: int = 1,
    min_dynamic_range: float = 0.05,
    min_grid_completeness: float = 0.8,
) -> QCReport:
    required = {"c", "m", "response_mean", "n"}
    _require_columns(summary_df, required, "endpoint summary")

    warnings: list[str] = []
    n_unique_c = int(summary_df["c"].nunique())
    n_unique_m = int(summary_df["m"].nunique())
    min_n_per_condition = int(summary_df["n"].min()) if not summary_df.empty else 0
    median_n_per_condition = float(summary_df["n"].median()) if not summary_df.empty else 0.0
    dynamic_range = (
        float(summary_df["response_mean"].max() - summary_df["response_mean"].min())
        if not summary_df.empty
        else 0.0
    )
    completeness = grid_completeness(summary_df, "response_mean")

    if n_unique_c < min_unique_c:
        warnings.append(
            f"Only {n_unique_c} unique concentration levels detected; EC50 estimates may be unreliable."
        )
    if n_unique_m < min_unique_m:
        warnings.append(
            f"Only {n_unique_m} unique mechanical levels detected; mechanical fingerprints may be unreliable."
        )
    if min_n_per_condition < min_replicates:
        warnings.append(
            f"Some endpoint conditions have fewer than {min_replicates} replicates."
        )
    if completeness < min_grid_completeness:
        warnings.append(
            f"Endpoint grid completeness is {completeness:.2f}; optimum calls may be unstable."
        )
    if dynamic_range < min_dynamic_range:
        warnings.append(
            f"Endpoint dynamic range is only {dynamic_range:.3f}; shift detection may be unreliable."
        )

    passed = not warnings
    metrics: dict[str, float | int | bool] = {
        "n_unique_c": n_unique_c,
        "n_unique_m": n_unique_m,
        "n_rows": int(len(summary_df)),
        "min_n_per_condition": min_n_per_condition,
        "median_n_per_condition": median_n_per_condition,
        "grid_completeness": completeness,
        "dynamic_range": dynamic_range,
    }
    return QCReport(kind="endpoint", passed=passed, warnings=warnings, metrics=metrics)


def summarize_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "c", "m", "value"}
    _require_columns(df, required, "timecourse")

    group_cols = ["time", "c", "m"]
    grouped = df.groupby(group_cols, as_index=False)["value"]

    if "replicate" in df.columns:
        out = grouped.agg(
            value_mean="mean",
            value_sd="std",
            n="count",
        )
    else:
        out = grouped.agg(
            value_mean="mean",
            n="count",
        )
        out["value_sd"] = np.nan

    out = out[["time", "c", "m", "value_mean", "value_sd", "n"]]
    return out.sort_values(["c", "m", "time"]).reset_index(drop=True)


def split_timecourses_by_condition(summary_df: pd.DataFrame) -> dict[tuple[float, float], pd.DataFrame]:
    required = {"time", "c", "m", "value_mean"}
    _require_columns(summary_df, required, "timecourse summary")

    out: dict[tuple[float, float], pd.DataFrame] = {}
    for (c, m), sub in summary_df.groupby(["c", "m"]):
        out[(float(c), float(m))] = (
            sub.sort_values("time")
            .reset_index(drop=True)
            .copy()
        )
    return out


def check_timecourse_qc(
    summary_df: pd.DataFrame,
    min_timepoints_per_condition: int = 3,
    min_duration: float = 0.0,
) -> QCReport:
    required = {"time", "c", "m", "value_mean", "n"}
    _require_columns(summary_df, required, "timecourse summary")

    warnings: list[str] = []
    grouped = summary_df.groupby(["c", "m"], as_index=False)

    n_conditions = int(summary_df[["c", "m"]].drop_duplicates().shape[0])
    point_counts = grouped.size().rename(columns={"size": "n_timepoints"})
    durations = grouped["time"].agg(lambda s: float(np.max(s) - np.min(s))).rename(columns={"time": "duration"})
    per_condition = point_counts.merge(durations, on=["c", "m"], how="inner")

    min_points = int(per_condition["n_timepoints"].min()) if not per_condition.empty else 0
    median_points = float(per_condition["n_timepoints"].median()) if not per_condition.empty else 0.0
    min_observed_duration = float(per_condition["duration"].min()) if not per_condition.empty else 0.0
    median_observed_duration = float(per_condition["duration"].median()) if not per_condition.empty else 0.0

    if min_points < min_timepoints_per_condition:
        warnings.append(
            f"Several conditions have fewer than {min_timepoints_per_condition} time points; peak detection may be unreliable."
        )
    if min_observed_duration <= min_duration:
        warnings.append(
            "Some timecourse conditions have no observable duration beyond a single timepoint window."
        )

    replicate_min = int(summary_df["n"].min()) if not summary_df.empty else 0
    if replicate_min < 1:
        warnings.append("Some timecourse rows have no valid replicate count.")

    passed = not warnings
    metrics: dict[str, float | int | bool] = {
        "n_conditions": n_conditions,
        "n_rows": int(len(summary_df)),
        "min_timepoints_per_condition": min_points,
        "median_timepoints_per_condition": median_points,
        "min_duration_observed": min_observed_duration,
        "median_duration_observed": median_observed_duration,
        "min_n_per_timepoint": replicate_min,
    }
    return QCReport(kind="timecourse", passed=passed, warnings=warnings, metrics=metrics)
