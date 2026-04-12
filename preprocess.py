from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    required = {"c", "m", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"endpoint dataframe is missing required columns: {sorted(missing)}")

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
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary dataframe is missing required columns: {sorted(missing)}")

    c_grid = np.sort(summary_df["c"].unique())
    m_grid = np.sort(summary_df["m"].unique())

    pivot = summary_df.pivot(index="m", columns="c", values="response_mean")
    pivot = pivot.reindex(index=m_grid, columns=c_grid)

    response = pivot.to_numpy(dtype=float)
    return c_grid, m_grid, response


def summarize_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "c", "m", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"timecourse dataframe is missing required columns: {sorted(missing)}")

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
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"timecourse summary is missing required columns: {sorted(missing)}")

    out: dict[tuple[float, float], pd.DataFrame] = {}
    for (c, m), sub in summary_df.groupby(["c", "m"]):
        out[(float(c), float(m))] = (
            sub.sort_values("time")
            .reset_index(drop=True)
            .copy()
        )
    return out
