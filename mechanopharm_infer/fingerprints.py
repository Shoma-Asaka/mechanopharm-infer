from __future__ import annotations

import numpy as np
import pandas as pd


def ec50_from_curve(c: np.ndarray, y: np.ndarray) -> float:
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)

    if c.ndim != 1 or y.ndim != 1 or len(c) != len(y):
        raise ValueError("c and y must be 1D arrays with the same length")

    finite = np.isfinite(c) & np.isfinite(y)
    c = c[finite]
    y = y[finite]

    if len(c) < 3:
        return float("nan")

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    target = ymin + 0.5 * (ymax - ymin)

    increasing = bool(y[-1] >= y[0])
    if not increasing:
        c = c[::-1]
        y = y[::-1]

    if target < np.min(y) or target > np.max(y):
        return float("nan")

    return float(np.interp(target, y, c))


def ec50_vs_m(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray) -> pd.DataFrame:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)

    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")

    values = [ec50_from_curve(c_grid, response[i, :]) for i in range(len(m_grid))]
    return pd.DataFrame({"m": m_grid, "ec50": values})


def find_mechanical_optima(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    response: np.ndarray,
) -> pd.DataFrame:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)

    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")

    rows: list[dict[str, float | int | bool]] = []
    n_m = len(m_grid)

    for j, c in enumerate(c_grid):
        col = response[:, j]
        finite = np.isfinite(col)

        if not finite.any():
            rows.append(
                {
                    "c": float(c),
                    "m_opt": float("nan"),
                    "optimum_index": -1,
                    "is_edge": False,
                    "is_interior": False,
                }
            )
            continue

        valid_idx = np.where(finite)[0]
        valid_vals = col[finite]
        local_argmax = int(np.argmax(valid_vals))
        idx = int(valid_idx[local_argmax])

        is_edge = idx == 0 or idx == (n_m - 1)
        rows.append(
            {
                "c": float(c),
                "m_opt": float(m_grid[idx]),
                "optimum_index": idx,
                "is_edge": bool(is_edge),
                "is_interior": bool(not is_edge),
            }
        )

    return pd.DataFrame(rows)


def mechanical_sign_reversal(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    response: np.ndarray,
    slope_threshold: float = 1e-8,
) -> dict[str, float | bool]:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)

    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")

    if len(m_grid) < 2 or len(c_grid) < 2:
        return {
            "low_c_mean_slope": float("nan"),
            "high_c_mean_slope": float("nan"),
            "has_reversal": False,
        }

    dm = np.gradient(m_grid)
    dRdm = np.gradient(response, axis=0) / dm[:, None]

    n_q = max(2, len(c_grid) // 4)
    low = float(np.nanmean(dRdm[:, :n_q]))
    high = float(np.nanmean(dRdm[:, -n_q:]))

    has_reversal = (
        np.isfinite(low)
        and np.isfinite(high)
        and np.sign(low) != np.sign(high)
        and abs(low) > slope_threshold
        and abs(high) > slope_threshold
    )

    return {
        "low_c_mean_slope": low,
        "high_c_mean_slope": high,
        "has_reversal": bool(has_reversal),
    }


def peak_metrics_by_condition(
    timecourse_by_condition: dict[tuple[float, float], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for (c, m), df in timecourse_by_condition.items():
        if "time" not in df.columns or "value_mean" not in df.columns:
            raise ValueError("each timecourse dataframe must contain 'time' and 'value_mean'")

        time = df["time"].to_numpy(dtype=float)
        value = df["value_mean"].to_numpy(dtype=float)

        finite = np.isfinite(time) & np.isfinite(value)
        time = time[finite]
        value = value[finite]

        if len(time) == 0:
            rows.append(
                {
                    "c": float(c),
                    "m": float(m),
                    "peak_value": float("nan"),
                    "peak_time": float("nan"),
                }
            )
            continue

        idx = int(np.argmax(value))
        rows.append(
            {
                "c": float(c),
                "m": float(m),
                "peak_value": float(value[idx]),
                "peak_time": float(time[idx]),
            }
        )

    return pd.DataFrame(rows)


def endpoint_final_response(
    timecourse_by_condition: dict[tuple[float, float], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for (c, m), df in timecourse_by_condition.items():
        if "time" not in df.columns or "value_mean" not in df.columns:
            raise ValueError("each timecourse dataframe must contain 'time' and 'value_mean'")

        sub = df.sort_values("time").reset_index(drop=True)
        sub = sub[np.isfinite(sub["time"]) & np.isfinite(sub["value_mean"])]

        if len(sub) == 0:
            rows.append(
                {
                    "c": float(c),
                    "m": float(m),
                    "e_final": float("nan"),
                }
            )
            continue

        last = sub.iloc[-1]
        rows.append(
            {
                "c": float(c),
                "m": float(m),
                "e_final": float(last["value_mean"]),
            }
        )

    return pd.DataFrame(rows)



def delayed_protection_metrics(
    peak_df: pd.DataFrame,
    final_df: pd.DataFrame,
    attenuation_threshold: float = 0.05,
) -> pd.DataFrame:
    merged = peak_df.merge(final_df, on=["c", "m"], how="inner", suffixes=("_peak", "_final"))
    rows: list[dict[str, float | bool | str | None]] = []

    for _, row in merged.iterrows():
        peak_value = float(row["peak_value"]) if pd.notna(row.get("peak_value")) else float("nan")
        e_final = float(row["e_final"]) if pd.notna(row.get("e_final")) else float("nan")
        attenuation = peak_value - e_final if np.isfinite(peak_value) and np.isfinite(e_final) else float("nan")

        peak_reliable = bool(row.get("is_reliable_peak", row.get("is_reliable", True)))
        final_reliable = bool(row.get("is_reliable_final", row.get("is_reliable", True)))
        has_clear_peak = bool(row.get("has_clear_peak", True))
        is_terminal_peak = bool(row.get("is_terminal_peak", False))

        is_reliable = peak_reliable and final_reliable
        delayed = bool(
            is_reliable
            and has_clear_peak
            and (not is_terminal_peak)
            and np.isfinite(attenuation)
            and attenuation > attenuation_threshold
        )

        warning = None
        if not is_reliable:
            warning = "Peak or final response is unreliable."
        elif is_terminal_peak:
            warning = "Peak occurs at terminal timepoint."
        elif not has_clear_peak:
            warning = "No clear transient peak detected."
        elif not np.isfinite(attenuation):
            warning = "Peak-to-final attenuation could not be computed."

        rows.append(
            {
                "c": float(row["c"]),
                "m": float(row["m"]),
                "peak_value": peak_value,
                "e_final": e_final,
                "attenuation": attenuation,
                "delayed_protection_detected": delayed,
                "is_reliable": is_reliable,
                "warning": warning,
            }
        )

    return pd.DataFrame(rows)
