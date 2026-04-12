from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_curve(c: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(c) & np.isfinite(y)
    c = np.asarray(c, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if len(c) == 0:
        return c, y
    order = np.argsort(c)
    return c[order], y[order]


def _is_effectively_monotone(y: np.ndarray, atol: float = 1e-8) -> bool:
    if len(y) < 2:
        return False
    dy = np.diff(y)
    return bool(np.all(dy >= -atol) or np.all(dy <= atol))


def _curve_dynamic_range(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.nanmax(y) - np.nanmin(y))


def ec50_from_curve(c: np.ndarray, y: np.ndarray, min_dynamic_range: float = 0.05) -> dict[str, float | bool | str | None]:
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    if c.ndim != 1 or y.ndim != 1 or len(c) != len(y):
        raise ValueError("c and y must be 1D arrays with the same length")
    c, y = _clean_curve(c, y)
    if len(c) < 3:
        return {"ec50": float("nan"), "is_monotone": False, "has_sufficient_range": False, "is_reliable": False, "warning": "Too few finite points for EC50 estimation."}
    dyn = _curve_dynamic_range(y)
    mono = _is_effectively_monotone(y)
    if not mono:
        return {"ec50": float("nan"), "is_monotone": False, "has_sufficient_range": dyn >= min_dynamic_range, "is_reliable": False, "warning": "Dose-response curve is not effectively monotone."}
    increasing = bool(y[-1] >= y[0])
    if not increasing:
        c = c[::-1]
        y = y[::-1]
    target = float(np.min(y) + 0.5 * (np.max(y) - np.min(y)))
    if dyn < min_dynamic_range:
        return {"ec50": float("nan"), "is_monotone": True, "has_sufficient_range": False, "is_reliable": False, "warning": "Dynamic range is too small for reliable EC50 estimation."}
    if target < np.min(y) or target > np.max(y):
        return {"ec50": float("nan"), "is_monotone": True, "has_sufficient_range": True, "is_reliable": False, "warning": "Half-max target falls outside the observed response range."}
    return {"ec50": float(np.interp(target, y, c)), "is_monotone": True, "has_sufficient_range": True, "is_reliable": True, "warning": None}


def ec50_vs_m(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray, min_dynamic_range: float = 0.05) -> pd.DataFrame:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    rows = []
    for i, m in enumerate(m_grid):
        out = ec50_from_curve(c_grid, response[i, :], min_dynamic_range=min_dynamic_range)
        rows.append({"m": float(m), **out})
    return pd.DataFrame(rows)


def _optimum_prominence(values: np.ndarray, idx: int) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return float("nan")
    edge_candidates = np.array([valid[0], valid[-1]], dtype=float)
    return float(values[idx] - np.nanmax(edge_candidates))


def find_mechanical_optima(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray, prominence_threshold: float = 0.03) -> pd.DataFrame:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    rows = []
    n_m = len(m_grid)
    for j, c in enumerate(c_grid):
        col = response[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 1:
            rows.append({"c": float(c), "m_opt": float("nan"), "optimum_index": -1, "is_edge": False, "is_interior": False, "prominence": float("nan"), "is_reliable": False, "warning": "No finite response values."})
            continue
        valid_idx = np.where(finite)[0]
        idx = int(valid_idx[int(np.argmax(col[finite]))])
        is_edge = idx == 0 or idx == (n_m - 1)
        prominence = _optimum_prominence(col, idx)
        is_reliable = bool((not is_edge) and np.isfinite(prominence) and prominence >= prominence_threshold and finite.sum() >= 3)
        warning = None if is_reliable else ("Optimum is on the boundary." if is_edge else "Interior optimum prominence is too small or sampling is sparse.")
        rows.append({"c": float(c), "m_opt": float(m_grid[idx]), "optimum_index": idx, "is_edge": bool(is_edge), "is_interior": bool(not is_edge), "prominence": prominence, "is_reliable": is_reliable, "warning": warning})
    return pd.DataFrame(rows)


def mechanical_sign_reversal(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray, slope_threshold: float = 1e-8, min_reliable_columns: int = 2) -> dict[str, float | bool | str | None]:
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    if len(m_grid) < 2 or len(c_grid) < max(2, min_reliable_columns):
        return {"low_c_mean_slope": float("nan"), "high_c_mean_slope": float("nan"), "has_reversal": False, "is_reliable": False, "warning": "Grid is too small for reversal detection."}
    dRdm = np.gradient(response, axis=0) / np.gradient(m_grid)[:, None]
    n_q = max(2, len(c_grid) // 4)
    low = float(np.nanmean(dRdm[:, :n_q]))
    high = float(np.nanmean(dRdm[:, -n_q:]))
    has_reversal = bool(np.isfinite(low) and np.isfinite(high) and np.sign(low) != np.sign(high) and abs(low) > slope_threshold and abs(high) > slope_threshold)
    return {"low_c_mean_slope": low, "high_c_mean_slope": high, "has_reversal": has_reversal, "is_reliable": True, "warning": None}


def peak_metrics_by_condition(timecourse_by_condition: dict[tuple[float, float], pd.DataFrame], peak_prominence_threshold: float = 0.03) -> pd.DataFrame:
    rows = []
    for (c, m), df in timecourse_by_condition.items():
        if "time" not in df.columns or "value_mean" not in df.columns:
            raise ValueError("each timecourse dataframe must contain 'time' and 'value_mean'")
        time = df["time"].to_numpy(dtype=float)
        value = df["value_mean"].to_numpy(dtype=float)
        finite = np.isfinite(time) & np.isfinite(value)
        time = time[finite]
        value = value[finite]
        if len(time) == 0:
            rows.append({"c": float(c), "m": float(m), "peak_value": float("nan"), "peak_time": float("nan"), "peak_index": -1, "is_terminal_peak": False, "peak_prominence": float("nan"), "has_clear_peak": False, "is_reliable": False, "warning": "No finite timecourse values."})
            continue
        idx = int(np.argmax(value))
        terminal = idx == len(value) - 1
        final = float(value[-1])
        prominence = float(value[idx] - final)
        clear = bool((not terminal) and prominence > peak_prominence_threshold)
        rows.append({"c": float(c), "m": float(m), "peak_value": float(value[idx]), "peak_time": float(time[idx]), "peak_index": idx, "is_terminal_peak": terminal, "peak_prominence": prominence, "has_clear_peak": clear, "is_reliable": len(value) >= 3, "warning": None if len(value) >= 3 else "Too few time points for reliable peak detection."})
    return pd.DataFrame(rows)


def endpoint_final_response(timecourse_by_condition: dict[tuple[float, float], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (c, m), df in timecourse_by_condition.items():
        if "time" not in df.columns or "value_mean" not in df.columns:
            raise ValueError("each timecourse dataframe must contain 'time' and 'value_mean'")
        sub = df.sort_values("time").reset_index(drop=True)
        sub = sub[np.isfinite(sub["time"]) & np.isfinite(sub["value_mean"])]
        if len(sub) == 0:
            rows.append({"c": float(c), "m": float(m), "e_final": float("nan"), "final_time": float("nan"), "is_reliable": False, "warning": "No finite timecourse values."})
            continue
        last = sub.iloc[-1]
        rows.append({"c": float(c), "m": float(m), "e_final": float(last["value_mean"]), "final_time": float(last["time"]), "is_reliable": True, "warning": None})
    return pd.DataFrame(rows)


def delayed_protection_metrics(peak_df: pd.DataFrame, final_df: pd.DataFrame, attenuation_threshold: float = 0.05) -> pd.DataFrame:
    merged = peak_df.merge(final_df, on=["c", "m"], how="inner", suffixes=("_peak", "_final"))
    if merged.empty:
        return pd.DataFrame(columns=["c", "m", "peak_value", "e_final", "attenuation", "delayed_protection_detected", "is_reliable", "warning"])
    merged["attenuation"] = merged["peak_value"] - merged["e_final"]
    merged["delayed_protection_detected"] = merged["attenuation"] > attenuation_threshold
    merged["is_reliable"] = merged.get("is_reliable_peak", True).fillna(False).astype(bool) & merged.get("is_reliable_final", True).fillna(False).astype(bool)
    merged["warning"] = None
    return merged[["c", "m", "peak_value", "e_final", "attenuation", "delayed_protection_detected", "is_reliable", "warning"]]
