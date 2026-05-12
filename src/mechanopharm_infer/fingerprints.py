"""Response-landscape fingerprint extraction.

This module extracts the practical fingerprint set defined by the theory:

* ``EC50(m)``                : mechanically conditioned half-effective concentration
* ``c_rev``                  : concentration at which mechanical sensitivity reverses sign
                               (``c_rev = -Delta_lambda / Delta_mu`` in the two-state theory)
* ``m*(c)``                  : optimal mechanical condition at fixed concentration
* ``E_peak(c, m)``           : peak amplitude of the proximal signal over time
* ``t_peak(c, m)``           : time at which the peak is reached
* ``E_inf(c, m)``            : long-time / endpoint response
* delayed-protection metric  : ``E_peak - E_inf`` (attenuation)

Output column names follow the unified glossary.  Boundary cases are flagged
through ``is_reliable`` / ``evidence_strength`` so that downstream consumers
can treat unreliable estimates conservatively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


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


def _monotonicity_score(y: np.ndarray, atol: float = 1e-8) -> float:
    if len(y) < 2:
        return 0.0
    dy = np.diff(y)
    if np.all(dy >= -atol) or np.all(dy <= atol):
        return 1.0
    total = float(np.sum(np.abs(dy)))
    if total <= atol:
        return 1.0
    positive = float(np.sum(np.clip(dy, 0.0, None)))
    negative = float(np.sum(np.clip(-dy, 0.0, None)))
    return max(positive, negative) / total


def _evidence_from_reliability(is_reliable: bool, warning: str | None) -> str:
    if is_reliable:
        return "moderate"
    if warning and "too few" in warning.lower():
        return "not_assessable"
    return "weak"


# ---------------------------------------------------------------------------
# EC50(m)
# ---------------------------------------------------------------------------


def ec50_from_curve(
    c: np.ndarray,
    y: np.ndarray,
    min_dynamic_range: float = 0.05,
) -> dict[str, float | bool | str | int | None]:
    """Estimate the half-effective concentration ``EC50`` for one ``c``-curve.

    Uses linear interpolation between the two grid points bracketing the
    half-maximum response.  Returns reliability flags so that unreliable
    estimates can be filtered downstream.
    """

    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    if c.ndim != 1 or y.ndim != 1 or len(c) != len(y):
        raise ValueError("c and y must be 1D arrays with the same length")
    c, y = _clean_curve(c, y)
    n_points = int(len(c))
    monotonicity_score = _monotonicity_score(y)
    dyn = _curve_dynamic_range(y)
    base = {
        "n_points": n_points,
        "dynamic_range": dyn,
        "monotonicity_score": monotonicity_score,
    }
    if len(c) < 3:
        warning = "Too few finite points for EC50 estimation."
        return {
            "ec50": float("nan"),
            "is_monotone": False,
            "has_sufficient_range": False,
            "is_reliable": False,
            "warning": warning,
            "halfmax_response": float("nan"),
            "response_min": float(np.nanmin(y)) if len(y) else float("nan"),
            "response_max": float(np.nanmax(y)) if len(y) else float("nan"),
            "evidence_strength": _evidence_from_reliability(False, warning),
            **base,
        }
    mono = _is_effectively_monotone(y)
    response_min = float(np.min(y))
    response_max = float(np.max(y))
    target = float(response_min + 0.5 * (response_max - response_min))
    if not mono:
        warning = "Dose-response curve is not effectively monotone."
        return {
            "ec50": float("nan"),
            "is_monotone": False,
            "has_sufficient_range": dyn >= min_dynamic_range,
            "is_reliable": False,
            "warning": warning,
            "halfmax_response": target,
            "response_min": response_min,
            "response_max": response_max,
            "evidence_strength": _evidence_from_reliability(False, warning),
            **base,
        }
    increasing = bool(y[-1] >= y[0])
    if not increasing:
        c = c[::-1]
        y = y[::-1]
    if dyn < min_dynamic_range:
        warning = "Dynamic range is too small for reliable EC50 estimation."
        return {
            "ec50": float("nan"),
            "is_monotone": True,
            "has_sufficient_range": False,
            "is_reliable": False,
            "warning": warning,
            "halfmax_response": target,
            "response_min": response_min,
            "response_max": response_max,
            "evidence_strength": _evidence_from_reliability(False, warning),
            **base,
        }
    if target < np.min(y) or target > np.max(y):
        warning = "Half-max target falls outside the observed response range."
        return {
            "ec50": float("nan"),
            "is_monotone": True,
            "has_sufficient_range": True,
            "is_reliable": False,
            "warning": warning,
            "halfmax_response": target,
            "response_min": response_min,
            "response_max": response_max,
            "evidence_strength": _evidence_from_reliability(False, warning),
            **base,
        }
    warning = None
    return {
        "ec50": float(np.interp(target, y, c)),
        "is_monotone": True,
        "has_sufficient_range": True,
        "is_reliable": True,
        "warning": warning,
        "halfmax_response": target,
        "response_min": response_min,
        "response_max": response_max,
        "evidence_strength": _evidence_from_reliability(True, warning),
        **base,
    }


def ec50_vs_m(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    response: np.ndarray,
    min_dynamic_range: float = 0.05,
) -> pd.DataFrame:
    """Compute ``EC50(m)`` row-by-row over a ``(m, c)`` response grid."""

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


# ---------------------------------------------------------------------------
# m*(c) -- interior optimum over mechanical condition
# ---------------------------------------------------------------------------


def _optimum_prominence(values: np.ndarray, idx: int) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return float("nan")
    edge_candidates = np.array([valid[0], valid[-1]], dtype=float)
    return float(values[idx] - np.nanmax(edge_candidates))


def _neighbor_margin(values: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(values) - 1:
        return float("nan")
    neighbors = np.array([values[idx - 1], values[idx + 1]], dtype=float)
    neighbors = neighbors[np.isfinite(neighbors)]
    if len(neighbors) == 0:
        return float("nan")
    return float(values[idx] - np.nanmax(neighbors))


def _parabolic_refinement(
    m_minus: float,
    m_zero: float,
    m_plus: float,
    y_minus: float,
    y_zero: float,
    y_plus: float,
) -> float:
    """Refine an interior optimum by fitting a parabola through three points.

    Returns ``m_zero`` if the quadratic fit is ill-conditioned or the refined
    optimum falls outside the bracketing interval.
    """

    if not (np.isfinite(y_minus) and np.isfinite(y_zero) and np.isfinite(y_plus)):
        return float(m_zero)
    # Solve for vertex of y = a (m - m_zero)^2 + b (m - m_zero) + y_zero on a
    # non-uniform grid using divided differences.
    h_minus = m_zero - m_minus
    h_plus = m_plus - m_zero
    if h_minus <= 0.0 or h_plus <= 0.0:
        return float(m_zero)
    # Newton form: a = ((y_plus - y_zero)/h_plus - (y_zero - y_minus)/h_minus) / (h_minus + h_plus)
    slope_right = (y_plus - y_zero) / h_plus
    slope_left = (y_zero - y_minus) / h_minus
    a = (slope_right - slope_left) / (h_minus + h_plus)
    if not np.isfinite(a) or abs(a) < 1e-12:
        return float(m_zero)
    # First derivative at m_zero (centered): b = (y_plus - y_minus)/(h_minus + h_plus)
    b = (y_plus - y_minus) / (h_minus + h_plus)
    delta = -b / (2.0 * a)
    refined = float(m_zero + delta)
    lo = min(m_minus, m_plus)
    hi = max(m_minus, m_plus)
    if not (lo <= refined <= hi):
        return float(m_zero)
    return refined


def find_mechanical_optima(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    response: np.ndarray,
    prominence_threshold: float = 0.03,
) -> pd.DataFrame:
    """Locate the optimal mechanical condition ``m*(c)`` at each concentration.

    For interior optima, a parabolic refinement is reported in ``m_opt`` (a
    smooth sub-grid estimate consistent with ``m*(c) = (Delta_lambda +
    Delta_mu * c)/kappa`` of the theory).  The raw grid maximum is preserved in
    ``m_opt_grid`` for reference.
    """

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
        n_points = int(finite.sum())
        if n_points < 1:
            rows.append({
                "c": float(c),
                "m_opt": float("nan"),
                "m_opt_grid": float("nan"),
                "optimum_index": -1,
                "is_edge": False,
                "is_interior": False,
                "prominence": float("nan"),
                "neighbor_margin": float("nan"),
                "response_span": 0.0,
                "is_reliable": False,
                "warning": "No finite response values.",
                "evidence_strength": "not_assessable",
                "n_mechanics_points": n_points,
            })
            continue
        valid_idx = np.where(finite)[0]
        idx = int(valid_idx[int(np.argmax(col[finite]))])
        is_edge = idx == 0 or idx == (n_m - 1)
        prominence = _optimum_prominence(col, idx)
        neighbor_margin = _neighbor_margin(col, idx)
        response_span = _curve_dynamic_range(col[finite])
        is_reliable = bool(
            (not is_edge)
            and np.isfinite(prominence)
            and prominence >= prominence_threshold
            and n_points >= 3
        )
        m_opt_grid = float(m_grid[idx])
        if (not is_edge) and idx >= 1 and idx <= n_m - 2:
            m_opt_refined = _parabolic_refinement(
                float(m_grid[idx - 1]),
                float(m_grid[idx]),
                float(m_grid[idx + 1]),
                float(col[idx - 1]),
                float(col[idx]),
                float(col[idx + 1]),
            )
        else:
            m_opt_refined = m_opt_grid
        warning = None if is_reliable else (
            "Optimum is on the boundary." if is_edge else "Interior optimum prominence is too small or sampling is sparse."
        )
        rows.append({
            "c": float(c),
            "m_opt": m_opt_refined,
            "m_opt_grid": m_opt_grid,
            "optimum_index": idx,
            "is_edge": bool(is_edge),
            "is_interior": bool(not is_edge),
            "prominence": prominence,
            "neighbor_margin": neighbor_margin,
            "response_span": response_span,
            "is_reliable": is_reliable,
            "warning": warning,
            "evidence_strength": _evidence_from_reliability(is_reliable, warning),
            "n_mechanics_points": n_points,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mechanical sensitivity sign reversal and c_rev estimation
# ---------------------------------------------------------------------------


def _mechanics_slope_by_concentration(
    c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray
) -> np.ndarray:
    dmd = np.gradient(m_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.gradient(response, axis=0) / dmd[:, None]


def _column_mean_slopes(dRdm: np.ndarray) -> np.ndarray:
    return np.nanmean(dRdm, axis=0)


def _slope_sign(x: float, threshold: float) -> int:
    if not np.isfinite(x) or abs(x) <= threshold:
        return 0
    return 1 if x > 0 else -1


def _estimate_c_rev(
    c_grid: np.ndarray, mean_slopes: np.ndarray
) -> tuple[float, float, float, str | None]:
    """Estimate the reversal concentration ``c_rev = -Delta_lambda/Delta_mu``.

    Fits the column-mean mechanical slope to ``s(c) = a + b * c`` and returns
    the root ``-a/b`` together with the fitted coefficients.  Returns
    ``(nan, a, b, warning)`` if the fit is ill-conditioned or the implied
    crossing lies outside the observed concentration range.
    """

    finite = np.isfinite(mean_slopes)
    c = np.asarray(c_grid, dtype=float)[finite]
    s = np.asarray(mean_slopes, dtype=float)[finite]
    if len(c) < 3:
        return float("nan"), float("nan"), float("nan"), "Too few finite columns for c_rev fit."
    try:
        b, a = np.polyfit(c, s, 1)
    except Exception as exc:  # pragma: no cover - polyfit failures
        return float("nan"), float("nan"), float("nan"), f"c_rev fit failed: {exc}"
    if not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan"), float(a), float(b), "Mechanochemical coupling slope (Delta_mu) is indistinguishable from zero."
    c_rev = -a / b
    if not np.isfinite(c_rev):
        return float("nan"), float(a), float(b), "c_rev estimate is not finite."
    c_min, c_max = float(np.min(c)), float(np.max(c))
    if c_rev < c_min or c_rev > c_max:
        return float(c_rev), float(a), float(b), "c_rev lies outside the sampled concentration range."
    return float(c_rev), float(a), float(b), None


def mechanical_sign_reversal(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    response: np.ndarray,
    slope_threshold: float = 1e-8,
    min_reliable_columns: int = 3,
) -> dict[str, float | bool | str | int | None]:
    """Detect a concentration-dependent sign reversal of mechanical sensitivity.

    Following the two-state theory, the sign of ``partial p* / partial m`` is
    governed by ``Delta_lambda + Delta_mu * c``, so a reversal at finite
    concentration implies ``c_rev = -Delta_lambda / Delta_mu``.  This function
    returns both the qualitative reversal flag and a regression-based estimate
    of ``c_rev``.
    """

    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    if len(m_grid) < 2 or len(c_grid) < max(2, min_reliable_columns):
        return {
            "low_c_mean_slope": float("nan"),
            "high_c_mean_slope": float("nan"),
            "has_reversal": False,
            "is_reliable": False,
            "warning": "Grid is too small for reversal detection.",
            "n_reliable_columns": 0,
            "first_positive_c": float("nan"),
            "first_negative_c": float("nan"),
            "reversal_window_center": float("nan"),
            "c_rev_estimate": float("nan"),
            "delta_lambda_proxy": float("nan"),
            "delta_mu_proxy": float("nan"),
            "c_rev_warning": "Grid is too small for c_rev estimation.",
            "evidence_strength": "not_assessable",
        }
    dRdm = _mechanics_slope_by_concentration(c_grid, m_grid, response)
    mean_slopes = _column_mean_slopes(dRdm)
    signs = np.array([_slope_sign(x, slope_threshold) for x in mean_slopes], dtype=int)
    nonzero_mask = signs != 0
    n_reliable_columns = int(nonzero_mask.sum())
    reliable_signs = signs[nonzero_mask]
    reliable_c = c_grid[nonzero_mask]
    low_idx = int(np.argmax(nonzero_mask)) if n_reliable_columns else 0
    high_idx = (
        int(len(nonzero_mask) - 1 - np.argmax(nonzero_mask[::-1])) if n_reliable_columns else 0
    )
    low = float(mean_slopes[low_idx]) if n_reliable_columns else float("nan")
    high = float(mean_slopes[high_idx]) if n_reliable_columns else float("nan")
    first_positive_c = (
        float(reliable_c[np.where(reliable_signs > 0)[0][0]])
        if np.any(reliable_signs > 0)
        else float("nan")
    )
    first_negative_c = (
        float(reliable_c[np.where(reliable_signs < 0)[0][0]])
        if np.any(reliable_signs < 0)
        else float("nan")
    )
    has_both_signs = bool(np.any(reliable_signs > 0) and np.any(reliable_signs < 0))
    switch_indices = (
        np.where(np.diff(reliable_signs) != 0)[0]
        if len(reliable_signs) >= 2
        else np.array([], dtype=int)
    )
    has_reversal = bool(has_both_signs and len(switch_indices) >= 1)
    reversal_center = float("nan")
    if len(switch_indices) >= 1:
        i = int(switch_indices[0])
        reversal_center = float(0.5 * (reliable_c[i] + reliable_c[i + 1]))
    is_reliable = bool(n_reliable_columns >= min_reliable_columns and has_both_signs)
    warning = None
    if not is_reliable:
        warning = "Mechanical-sensitivity sign is not estimable across enough concentration columns."
    elif not has_reversal:
        warning = "Mechanical sensitivity does not show a stable sign change across concentration."

    c_rev_estimate, intercept, coupling, c_rev_warning = _estimate_c_rev(c_grid, mean_slopes)

    return {
        "low_c_mean_slope": low,
        "high_c_mean_slope": high,
        "has_reversal": has_reversal,
        "is_reliable": is_reliable,
        "warning": warning,
        "n_reliable_columns": n_reliable_columns,
        "first_positive_c": first_positive_c,
        "first_negative_c": first_negative_c,
        "reversal_window_center": reversal_center,
        # Theory-aligned reversal-concentration estimate (Eq. crev).
        "c_rev_estimate": c_rev_estimate,
        # Proxies for the two-state state-bias coefficients Delta_lambda / Delta_mu.
        # They are *not* identified as absolute energetic parameters because
        # ``response`` may not equal occupancy, but the ratio ``-a/b`` recovers
        # c_rev consistently up to a common scaling.
        "delta_lambda_proxy": intercept,
        "delta_mu_proxy": coupling,
        "c_rev_warning": c_rev_warning,
        "evidence_strength": "moderate"
        if (is_reliable and has_reversal)
        else ("weak" if is_reliable else "not_assessable"),
    }


# ---------------------------------------------------------------------------
# Time-resolved fingerprints: E_peak, t_peak, E_inf, delayed protection
# ---------------------------------------------------------------------------


def peak_metrics_by_condition(
    timecourse_by_condition: dict[tuple[float, float], pd.DataFrame],
    peak_prominence_threshold: float = 0.03,
) -> pd.DataFrame:
    """Extract ``E_peak(c, m)`` and ``t_peak(c, m)`` per condition.

    Output columns
    --------------
    e_peak, t_peak
        Peak amplitude and time as defined by the theory.
    peak_index, is_terminal_peak
        Index of the maximum and a flag indicating whether the maximum is the
        final observation (in which case ``E_peak`` may be limited by the
        observation window).
    peak_prominence
        ``E_peak - E_inf``, a coarse-grained measure of transient amplification.
    has_clear_peak
        ``peak_prominence > peak_prominence_threshold`` and not terminal.
    """

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
            rows.append({
                "c": float(c),
                "m": float(m),
                "e_peak": float("nan"),
                "t_peak": float("nan"),
                "peak_index": -1,
                "is_terminal_peak": False,
                "peak_prominence": float("nan"),
                "has_clear_peak": False,
                "is_reliable": False,
                "warning": "No finite timecourse values.",
                "evidence_strength": "not_assessable",
            })
            continue
        idx = int(np.argmax(value))
        terminal = idx == len(value) - 1
        final = float(value[-1])
        prominence = float(value[idx] - final)
        clear = bool((not terminal) and prominence > peak_prominence_threshold)
        is_reliable = len(value) >= 3
        warning = None if is_reliable else "Too few time points for reliable peak detection."
        rows.append({
            "c": float(c),
            "m": float(m),
            "e_peak": float(value[idx]),
            "t_peak": float(time[idx]),
            "peak_index": idx,
            "is_terminal_peak": terminal,
            "peak_prominence": prominence,
            "has_clear_peak": clear,
            "is_reliable": is_reliable,
            "warning": warning,
            "evidence_strength": _evidence_from_reliability(is_reliable and clear, warning),
        })
    return pd.DataFrame(rows)


def endpoint_final_response(
    timecourse_by_condition: dict[tuple[float, float], pd.DataFrame],
) -> pd.DataFrame:
    """Extract ``E_inf(c, m)`` -- the latest-time response per condition.

    Output columns
    --------------
    e_inf
        Value of the proximal signal at the latest available time point.
        Interpreted as the theory's ``E_infty`` when the observation window is
        long enough to capture late-time behavior.
    t_inf
        The time at which ``e_inf`` was observed.
    """

    rows = []
    for (c, m), df in timecourse_by_condition.items():
        if "time" not in df.columns or "value_mean" not in df.columns:
            raise ValueError("each timecourse dataframe must contain 'time' and 'value_mean'")
        sub = df.sort_values("time").reset_index(drop=True)
        sub = sub[np.isfinite(sub["time"]) & np.isfinite(sub["value_mean"])]
        if len(sub) == 0:
            rows.append({
                "c": float(c),
                "m": float(m),
                "e_inf": float("nan"),
                "t_inf": float("nan"),
                "is_reliable": False,
                "warning": "No finite timecourse values.",
            })
            continue
        last = sub.iloc[-1]
        rows.append({
            "c": float(c),
            "m": float(m),
            "e_inf": float(last["value_mean"]),
            "t_inf": float(last["time"]),
            "is_reliable": True,
            "warning": None,
        })
    return pd.DataFrame(rows)


def delayed_protection_metrics(
    peak_df: pd.DataFrame,
    final_df: pd.DataFrame,
    attenuation_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compare ``E_peak`` with ``E_inf`` to flag delayed-protection signatures."""

    merged = peak_df.merge(final_df, on=["c", "m"], how="inner", suffixes=("_peak", "_final"))
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "c",
                "m",
                "e_peak",
                "e_inf",
                "attenuation",
                "delayed_protection_detected",
                "is_reliable",
                "warning",
            ]
        )
    merged["attenuation"] = merged["e_peak"] - merged["e_inf"]
    merged["delayed_protection_detected"] = merged["attenuation"] > attenuation_threshold
    merged["is_reliable"] = (
        merged.get("is_reliable_peak", True).fillna(False).astype(bool)
        & merged.get("is_reliable_final", True).fillna(False).astype(bool)
    )
    merged["warning"] = None
    return merged[
        [
            "c",
            "m",
            "e_peak",
            "e_inf",
            "attenuation",
            "delayed_protection_detected",
            "is_reliable",
            "warning",
        ]
    ]
