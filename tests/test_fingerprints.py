import numpy as np
import pandas as pd

from mechanopharm_infer.fingerprints import (
    delayed_protection_metrics,
    ec50_from_curve,
    ec50_vs_m,
    endpoint_final_response,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
)


def test_ec50_from_curve_basic_monotone_case():
    c = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    out = ec50_from_curve(c, y)
    assert out["is_reliable"] is True
    assert np.isfinite(out["ec50"])
    assert 0.9 <= out["ec50"] <= 1.1


def test_ec50_from_curve_returns_unreliable_for_too_few_points():
    out = ec50_from_curve(np.array([0.0, 1.0]), np.array([0.1, 0.9]))
    assert out["is_reliable"] is False
    assert np.isnan(out["ec50"])


def test_ec50_vs_m_returns_expected_columns():
    c_grid = np.linspace(0.0, 2.0, 21)
    m_grid = np.array([0.0, 0.5, 1.0])
    response = np.vstack(
        [
            1.0 / (1.0 + np.exp(-(c_grid - 0.8))),
            1.0 / (1.0 + np.exp(-(c_grid - 1.0))),
            1.0 / (1.0 + np.exp(-(c_grid - 1.2))),
        ]
    )
    out = ec50_vs_m(c_grid, m_grid, response)
    assert isinstance(out, pd.DataFrame)
    assert {
        "m",
        "ec50",
        "is_monotone",
        "has_sufficient_range",
        "is_reliable",
        "warning",
        "n_points",
        "dynamic_range",
        "monotonicity_score",
        "evidence_strength",
    }.issubset(set(out.columns))
    assert len(out) == len(m_grid)


def test_find_mechanical_optima_returns_expected_columns_and_subgrid_estimate():
    c_grid = np.array([0.1, 0.5, 1.0])
    m_grid = np.array([0.0, 0.5, 1.0])
    response = np.array([[0.1, 0.1, 0.1], [0.5, 0.7, 0.9], [0.2, 0.2, 0.2]])
    out = find_mechanical_optima(c_grid, m_grid, response)
    assert {
        "c",
        "m_opt",
        "m_opt_grid",
        "optimum_index",
        "is_edge",
        "is_interior",
        "prominence",
        "neighbor_margin",
        "response_span",
        "is_reliable",
        "warning",
        "evidence_strength",
        "n_mechanics_points",
    }.issubset(set(out.columns))
    assert np.allclose(out["m_opt_grid"].to_numpy(), [0.5, 0.5, 0.5])
    assert np.all(out["is_interior"].to_numpy())
    # m_opt must lie inside the bracketing interval (0.0, 1.0).
    assert np.all(out["m_opt"].between(0.0, 1.0))


def test_parabolic_refinement_recovers_known_peak():
    # f(m) = 1 - (m - 0.3)^2.  Grid maxes at m=0.4 but the true optimum is m=0.3.
    m_grid = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    c_grid = np.array([1.0])
    f = 1.0 - (m_grid - 0.3) ** 2
    response = f[:, None]
    out = find_mechanical_optima(c_grid, m_grid, response)
    assert abs(float(out["m_opt"].iloc[0]) - 0.3) < 0.05


def test_mechanical_sign_reversal_detects_reversal_and_estimates_c_rev():
    c_grid = np.linspace(0.0, 2.0, 41)
    m_grid = np.linspace(-1.0, 1.0, 31)
    cc, mm = np.meshgrid(c_grid, m_grid)
    # E = 0.4 c + 0.6 m (1 - c).  Mechanical slope is 0.6 - 0.6 c, so c_rev = 1.
    response = 0.4 * cc + 0.6 * mm * (1.0 - cc)
    out = mechanical_sign_reversal(c_grid, m_grid, response)
    assert out["has_reversal"] is True
    assert out["is_reliable"] is True
    assert out["n_reliable_columns"] >= 3
    assert abs(float(out["c_rev_estimate"]) - 1.0) < 0.05
    assert "delta_lambda_proxy" in out
    assert "delta_mu_proxy" in out


def test_peak_and_delayed_protection_metrics_use_canonical_columns():
    tc = {(1.0, 0.5): pd.DataFrame({"time": [0.0, 1.0, 2.0], "value_mean": [0.0, 0.8, 0.4]})}
    peak_df = peak_metrics_by_condition(tc)
    final_df = endpoint_final_response(tc)
    delayed_df = delayed_protection_metrics(peak_df, final_df)
    assert "e_peak" in peak_df.columns
    assert "t_peak" in peak_df.columns
    assert "e_inf" in final_df.columns
    assert "t_inf" in final_df.columns
    assert bool(peak_df.iloc[0]["has_clear_peak"]) is True
    assert bool(delayed_df.iloc[0]["delayed_protection_detected"]) is True
    assert {"e_peak", "e_inf", "attenuation"}.issubset(delayed_df.columns)
