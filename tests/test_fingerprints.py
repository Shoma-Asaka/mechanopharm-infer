import numpy as np
import pandas as pd
import pytest

from mechanopharm_infer.fingerprints import (
    ec50_from_curve,
    ec50_vs_m,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
    endpoint_final_response,
    delayed_protection_metrics,
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
    response = np.vstack([
        1.0 / (1.0 + np.exp(-(c_grid - 0.8))),
        1.0 / (1.0 + np.exp(-(c_grid - 1.0))),
        1.0 / (1.0 + np.exp(-(c_grid - 1.2))),
    ])
    out = ec50_vs_m(c_grid, m_grid, response)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["m", "ec50", "is_monotone", "has_sufficient_range", "is_reliable", "warning"]
    assert len(out) == len(m_grid)


def test_find_mechanical_optima_returns_expected_columns():
    c_grid = np.array([0.1, 0.5, 1.0])
    m_grid = np.array([0.0, 0.5, 1.0])
    response = np.array([[0.1, 0.1, 0.1], [0.5, 0.7, 0.9], [0.2, 0.2, 0.2]])
    out = find_mechanical_optima(c_grid, m_grid, response)
    assert list(out.columns) == ["c", "m_opt", "optimum_index", "is_edge", "is_interior", "prominence", "is_reliable", "warning"]
    assert np.allclose(out["m_opt"].to_numpy(), [0.5, 0.5, 0.5])
    assert np.all(out["is_interior"].to_numpy())


def test_mechanical_sign_reversal_detects_reversal():
    c_grid = np.linspace(0.0, 2.0, 41)
    m_grid = np.linspace(-1.0, 1.0, 31)
    C, M = np.meshgrid(c_grid, m_grid)
    response = 0.4 * C + 0.6 * M * (1.0 - C)
    out = mechanical_sign_reversal(c_grid, m_grid, response)
    assert out["has_reversal"] is True
    assert out["is_reliable"] is True


def test_peak_and_delayed_protection_metrics():
    tc = {(1.0, 0.5): pd.DataFrame({"time": [0.0, 1.0, 2.0], "value_mean": [0.0, 0.8, 0.4]})}
    peak_df = peak_metrics_by_condition(tc)
    final_df = endpoint_final_response(tc)
    delayed_df = delayed_protection_metrics(peak_df, final_df)
    assert bool(peak_df.iloc[0]["has_clear_peak"]) is True
    assert bool(delayed_df.iloc[0]["delayed_protection_detected"]) is True
