import numpy as np
import pandas as pd
import pytest

from mechanopharm_infer.fingerprints import (
    ec50_from_curve,
    ec50_vs_m,
    find_mechanical_optima,
    mechanical_sign_reversal,
)


def test_ec50_from_curve_basic_monotone_case():
    c = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

    ec50 = ec50_from_curve(c, y)

    assert np.isfinite(ec50)
    assert 0.9 <= ec50 <= 1.1


def test_ec50_from_curve_returns_nan_for_too_few_points():
    c = np.array([0.0, 1.0])
    y = np.array([0.1, 0.9])

    ec50 = ec50_from_curve(c, y)

    assert np.isnan(ec50)


def test_ec50_from_curve_handles_descending_curve():
    c = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y = np.array([1.0, 0.8, 0.5, 0.2, 0.0])

    ec50 = ec50_from_curve(c, y)

    assert np.isfinite(ec50)
    assert 0.9 <= ec50 <= 1.1


def test_ec50_from_curve_bad_shape_raises():
    c = np.array([[0.0, 0.5]])
    y = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="1D arrays"):
        ec50_from_curve(c, y)


def test_ec50_vs_m_returns_expected_shape():
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
    assert list(out.columns) == ["m", "ec50"]
    assert len(out) == len(m_grid)
    assert np.allclose(out["m"].to_numpy(), m_grid)


def test_ec50_vs_m_bad_shape_raises():
    c_grid = np.linspace(0.0, 2.0, 10)
    m_grid = np.linspace(0.0, 1.0, 5)
    response = np.zeros((4, 10))

    with pytest.raises(ValueError, match="shape"):
        ec50_vs_m(c_grid, m_grid, response)


def test_find_mechanical_optima_returns_expected_columns():
    c_grid = np.array([0.1, 0.5, 1.0])
    m_grid = np.array([0.0, 0.5, 1.0])

    response = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.5, 0.7, 0.9],
            [0.2, 0.2, 0.2],
        ]
    )

    out = find_mechanical_optima(c_grid, m_grid, response)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["c", "m_opt", "optimum_index", "is_edge", "is_interior"]
    assert len(out) == len(c_grid)
    assert np.allclose(out["m_opt"].to_numpy(), [0.5, 0.5, 0.5])
    assert np.all(out["is_interior"].to_numpy())


def test_find_mechanical_optima_detects_edge_case():
    c_grid = np.array([0.1, 0.5])
    m_grid = np.array([0.0, 0.5, 1.0])

    response = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.2],
            [0.1, 0.8],
        ]
    )

    out = find_mechanical_optima(c_grid, m_grid, response)

    assert out.iloc[0]["is_edge"]
    assert out.iloc[1]["is_edge"]


def test_find_mechanical_optima_bad_shape_raises():
    c_grid = np.linspace(0.0, 2.0, 5)
    m_grid = np.linspace(0.0, 1.0, 4)
    response = np.zeros((4, 4))

    with pytest.raises(ValueError, match="shape"):
        find_mechanical_optima(c_grid, m_grid, response)


def test_mechanical_sign_reversal_detects_reversal():
    c_grid = np.linspace(0.0, 2.0, 41)
    m_grid = np.linspace(-1.0, 1.0, 31)

    C, M = np.meshgrid(c_grid, m_grid)
    response = 0.4 * C + 0.6 * M * (1.0 - C)

    out = mechanical_sign_reversal(c_grid, m_grid, response)

    assert isinstance(out, dict)
    assert "low_c_mean_slope" in out
    assert "high_c_mean_slope" in out
    assert "has_reversal" in out
    assert out["has_reversal"] is True


def test_mechanical_sign_reversal_no_reversal_case():
    c_grid = np.linspace(0.0, 2.0, 41)
    m_grid = np.linspace(-1.0, 1.0, 31)

    C, M = np.meshgrid(c_grid, m_grid)
    response = 0.3 * C + 0.2 * M

    out = mechanical_sign_reversal(c_grid, m_grid, response)

    assert out["has_reversal"] is False


def test_mechanical_sign_reversal_bad_shape_raises():
    c_grid = np.linspace(0.0, 2.0, 10)
    m_grid = np.linspace(-1.0, 1.0, 8)
    response = np.zeros((7, 10))

    with pytest.raises(ValueError, match="shape"):
        mechanical_sign_reversal(c_grid, m_grid, response)
