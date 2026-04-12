import numpy as np
import pandas as pd
import pytest

from mechanopharm_infer.preprocess import (
    summarize_endpoint,
    endpoint_to_grid,
    summarize_timecourse,
    split_timecourses_by_condition,
)


def test_summarize_endpoint_without_replicate():
    df = pd.DataFrame(
        {
            "c": [0.1, 0.5, 0.1, 0.5],
            "m": [0.0, 0.0, 0.5, 0.5],
            "response": [0.2, 0.6, 0.3, 0.7],
        }
    )

    summary = summarize_endpoint(df)

    assert list(summary.columns) == ["c", "m", "response_mean", "response_sd", "n"]
    assert summary.shape[0] == 4
    assert np.all(summary["n"].to_numpy() == 1)
    assert summary["response_mean"].notnull().all()


def test_summarize_endpoint_with_replicate():
    df = pd.DataFrame(
        {
            "c": [0.1, 0.1, 0.5, 0.5],
            "m": [0.0, 0.0, 0.5, 0.5],
            "response": [0.2, 0.4, 0.6, 0.8],
            "replicate": [1, 2, 1, 2],
        }
    )

    summary = summarize_endpoint(df)

    assert summary.shape[0] == 2
    assert np.all(summary["n"].to_numpy() == 2)
    row1 = summary.iloc[0]
    row2 = summary.iloc[1]
    assert np.isclose(row1["response_mean"], 0.3)
    assert np.isclose(row2["response_mean"], 0.7)


def test_summarize_endpoint_missing_required_column():
    df = pd.DataFrame(
        {
            "c": [0.1],
            "m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        summarize_endpoint(df)


def test_endpoint_to_grid_shape_and_values():
    summary = pd.DataFrame(
        {
            "c": [0.1, 0.5, 0.1, 0.5],
            "m": [0.0, 0.0, 0.5, 0.5],
            "response_mean": [0.2, 0.6, 0.3, 0.7],
            "response_sd": [np.nan, np.nan, np.nan, np.nan],
            "n": [1, 1, 1, 1],
        }
    )

    c_grid, m_grid, response = endpoint_to_grid(summary)

    assert np.allclose(c_grid, [0.1, 0.5])
    assert np.allclose(m_grid, [0.0, 0.5])
    assert response.shape == (2, 2)
    assert np.isclose(response[0, 0], 0.2)
    assert np.isclose(response[0, 1], 0.6)
    assert np.isclose(response[1, 0], 0.3)
    assert np.isclose(response[1, 1], 0.7)


def test_endpoint_to_grid_preserves_missing_cells_as_nan():
    summary = pd.DataFrame(
        {
            "c": [0.1, 0.5, 0.1],
            "m": [0.0, 0.0, 0.5],
            "response_mean": [0.2, 0.6, 0.3],
            "response_sd": [np.nan, np.nan, np.nan],
            "n": [1, 1, 1],
        }
    )

    c_grid, m_grid, response = endpoint_to_grid(summary)

    assert response.shape == (2, 2)
    assert np.isnan(response[1, 1])


def test_summarize_timecourse_without_replicate():
    df = pd.DataFrame(
        {
            "time": [0.0, 1.0, 0.0, 1.0],
            "c": [0.5, 0.5, 1.0, 1.0],
            "m": [0.3, 0.3, 0.8, 0.8],
            "value": [0.0, 0.2, 0.1, 0.4],
        }
    )

    summary = summarize_timecourse(df)

    assert list(summary.columns) == ["time", "c", "m", "value_mean", "value_sd", "n"]
    assert summary.shape[0] == 4
    assert np.all(summary["n"].to_numpy() == 1)


def test_summarize_timecourse_with_replicate():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0],
            "c": [0.5, 0.5, 0.5, 0.5],
            "m": [0.3, 0.3, 0.3, 0.3],
            "value": [0.0, 0.1, 0.2, 0.4],
            "replicate": [1, 2, 1, 2],
        }
    )

    summary = summarize_timecourse(df)

    assert summary.shape[0] == 2
    assert np.all(summary["n"].to_numpy() == 2)
    assert np.isclose(summary.iloc[0]["value_mean"], 0.05)
    assert np.isclose(summary.iloc[1]["value_mean"], 0.3)


def test_summarize_timecourse_missing_required_column():
    df = pd.DataFrame(
        {
            "time": [0.0],
            "c": [0.5],
            "m": [0.3],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        summarize_timecourse(df)


def test_split_timecourses_by_condition():
    summary = pd.DataFrame(
        {
            "time": [1.0, 0.0, 1.0, 0.0],
            "c": [0.5, 0.5, 1.0, 1.0],
            "m": [0.3, 0.3, 0.8, 0.8],
            "value_mean": [0.2, 0.0, 0.4, 0.1],
            "value_sd": [np.nan, np.nan, np.nan, np.nan],
            "n": [1, 1, 1, 1],
        }
    )

    split = split_timecourses_by_condition(summary)

    assert len(split) == 2
    assert (0.5, 0.3) in split
    assert (1.0, 0.8) in split

    first = split[(0.5, 0.3)]
    assert np.allclose(first["time"].to_numpy(), [0.0, 1.0])
    assert np.allclose(first["value_mean"].to_numpy(), [0.0, 0.2])
