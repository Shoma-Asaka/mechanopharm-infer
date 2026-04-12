import numpy as np
import pandas as pd
import pytest

from mechanopharm_infer.preprocess import (
    summarize_endpoint,
    endpoint_to_grid,
    summarize_timecourse,
    split_timecourses_by_condition,
    check_endpoint_qc,
    check_timecourse_qc,
)


def test_summarize_endpoint_without_replicate():
    df = pd.DataFrame({"c": [0.1, 0.5, 0.1, 0.5], "m": [0.0, 0.0, 0.5, 0.5], "response": [0.2, 0.6, 0.3, 0.7]})
    summary = summarize_endpoint(df)
    assert list(summary.columns) == ["c", "m", "response_mean", "response_sd", "n"]
    assert summary.shape[0] == 4
    assert np.all(summary["n"].to_numpy() == 1)


def test_endpoint_to_grid_preserves_missing_cells_as_nan():
    summary = pd.DataFrame({"c": [0.1, 0.5, 0.1], "m": [0.0, 0.0, 0.5], "response_mean": [0.2, 0.6, 0.3], "response_sd": [np.nan, np.nan, np.nan], "n": [1, 1, 1]})
    _, _, response = endpoint_to_grid(summary)
    assert response.shape == (2, 2)
    assert np.isnan(response[1, 1])


def test_summarize_timecourse_with_replicate():
    df = pd.DataFrame({"time": [0.0, 0.0, 1.0, 1.0], "c": [0.5, 0.5, 0.5, 0.5], "m": [0.3, 0.3, 0.3, 0.3], "value": [0.0, 0.1, 0.2, 0.4], "replicate": [1, 2, 1, 2]})
    summary = summarize_timecourse(df)
    assert summary.shape[0] == 2
    assert np.all(summary["n"].to_numpy() == 2)


def test_split_timecourses_by_condition():
    summary = pd.DataFrame({"time": [1.0, 0.0, 1.0, 0.0], "c": [0.5, 0.5, 1.0, 1.0], "m": [0.3, 0.3, 0.8, 0.8], "value_mean": [0.2, 0.0, 0.4, 0.1], "value_sd": [np.nan]*4, "n": [1]*4})
    split = split_timecourses_by_condition(summary)
    assert len(split) == 2
    first = split[(0.5, 0.3)]
    assert np.allclose(first["time"].to_numpy(), [0.0, 1.0])


def test_check_endpoint_qc_passes_reasonable_grid():
    summary = pd.DataFrame({
        "c": [0.1, 0.5, 1.0, 0.1, 0.5, 1.0],
        "m": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        "response_mean": [0.2, 0.5, 0.8, 0.3, 0.6, 0.7],
        "response_sd": [np.nan]*6,
        "n": [1]*6,
    })
    qc = check_endpoint_qc(summary)
    assert qc.kind == "endpoint"
    assert qc.passed is True
    assert qc.metrics["n_unique_c"] == 3


def test_check_timecourse_qc_fails_short_series():
    summary = pd.DataFrame({"time": [0.0, 1.0], "c": [0.5, 0.5], "m": [0.3, 0.3], "value_mean": [0.0, 0.2], "value_sd": [np.nan, np.nan], "n": [1, 1]})
    qc = check_timecourse_qc(summary)
    assert qc.kind == "timecourse"
    assert qc.passed is False
    assert qc.warnings
