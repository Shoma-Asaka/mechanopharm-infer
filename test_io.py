from pathlib import Path
import pandas as pd
import pytest

from mechanopharm_infer.io import load_endpoint_csv, load_timecourse_csv


def test_load_endpoint_csv_valid(tmp_path: Path):
    path = tmp_path / "endpoint.csv"
    pd.DataFrame(
        {
            "c": [0.1, 0.5],
            "m": [0.0, 0.5],
            "response": [0.2, 0.7],
        }
    ).to_csv(path, index=False)

    df = load_endpoint_csv(path)

    assert list(df.columns) == ["c", "m", "response"]
    assert df.shape == (2, 3)


def test_load_endpoint_csv_missing_column(tmp_path: Path):
    path = tmp_path / "bad_endpoint.csv"
    pd.DataFrame(
        {
            "c": [0.1],
            "m": [0.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_endpoint_csv(path)


def test_load_endpoint_csv_missing_value(tmp_path: Path):
    path = tmp_path / "bad_endpoint_missing.csv"
    pd.DataFrame(
        {
            "c": [0.1, None],
            "m": [0.0, 0.5],
            "response": [0.2, 0.7],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_endpoint_csv(path)


def test_load_timecourse_csv_valid(tmp_path: Path):
    path = tmp_path / "timecourse.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "c": [0.5, 0.5],
            "m": [0.3, 0.3],
            "value": [0.0, 0.2],
        }
    ).to_csv(path, index=False)

    df = load_timecourse_csv(path)

    assert list(df.columns) == ["time", "c", "m", "value"]
    assert df.shape == (2, 4)


def test_load_timecourse_csv_missing_column(tmp_path: Path):
    path = tmp_path / "bad_timecourse.csv"
    pd.DataFrame(
        {
            "time": [0.0],
            "c": [0.5],
            "m": [0.3],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_timecourse_csv(path)


def test_load_timecourse_csv_missing_value(tmp_path: Path):
    path = tmp_path / "bad_timecourse_missing.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "c": [0.5, 0.5],
            "m": [0.3, None],
            "value": [0.0, 0.2],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_timecourse_csv(path)
