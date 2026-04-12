from __future__ import annotations

from pathlib import Path
import pandas as pd


ENDPOINT_REQUIRED_COLUMNS = {"c", "m", "response"}
TIMECOURSE_REQUIRED_COLUMNS = {"time", "c", "m", "value"}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _validate_required_columns(
    df: pd.DataFrame,
    required: set[str],
    kind: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{kind} data is missing required columns: {sorted(missing)}"
        )


def _validate_no_missing_required_values(
    df: pd.DataFrame,
    required: set[str],
    kind: str,
) -> None:
    missing_mask = df[list(required)].isnull().any(axis=1)
    if bool(missing_mask.any()):
        n_bad = int(missing_mask.sum())
        raise ValueError(
            f"{kind} data contains missing values in required columns "
            f"for {n_bad} row(s)"
        )


def _coerce_numeric_columns(
    df: pd.DataFrame,
    columns: set[str],
    kind: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception as exc:
            raise ValueError(
                f"{kind} column '{col}' could not be converted to numeric"
            ) from exc
    return out


def load_endpoint_csv(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_required_columns(df, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    df = _coerce_numeric_columns(df, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    _validate_no_missing_required_values(df, ENDPOINT_REQUIRED_COLUMNS, "endpoint")
    return df.copy()


def load_timecourse_csv(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_required_columns(df, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    df = _coerce_numeric_columns(df, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    _validate_no_missing_required_values(df, TIMECOURSE_REQUIRED_COLUMNS, "timecourse")
    return df.copy()
