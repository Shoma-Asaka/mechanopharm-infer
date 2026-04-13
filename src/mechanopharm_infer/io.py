from __future__ import annotations

from pathlib import Path
import pandas as pd

from .schema import standardize_endpoint_schema, standardize_timecourse_schema


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def load_endpoint_csv(path: str | Path) -> pd.DataFrame:
    """Load endpoint data and coerce it into the canonical schema.

    The loader accepts the original minimal columns (c, m, response) and a small
    set of aliases such as concentration/mechanics/effect for literature-derived
    tables. Optional metadata columns are preserved or added with conservative
    defaults.
    """

    df = _read_csv(path)
    return standardize_endpoint_schema(df)


def load_timecourse_csv(path: str | Path) -> pd.DataFrame:
    """Load timecourse data and coerce it into the canonical schema."""

    df = _read_csv(path)
    return standardize_timecourse_schema(df)
