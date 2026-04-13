import pandas as pd
import pytest

from mechanopharm_infer.schema import standardize_endpoint_schema, standardize_timecourse_schema
from mechanopharm_infer.types import AssayMetadata, coerce_assay_metadata


def test_standardize_endpoint_schema_accepts_aliases_and_adds_metadata_columns():
    df = pd.DataFrame(
        {
            "concentration": [0.1, 0.5],
            "mechanics": [0.0, 0.5],
            "effect": [0.2, 0.7],
        }
    )
    out = standardize_endpoint_schema(df)
    assert {"dataset_id", "system", "assay", "control_flag", "c", "m", "response"}.issubset(out.columns)
    assert out["control_flag"].dtype == bool


def test_standardize_timecourse_schema_accepts_response_alias():
    df = pd.DataFrame(
        {
            "timepoint": [0.0, 1.0],
            "concentration": [0.5, 0.5],
            "mechanics": [0.3, 0.3],
            "response": [0.0, 0.2],
        }
    )
    out = standardize_timecourse_schema(df)
    assert {"time", "c", "m", "value"}.issubset(out.columns)


def test_coerce_assay_metadata_rejects_invalid_response_mode():
    with pytest.raises(ValueError, match="response_mode"):
        coerce_assay_metadata({"response_mode": "bad"})


def test_assay_metadata_defaults_are_valid():
    md = coerce_assay_metadata(AssayMetadata())
    assert md.normalization_mode == "raw"
