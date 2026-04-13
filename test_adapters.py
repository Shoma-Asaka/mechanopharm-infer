import pandas as pd

from mechanopharm_infer.adapters import (
    prepare_jem_endpoint,
    prepare_kalli_timecourse,
    prepare_novak_endpoint,
)


def test_prepare_jem_endpoint_maps_soft_stiff_and_sets_metadata():
    df = pd.DataFrame(
        {
            "concentration": [0.1, 0.1, 1.0, 1.0],
            "stiffness": ["soft", "stiff", "soft", "stiff"],
            "response": [0.2, 0.5, 0.3, 0.7],
        }
    )
    out = prepare_jem_endpoint(df)
    assert set(out["m"].tolist()) == {0.0, 1.0}
    assert set(out["dataset_id"].unique()) == {"JEM_20191360"}
    assert set(out["system"].unique()) == {"breast_cancer_tnbc"}
    assert out["metadata_assay_family"].iloc[0] == "apoptosis"


def test_prepare_novak_endpoint_can_order_named_compression_levels_and_flip_response():
    df = pd.DataFrame(
        {
            "dose": [1, 1, 10, 10],
            "compression": ["control", "compressed", "control", "compressed"],
            "response": [0.9, 0.7, 0.4, 0.2],
        }
    )
    out = prepare_novak_endpoint(df, ordered_mechanics_levels=["control", "compressed"])
    assert set(out["m"].tolist()) == {0.0, 1.0}
    assert abs(out.loc[(out["c"] == 1.0) & (out["m"] == 0.0), "response"].iloc[0] - 0.1) < 1e-9
    assert set(out["system"].unique()) == {"ovarian_cancer_compression"}


def test_prepare_kalli_timecourse_accepts_pressure_alias_and_preserves_time():
    df = pd.DataFrame(
        {
            "concentration": [0.0, 0.0, 0.0, 0.0],
            "pressure": [0, 4, 0, 4],
            "timepoint": [0, 0, 48, 48],
            "response": [0.1, 0.15, 0.2, 0.3],
        }
    )
    out = prepare_kalli_timecourse(df)
    assert set(out["m"].tolist()) == {0.0, 4.0}
    assert set(out["time"].tolist()) == {0.0, 48.0}
    assert set(out["dataset_id"].unique()) == {"s42003_024_07268_1"}
