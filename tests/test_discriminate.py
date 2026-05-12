import pandas as pd

from mechanopharm_infer.discriminate import (
    build_evidence_flags,
    build_evidence_table,
    build_fingerprint_values,
    discriminate_architecture,
)
from mechanopharm_infer.types import QCReport


def test_build_evidence_flags_from_reversal_only():
    reversal = {
        "low_c_mean_slope": 0.2,
        "high_c_mean_slope": -0.1,
        "has_reversal": True,
        "is_reliable": True,
        "evidence_strength": "moderate",
    }
    ec50_df = pd.DataFrame(
        {
            "m": [0.0, 0.5],
            "ec50": [0.8, 1.2],
            "is_reliable": [True, True],
            "evidence_strength": ["moderate", "strong"],
        }
    )
    flags = build_evidence_flags(reversal=reversal, ec50_df=ec50_df)
    assert flags["shift_detected"] is True
    assert flags["sign_reversal_detected"] is True
    assert flags["interior_optimum_detected"] is False


def test_build_evidence_table_contains_strengths():
    reversal = {"has_reversal": True, "is_reliable": True, "evidence_strength": "moderate"}
    ec50_df = pd.DataFrame(
        {
            "m": [0.0, 0.5],
            "ec50": [0.8, 1.2],
            "is_reliable": [True, True],
            "evidence_strength": ["weak", "moderate"],
        }
    )
    table = build_evidence_table(reversal=reversal, ec50_df=ec50_df)
    assert {"fingerprint", "supported", "evidence_strength", "source", "notes"}.issubset(table.columns)
    shift_row = table.loc[table["fingerprint"] == "shift"].iloc[0]
    assert shift_row["supported"] in [True, False]
    assert shift_row["evidence_strength"] in ["weak", "moderate", "strong", "none", "not_assessable"]


def test_discriminate_two_state_supported():
    reversal = {
        "low_c_mean_slope": 0.2,
        "high_c_mean_slope": -0.1,
        "has_reversal": True,
        "is_reliable": True,
        "evidence_strength": "moderate",
        "c_rev_estimate": 1.0,
        "delta_lambda_proxy": 0.2,
        "delta_mu_proxy": -0.2,
    }
    ec50_df = pd.DataFrame(
        {
            "m": [0.0, 0.5],
            "ec50": [0.8, 1.2],
            "is_reliable": [True, True],
            "evidence_strength": ["moderate", "strong"],
        }
    )
    endpoint_qc = QCReport(kind="endpoint", passed=True, warnings=[], metrics={})
    result = discriminate_architecture(reversal=reversal, ec50_df=ec50_df, endpoint_qc=endpoint_qc)
    assert result.label == "two_state_supported"
    assert result.supporting_evidence is not None
    assert any("sign_reversal" in x or "shift" in x for x in result.supporting_evidence)
    assert result.fingerprint_values is not None
    assert "c_rev" in result.fingerprint_values
    assert result.fingerprint_values["c_rev"]["estimate"] == 1.0


def test_discriminate_protected_state_suggested():
    reversal = {
        "low_c_mean_slope": 0.1,
        "high_c_mean_slope": 0.1,
        "has_reversal": False,
        "is_reliable": True,
        "evidence_strength": "none",
    }
    mopt_df = pd.DataFrame(
        {
            "c": [0.5, 1.0, 1.5],
            "m_opt": [0.4, 0.8, 1.2],
            "optimum_index": [1, 2, 3],
            "is_edge": [False, False, False],
            "is_interior": [True, True, True],
            "prominence": [0.1, 0.1, 0.1],
            "is_reliable": [True, True, True],
            "evidence_strength": ["moderate", "moderate", "strong"],
        }
    )
    peak_df = pd.DataFrame(
        {
            "c": [1.0],
            "m": [0.8],
            "e_peak": [0.85],
            "t_peak": [2.0],
            "peak_index": [1],
            "is_terminal_peak": [False],
            "peak_prominence": [0.45],
            "has_clear_peak": [True],
            "is_reliable": [True],
            "warning": [None],
            "evidence_strength": ["moderate"],
        }
    )
    final_df = pd.DataFrame(
        {
            "c": [1.0],
            "m": [0.8],
            "e_inf": [0.40],
            "t_inf": [3.0],
            "is_reliable": [True],
            "warning": [None],
        }
    )
    delayed_df = pd.DataFrame(
        {
            "c": [1.0],
            "m": [0.8],
            "e_peak": [0.85],
            "e_inf": [0.40],
            "attenuation": [0.45],
            "delayed_protection_detected": [True],
            "is_reliable": [True],
            "warning": [None],
            "evidence_strength": ["strong"],
        }
    )
    endpoint_qc = QCReport(kind="endpoint", passed=True, warnings=[], metrics={})
    result = discriminate_architecture(
        reversal=reversal,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
        delayed_df=delayed_df,
        endpoint_qc=endpoint_qc,
    )
    assert result.label == "protected_state_suggested"
    assert result.evidence_strengths is not None
    assert result.evidence_strengths["interior_optimum"] in ["moderate", "strong"]
    assert result.fingerprint_values is not None
    assert "m_star_vs_c" in result.fingerprint_values
    assert len(result.fingerprint_values["m_star_vs_c"]) == 3


def test_discriminate_inconclusive_when_endpoint_qc_fails():
    reversal = {
        "low_c_mean_slope": 0.2,
        "high_c_mean_slope": -0.1,
        "has_reversal": True,
        "is_reliable": True,
        "evidence_strength": "moderate",
    }
    endpoint_qc = QCReport(kind="endpoint", passed=False, warnings=["bad grid"], metrics={})
    result = discriminate_architecture(reversal=reversal, endpoint_qc=endpoint_qc)
    assert result.label == "inconclusive"
    assert result.warnings == ["bad grid"]


def test_build_fingerprint_values_minimal():
    reversal = {
        "c_rev_estimate": 1.5,
        "delta_lambda_proxy": 0.3,
        "delta_mu_proxy": -0.2,
        "has_reversal": True,
        "reversal_window_center": 1.4,
    }
    payload = build_fingerprint_values(reversal)
    assert payload["c_rev"]["estimate"] == 1.5
    assert payload["c_rev"]["has_reversal"] is True
    assert payload["EC50_vs_m"] == []
    assert payload["m_star_vs_c"] == []
