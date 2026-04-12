import pandas as pd

from mechanopharm_infer.discriminate import build_evidence_flags, discriminate_architecture
from mechanopharm_infer.types import QCReport


def test_build_evidence_flags_from_reversal_only():
    reversal = {"low_c_mean_slope": 0.2, "high_c_mean_slope": -0.1, "has_reversal": True, "is_reliable": True}
    ec50_df = pd.DataFrame({"m": [0.0, 0.5], "ec50": [0.8, 1.2], "is_reliable": [True, True]})
    flags = build_evidence_flags(reversal=reversal, ec50_df=ec50_df)
    assert flags["shift_detected"] is True
    assert flags["sign_reversal_detected"] is True
    assert flags["interior_optimum_detected"] is False


def test_discriminate_two_state_supported():
    reversal = {"low_c_mean_slope": 0.2, "high_c_mean_slope": -0.1, "has_reversal": True, "is_reliable": True}
    ec50_df = pd.DataFrame({"m": [0.0, 0.5], "ec50": [0.8, 1.2], "is_reliable": [True, True]})
    endpoint_qc = QCReport(kind="endpoint", passed=True, warnings=[], metrics={})
    result = discriminate_architecture(reversal=reversal, ec50_df=ec50_df, endpoint_qc=endpoint_qc)
    assert result.label == "two_state_supported"


def test_discriminate_protected_state_suggested():
    reversal = {"low_c_mean_slope": 0.1, "high_c_mean_slope": 0.1, "has_reversal": False, "is_reliable": True}
    mopt_df = pd.DataFrame({"c": [0.5, 1.0, 1.5], "m_opt": [0.4, 0.8, 1.2], "optimum_index": [1, 2, 3], "is_edge": [False, False, False], "is_interior": [True, True, True], "prominence": [0.1, 0.1, 0.1], "is_reliable": [True, True, True]})
    peak_df = pd.DataFrame({"c": [1.0], "m": [0.8], "peak_value": [0.85], "peak_time": [2.0], "peak_index": [1], "is_terminal_peak": [False], "peak_prominence": [0.45], "has_clear_peak": [True], "is_reliable": [True], "warning": [None]})
    final_df = pd.DataFrame({"c": [1.0], "m": [0.8], "e_final": [0.40], "final_time": [3.0], "is_reliable": [True], "warning": [None]})
    delayed_df = pd.DataFrame({"c": [1.0], "m": [0.8], "peak_value": [0.85], "e_final": [0.40], "attenuation": [0.45], "delayed_protection_detected": [True], "is_reliable": [True], "warning": [None]})
    endpoint_qc = QCReport(kind="endpoint", passed=True, warnings=[], metrics={})
    result = discriminate_architecture(reversal=reversal, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc)
    assert result.label == "protected_state_suggested"


def test_discriminate_inconclusive_when_endpoint_qc_fails():
    reversal = {"low_c_mean_slope": 0.2, "high_c_mean_slope": -0.1, "has_reversal": True, "is_reliable": True}
    endpoint_qc = QCReport(kind="endpoint", passed=False, warnings=["bad grid"], metrics={})
    result = discriminate_architecture(reversal=reversal, endpoint_qc=endpoint_qc)
    assert result.label == "inconclusive"
