import pandas as pd

from mechanopharm_infer.discriminate import (
    build_evidence_flags,
    discriminate_architecture,
)


def test_build_evidence_flags_from_reversal_only():
    reversal = {
        "low_c_mean_slope": 0.2,
        "high_c_mean_slope": -0.1,
        "has_reversal": True,
    }

    flags = build_evidence_flags(reversal=reversal)

    assert flags["shift_detected"] is True
    assert flags["sign_reversal_detected"] is True
    assert flags["interior_optimum_detected"] is False
    assert flags["moving_optimum_detected"] is False
    assert flags["transient_peak_detected"] is False
    assert flags["delayed_protection_detected"] is False


def test_discriminate_two_state_supported():
    reversal = {
        "low_c_mean_slope": 0.2,
        "high_c_mean_slope": -0.1,
        "has_reversal": True,
    }

    result = discriminate_architecture(reversal=reversal)

    assert result.label == "two_state_supported"
    assert result.confidence == "moderate"
    assert result.evidence_flags["sign_reversal_detected"] is True


def test_discriminate_protected_state_suggested_from_moving_optimum_and_peak():
    reversal = {
        "low_c_mean_slope": 0.1,
        "high_c_mean_slope": 0.1,
        "has_reversal": False,
    }

    mopt_df = pd.DataFrame(
        {
            "c": [0.5, 1.0, 1.5],
            "m_opt": [0.4, 0.8, 1.2],
            "optimum_index": [1, 2, 3],
            "is_edge": [False, False, False],
            "is_interior": [True, True, True],
        }
    )

    peak_df = pd.DataFrame(
        {
            "c": [1.0],
            "m": [0.8],
            "peak_value": [0.85],
            "peak_time": [2.0],
        }
    )

    final_df = pd.DataFrame(
        {
            "c": [1.0],
            "m": [0.8],
            "e_final": [0.40],
        }
    )

    result = discriminate_architecture(
        reversal=reversal,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
    )

    assert result.label == "protected_state_suggested"
    assert result.evidence_flags["interior_optimum_detected"] is True
    assert result.evidence_flags["moving_optimum_detected"] is True
    assert result.evidence_flags["transient_peak_detected"] is True
    assert result.evidence_flags["delayed_protection_detected"] is True


def test_discriminate_inconclusive():
    reversal = {
        "low_c_mean_slope": 0.01,
        "high_c_mean_slope": 0.01,
        "has_reversal": False,
    }

    mopt_df = pd.DataFrame(
        {
            "c": [0.5, 1.0, 1.5],
            "m_opt": [0.0, 0.0, 0.0],
            "optimum_index": [0, 0, 0],
            "is_edge": [True, True, True],
            "is_interior": [False, False, False],
        }
    )

    result = discriminate_architecture(
        reversal=reversal,
        mopt_df=mopt_df,
    )

    assert result.label == "inconclusive"
    assert result.confidence == "low"
