from __future__ import annotations

import pandas as pd

from .types import DiscriminationResult


def _detect_interior_optimum(
    mopt_df: pd.DataFrame,
    tol: float = 1e-8,
) -> bool:
    if "is_interior" in mopt_df.columns:
        return bool(mopt_df["is_interior"].fillna(False).any())

    if "m_opt" not in mopt_df.columns:
        return False

    finite = mopt_df["m_opt"].dropna()
    if finite.empty:
        return False

    m_min = float(finite.min())
    m_max = float(finite.max())
    interior = (mopt_df["m_opt"] > m_min + tol) & (mopt_df["m_opt"] < m_max - tol)
    return bool(interior.any())


def _detect_moving_optimum(
    mopt_df: pd.DataFrame,
    span_threshold: float = 0.1,
) -> bool:
    if "m_opt" not in mopt_df.columns:
        return False

    finite = mopt_df["m_opt"].dropna()
    if len(finite) < 2:
        return False

    span = float(finite.max() - finite.min())
    return span >= span_threshold


def _detect_transient_peak(
    peak_df: pd.DataFrame,
    final_df: pd.DataFrame,
    peak_delta_threshold: float = 0.05,
) -> bool:
    merged = peak_df.merge(final_df, on=["c", "m"], how="inner")
    if merged.empty:
        return False

    delta = merged["peak_value"] - merged["e_final"]
    return bool((delta > peak_delta_threshold).any())


def build_evidence_flags(
    reversal: dict[str, float | bool],
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
) -> dict[str, bool]:
    flags = {
        "shift_detected": True,
        "sign_reversal_detected": bool(reversal.get("has_reversal", False)),
        "interior_optimum_detected": False,
        "moving_optimum_detected": False,
        "transient_peak_detected": False,
        "delayed_protection_detected": False,
    }

    if mopt_df is not None and not mopt_df.empty:
        flags["interior_optimum_detected"] = _detect_interior_optimum(mopt_df)
        flags["moving_optimum_detected"] = _detect_moving_optimum(mopt_df)

    if peak_df is not None and final_df is not None and not peak_df.empty and not final_df.empty:
        transient = _detect_transient_peak(peak_df, final_df)
        flags["transient_peak_detected"] = transient
        flags["delayed_protection_detected"] = transient

    return flags


def discriminate_architecture(
    reversal: dict[str, float | bool],
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
) -> DiscriminationResult:
    evidence_flags = build_evidence_flags(
        reversal=reversal,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
    )

    notes: list[str] = []

    protected_votes = sum(
        [
            evidence_flags["interior_optimum_detected"],
            evidence_flags["moving_optimum_detected"],
            evidence_flags["transient_peak_detected"],
            evidence_flags["delayed_protection_detected"],
        ]
    )

    if protected_votes >= 2:
        label = "protected_state_suggested"
        confidence = "moderate" if protected_votes == 2 else "high"
        notes.append("Protected-state-like signatures are present.")
    elif evidence_flags["sign_reversal_detected"]:
        label = "two_state_supported"
        confidence = "moderate"
        notes.append("Observed signatures remain compatible with a strict two-state architecture.")
    else:
        label = "inconclusive"
        confidence = "low"
        notes.append("Current fingerprints do not yet distinguish the minimal architecture.")

    return DiscriminationResult(
        label=label,
        evidence_flags=evidence_flags,
        notes=notes,
        confidence=confidence,
    )
