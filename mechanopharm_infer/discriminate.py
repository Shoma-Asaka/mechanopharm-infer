from __future__ import annotations

import pandas as pd

from .types import DiscriminationResult, QCReport


def _reliable_mask(df: pd.DataFrame, column: str = "is_reliable") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(False).astype(bool)
    return pd.Series([True] * len(df), index=df.index)


def _detect_shift(ec50_df: pd.DataFrame, span_threshold: float = 0.1) -> bool:
    if "ec50" not in ec50_df.columns:
        return False
    finite = ec50_df.loc[_reliable_mask(ec50_df), "ec50"].dropna()
    return bool(len(finite) >= 2 and float(finite.max() - finite.min()) >= span_threshold)


def _detect_interior_optimum(mopt_df: pd.DataFrame, prominence_threshold: float = 0.03) -> bool:
    if "is_interior" not in mopt_df.columns:
        return False
    reliable = mopt_df[_reliable_mask(mopt_df)]
    if reliable.empty:
        return False
    prom_ok = reliable["prominence"].fillna(float("-inf")) >= prominence_threshold if "prominence" in reliable.columns else True
    return bool((reliable["is_interior"].fillna(False).astype(bool) & prom_ok).any())


def _detect_moving_optimum(mopt_df: pd.DataFrame, span_threshold: float = 0.1) -> bool:
    if "m_opt" not in mopt_df.columns:
        return False
    finite = mopt_df.loc[_reliable_mask(mopt_df), "m_opt"].dropna()
    return bool(len(finite) >= 2 and float(finite.max() - finite.min()) >= span_threshold)


def _detect_transient_peak(peak_df: pd.DataFrame, final_df: pd.DataFrame, peak_delta_threshold: float = 0.05) -> bool:
    merged = peak_df.merge(final_df, on=["c", "m"], how="inner", suffixes=("_peak", "_final"))
    if merged.empty:
        return False
    mask = (merged.get("has_clear_peak", True).fillna(False).astype(bool) & ~merged.get("is_terminal_peak", False).fillna(True).astype(bool))
    mask &= merged.get("is_reliable_peak", True).fillna(True).astype(bool)
    mask &= merged.get("is_reliable_final", True).fillna(True).astype(bool)
    return bool((mask & ((merged["peak_value"] - merged["e_final"]) > peak_delta_threshold)).any())


def _detect_delayed_protection(delayed_df: pd.DataFrame) -> bool:
    if "delayed_protection_detected" not in delayed_df.columns:
        return False
    reliable = delayed_df[_reliable_mask(delayed_df)]
    return bool(not reliable.empty and reliable["delayed_protection_detected"].fillna(False).astype(bool).any())


def _qc_passed(qc: QCReport | None) -> bool:
    return True if qc is None else bool(qc.passed)


def build_evidence_flags(reversal: dict[str, float | bool | str | None], ec50_df: pd.DataFrame | None = None, mopt_df: pd.DataFrame | None = None, peak_df: pd.DataFrame | None = None, final_df: pd.DataFrame | None = None, delayed_df: pd.DataFrame | None = None, endpoint_qc: QCReport | None = None, timecourse_qc: QCReport | None = None) -> dict[str, bool]:
    flags = {
        "endpoint_qc_passed": _qc_passed(endpoint_qc),
        "timecourse_qc_passed": _qc_passed(timecourse_qc),
        "shift_detected": False,
        "sign_reversal_detected": bool(reversal.get("has_reversal", False)) and bool(reversal.get("is_reliable", True)),
        "interior_optimum_detected": False,
        "moving_optimum_detected": False,
        "transient_peak_detected": False,
        "delayed_protection_detected": False,
    }
    if ec50_df is not None and not ec50_df.empty:
        flags["shift_detected"] = _detect_shift(ec50_df)
    if mopt_df is not None and not mopt_df.empty:
        flags["interior_optimum_detected"] = _detect_interior_optimum(mopt_df)
        flags["moving_optimum_detected"] = _detect_moving_optimum(mopt_df)
    if peak_df is not None and final_df is not None and not peak_df.empty and not final_df.empty:
        flags["transient_peak_detected"] = _detect_transient_peak(peak_df, final_df)
    if delayed_df is not None and not delayed_df.empty:
        flags["delayed_protection_detected"] = _detect_delayed_protection(delayed_df)
    return flags


def discriminate_architecture(reversal: dict[str, float | bool | str | None], ec50_df: pd.DataFrame | None = None, mopt_df: pd.DataFrame | None = None, peak_df: pd.DataFrame | None = None, final_df: pd.DataFrame | None = None, delayed_df: pd.DataFrame | None = None, endpoint_qc: QCReport | None = None, timecourse_qc: QCReport | None = None) -> DiscriminationResult:
    evidence_flags = build_evidence_flags(reversal, ec50_df, mopt_df, peak_df, final_df, delayed_df, endpoint_qc, timecourse_qc)
    notes: list[str] = []
    if not evidence_flags["endpoint_qc_passed"]:
        notes.append("Endpoint QC did not pass; architecture call withheld.")
        if endpoint_qc and endpoint_qc.warnings:
            notes.extend(endpoint_qc.warnings)
        return DiscriminationResult(label="inconclusive", evidence_flags=evidence_flags, notes=notes, confidence="low")
    protected_score = 2 * evidence_flags["interior_optimum_detected"] + 1 * evidence_flags["moving_optimum_detected"] + 1 * evidence_flags["transient_peak_detected"] + 2 * evidence_flags["delayed_protection_detected"]
    if protected_score >= 3:
        label = "protected_state_suggested"
        confidence = "high" if protected_score >= 4 else "moderate"
        notes.append("Protected-state-like signatures are present with nontrivial evidence weight.")
    elif evidence_flags["sign_reversal_detected"] or evidence_flags["shift_detected"]:
        label = "two_state_supported"
        confidence = "moderate" if evidence_flags["sign_reversal_detected"] else "low"
        notes.append("Observed signatures remain compatible with a strict two-state architecture.")
    else:
        label = "inconclusive"
        confidence = "low"
        notes.append("Current fingerprints do not yet distinguish the minimal architecture.")
    return DiscriminationResult(label=label, evidence_flags=evidence_flags, notes=notes, confidence=confidence)
