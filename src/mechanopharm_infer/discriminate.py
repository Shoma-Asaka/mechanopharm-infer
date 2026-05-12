"""Architecture-class discrimination from response-landscape fingerprints.

Implements the signature-level inference logic of Section 5 of the theory
paper.  The discrimination is intentionally conservative: it makes an
architecture-class call (``two_state_supported`` /
``protected_state_suggested`` / ``inconclusive``) rather than attempting a
unique microscopic identification.

The result includes a structured ``fingerprint_values`` payload that records
the numerical values of each fingerprint together with their bootstrap CIs
when available.  This payload is the primary programmatic interface for
downstream consumers (web UIs, comparison tables, dashboards).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import DiscriminationResult, QCReport


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _reliable_mask(df: pd.DataFrame, column: str = "is_reliable") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(False).astype(bool)
    return pd.Series([True] * len(df), index=df.index)


def _strength_to_rank(value: str | None) -> int:
    order = {"not_assessable": 0, "none": 1, "weak": 2, "moderate": 3, "strong": 4}
    if value is None:
        return 0
    return order.get(str(value), 0)


def _max_strength_from_df(df: pd.DataFrame | None, default: str = "not_assessable") -> str:
    if df is None or df.empty:
        return default
    strengths = [str(x) for x in df.get("evidence_strength", pd.Series(dtype=object)).dropna().tolist()]
    base = max(strengths, key=_strength_to_rank) if strengths else default
    reliability_cols = [
        c
        for c in [
            "ec50_bootstrap_reliability",
            "mopt_bootstrap_reliability",
            "delayed_bootstrap_reliability",
        ]
        if c in df.columns
    ]
    if not reliability_cols:
        return base
    rel = pd.to_numeric(df[reliability_cols].stack(), errors="coerce").dropna()
    if rel.empty:
        return base
    rel_mean = float(rel.mean())
    if rel_mean >= 0.8 and _strength_to_rank(base) >= _strength_to_rank("moderate"):
        return "strong"
    if rel_mean < 0.35 and _strength_to_rank(base) > _strength_to_rank("weak"):
        return "weak"
    return base


def _reversal_strength(reversal: dict[str, Any]) -> str:
    return str(reversal.get("evidence_strength", "not_assessable"))


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
    prom_ok = (
        reliable["prominence"].fillna(float("-inf")) >= prominence_threshold
        if "prominence" in reliable.columns
        else True
    )
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
    mask = (
        merged.get("has_clear_peak", True).fillna(False).astype(bool)
        & ~merged.get("is_terminal_peak", False).fillna(True).astype(bool)
    )
    mask &= merged.get("is_reliable_peak", True).fillna(True).astype(bool)
    mask &= merged.get("is_reliable_final", True).fillna(True).astype(bool)
    return bool((mask & ((merged["e_peak"] - merged["e_inf"]) > peak_delta_threshold)).any())


def _detect_delayed_protection(delayed_df: pd.DataFrame) -> bool:
    if "delayed_protection_detected" not in delayed_df.columns:
        return False
    reliable = delayed_df[_reliable_mask(delayed_df)]
    return bool(not reliable.empty and reliable["delayed_protection_detected"].fillna(False).astype(bool).any())


def _qc_passed(qc: QCReport | None) -> bool:
    return True if qc is None else bool(qc.passed)


# ---------------------------------------------------------------------------
# Evidence flags / table / structured payload
# ---------------------------------------------------------------------------


def build_evidence_flags(
    reversal: dict[str, Any],
    ec50_df: pd.DataFrame | None = None,
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
    delayed_df: pd.DataFrame | None = None,
    endpoint_qc: QCReport | None = None,
    timecourse_qc: QCReport | None = None,
) -> dict[str, bool]:
    flags = {
        "endpoint_qc_passed": _qc_passed(endpoint_qc),
        "timecourse_qc_passed": _qc_passed(timecourse_qc),
        "shift_detected": False,
        "sign_reversal_detected": bool(reversal.get("has_reversal", False))
        and bool(reversal.get("is_reliable", True)),
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


def build_evidence_table(
    reversal: dict[str, Any],
    ec50_df: pd.DataFrame | None = None,
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
    delayed_df: pd.DataFrame | None = None,
    endpoint_qc: QCReport | None = None,
    timecourse_qc: QCReport | None = None,
) -> pd.DataFrame:
    flags = build_evidence_flags(
        reversal, ec50_df, mopt_df, peak_df, final_df, delayed_df, endpoint_qc, timecourse_qc
    )

    shift_strength = _max_strength_from_df(ec50_df)
    reversal_strength = _reversal_strength(reversal)
    interior_strength = _max_strength_from_df(mopt_df)
    moving_strength = (
        _max_strength_from_df(mopt_df)
        if flags["moving_optimum_detected"]
        else ("not_assessable" if mopt_df is None or mopt_df.empty else "none")
    )
    transient_strength = (
        _max_strength_from_df(peak_df) if peak_df is not None and final_df is not None else "not_assessable"
    )
    delayed_strength = _max_strength_from_df(delayed_df)

    rows = [
        {
            "fingerprint": "shift",
            "supported": flags["shift_detected"],
            "evidence_strength": shift_strength,
            "source": "EC50(m)",
            "notes": "Mechanically shifted dose-response family (theory Eq. c12).",
        },
        {
            "fingerprint": "sign_reversal",
            "supported": flags["sign_reversal_detected"],
            "evidence_strength": reversal_strength,
            "source": "mechanical_sign_reversal",
            "notes": "Concentration-dependent sign change in mechanical sensitivity; c_rev = -Delta_lambda/Delta_mu (theory Eq. crev).",
        },
        {
            "fingerprint": "interior_optimum",
            "supported": flags["interior_optimum_detected"],
            "evidence_strength": interior_strength,
            "source": "m*(c)",
            "notes": "Interior mechanical optimum (theory Eq. mstar-opt) -- protected-state class.",
        },
        {
            "fingerprint": "moving_optimum",
            "supported": flags["moving_optimum_detected"],
            "evidence_strength": moving_strength,
            "source": "m*(c)",
            "notes": "Optimal mechanical condition varies with concentration -- protected-state class.",
        },
        {
            "fingerprint": "transient_peak",
            "supported": flags["transient_peak_detected"],
            "evidence_strength": transient_strength,
            "source": "E_peak(c, m)",
            "notes": "Non-terminal transient amplification (theory Eq. p1approx) -- protected-state class.",
        },
        {
            "fingerprint": "delayed_protection",
            "supported": flags["delayed_protection_detected"],
            "evidence_strength": delayed_strength,
            "source": "E_peak - E_inf",
            "notes": "Late-time attenuation consistent with responsive-to-protected branching.",
        },
        {
            "fingerprint": "endpoint_qc",
            "supported": flags["endpoint_qc_passed"],
            "evidence_strength": "strong" if flags["endpoint_qc_passed"] else "none",
            "source": "QC",
            "notes": "Endpoint grid quality gate.",
        },
        {
            "fingerprint": "timecourse_qc",
            "supported": flags["timecourse_qc_passed"],
            "evidence_strength": "strong"
            if flags["timecourse_qc_passed"]
            else ("not_assessable" if timecourse_qc is None else "none"),
            "source": "QC",
            "notes": "Timecourse quality gate.",
        },
    ]
    return pd.DataFrame(rows)


def _fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _ec50_payload(ec50_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if ec50_df is None or ec50_df.empty:
        return []
    cols = [
        "m",
        "ec50",
        "is_reliable",
        "evidence_strength",
        "ec50_ci_low",
        "ec50_ci_high",
        "ec50_bootstrap_reliability",
    ]
    available = [c for c in cols if c in ec50_df.columns]
    return ec50_df[available].to_dict(orient="records")


def _mopt_payload(mopt_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if mopt_df is None or mopt_df.empty:
        return []
    cols = [
        "c",
        "m_opt",
        "m_opt_grid",
        "is_interior",
        "prominence",
        "is_reliable",
        "evidence_strength",
        "m_opt_ci_low",
        "m_opt_ci_high",
        "interior_optimum_fraction",
    ]
    available = [c for c in cols if c in mopt_df.columns]
    return mopt_df[available].to_dict(orient="records")


def _peak_payload(peak_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if peak_df is None or peak_df.empty:
        return []
    cols = [
        "c",
        "m",
        "e_peak",
        "t_peak",
        "has_clear_peak",
        "is_terminal_peak",
        "peak_prominence",
        "is_reliable",
    ]
    available = [c for c in cols if c in peak_df.columns]
    return peak_df[available].to_dict(orient="records")


def _final_payload(final_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if final_df is None or final_df.empty:
        return []
    cols = ["c", "m", "e_inf", "t_inf", "is_reliable"]
    available = [c for c in cols if c in final_df.columns]
    return final_df[available].to_dict(orient="records")


def _delayed_payload(delayed_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if delayed_df is None or delayed_df.empty:
        return []
    cols = [
        "c",
        "m",
        "e_peak",
        "e_inf",
        "attenuation",
        "delayed_protection_detected",
        "attenuation_ci_low",
        "attenuation_ci_high",
        "delayed_bootstrap_reliability",
        "delayed_protection_fraction",
    ]
    available = [c for c in cols if c in delayed_df.columns]
    return delayed_df[available].to_dict(orient="records")


def build_fingerprint_values(
    reversal: dict[str, Any],
    ec50_df: pd.DataFrame | None = None,
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
    delayed_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable payload of fingerprint values for export."""

    c_rev_payload: dict[str, Any] = {
        "estimate": _fnum(reversal.get("c_rev_estimate")),
        "ci_low": _fnum(reversal.get("c_rev_ci_low", float("nan"))),
        "ci_high": _fnum(reversal.get("c_rev_ci_high", float("nan"))),
        "delta_lambda_proxy": _fnum(reversal.get("delta_lambda_proxy")),
        "delta_mu_proxy": _fnum(reversal.get("delta_mu_proxy")),
        "warning": reversal.get("c_rev_warning"),
        "has_reversal": bool(reversal.get("has_reversal", False)),
        "reversal_window_center": _fnum(reversal.get("reversal_window_center")),
    }

    return {
        "EC50_vs_m": _ec50_payload(ec50_df),
        "m_star_vs_c": _mopt_payload(mopt_df),
        "c_rev": c_rev_payload,
        "E_peak_t_peak": _peak_payload(peak_df),
        "E_inf": _final_payload(final_df),
        "delayed_protection": _delayed_payload(delayed_df),
    }


# ---------------------------------------------------------------------------
# Architecture call
# ---------------------------------------------------------------------------


def discriminate_architecture(
    reversal: dict[str, Any],
    ec50_df: pd.DataFrame | None = None,
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
    delayed_df: pd.DataFrame | None = None,
    endpoint_qc: QCReport | None = None,
    timecourse_qc: QCReport | None = None,
) -> DiscriminationResult:
    """Assemble fingerprint evidence into an architecture-class call.

    Returns a :class:`DiscriminationResult` whose ``fingerprint_values``
    attribute carries the structured numerical payload (incl. CIs) so that
    downstream UIs do not need to re-parse the CSVs.
    """

    evidence_flags = build_evidence_flags(
        reversal, ec50_df, mopt_df, peak_df, final_df, delayed_df, endpoint_qc, timecourse_qc
    )
    evidence_table = build_evidence_table(
        reversal, ec50_df, mopt_df, peak_df, final_df, delayed_df, endpoint_qc, timecourse_qc
    )
    strengths = dict(zip(evidence_table["fingerprint"], evidence_table["evidence_strength"]))
    fingerprint_values = build_fingerprint_values(
        reversal, ec50_df, mopt_df, peak_df, final_df, delayed_df
    )
    notes: list[str] = []
    supporting: list[str] = []
    counterpoints: list[str] = []
    warnings: list[str] = []

    if endpoint_qc is not None and endpoint_qc.warnings:
        warnings.extend(endpoint_qc.warnings)
    if timecourse_qc is not None and timecourse_qc.warnings:
        warnings.extend(timecourse_qc.warnings)

    if not evidence_flags["endpoint_qc_passed"]:
        notes.append("Endpoint QC did not pass; architecture call withheld.")
        return DiscriminationResult(
            label="inconclusive",
            evidence_flags=evidence_flags,
            notes=notes,
            confidence="low",
            evidence_strengths=strengths,
            supporting_evidence=supporting,
            counterpoints=counterpoints,
            warnings=warnings,
            fingerprint_values=fingerprint_values,
        )

    for fp in ["shift", "sign_reversal", "interior_optimum", "moving_optimum", "transient_peak", "delayed_protection"]:
        row = evidence_table.loc[evidence_table["fingerprint"] == fp].iloc[0]
        status = bool(row["supported"])
        strength = str(row["evidence_strength"])
        if status:
            supporting.append(f"{fp} evidence {strength}")
        elif strength == "not_assessable":
            counterpoints.append(f"{fp} not assessable")
        else:
            counterpoints.append(f"{fp} evidence {strength}")

    protected_score = (
        2 * evidence_flags["interior_optimum_detected"]
        + evidence_flags["moving_optimum_detected"]
        + evidence_flags["transient_peak_detected"]
        + 2 * evidence_flags["delayed_protection_detected"]
    )

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

    if not evidence_flags["timecourse_qc_passed"]:
        notes.append("Timecourse QC did not pass; dynamic evidence should be interpreted cautiously.")

    return DiscriminationResult(
        label=label,
        evidence_flags=evidence_flags,
        notes=notes,
        confidence=confidence,
        evidence_strengths=strengths,
        supporting_evidence=supporting,
        counterpoints=counterpoints,
        warnings=warnings,
        fingerprint_values=fingerprint_values,
    )
