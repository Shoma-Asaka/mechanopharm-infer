from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_endpoint_csv, load_timecourse_csv
from .preprocess import summarize_endpoint, endpoint_to_grid, summarize_timecourse, split_timecourses_by_condition, check_endpoint_qc, check_timecourse_qc
from .fingerprints import ec50_vs_m, find_mechanical_optima, mechanical_sign_reversal, peak_metrics_by_condition, endpoint_final_response, delayed_protection_metrics
from .discriminate import discriminate_architecture, build_evidence_table
from .bootstrap import bootstrap_ec50_vs_m, bootstrap_mopt, bootstrap_delayed_protection
from .diagnostics import combine_diagnostics, diagnostics_messages, endpoint_diagnostics, timecourse_diagnostics
from .plotting import plot_endpoint_landscape, plot_ec50_vs_m, plot_mopt_vs_c
from .report import write_text_report


def _write_qc_json(outpath: Path, qc) -> None:
    outpath.write_text(json.dumps({"kind": qc.kind, "passed": qc.passed, "warnings": qc.warnings, "metrics": qc.metrics}, indent=2), encoding="utf-8")


def _write_architecture_json(outpath: Path, result) -> None:
    payload = {
        "call": result.label,
        "confidence": result.confidence,
        "supporting_evidence": result.supporting_evidence or [],
        "counterpoints": result.counterpoints or [],
        "warnings": result.warnings or [],
        "notes": result.notes or [],
        "evidence_flags": result.evidence_flags,
        "evidence_strengths": result.evidence_strengths or {},
    }
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze(endpoint_path: str, outdir: str, timecourse_path: str | None = None, n_boot: int = 200, random_seed: int = 0) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    endpoint_df = load_endpoint_csv(endpoint_path)
    endpoint_summary = summarize_endpoint(endpoint_df)
    endpoint_qc = check_endpoint_qc(endpoint_summary)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    ec50_df = ec50_vs_m(c_grid, m_grid, response)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    ec50_boot = bootstrap_ec50_vs_m(endpoint_df, n_boot=n_boot, random_seed=random_seed)
    mopt_boot = bootstrap_mopt(endpoint_df, n_boot=n_boot, random_seed=random_seed)
    ec50_df = ec50_df.merge(ec50_boot, on="m", how="left")
    mopt_df = mopt_df.merge(mopt_boot, on="c", how="left")
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)
    peak_df = None
    final_df = None
    delayed_df = None
    timecourse_qc = None
    timecourse_summary = None
    if timecourse_path is not None:
        timecourse_df = load_timecourse_csv(timecourse_path)
        timecourse_summary = summarize_timecourse(timecourse_df)
        timecourse_qc = check_timecourse_qc(timecourse_summary)
        timecourse_by_condition = split_timecourses_by_condition(timecourse_summary)
        peak_df = peak_metrics_by_condition(timecourse_by_condition)
        final_df = endpoint_final_response(timecourse_by_condition)
        delayed_df = delayed_protection_metrics(peak_df, final_df)
        delayed_boot = bootstrap_delayed_protection(timecourse_df, n_boot=n_boot, random_seed=random_seed)
        delayed_df = delayed_df.merge(delayed_boot, on=["c", "m"], how="left")
    evidence_df = build_evidence_table(reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc, timecourse_qc=timecourse_qc)
    endpoint_diag = endpoint_diagnostics(endpoint_summary, ec50_df=ec50_df, mopt_df=mopt_df)
    timecourse_diag = timecourse_diagnostics(timecourse_summary if timecourse_path is not None else None, peak_df=peak_df, delayed_df=delayed_df)
    diagnostics_df = combine_diagnostics(endpoint_diag, timecourse_diag)
    result = discriminate_architecture(reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc, timecourse_qc=timecourse_qc)
    extra_warnings = diagnostics_messages(diagnostics_df, min_severity="medium")
    if extra_warnings:
        existing = list(result.warnings or [])
        result.warnings = existing + [w for w in extra_warnings if w not in existing]
    endpoint_summary.to_csv(out / "endpoint_summary.csv", index=False)
    ec50_df.to_csv(out / "ec50_vs_m.csv", index=False)
    mopt_df.to_csv(out / "mopt_vs_c.csv", index=False)
    evidence_df.to_csv(out / "fingerprint_evidence.csv", index=False)
    diagnostics_df.to_csv(out / "diagnostics.csv", index=False)
    ec50_boot.to_csv(out / "ec50_bootstrap.csv", index=False)
    mopt_boot.to_csv(out / "mopt_bootstrap.csv", index=False)
    if peak_df is not None:
        peak_df.to_csv(out / "peak_metrics.csv", index=False)
    if final_df is not None:
        final_df.to_csv(out / "final_response.csv", index=False)
    if delayed_df is not None:
        delayed_df.to_csv(out / "delayed_protection.csv", index=False)
        delayed_boot.to_csv(out / "delayed_protection_bootstrap.csv", index=False)
    _write_qc_json(out / "endpoint_qc.json", endpoint_qc)
    if timecourse_qc is not None:
        _write_qc_json(out / "timecourse_qc.json", timecourse_qc)
    _write_architecture_json(out / "architecture_call.json", result)
    plot_endpoint_landscape(c_grid, m_grid, response, savepath=str(out / "endpoint_landscape.png"))
    plot_ec50_vs_m(ec50_df, savepath=str(out / "ec50_vs_m.png"))
    plot_mopt_vs_c(mopt_df, savepath=str(out / "mopt_vs_c.png"))
    write_text_report(out / "report.txt", result=result, reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc, timecourse_qc=timecourse_qc, diagnostics_df=diagnostics_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanopharmacology response-landscape inference toolkit")
    parser.add_argument("--endpoint", required=True, help="Path to endpoint CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--timecourse", required=False, default=None, help="Optional path to timecourse CSV")
    parser.add_argument("--n-boot", required=False, default=200, type=int, help="Number of bootstrap resamples")
    parser.add_argument("--random-seed", required=False, default=0, type=int, help="Random seed for bootstrap")
    args = parser.parse_args()
    analyze(endpoint_path=args.endpoint, outdir=args.out, timecourse_path=args.timecourse, n_boot=args.n_boot, random_seed=args.random_seed)


if __name__ == "__main__":
    main()
