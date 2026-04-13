from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .discriminate import build_evidence_table
from .types import DiscriminationResult, QCReport


def _write_dataframe_block(f, title: str, df: pd.DataFrame | None) -> None:
    f.write(f"{title}\n")
    f.write("-" * len(title) + "\n")
    if df is None or df.empty:
        f.write("(none)\n\n")
        return
    f.write(df.to_string(index=False))
    f.write("\n\n")


def _write_qc_block(f, title: str, qc: QCReport | None) -> None:
    f.write(f"{title}\n")
    f.write("-" * len(title) + "\n")
    if qc is None:
        f.write("(none)\n\n")
        return
    f.write(f"Passed: {qc.passed}\n")
    for k, v in qc.metrics.items():
        f.write(f"- {k}: {v}\n")
    if qc.warnings:
        f.write("Warnings:\n")
        for w in qc.warnings:
            f.write(f"- {w}\n")
    f.write("\n")


def write_text_report(
    outpath: str | Path,
    result: DiscriminationResult,
    reversal: dict[str, float | bool | str | None],
    ec50_df: pd.DataFrame | None = None,
    mopt_df: pd.DataFrame | None = None,
    peak_df: pd.DataFrame | None = None,
    final_df: pd.DataFrame | None = None,
    delayed_df: pd.DataFrame | None = None,
    endpoint_qc: QCReport | None = None,
    timecourse_qc: QCReport | None = None,
    diagnostics_df: pd.DataFrame | None = None,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    evidence_df = build_evidence_table(
        reversal=reversal,
        ec50_df=ec50_df,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
        delayed_df=delayed_df,
        endpoint_qc=endpoint_qc,
        timecourse_qc=timecourse_qc,
    )
    with outpath.open("w", encoding="utf-8") as f:
        f.write("mechanopharm-infer report\n=========================\n\n")
        _write_qc_block(f, "Endpoint QC", endpoint_qc)
        _write_qc_block(f, "Timecourse QC", timecourse_qc)
        f.write("Architecture discrimination\n---------------------------\n")
        f.write(f"Label: {result.label}\nConfidence: {result.confidence}\n")
        if result.supporting_evidence:
            f.write("Supporting evidence:\n")
            for item in result.supporting_evidence:
                f.write(f"- {item}\n")
        if result.counterpoints:
            f.write("Counterpoints / limitations:\n")
            for item in result.counterpoints:
                f.write(f"- {item}\n")
        if result.warnings:
            f.write("Warnings:\n")
            for item in result.warnings:
                f.write(f"- {item}\n")
        if result.notes:
            f.write("Notes:\n")
            for note in result.notes:
                f.write(f"- {note}\n")
        f.write("\nEvidence flags\n--------------\n")
        for k, v in result.evidence_flags.items():
            f.write(f"- {k}: {v}\n")
        f.write("\nMechanical sign reversal\n------------------------\n")
        for k, v in reversal.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
        _write_dataframe_block(f, "Fingerprint evidence", evidence_df)
        _write_dataframe_block(f, "Diagnostics", diagnostics_df)
        _write_dataframe_block(f, "EC50(m)", ec50_df)
        _write_dataframe_block(f, "m*(c)", mopt_df)
        _write_dataframe_block(f, "Peak metrics", peak_df)
        _write_dataframe_block(f, "Final response", final_df)
        _write_dataframe_block(f, "Delayed protection metrics", delayed_df)


def write_benchmark_report(
    outdir: str | Path,
    benchmark_df: pd.DataFrame,
    config: dict[str, object] | None = None,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(outdir / "benchmark_summary.csv", index=False)
    matched = int(benchmark_df["matched_expected"].sum()) if not benchmark_df.empty else 0
    total = int(len(benchmark_df))
    payload = {
        "n_cases": total,
        "n_matched": matched,
        "match_rate": (matched / total) if total else None,
        "config": config or {},
    }
    (outdir / "benchmark_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (outdir / "benchmark_report.txt").open("w", encoding="utf-8") as f:
        f.write("Synthetic benchmark report\n==========================\n\n")
        f.write(f"Cases: {total}\nMatched expected: {matched}\n")
        if total:
            f.write(f"Match rate: {matched / total:.3f}\n")
        if config:
            f.write("\nConfig\n------\n")
            for k, v in config.items():
                f.write(f"- {k}: {v}\n")
        f.write("\nCase summary\n------------\n")
        if benchmark_df.empty:
            f.write("(none)\n")
        else:
            f.write(benchmark_df.to_string(index=False))
            f.write("\n")
