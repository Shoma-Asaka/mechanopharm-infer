from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_endpoint_csv, load_timecourse_csv
from .preprocess import (
    summarize_endpoint,
    endpoint_to_grid,
    summarize_timecourse,
    split_timecourses_by_condition,
    check_endpoint_qc,
    check_timecourse_qc,
)
from .fingerprints import (
    ec50_vs_m,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
    endpoint_final_response,
    delayed_protection_metrics,
)
from .discriminate import discriminate_architecture
from .plotting import (
    plot_endpoint_landscape,
    plot_ec50_vs_m,
    plot_mopt_vs_c,
)
from .report import write_text_report


def _write_qc_json(outpath: Path, qc) -> None:
    payload = {
        "kind": qc.kind,
        "passed": qc.passed,
        "warnings": qc.warnings,
        "metrics": qc.metrics,
    }
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze(
    endpoint_path: str,
    outdir: str,
    timecourse_path: str | None = None,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    endpoint_df = load_endpoint_csv(endpoint_path)
    endpoint_summary = summarize_endpoint(endpoint_df)
    endpoint_qc = check_endpoint_qc(endpoint_summary)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)

    ec50_df = ec50_vs_m(c_grid, m_grid, response)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)

    peak_df = None
    final_df = None
    delayed_df = None
    timecourse_qc = None
    if timecourse_path is not None:
        timecourse_df = load_timecourse_csv(timecourse_path)
        timecourse_summary = summarize_timecourse(timecourse_df)
        timecourse_qc = check_timecourse_qc(timecourse_summary)
        timecourse_by_condition = split_timecourses_by_condition(timecourse_summary)

        peak_df = peak_metrics_by_condition(timecourse_by_condition)
        final_df = endpoint_final_response(timecourse_by_condition)
        delayed_df = delayed_protection_metrics(peak_df, final_df)

    result = discriminate_architecture(
        reversal=reversal,
        ec50_df=ec50_df,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
        delayed_df=delayed_df,
        endpoint_qc=endpoint_qc,
        timecourse_qc=timecourse_qc,
    )

    endpoint_summary.to_csv(out / "endpoint_summary.csv", index=False)
    ec50_df.to_csv(out / "ec50_vs_m.csv", index=False)
    mopt_df.to_csv(out / "mopt_vs_c.csv", index=False)
    if peak_df is not None:
        peak_df.to_csv(out / "peak_metrics.csv", index=False)
    if final_df is not None:
        final_df.to_csv(out / "final_response.csv", index=False)
    if delayed_df is not None:
        delayed_df.to_csv(out / "delayed_protection.csv", index=False)

    _write_qc_json(out / "endpoint_qc.json", endpoint_qc)
    if timecourse_qc is not None:
        _write_qc_json(out / "timecourse_qc.json", timecourse_qc)

    plot_endpoint_landscape(
        c_grid,
        m_grid,
        response,
        savepath=str(out / "endpoint_landscape.png"),
    )
    plot_ec50_vs_m(
        ec50_df,
        savepath=str(out / "ec50_vs_m.png"),
    )
    plot_mopt_vs_c(
        mopt_df,
        savepath=str(out / "mopt_vs_c.png"),
    )

    write_text_report(
        out / "report.txt",
        result=result,
        reversal=reversal,
        ec50_df=ec50_df,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
        delayed_df=delayed_df,
        endpoint_qc=endpoint_qc,
        timecourse_qc=timecourse_qc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanopharmacology response-landscape inference toolkit"
    )
    parser.add_argument("--endpoint", required=True, help="Path to endpoint CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--timecourse",
        required=False,
        default=None,
        help="Optional path to timecourse CSV",
    )

    args = parser.parse_args()

    analyze(
        endpoint_path=args.endpoint,
        outdir=args.out,
        timecourse_path=args.timecourse,
    )


if __name__ == "__main__":
    main()
