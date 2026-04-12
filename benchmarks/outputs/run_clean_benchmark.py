from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechanopharm_infer.cli import analyze  # noqa: E402
from mechanopharm_infer.io import load_endpoint_csv, load_timecourse_csv  # noqa: E402
from mechanopharm_infer.preprocess import (  # noqa: E402
    summarize_endpoint,
    endpoint_to_grid,
    summarize_timecourse,
    split_timecourses_by_condition,
)
from mechanopharm_infer.fingerprints import (  # noqa: E402
    ec50_vs_m,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
    endpoint_final_response,
)
from mechanopharm_infer.discriminate import discriminate_architecture  # noqa: E402

from generate_synthetic import generate_clean_benchmark_bundle  # noqa: E402


OUTDIR = ROOT / "benchmarks" / "outputs" / "clean_benchmark"
OUTDIR.mkdir(parents=True, exist_ok=True)


def infer_endpoint_only(endpoint_path: Path, dataset_name: str) -> dict[str, object]:
    endpoint_df = load_endpoint_csv(endpoint_path)
    endpoint_summary = summarize_endpoint(endpoint_df)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    ec50_df = ec50_vs_m(c_grid, m_grid, response)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)
    result = discriminate_architecture(reversal=reversal, mopt_df=mopt_df)

    dataset_out = OUTDIR / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)
    endpoint_summary.to_csv(dataset_out / "endpoint_summary.csv", index=False)
    ec50_df.to_csv(dataset_out / "ec50_vs_m.csv", index=False)
    mopt_df.to_csv(dataset_out / "mopt_vs_c.csv", index=False)

    analyze(endpoint_path=str(endpoint_path), outdir=str(dataset_out))

    return {
        "dataset": dataset_name,
        "truth": "two_state" if "two_state" in dataset_name else "protected_state",
        "used_timecourse": False,
        "label": result.label,
        "confidence": result.confidence,
        "sign_reversal_detected": result.evidence_flags["sign_reversal_detected"],
        "interior_optimum_detected": result.evidence_flags["interior_optimum_detected"],
        "moving_optimum_detected": result.evidence_flags["moving_optimum_detected"],
        "transient_peak_detected": result.evidence_flags["transient_peak_detected"],
        "delayed_protection_detected": result.evidence_flags["delayed_protection_detected"],
    }


def infer_endpoint_plus_timecourse(endpoint_path: Path, timecourse_path: Path, dataset_name: str) -> dict[str, object]:
    endpoint_df = load_endpoint_csv(endpoint_path)
    endpoint_summary = summarize_endpoint(endpoint_df)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)

    timecourse_df = load_timecourse_csv(timecourse_path)
    timecourse_summary = summarize_timecourse(timecourse_df)
    timecourse_by_condition = split_timecourses_by_condition(timecourse_summary)
    peak_df = peak_metrics_by_condition(timecourse_by_condition)
    final_df = endpoint_final_response(timecourse_by_condition)

    result = discriminate_architecture(
        reversal=reversal,
        mopt_df=mopt_df,
        peak_df=peak_df,
        final_df=final_df,
    )

    dataset_out = OUTDIR / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)
    analyze(endpoint_path=str(endpoint_path), timecourse_path=str(timecourse_path), outdir=str(dataset_out))

    return {
        "dataset": dataset_name,
        "truth": "protected_state",
        "used_timecourse": True,
        "label": result.label,
        "confidence": result.confidence,
        "sign_reversal_detected": result.evidence_flags["sign_reversal_detected"],
        "interior_optimum_detected": result.evidence_flags["interior_optimum_detected"],
        "moving_optimum_detected": result.evidence_flags["moving_optimum_detected"],
        "transient_peak_detected": result.evidence_flags["transient_peak_detected"],
        "delayed_protection_detected": result.evidence_flags["delayed_protection_detected"],
    }


def main() -> None:
    bundle = generate_clean_benchmark_bundle(ROOT / "benchmarks" / "outputs")

    rows: list[dict[str, object]] = []
    rows.append(infer_endpoint_only(bundle["two_state_endpoint"], "two_state_endpoint_clean"))
    rows.append(infer_endpoint_only(bundle["protected_state_endpoint"], "protected_state_endpoint_clean_endpoint_only"))
    rows.append(
        infer_endpoint_plus_timecourse(
            bundle["protected_state_endpoint"],
            bundle["protected_state_timecourse"],
            "protected_state_clean_with_timecourse",
        )
    )

    summary = pd.DataFrame(rows)
    summary_path = OUTDIR / "clean_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    txt_path = OUTDIR / "clean_benchmark_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("clean benchmark summary\n")
        f.write("=======================\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved text summary to: {txt_path}")


if __name__ == "__main__":
    main()
