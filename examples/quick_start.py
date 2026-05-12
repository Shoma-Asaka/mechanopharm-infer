"""Quick-start example for mechanopharm-infer.

Run from the repository root:

    python examples/quick_start.py

This script demonstrates the typical workflow on bundled demo data:

1. Load endpoint + timecourse CSVs.
2. Extract fingerprints ``EC50(m)``, ``m*(c)``, ``c_rev``, ``E_peak``,
   ``t_peak``, ``E_inf``.
3. Run architecture discrimination.
4. Inspect the structured ``fingerprint_values`` payload that is also written
   to ``architecture_call.json`` by the CLI.
"""

from __future__ import annotations

from pathlib import Path

from mechanopharm_infer import (
    AssayMetadata,
    discriminate_architecture,
    ec50_vs_m,
    find_mechanical_optima,
    load_endpoint_csv,
    load_timecourse_csv,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
    endpoint_final_response,
    delayed_protection_metrics,
)
from mechanopharm_infer.preprocess import (
    check_endpoint_qc,
    check_timecourse_qc,
    endpoint_to_grid,
    split_timecourses_by_condition,
    summarize_endpoint,
    summarize_timecourse,
)


def main() -> None:
    examples = Path(__file__).resolve().parent
    endpoint_df = load_endpoint_csv(examples / "demo_endpoint.csv")
    timecourse_df = load_timecourse_csv(examples / "demo_timecourse.csv")

    metadata = AssayMetadata(
        response_mode="higher_is_stronger_effect",
        assay_family="cell_substrate",
        readout_level="phenotypic",
    )
    print("Assay metadata:", metadata.to_dict())

    endpoint_summary = summarize_endpoint(endpoint_df)
    endpoint_qc = check_endpoint_qc(endpoint_summary)
    print(f"Endpoint QC passed: {endpoint_qc.passed}")

    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    ec50_df = ec50_vs_m(c_grid, m_grid, response)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)
    print(
        "Sign reversal:",
        f"has_reversal={reversal['has_reversal']}, c_rev≈{reversal['c_rev_estimate']:.3f}",
    )

    timecourse_summary = summarize_timecourse(timecourse_df)
    timecourse_qc = check_timecourse_qc(timecourse_summary)
    by_condition = split_timecourses_by_condition(timecourse_summary)
    peak_df = peak_metrics_by_condition(by_condition)
    final_df = endpoint_final_response(by_condition)
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
    print(f"Architecture call: {result.label} (confidence: {result.confidence})")
    print("Evidence flags:")
    for k, v in result.evidence_flags.items():
        print(f"  {k}: {v}")
    if result.fingerprint_values is not None:
        c_rev = result.fingerprint_values["c_rev"]
        print(f"  c_rev payload: estimate={c_rev['estimate']}, has_reversal={c_rev['has_reversal']}")


if __name__ == "__main__":
    main()
