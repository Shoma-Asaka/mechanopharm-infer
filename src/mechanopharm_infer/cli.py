"""Command-line entry point.

Two invocation styles are supported:

1. Explicit flags (legacy form)::

    mechanopharm-infer --endpoint endpoint.csv --out outputs --timecourse tc.csv

2. Config-driven (recommended for reproducibility)::

    mechanopharm-infer --config analysis.yaml

Where ``analysis.yaml`` looks like::

    endpoint: examples/demo_endpoint.csv
    timecourse: examples/demo_timecourse.csv
    out: outputs_demo
    n_boot: 500
    random_seed: 1
    assay_metadata:
      response_mode: higher_is_stronger_effect
      readout_level: phenotypic
      assay_family: cell_substrate
    thresholds:
      ec50_min_dynamic_range: 0.05
      mopt_prominence_threshold: 0.03
      delayed_attenuation_threshold: 0.05

JSON is also accepted (when the file ends with ``.json``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bootstrap import (
    bootstrap_c_rev,
    bootstrap_delayed_protection,
    bootstrap_ec50_vs_m,
    bootstrap_mopt,
)
from .diagnostics import combine_diagnostics, diagnostics_messages, endpoint_diagnostics, timecourse_diagnostics
from .discriminate import build_evidence_table, discriminate_architecture
from .fingerprints import (
    delayed_protection_metrics,
    ec50_vs_m,
    endpoint_final_response,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
)
from .io import load_endpoint_csv, load_timecourse_csv
from .plotting import (
    plot_dose_response_family,
    plot_ec50_vs_m,
    plot_endpoint_landscape,
    plot_evidence_summary,
    plot_mopt_vs_c,
    plot_timecourse_panel,
)
from .preprocess import (
    check_endpoint_qc,
    check_timecourse_qc,
    endpoint_to_grid,
    prepare_endpoint_data,
    prepare_timecourse_data,
    split_timecourses_by_condition,
    summarize_endpoint,
    summarize_timecourse,
)
from .report import write_text_report
from .types import AssayMetadata, coerce_assay_metadata


def _write_qc_json(outpath: Path, qc) -> None:
    outpath.write_text(
        json.dumps({"kind": qc.kind, "passed": qc.passed, "warnings": qc.warnings, "metrics": qc.metrics}, indent=2),
        encoding="utf-8",
    )


def _write_architecture_json(outpath: Path, result, assay_metadata: AssayMetadata) -> None:
    payload = {
        "call": result.label,
        "confidence": result.confidence,
        "supporting_evidence": result.supporting_evidence or [],
        "counterpoints": result.counterpoints or [],
        "warnings": result.warnings or [],
        "notes": result.notes or [],
        "evidence_flags": result.evidence_flags,
        "evidence_strengths": result.evidence_strengths or {},
        "fingerprint_values": result.fingerprint_values or {},
        "assay_metadata": assay_metadata.to_dict(),
    }
    outpath.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with `pip install pyyaml` or supply a JSON config."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config file must define a top-level mapping.")
    return data


def analyze(
    endpoint_path: str,
    outdir: str,
    timecourse_path: str | None = None,
    n_boot: int = 200,
    random_seed: int = 0,
    *,
    assay_metadata: AssayMetadata | dict[str, Any] | None = None,
    ec50_min_dynamic_range: float = 0.05,
    mopt_prominence_threshold: float = 0.03,
    delayed_attenuation_threshold: float = 0.05,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    md = coerce_assay_metadata(assay_metadata)

    endpoint_df = load_endpoint_csv(endpoint_path)
    if assay_metadata is not None and md.normalization_mode != "raw":
        endpoint_df = prepare_endpoint_data(endpoint_df, md)
    endpoint_summary = summarize_endpoint(endpoint_df)
    endpoint_qc = check_endpoint_qc(endpoint_summary)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    ec50_df = ec50_vs_m(c_grid, m_grid, response, min_dynamic_range=ec50_min_dynamic_range)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response, prominence_threshold=mopt_prominence_threshold)
    ec50_boot = bootstrap_ec50_vs_m(endpoint_df, n_boot=n_boot, random_seed=random_seed, min_dynamic_range=ec50_min_dynamic_range)
    mopt_boot = bootstrap_mopt(endpoint_df, n_boot=n_boot, random_seed=random_seed, prominence_threshold=mopt_prominence_threshold)
    ec50_df = ec50_df.merge(ec50_boot, on="m", how="left")
    mopt_df = mopt_df.merge(mopt_boot, on="c", how="left")
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)
    c_rev_boot = bootstrap_c_rev(endpoint_df, n_boot=n_boot, random_seed=random_seed)
    reversal.update(c_rev_boot)
    peak_df = None
    final_df = None
    delayed_df = None
    timecourse_qc = None
    timecourse_summary = None
    delayed_boot = None
    if timecourse_path is not None:
        timecourse_df = load_timecourse_csv(timecourse_path)
        if assay_metadata is not None and md.normalization_mode != "raw":
            timecourse_df = prepare_timecourse_data(timecourse_df, md)
        timecourse_summary = summarize_timecourse(timecourse_df)
        timecourse_qc = check_timecourse_qc(timecourse_summary)
        timecourse_by_condition = split_timecourses_by_condition(timecourse_summary)
        peak_df = peak_metrics_by_condition(timecourse_by_condition)
        final_df = endpoint_final_response(timecourse_by_condition)
        delayed_df = delayed_protection_metrics(peak_df, final_df, attenuation_threshold=delayed_attenuation_threshold)
        delayed_boot = bootstrap_delayed_protection(
            timecourse_df, n_boot=n_boot, random_seed=random_seed, attenuation_threshold=delayed_attenuation_threshold
        )
        delayed_df = delayed_df.merge(delayed_boot, on=["c", "m"], how="left")
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
    endpoint_diag = endpoint_diagnostics(endpoint_summary, ec50_df=ec50_df, mopt_df=mopt_df)
    timecourse_diag = timecourse_diagnostics(
        timecourse_summary if timecourse_path is not None else None,
        peak_df=peak_df,
        delayed_df=delayed_df,
    )
    diagnostics_df = combine_diagnostics(endpoint_diag, timecourse_diag)
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
    # Write the reversal payload as a small CSV-shaped record for inspection.
    import pandas as pd  # local import to avoid global pandas usage in helpers

    pd.DataFrame([reversal]).to_csv(out / "sign_reversal.csv", index=False)
    if peak_df is not None:
        peak_df.to_csv(out / "peak_metrics.csv", index=False)
    if final_df is not None:
        final_df.to_csv(out / "final_response.csv", index=False)
    if delayed_df is not None:
        delayed_df.to_csv(out / "delayed_protection.csv", index=False)
        if delayed_boot is not None:
            delayed_boot.to_csv(out / "delayed_protection_bootstrap.csv", index=False)
    _write_qc_json(out / "endpoint_qc.json", endpoint_qc)
    if timecourse_qc is not None:
        _write_qc_json(out / "timecourse_qc.json", timecourse_qc)
    _write_architecture_json(out / "architecture_call.json", result, md)
    plot_endpoint_landscape(c_grid, m_grid, response, savepath=str(out / "endpoint_landscape.png"))
    plot_ec50_vs_m(ec50_df, savepath=str(out / "ec50_vs_m.png"))
    plot_mopt_vs_c(mopt_df, savepath=str(out / "mopt_vs_c.png"))
    plot_dose_response_family(endpoint_summary, savepath=str(out / "dose_response_family.png"))
    plot_evidence_summary(evidence_df, savepath=str(out / "evidence_summary.png"))
    if timecourse_summary is not None:
        plot_timecourse_panel(timecourse_summary, savepath=str(out / "timecourse_panel.png"))
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
        diagnostics_df=diagnostics_df,
        assay_metadata=md,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "mechanopharm-infer: response-landscape inference for mechanopharmacology. "
            "Extracts EC50(m), m*(c), c_rev, E_peak, t_peak, E_inf, and makes an "
            "architecture-class call (two-state vs protected-state)."
        )
    )
    parser.add_argument("--config", default=None, help="Path to a YAML/JSON config file (overridden by explicit flags)")
    parser.add_argument("--endpoint", default=None, help="Path to endpoint CSV")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--timecourse", default=None, help="Optional path to timecourse CSV")
    parser.add_argument("--n-boot", default=None, type=int, help="Number of bootstrap resamples")
    parser.add_argument("--random-seed", default=None, type=int, help="Random seed for bootstrap")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config is not None:
        config = _load_config(args.config)

    endpoint = args.endpoint or config.get("endpoint")
    outdir = args.out or config.get("out")
    timecourse = args.timecourse or config.get("timecourse")
    n_boot = args.n_boot if args.n_boot is not None else int(config.get("n_boot", 200))
    random_seed = args.random_seed if args.random_seed is not None else int(config.get("random_seed", 0))
    thresholds = config.get("thresholds", {}) or {}
    assay_metadata = config.get("assay_metadata")

    if endpoint is None:
        parser.error("--endpoint (or `endpoint:` in the config) is required")
    if outdir is None:
        parser.error("--out (or `out:` in the config) is required")

    analyze(
        endpoint_path=endpoint,
        outdir=outdir,
        timecourse_path=timecourse,
        n_boot=n_boot,
        random_seed=random_seed,
        assay_metadata=assay_metadata,
        ec50_min_dynamic_range=float(thresholds.get("ec50_min_dynamic_range", 0.05)),
        mopt_prominence_threshold=float(thresholds.get("mopt_prominence_threshold", 0.03)),
        delayed_attenuation_threshold=float(thresholds.get("delayed_attenuation_threshold", 0.05)),
    )


if __name__ == "__main__":
    main()
