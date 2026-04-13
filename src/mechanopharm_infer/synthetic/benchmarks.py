from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd

from ..bootstrap import bootstrap_delayed_protection, bootstrap_ec50_vs_m, bootstrap_mopt
from ..diagnostics import combine_diagnostics, endpoint_diagnostics, timecourse_diagnostics
from ..discriminate import build_evidence_table, discriminate_architecture
from ..fingerprints import delayed_protection_metrics, ec50_vs_m, endpoint_final_response, find_mechanical_optima, mechanical_sign_reversal, peak_metrics_by_condition
from ..plotting import plot_benchmark_summary
from ..preprocess import check_endpoint_qc, check_timecourse_qc, endpoint_to_grid, split_timecourses_by_condition, summarize_endpoint, summarize_timecourse
from ..report import write_benchmark_report
from .generators import (
    generate_protected_state_endpoint,
    generate_protected_state_timecourse,
    generate_two_state_endpoint,
    generate_two_state_timecourse,
)


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    n_boot: int = 100
    random_seed: int = 0
    endpoint_noise_sd: float = 0.02
    timecourse_noise_sd: float = 0.015
    n_replicates: int = 4


def analyze_synthetic_dataset(endpoint_df: pd.DataFrame, timecourse_df: pd.DataFrame | None = None, *, n_boot: int = 100, random_seed: int = 0) -> dict[str, object]:
    endpoint_summary = summarize_endpoint(endpoint_df)
    endpoint_qc = check_endpoint_qc(endpoint_summary)
    c_grid, m_grid, response = endpoint_to_grid(endpoint_summary)
    ec50_df = ec50_vs_m(c_grid, m_grid, response)
    mopt_df = find_mechanical_optima(c_grid, m_grid, response)
    reversal = mechanical_sign_reversal(c_grid, m_grid, response)
    ec50_boot = bootstrap_ec50_vs_m(endpoint_df, n_boot=n_boot, random_seed=random_seed)
    mopt_boot = bootstrap_mopt(endpoint_df, n_boot=n_boot, random_seed=random_seed)
    ec50_df = ec50_df.merge(ec50_boot, on='m', how='left')
    mopt_df = mopt_df.merge(mopt_boot, on='c', how='left')

    peak_df = None
    final_df = None
    delayed_df = None
    timecourse_qc = None
    diagnostics_df = None
    if timecourse_df is not None:
        timecourse_summary = summarize_timecourse(timecourse_df)
        timecourse_qc = check_timecourse_qc(timecourse_summary)
        tc_by_cond = split_timecourses_by_condition(timecourse_summary)
        peak_df = peak_metrics_by_condition(tc_by_cond)
        final_df = endpoint_final_response(tc_by_cond)
        delayed_df = delayed_protection_metrics(peak_df, final_df)
        delayed_boot = bootstrap_delayed_protection(timecourse_df, n_boot=n_boot, random_seed=random_seed)
        delayed_df = delayed_df.merge(delayed_boot, on=['c', 'm'], how='left')
        diagnostics_df = combine_diagnostics(
            endpoint_diagnostics(endpoint_summary, ec50_df=ec50_df, mopt_df=mopt_df),
            timecourse_diagnostics(timecourse_summary, peak_df=peak_df, delayed_df=delayed_df),
        )
    else:
        diagnostics_df = combine_diagnostics(endpoint_diagnostics(endpoint_summary, ec50_df=ec50_df, mopt_df=mopt_df), None)

    evidence_df = build_evidence_table(reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc, timecourse_qc=timecourse_qc)
    result = discriminate_architecture(reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df, peak_df=peak_df, final_df=final_df, delayed_df=delayed_df, endpoint_qc=endpoint_qc, timecourse_qc=timecourse_qc)
    return {
        'endpoint_summary': endpoint_summary,
        'ec50_df': ec50_df,
        'mopt_df': mopt_df,
        'reversal': reversal,
        'peak_df': peak_df,
        'final_df': final_df,
        'delayed_df': delayed_df,
        'evidence_df': evidence_df,
        'diagnostics_df': diagnostics_df,
        'result': asdict(result),
    }


def run_synthetic_benchmark(config: SyntheticBenchmarkConfig | None = None) -> pd.DataFrame:
    cfg = config or SyntheticBenchmarkConfig()
    cases = []
    for label, endpoint_fun, timecourse_fun, expected in [
        ('two_state', generate_two_state_endpoint, generate_two_state_timecourse, 'two_state_supported'),
        ('protected_state', generate_protected_state_endpoint, generate_protected_state_timecourse, 'protected_state_suggested'),
    ]:
        endpoint_df = endpoint_fun(n_replicates=cfg.n_replicates, noise_sd=cfg.endpoint_noise_sd, random_seed=cfg.random_seed)
        timecourse_df = timecourse_fun(n_replicates=cfg.n_replicates, noise_sd=cfg.timecourse_noise_sd, random_seed=cfg.random_seed + 1)
        out = analyze_synthetic_dataset(endpoint_df, timecourse_df, n_boot=cfg.n_boot, random_seed=cfg.random_seed)
        result = out['result']
        cases.append({
            'benchmark_case': label,
            'expected_label': expected,
            'predicted_label': result['label'],
            'confidence': result['confidence'],
            'matched_expected': result['label'] == expected,
        })
    return pd.DataFrame(cases)


def write_benchmark_outputs(outdir: str | Path, config: SyntheticBenchmarkConfig | None = None) -> pd.DataFrame:
    cfg = config or SyntheticBenchmarkConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    benchmark_df = run_synthetic_benchmark(cfg)
    write_benchmark_report(outdir, benchmark_df, config=asdict(cfg))
    plot_benchmark_summary(benchmark_df, savepath=str(outdir / "benchmark_summary.png"))
    return benchmark_df
