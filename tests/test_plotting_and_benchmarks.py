import pandas as pd

from mechanopharm_infer.plotting import (
    plot_benchmark_summary,
    plot_dose_response_family,
    plot_evidence_summary,
    plot_timecourse_panel,
)
from mechanopharm_infer.synthetic import SyntheticBenchmarkConfig, write_benchmark_outputs


def test_plot_helpers_return_figures(tmp_path):
    endpoint_summary = pd.DataFrame(
        {
            "c": [0.1, 1.0, 0.1, 1.0],
            "m": [0.0, 0.0, 1.0, 1.0],
            "response_mean": [0.2, 0.6, 0.1, 0.4],
        }
    )
    timecourse_summary = pd.DataFrame(
        {
            "c": [0.1, 0.1, 1.0, 1.0],
            "m": [0.0, 0.0, 1.0, 1.0],
            "t": [0.0, 1.0, 0.0, 1.0],
            "response_mean": [0.2, 0.5, 0.1, 0.3],
        }
    )
    evidence_df = pd.DataFrame(
        {
            "fingerprint": ["shift", "interior_optimum"],
            "evidence_strength": ["strong", "weak"],
        }
    )
    benchmark_df = pd.DataFrame(
        {
            "benchmark_case": ["two_state", "protected_state"],
            "matched_expected": [True, False],
        }
    )

    fig, _ = plot_dose_response_family(endpoint_summary, savepath=str(tmp_path / "dose_family.png"))
    assert fig is not None
    fig, _ = plot_timecourse_panel(timecourse_summary, savepath=str(tmp_path / "timecourse_panel.png"))
    assert fig is not None
    fig, _ = plot_evidence_summary(evidence_df, savepath=str(tmp_path / "evidence_summary.png"))
    assert fig is not None
    fig, _ = plot_benchmark_summary(benchmark_df, savepath=str(tmp_path / "benchmark_summary.png"))
    assert fig is not None


def test_write_benchmark_outputs(tmp_path):
    cfg = SyntheticBenchmarkConfig(n_boot=10, n_replicates=3, endpoint_noise_sd=0.01, timecourse_noise_sd=0.01)
    df = write_benchmark_outputs(tmp_path, cfg)
    assert not df.empty
    assert (tmp_path / "benchmark_summary.csv").exists()
    assert (tmp_path / "benchmark_report.json").exists()
    assert (tmp_path / "benchmark_report.txt").exists()
    assert (tmp_path / "benchmark_summary.png").exists()
