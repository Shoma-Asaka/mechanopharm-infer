from pathlib import Path
import pandas as pd

from mechanopharm_infer.cli import analyze


def test_import_smoke():
    import mechanopharm_infer
    assert mechanopharm_infer is not None


def test_example_files_exist():
    assert Path("/mnt/data/examples/demo_endpoint.csv").exists()
    assert Path("/mnt/data/examples/demo_timecourse.csv").exists()


def test_analyze_end_to_end(tmp_path: Path):
    endpoint_path = tmp_path / "endpoint.csv"
    timecourse_path = tmp_path / "timecourse.csv"
    outdir = tmp_path / "outputs"
    pd.DataFrame({"c": [0.1, 0.5, 1.0, 0.1, 0.5, 1.0], "m": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5], "response": [0.2, 0.5, 0.8, 0.3, 0.6, 0.7]}).to_csv(endpoint_path, index=False)
    pd.DataFrame({"time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], "c": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0], "m": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "value": [0.0, 0.8, 0.4, 0.0, 0.7, 0.6]}).to_csv(timecourse_path, index=False)
    analyze(endpoint_path=str(endpoint_path), timecourse_path=str(timecourse_path), outdir=str(outdir))
    assert (outdir / "endpoint_summary.csv").exists()
    assert (outdir / "ec50_vs_m.csv").exists()
    assert (outdir / "mopt_vs_c.csv").exists()
    assert (outdir / "peak_metrics.csv").exists()
    assert (outdir / "final_response.csv").exists()
    assert (outdir / "delayed_protection.csv").exists()
    assert (outdir / "endpoint_qc.json").exists()
    assert (outdir / "timecourse_qc.json").exists()
    assert (outdir / "endpoint_landscape.png").exists()
    assert (outdir / "ec50_vs_m.png").exists()
    assert (outdir / "mopt_vs_c.png").exists()
    assert (outdir / "report.txt").exists()
