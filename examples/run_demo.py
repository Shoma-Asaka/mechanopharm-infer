from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "examples" / "demo_outputs"


def run_demo(with_timecourse: bool = True) -> None:
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    cmd = [
        sys.executable,
        "-m",
        "mechanopharm_infer.cli",
        "--endpoint",
        str(ROOT / "examples" / "demo_endpoint.csv"),
        "--out",
        str(OUTDIR),
        "--n-boot",
        "50",
        "--random-seed",
        "1",
    ]
    if with_timecourse:
        cmd.extend(["--timecourse", str(ROOT / "examples" / "demo_timecourse.csv")])
    subprocess.run(cmd, check=True)
    print(f"Wrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    run_demo(with_timecourse=True)
