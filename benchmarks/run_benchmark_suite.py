from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechanopharm_infer import SyntheticBenchmarkConfig, write_benchmark_outputs  # noqa: E402


def main() -> None:
    outdir = ROOT / "benchmarks" / "outputs" / "benchmark_suite"
    config = SyntheticBenchmarkConfig(n_boot=50, random_seed=1)
    summary = write_benchmark_outputs(outdir=outdir, config=config)

    print("Benchmark outputs written:")
    print(summary.to_string(index=False))
    print(f"- outdir: {outdir}")


if __name__ == "__main__":
    main()
