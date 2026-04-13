from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechanopharm_infer import SyntheticBenchmarkConfig, run_synthetic_benchmark, write_benchmark_outputs  # noqa: E402


def main() -> None:
    outdir = ROOT / "benchmarks" / "outputs" / "benchmark_suite"
    config = SyntheticBenchmarkConfig(n_boot=50, random_seed=1)
    summary = run_synthetic_benchmark(config=config)
    paths = write_benchmark_outputs(summary, outdir=outdir, prefix="benchmark_suite")
    print("Benchmark outputs written:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
