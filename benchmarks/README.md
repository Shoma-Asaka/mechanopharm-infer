# Benchmarks

This directory contains synthetic benchmark utilities and example outputs for `mechanopharm-infer`.

## Main scripts

- `generate_synthetic.py`
  - generates clean and noisy endpoint/timecourse CSV files for two-state and protected-state synthetic cases.
- `run_clean_benchmark.py`
  - runs a small development-facing benchmark on clean synthetic datasets and writes per-dataset outputs.
- `run_benchmark_suite.py`
  - runs the current benchmark suite through the public benchmark API and writes summary tables, JSON/TXT reports, and a summary figure.

## Outputs

Example outputs are written under `benchmarks/outputs/`.
These are development-facing artifacts intended to illustrate:

- current behavior of the inference pipeline,
- evidence-first architecture calls,
- benchmark summary aggregation,
- known limitations under clean and noisy synthetic cases.

## Notes

These benchmarks are not intended as definitive performance claims.
They are scaffolding for the methods workflow and are expected to evolve as fingerprint evidence rules and diagnostics mature.
