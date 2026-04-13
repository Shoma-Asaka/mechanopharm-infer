# mechanopharm-infer

`mechanopharm-infer` is a lightweight inference toolkit for mechanopharmacology response-landscape analysis.

It is designed as an actively developed research prototype for **data-facing, architecture-level inference** from endpoint and timecourse datasets. The current focus is conservative first-pass analysis: preprocessing, fingerprint extraction, uncertainty summaries, evidence assembly, and cautious architecture-class discrimination.

## Status

This repository is an **early public prototype**.

It is intended for methods development around mechanopharmacology inference workflows, not for full mechanistic identification or production-grade model fitting. API details, evidence rules, and output formats may still evolve across pre-1.0 releases.

## What the package does

The current toolkit supports the following workflow:

1. standardize endpoint and timecourse tables into a canonical schema,
2. apply assay-aware preprocessing and response-direction normalization,
3. perform QC and dataset diagnostics,
4. extract response fingerprints such as `EC50(m)`, `m*(c)`, sign-reversal diagnostics, and timecourse-derived peak/final metrics,
5. attach uncertainty summaries and warning metadata,
6. assemble fingerprint evidence,
7. make a conservative architecture-class call:
   - `two_state_supported`
   - `protected_state_suggested`
   - `inconclusive`

The package is intended as an **architecture-inference layer**, not as a full parameter-identification framework.

## Current scope

Included in the current prototype:

- canonical schema handling for endpoint and timecourse CSV files,
- assay metadata handling and response-direction normalization,
- endpoint and timecourse QC,
- endpoint and timecourse diagnostics,
- reliability-aware `EC50(m)` extraction,
- reliability-aware `m*(c)` extraction,
- sign-reversal diagnostics,
- timecourse peak/final/delayed-protection fingerprints,
- bootstrap summaries for selected fingerprints,
- evidence-first architecture discrimination,
- lightweight plotting,
- literature-dataset adapters for the current three target papers,
- synthetic dataset generators and benchmark utilities,
- benchmark summary plots and reports,
- CLI and Python API entry points,
- examples, benchmark scripts, and tests.

Not included:

- full parameter inference,
- Bayesian model comparison,
- assay-specific calibration models,
- high-dimensional mechanistic fitting,
- GUI or web interface,
- time-dependent mechanics / feedback model classes.

## Installation

For development use:

```bash
pip install -e .[dev]
```

## Quick start

### Endpoint-only CLI run

```bash
mechanopharm-infer \
  --endpoint examples/demo_endpoint.csv \
  --out outputs_demo
```

### Endpoint + timecourse CLI run

```bash
mechanopharm-infer \
  --endpoint examples/demo_endpoint.csv \
  --timecourse examples/demo_timecourse.csv \
  --out outputs_demo_tc
```

### Synthetic benchmark utilities

```bash
python benchmarks/generate_synthetic.py
python benchmarks/run_clean_benchmark.py
python benchmarks/run_benchmark_suite.py
```

## Standard outputs

A standard analysis produces some or all of the following, depending on whether timecourse input is provided:

- `endpoint_summary.csv`
- `ec50_vs_m.csv`
- `mopt_vs_c.csv`
- `fingerprint_evidence.csv`
- `diagnostics.csv`
- `ec50_bootstrap.csv`
- `mopt_bootstrap.csv`
- `peak_metrics.csv`
- `final_response.csv`
- `delayed_protection.csv`
- `delayed_protection_bootstrap.csv`
- `endpoint_qc.json`
- `timecourse_qc.json`
- `architecture_call.json`
- `report.txt`
- `endpoint_landscape.png`
- `ec50_vs_m.png`
- `mopt_vs_c.png`

Benchmark utilities additionally write:

- `benchmark_summary.csv`
- `benchmark_report.json`
- `benchmark_report.txt`
- `benchmark_summary.png`

## Canonical input schema

### Endpoint table

Required columns after standardization:

- `c` : chemical input
- `m` : mechanical input
- `response` : endpoint response

Optional columns:

- `replicate`
- `dataset_id`
- `system`
- `assay`
- `condition_label`
- `unit_concentration`
- `unit_mechanics`
- `batch`
- `control_flag`

Common aliases such as `concentration`, `mechanics`, and `effect` are accepted by the schema standardizer.

### Timecourse table

Required columns after standardization:

- `time`
- `c`
- `m`
- `value`

Optional columns mirror the endpoint case where relevant.

## Assay metadata

The preprocessing layer can use assay metadata to orient and normalize responses.

Key metadata fields are:

- `response_mode`
  - `higher_is_stronger_effect`
  - `lower_is_stronger_effect`
- `normalization_mode`
  - `raw`
  - `control_subtracted`
  - `vehicle_normalized`
  - `min_max`
  - `within_mechanics_min_max`
- `baseline_definition`

This is useful when mixing apoptosis-like, viability-like, and survival-like readouts in a shared workflow.

## Literature adapters

The current release includes thin adapters for the three papers currently targeted in the methods workflow:

- JEM TNBC stiffness dataset
- Novak ovarian compression dataset
- Kalli pancreatic compression/autophagy dataset

These adapters are intentionally lightweight. Their role is to map paper-specific column names and condition labels into the common schema before passing the data into the shared inference engine.

## Python API examples

### Canonical loading and preprocessing

```python
from mechanopharm_infer import load_endpoint_csv, AssayMetadata, prepare_endpoint_data

endpoint = load_endpoint_csv("examples/demo_endpoint.csv")
meta = AssayMetadata(response_mode="higher_is_stronger_effect")
prepared = prepare_endpoint_data(endpoint, assay_metadata=meta)
```

### Literature adapter

```python
from mechanopharm_infer import prepare_jem_endpoint

endpoint = prepare_jem_endpoint("examples/jem_like_endpoint.csv")
```

### Synthetic benchmark generation

```python
from mechanopharm_infer import (
    generate_two_state_endpoint,
    SyntheticBenchmarkConfig,
    run_synthetic_benchmark,
)

endpoint = generate_two_state_endpoint()
config = SyntheticBenchmarkConfig(n_boot=50, random_seed=1)
summary = run_synthetic_benchmark(config=config)
```

## Examples and benchmark scripts

- `examples/README.md` describes the demo files and adapter-flavored example tables.
- `examples/run_demo.py` runs a local endpoint or endpoint+timecourse demonstration.
- `benchmarks/generate_synthetic.py` creates clean and noisy synthetic benchmark inputs.
- `benchmarks/run_clean_benchmark.py` runs a small development-oriented synthetic benchmark.
- `benchmarks/run_benchmark_suite.py` writes a benchmark summary table, JSON/TXT reports, and a summary plot.

## Relation to `mechanopharm-minimal`

This repository is distinct from `mechanopharm-minimal`.

- `mechanopharm-minimal` provides a minimal reference implementation for the theory layer.
- `mechanopharm-infer` focuses on data-facing fingerprint extraction, evidence assembly, uncertainty summaries, and cautious architecture-class inference.

## Current limitations

This remains an early prototype release.

Important limitations:

- evidence rules are still evolving,
- sparse mechanics grids can limit interior-optimum assessment,
- short time windows can limit delayed-protection assessment,
- architecture calls are intentionally conservative,
- future releases may change API details, evidence weighting, and file outputs.

## Citation and code availability

If you use this repository in academic work, please cite the corresponding archived software release for the version you used. A manuscript-specific release and DOI should be used whenever available.

## License

MIT License
