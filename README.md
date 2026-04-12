# mechanopharm-infer

Inference toolkit for mechanopharmacology response-landscape analysis.

## Purpose

`mechanopharm-infer` is a data-facing companion package for architecture-level inference from mechanopharmacology experiments.

The package is designed to:
- load endpoint and timecourse datasets,
- summarize response landscapes over concentration and mechanical condition,
- extract compact response fingerprints,
- perform rule-based discrimination between minimal architecture classes,
- generate lightweight reports and plots for exploratory analysis.

## Relation to `mechanopharm-minimal`

This package is distinct from `mechanopharm-minimal`.

- `mechanopharm-minimal` provides the **minimal reference implementation** for the accompanying theory paper.
- `mechanopharm-infer` builds on that conceptual foundation and focuses on **data-facing inference workflows**.

## Current scope (v0.0.1)

Included:
- endpoint CSV loading,
- timecourse CSV loading,
- replicate summarization,
- response-matrix construction,
- EC50(m) extraction,
- m*(c) extraction,
- sign-reversal diagnostics,
- peak-metric extraction,
- final-response extraction,
- rule-based architecture discrimination,
- text report generation,
- minimal summary plots,
- end-to-end CLI analysis.

Not included:
- full parameter inference,
- Bayesian model comparison,
- assay-specific calibration,
- advanced uncertainty quantification,
- GUI tools,
- system-specific preprocessing pipelines,
- time-dependent mechanics / feedback extensions.

## Installation

```bash
pip install -e .[dev]
```

## Minimal usage

Endpoint-only analysis:

```bash
mechanopharm-infer --endpoint examples/demo_endpoint.csv --out outputs/
```

Endpoint + timecourse analysis:

```bash
mechanopharm-infer \
  --endpoint examples/demo_endpoint.csv \
  --timecourse examples/demo_timecourse.csv \
  --out outputs/
```

## Input format

### Endpoint CSV
Required columns:
- `c`
- `m`
- `response`

Optional columns:
- `replicate`

### Timecourse CSV
Required columns:
- `time`
- `c`
- `m`
- `value`

Optional columns:
- `replicate`

## Output

A standard analysis writes:
- `endpoint_summary.csv`
- `ec50_vs_m.csv`
- `mopt_vs_c.csv`
- `peak_metrics.csv` (if timecourse is provided)
- `final_response.csv` (if timecourse is provided)
- `endpoint_landscape.png`
- `ec50_vs_m.png`
- `mopt_vs_c.png`
- `report.txt`

## Architecture discrimination

The current release performs **rule-based** discrimination among:
- `two_state_supported`
- `protected_state_suggested`
- `inconclusive`

This is intended as a practical first-pass architecture-level workflow, not as a full mechanistic identifiability framework.


## Current limitations

This public release is an early working version (`v0.0.1`).
The current implementation uses rule-based architecture discrimination and should be treated as a first-pass exploratory workflow rather than a final inference framework.
In particular, evidence-flag definitions and discrimination thresholds are still being refined through synthetic benchmark analyses.
