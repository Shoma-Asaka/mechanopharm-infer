# mechanopharm-infer

Inference toolkit for mechanopharmacology response-landscape analysis.

## Status

`mechanopharm-infer` is currently an early research prototype for response-landscape analysis and architecture-class inference in mechanopharmacology. The package is intended to support method development, synthetic benchmarking, and early data-facing workflows. Its API, evidence rules, and decision logic may evolve substantially as the toolkit matures.

## Purpose

`mechanopharm-infer` is a data-facing companion package for architecture-level inference from mechanopharmacology experiments.

The package is designed to:
- load endpoint and timecourse datasets,
- perform basic QC checks on endpoint and timecourse inputs,
- summarize response landscapes over concentration and mechanical condition,
- extract compact response fingerprints,
- attach reliability and warning flags to key fingerprints,
- perform conservative rule-based discrimination between minimal architecture classes,
- generate lightweight reports and plots for exploratory analysis.

## Relation to `mechanopharm-minimal`

This package is distinct from `mechanopharm-minimal`.

- `mechanopharm-minimal` provides the **minimal reference implementation** for the accompanying theory paper.
- `mechanopharm-infer` builds on that conceptual foundation and focuses on **data-facing inference workflows**.

## Current scope (v0.0.2)

Included:
- endpoint CSV loading,
- timecourse CSV loading,
- endpoint and timecourse QC summaries,
- replicate summarization,
- response-matrix construction,
- reliability-aware `EC50(m)` extraction,
- reliability-aware `m*(c)` extraction,
- sign-reversal diagnostics,
- peak-metric extraction,
- final-response extraction,
- delayed-protection metrics,
- conservative rule-based architecture discrimination,
- `inconclusive` handling for weak or low-information cases,
- text report generation,
- minimal summary plots,
- end-to-end CLI analysis,
- example datasets,
- basic tests.

Not included:
- full parameter inference,
- Bayesian model comparison,
- assay-specific calibration,
- advanced uncertainty quantification,
- experimental-design recommendation tools,
- GUI tools,
- system-specific preprocessing pipelines,
- time-dependent mechanics / feedback extensions.

## Installation

For development use:

```bash
pip install -e .[dev]
```

## Minimal usage

Endpoint-only analysis:

```bash
mechanopharm-infer \
  --endpoint examples/demo_endpoint.csv \
  --out outputs/
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
- `delayed_protection.csv` (if timecourse is provided)
- `endpoint_qc.json`
- `timecourse_qc.json` (if timecourse is provided)
- `endpoint_landscape.png`
- `ec50_vs_m.png`
- `mopt_vs_c.png`
- `report.txt`

## Architecture discrimination

The current release performs conservative **rule-based** discrimination among:
- `two_state_supported`
- `protected_state_suggested`
- `inconclusive`

The toolkit is intended as a practical first-pass architecture-level workflow, not as a full mechanistic identifiability framework.

## Current limitations

This public release is an early working version (`v0.0.2`).

The current implementation should be treated as a QC-aware exploratory workflow rather than a final inference framework. In particular:
- evidence-flag definitions and discrimination thresholds are still being refined,
- some fingerprints may be marked unreliable for sparse or weakly informative data,
- architecture calls are intentionally conservative,
- future releases may change API details and decision logic.

## Development notes

This release is intended as a pre-methods-paper prototype baseline. A future `v0.1.0` should correspond to a more stable methods-paper-ready public baseline with broader validation, stronger uncertainty handling, and more mature documentation.
