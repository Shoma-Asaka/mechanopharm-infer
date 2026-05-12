# mechanopharm-infer

`mechanopharm-infer` is a lightweight inference toolkit for
mechanopharmacology response-landscape analysis.  It is the data-facing
companion to the theory paper.

The toolkit extracts the practical fingerprint set defined by the theory
(`EC50(m)`, `m*(c)`, `c_rev`, `E_peak`, `t_peak`, `E_inf`) from endpoint and
timecourse datasets and assembles them into a conservative architecture-class
call (two-state vs protected-state).

## Status

Active research prototype at v0.3.0.  This release introduces theory-aligned
column names, a direct estimate of the reversal concentration
`c_rev = -Delta_lambda / Delta_mu`, sub-grid refinement of `m*(c)`,
state-bias-parametrized synthetic generators, a structured
`fingerprint_values` payload in `architecture_call.json`, and `--config`
support in the CLI.  See `GLOSSARY.md` for the canonical vocabulary.

## What the package does

```
endpoint CSV  ─┐
               ├─► schema + assay-metadata ─► QC + diagnostics
timecourse CSV ┘                                │
                                                ▼
                       fingerprint extraction (EC50(m), m*(c), c_rev,
                                                E_peak, t_peak, E_inf,
                                                delayed protection)
                                                │
                                                ▼
                        bootstrap CIs + evidence-strength scoring
                                                │
                                                ▼
                       architecture call (two_state_supported /
                                          protected_state_suggested /
                                          inconclusive)
```

The package is intended as an **architecture-inference layer**, not as a full
parameter-identification framework.

## Theory ↔ code mapping

Every quantity in the table below is reproducible from the code base.

| Theory symbol | Code entry point | Output column / field |
|---------------|------------------|-----------------------|
| `Delta_G(c, m)` | `synthetic.TwoStateParams` | (generator parameters) |
| `p_1*(c, m)` (two-state) | `synthetic.generate_two_state_endpoint` | `response` |
| `p_1*(c, m)` (three-state) | `synthetic.generate_protected_state_endpoint` | `response` |
| `EC50(m)` | `fingerprints.ec50_vs_m` | `ec50_vs_m.csv` |
| `c_rev = -Delta_lambda / Delta_mu` | `fingerprints.mechanical_sign_reversal` | `sign_reversal.csv`, `architecture_call.json :: fingerprint_values.c_rev` |
| `m*(c)` (two-state, quadratic kappa) | `fingerprints.find_mechanical_optima` (parabolic refinement) | `mopt_vs_c.csv :: m_opt` |
| `m*(c)` (three-state) | `fingerprints.find_mechanical_optima` | `mopt_vs_c.csv :: m_opt` |
| `E_peak(c, m)` | `fingerprints.peak_metrics_by_condition` | `peak_metrics.csv :: e_peak` |
| `t_peak(c, m)` | `fingerprints.peak_metrics_by_condition` | `peak_metrics.csv :: t_peak` |
| `E_inf(c, m)` | `fingerprints.endpoint_final_response` | `final_response.csv :: e_inf` |
| delayed protection (`E_peak - E_inf`) | `fingerprints.delayed_protection_metrics` | `delayed_protection.csv :: attenuation` |
| architecture call | `discriminate.discriminate_architecture` | `architecture_call.json :: call` |

For the full glossary (state labels, evidence-strength ranks, etc.), see
[`GLOSSARY.md`](GLOSSARY.md).

## Installation

```bash
pip install -e .[dev]
```

YAML configs require `PyYAML`, which is included in the `dev` extra and is
also available as the `yaml` extra.

## Quick start

### Command line

Run with explicit flags:

```bash
mechanopharm-infer \
  --endpoint examples/demo_endpoint.csv \
  --timecourse examples/demo_timecourse.csv \
  --out outputs_demo
```

Or, recommended for reproducibility, with a config file:

```bash
mechanopharm-infer --config examples/analysis_config.yaml
```

`analysis_config.yaml` lets you pin assay metadata (`response_mode`,
`assay_family`, `readout_level`, ...) and fingerprint thresholds in a single
declarative file.

### Python API

```python
from mechanopharm_infer import (
    AssayMetadata,
    load_endpoint_csv, load_timecourse_csv,
    ec50_vs_m, find_mechanical_optima, mechanical_sign_reversal,
    peak_metrics_by_condition, endpoint_final_response, delayed_protection_metrics,
    discriminate_architecture,
)
from mechanopharm_infer.preprocess import (
    summarize_endpoint, summarize_timecourse,
    check_endpoint_qc, check_timecourse_qc,
    endpoint_to_grid, split_timecourses_by_condition,
)

endpoint = load_endpoint_csv("examples/demo_endpoint.csv")
timecourse = load_timecourse_csv("examples/demo_timecourse.csv")

summary = summarize_endpoint(endpoint)
c_grid, m_grid, response = endpoint_to_grid(summary)

ec50_df = ec50_vs_m(c_grid, m_grid, response)
mopt_df = find_mechanical_optima(c_grid, m_grid, response)
reversal = mechanical_sign_reversal(c_grid, m_grid, response)

tc = split_timecourses_by_condition(summarize_timecourse(timecourse))
peak_df = peak_metrics_by_condition(tc)
final_df = endpoint_final_response(tc)
delayed_df = delayed_protection_metrics(peak_df, final_df)

result = discriminate_architecture(
    reversal=reversal, ec50_df=ec50_df, mopt_df=mopt_df,
    peak_df=peak_df, final_df=final_df, delayed_df=delayed_df,
    endpoint_qc=check_endpoint_qc(summary),
)
print(result.label, result.fingerprint_values["c_rev"]["estimate"])
```

See `examples/quick_start.py` for a runnable version.

### Theory-parametrized synthetic data

```python
from mechanopharm_infer import (
    TwoStateParams, ProtectedStateParams,
    generate_two_state_endpoint, generate_protected_state_endpoint,
)

two_state = generate_two_state_endpoint(
    params=TwoStateParams(delta_g0=3.2, delta_alpha=4.0, delta_lambda=1.6, delta_mu=0.8),
)
protected = generate_protected_state_endpoint(
    params=ProtectedStateParams(a0=-2.0, b0=4.0, lambda0=1.2, lambda1=2.4),
)
```

The generator parameters are precisely the coefficients of Eq. (DeltaG) and
Eqs. (R01log, R12log) in the theory paper.

## Standard outputs

A standard analysis produces:

- `endpoint_summary.csv`, `endpoint_qc.json`
- `ec50_vs_m.csv` (+ `ec50_bootstrap.csv`)
- `mopt_vs_c.csv` (+ `mopt_bootstrap.csv`)
- `sign_reversal.csv` (with the regression-based `c_rev_estimate`)
- `fingerprint_evidence.csv`, `diagnostics.csv`
- `peak_metrics.csv`, `final_response.csv` (if timecourse is provided)
- `delayed_protection.csv` (+ `delayed_protection_bootstrap.csv`)
- `architecture_call.json` (includes the structured `fingerprint_values` payload and `assay_metadata` block)
- `report.txt`
- `endpoint_landscape.png`, `ec50_vs_m.png`, `mopt_vs_c.png`

## Canonical input schema

### Endpoint table

Required columns after standardization:

- `c` : chemical input
- `m` : mechanical input
- `response` : endpoint response

Optional columns: `replicate`, `dataset_id`, `system`, `assay`,
`condition_label`, `unit_concentration`, `unit_mechanics`, `batch`,
`control_flag`.  Common aliases (`concentration`, `mechanics`, `stiffness`,
`effect`) are accepted by the schema standardizer.

### Timecourse table

Required columns after standardization: `time`, `c`, `m`, `value`.  Optional
columns mirror the endpoint case.

## Assay metadata

`AssayMetadata` records assay-facing context so that fingerprints can be
interpreted unambiguously downstream:

| Field | Allowed values |
|-------|----------------|
| `response_mode` | `higher_is_stronger_effect`, `lower_is_stronger_effect` |
| `assay_family` | free form (`generic`, `membrane`, `cell_substrate`, `flow`, `confined_multicellular`, ...) |
| `normalization_mode` | `raw`, `control_subtracted`, `vehicle_normalized`, `min_max`, `within_mechanics_min_max` |
| `baseline_definition` | `none`, `control_flag`, `minimum_per_mechanics`, `global_minimum` |
| `readout_level` | `proximal`, `phenotypic`, `unspecified` |

The `readout_level` field captures the theory's distinction between the
proximal signaling output `S` and the phenotypic outcome `E` and is recorded
in `architecture_call.json` so that report consumers do not confuse them.

## Web app

A Streamlit MVP is available locally via:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run apps/streamlit_app.py
```

And on the public cloud at https://mechanopharm-infer.streamlit.app/.

## Current limitations

- Evidence rules remain conservative and may evolve.
- Sparse mechanics grids limit interior-optimum assessment.
- Short observation windows limit delayed-protection assessment.
- Architecture calls discriminate architecture *class*; they do not identify
  unique microscopic rate laws.
- Anisotropy, spatial transport, and explicit feedback dynamics are out of
  scope for this version.

## Citation and code availability

If you use this repository in academic work, please cite the corresponding
archived software release for the version you used.  Manuscript-specific
releases and DOIs are tracked at
https://doi.org/10.5281/zenodo.20136822.

## Acknowledgements

This project was revised with the assistance of `Claude Code`. All AI-generated contributions have been audited and approved by the developers, who maintain full accountability for this release.

## License

MIT License
