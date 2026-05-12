# Examples

This directory contains small development-facing examples for `mechanopharm-infer`.

## Included files

| File | Purpose |
|------|---------|
| `demo_endpoint.csv` | Minimal endpoint-only example used by CLI smoke tests. |
| `demo_timecourse.csv` | Minimal timecourse example used by CLI smoke tests. |
| `run_demo.py` | One-liner CLI invocation against the demo data. |
| `quick_start.py` | Python-API quick start: load, fingerprint, discriminate, print structured payload. |
| `analysis_config.yaml` | Reference YAML config consumed by `mechanopharm-infer --config`. |
| `minimal_matrix_template.csv` | Empty endpoint matrix scaffold (4 c x 3 m grid) for new datasets. |
| `minimal_matrix_timecourse_template.csv` | Empty timecourse matrix scaffold (6 time points x 2 conditions). |

These files are intentionally small and are not meant to represent complete
literature datasets.  Their purpose is to make installation, CLI execution,
Python-API usage, and output inspection easier during early development.

## Recommended starting points

1. **Just see it run**: `python examples/run_demo.py` (uses `demo_endpoint.csv` + `demo_timecourse.csv`).
2. **Inspect the Python API**: `python examples/quick_start.py`.
3. **Configure your own experiment**: copy `analysis_config.yaml` and edit
   `endpoint:`, `timecourse:`, `out:`, then run
   `mechanopharm-infer --config <your_config.yaml>`.
4. **Bring your own data**: copy `minimal_matrix_template.csv`
   (and `minimal_matrix_timecourse_template.csv` when relevant),
   fill in the `response` (or `value`) column, and point the config / CLI at the result.
