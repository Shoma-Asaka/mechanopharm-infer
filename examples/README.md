# Examples

This directory contains small development-facing examples for `mechanopharm-infer`.

## Included files

- `demo_endpoint.csv`
  - minimal endpoint-only example used by the CLI smoke tests.
- `demo_timecourse.csv`
  - minimal timecourse example used by the CLI smoke tests.
- `jem_like_endpoint.csv`
  - tiny adapter-flavored endpoint table using named stiffness levels (`soft`, `stiff`).
- `novak_like_endpoint.csv`
  - tiny compression-style endpoint table using named compression conditions.
- `kalli_like_endpoint.csv`
  - tiny endpoint table mimicking pressure/compression-oriented naming.
- `run_demo.py`
  - simple script that runs the CLI analysis pipeline on the demo files.

These files are intentionally small and are not meant to represent complete literature datasets.
Their purpose is to make installation, adapter behavior, and output inspection easier during early development.
