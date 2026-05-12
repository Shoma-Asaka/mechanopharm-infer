from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow local execution from a source checkout without requiring installation first.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechanopharm_infer.cli import analyze


APP_TITLE = "mechanopharm-infer"
DEFAULT_ENDPOINT = ROOT / "examples" / "demo_endpoint.csv"
DEFAULT_TIMECOURSE = ROOT / "examples" / "demo_timecourse.csv"


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


def _save_uploaded_file(uploaded_file, path: Path) -> None:
    path.write_bytes(uploaded_file.getvalue())


def _zip_directory(directory: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.where(pd.notnull(df), "")

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: round(x, 6) if x != "" else "")

    return df


def _show_table(path: Path, title: str, *, height: int | None = None) -> None:
    st.markdown(f"### {title}")
    df = _load_table(path)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


st.title(APP_TITLE)
st.caption("Mechanopharmacology response-landscape inference toolkit — Streamlit MVP")

with st.sidebar:
    st.header("Input")
    use_demo = st.checkbox("Use bundled demo data", value=True)

    endpoint_file = None
    timecourse_file = None
    if not use_demo:
        endpoint_file = st.file_uploader("Endpoint CSV", type=["csv"])
        timecourse_file = st.file_uploader("Timecourse CSV optional", type=["csv"])
    else:
        st.info("Demo endpoint and timecourse CSV files from examples/ will be used.")

    st.header("Analysis settings")
    n_boot = st.slider("Bootstrap resamples", min_value=10, max_value=500, value=100, step=10)
    random_seed = st.number_input("Random seed", min_value=0, value=0, step=1)

    run = st.button("Run analysis", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Citation")
    st.sidebar.markdown("If you use this software, please cite the archived release:")
    st.sidebar.markdown(
        "[mechanopharm-infer v0.3.0](https://doi.org/10.5281/zenodo.19780165)"
    )

st.markdown(
    """
This app runs the same core analysis as the command-line `mechanopharm-infer` workflow.
Upload an endpoint CSV, optionally upload a timecourse CSV, then inspect the architecture call,
QC outputs, response-landscape figures, and downloadable result files.
"""
)

if run:
    if not use_demo and endpoint_file is None:
        st.error("Endpoint CSV is required.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        endpoint_path = tmpdir / "endpoint.csv"
        timecourse_path = None
        outdir = tmpdir / "outputs"

        if use_demo:
            endpoint_path.write_bytes(DEFAULT_ENDPOINT.read_bytes())
            if DEFAULT_TIMECOURSE.exists():
                timecourse_path = tmpdir / "timecourse.csv"
                timecourse_path.write_bytes(DEFAULT_TIMECOURSE.read_bytes())
        else:
            _save_uploaded_file(endpoint_file, endpoint_path)
            if timecourse_file is not None:
                timecourse_path = tmpdir / "timecourse.csv"
                _save_uploaded_file(timecourse_file, timecourse_path)

        try:
            with st.spinner("Running mechanopharm-infer analysis..."):
                analyze(
                    endpoint_path=str(endpoint_path),
                    timecourse_path=str(timecourse_path) if timecourse_path else None,
                    outdir=str(outdir),
                    n_boot=int(n_boot),
                    random_seed=int(random_seed),
                )
        except Exception as exc:  # pragma: no cover - UI-level error handling
            st.error("Analysis failed. Please check that the uploaded CSV files follow the expected schema.")
            st.exception(exc)
            st.stop()

        st.success("Analysis complete.")

        arch_path = outdir / "architecture_call.json"
        report_path = outdir / "report.txt"

        if arch_path.exists():
            arch = _read_json(arch_path)

            call_value = arch.get("call", "NA")
            confidence_value = arch.get("confidence", "NA")
            warnings_list = arch.get("warnings", []) or []

            left, mid, right = st.columns([5, 2, 1])

            with left:
                st.markdown("**Architecture call**")
                st.code(str(call_value), language=None)

            with mid:
                st.metric("Confidence", confidence_value)

            with right:
                st.metric("Warnings", len(warnings_list))

            if warnings_list:
                with st.expander("Warnings details", expanded=False):
                    for i, warning in enumerate(warnings_list, start=1):
                        st.markdown(f"{i}. {warning}")

            with st.expander("Architecture call JSON", expanded=False):
                st.json(arch)

        if report_path.exists():
            with st.expander("Raw text report", expanded=False):
                st.code(report_path.read_text(encoding="utf-8"), language=None)

        st.subheader("Structured results")

        structured_tables = [
            ("fingerprint_evidence.csv", "Fingerprint evidence", 320),
            ("diagnostics.csv", "Diagnostics", 360),
            ("ec50_vs_m.csv", "EC50(m)", 220),
            ("mopt_vs_c.csv", "m*(c)", 220),
            ("peak_metrics.csv", "Peak metrics", 220),
            ("final_response.csv", "Final response", 220),
            ("delayed_protection.csv", "Delayed protection metrics", 220),
        ]

        for filename, title, height in structured_tables:
            path = outdir / filename
            if path.exists():
                _show_table(path, title, height=height)

        st.subheader("Figures")
        fig_cols = st.columns(3)
        for col, filename in zip(
            fig_cols,
            ["endpoint_landscape.png", "ec50_vs_m.png", "mopt_vs_c.png"],
        ):
            fig_path = outdir / filename
            if fig_path.exists():
                col.image(str(fig_path), caption=filename, use_container_width=True)

        with st.expander("All tables", expanded=False):
            for filename, title in [
                ("endpoint_summary.csv", "Endpoint summary"),
                ("ec50_vs_m.csv", "EC50 vs mechanics"),
                ("mopt_vs_c.csv", "Mechanical optimum vs concentration"),
                ("fingerprint_evidence.csv", "Fingerprint evidence"),
                ("diagnostics.csv", "Diagnostics"),
                ("peak_metrics.csv", "Peak metrics"),
                ("final_response.csv", "Final response"),
                ("delayed_protection.csv", "Delayed protection"),
            ]:
                path = outdir / filename
                if path.exists():
                    _show_table(path, title, height=280)

        st.subheader("Downloads")
        zip_path = tmpdir / "mechanopharm_outputs.zip"
        _zip_directory(outdir, zip_path)
        st.download_button(
            "Download all outputs as ZIP",
            data=zip_path.read_bytes(),
            file_name="mechanopharm_outputs.zip",
            mime="application/zip",
        )

        with st.expander("Download individual output files", expanded=False):
            for file_path in sorted(outdir.iterdir()):
                if file_path.is_file():
                    st.download_button(
                        label=f"Download {file_path.name}",
                        data=file_path.read_bytes(),
                        file_name=file_path.name,
                        mime="application/octet-stream",
                        key=f"download-{file_path.name}",
                    )
else:
    st.info("Choose input data in the sidebar and click **Run analysis**.")
