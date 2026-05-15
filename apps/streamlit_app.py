from __future__ import annotations

import inspect
import json
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make sure the in-repo `src/` wins over any pip-installed copy.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)
    for _mod_name in list(sys.modules):
        if _mod_name == "mechanopharm_infer" or _mod_name.startswith("mechanopharm_infer."):
            del sys.modules[_mod_name]

from mechanopharm_infer.cli import analyze

_ANALYZE_PARAMS = set(inspect.signature(analyze).parameters)
_ANALYZE_SUPPORTS_ASSAY_METADATA = "assay_metadata" in _ANALYZE_PARAMS
_ANALYZE_SUPPORTS_THRESHOLDS = {
    name: name in _ANALYZE_PARAMS
    for name in (
        "ec50_min_dynamic_range",
        "mopt_prominence_threshold",
        "delayed_attenuation_threshold",
    )
}

APP_TITLE = "mechanopharm-infer"
DEFAULT_ENDPOINT = ROOT / "examples" / "demo_endpoint.csv"
DEFAULT_TIMECOURSE = ROOT / "examples" / "demo_timecourse.csv"

ENDPOINT_REQUIRED = ("c", "m", "response")
TIMECOURSE_REQUIRED = ("time", "c", "m", "value")

RESPONSE_MODES = ["higher_is_stronger_effect", "lower_is_stronger_effect"]
NORMALIZATION_MODES = [
    "raw",
    "vehicle_normalized",
    "control_subtracted",
    "min_max",
    "within_mechanics_min_max",
]
BASELINE_DEFINITIONS = [
    "none",
    "control_flag",
    "minimum_per_mechanics",
    "global_minimum",
]
READOUT_LEVELS = ["unspecified", "proximal", "phenotypic"]

DEFAULT_ADVANCED: dict[str, Any] = {
    "n_boot": 100,
    "random_seed": 0,
    "response_mode": "higher_is_stronger_effect",
    "readout_level": "unspecified",
    "assay_family": "generic",
    "normalization_mode": "raw",
    "baseline_definition": "none",
    "ec50_min_dynamic_range": 0.05,
    "mopt_prominence_threshold": 0.03,
    "delayed_attenuation_threshold": 0.05,
}

# Architecture call display config
_CALL_CONFIG: dict[str, str] = {
    "two_state_supported": "Two-State Architecture Supported",
    "protected_state_suggested": "Protected-State Architecture Suggested",
    "inconclusive": "Inconclusive",
}


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _save_uploaded_file(uploaded_file, path: Path) -> None:
    path.write_bytes(uploaded_file.getvalue())


def _zip_directory(directory: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_safely(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return None


def _default_endpoint_editor_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="float") for col in ENDPOINT_REQUIRED})


def _default_timecourse_editor_df() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="float") for col in TIMECOURSE_REQUIRED})


def _parse_config_bytes(name: str, raw: bytes) -> dict[str, Any]:
    suffix = Path(name).suffix.lower()
    text = raw.decode("utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Upload a JSON config instead, or add PyYAML to requirements.txt."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config file must define a top-level mapping.")
    return data


def _validate_dataframe(
    df: pd.DataFrame | None, required: tuple[str, ...], *, label: str
) -> list[str]:
    errors: list[str] = []
    if df is None or df.empty:
        errors.append(f"{label}: no rows provided.")
        return errors
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{label}: required columns missing -> {missing}")
        return errors
    for col in required:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().all():
            errors.append(f"{label}: column '{col}' has no valid numeric values.")
        elif coerced.isna().any():
            n_bad = int(coerced.isna().sum())
            errors.append(
                f"{label}: column '{col}' has {n_bad} cell(s) that cannot be parsed as numbers."
            )
    return errors


def _apply_config_to_state(config: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    n_boot = config.get("n_boot")
    if n_boot is not None:
        st.session_state["n_boot"] = int(n_boot)
        messages.append(f"n_boot = {int(n_boot)}")
    random_seed = config.get("random_seed")
    if random_seed is not None:
        st.session_state["random_seed"] = int(random_seed)
        messages.append(f"random_seed = {int(random_seed)}")
    md = config.get("assay_metadata") or {}
    for key in ("response_mode", "readout_level", "assay_family", "normalization_mode", "baseline_definition"):
        if key in md and md[key] is not None:
            st.session_state[key] = md[key]
            messages.append(f"assay_metadata.{key} = {md[key]}")
    thresholds = config.get("thresholds") or {}
    for key in ("ec50_min_dynamic_range", "mopt_prominence_threshold", "delayed_attenuation_threshold"):
        if key in thresholds and thresholds[key] is not None:
            st.session_state[key] = float(thresholds[key])
            messages.append(f"thresholds.{key} = {float(thresholds[key])}")
    if config.get("endpoint") or config.get("timecourse"):
        messages.append(
            "(note) 'endpoint' and 'timecourse' paths in the config are ignored here; "
            "provide data through the Input section instead."
        )
    return messages


def _serialize_config(state: dict) -> tuple[str, str, str]:
    """Return (content, filename, mime_type) for the current settings."""
    data: dict[str, Any] = {
        "n_boot": int(state.get("n_boot", 100)),
        "random_seed": int(state.get("random_seed", 0)),
        "assay_metadata": {
            "response_mode": state.get("response_mode", "higher_is_stronger_effect"),
            "readout_level": state.get("readout_level", "unspecified"),
            "assay_family": state.get("assay_family", "generic") or "generic",
            "normalization_mode": state.get("normalization_mode", "raw"),
            "baseline_definition": state.get("baseline_definition", "none"),
        },
        "thresholds": {
            "ec50_min_dynamic_range": float(state.get("ec50_min_dynamic_range", 0.05)),
            "mopt_prominence_threshold": float(state.get("mopt_prominence_threshold", 0.03)),
            "delayed_attenuation_threshold": float(state.get("delayed_attenuation_threshold", 0.05)),
        },
    }
    try:
        import yaml  # type: ignore
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        return content, "analysis_config.yaml", "text/yaml"
    except ImportError:
        content = json.dumps(data, indent=2)
        return content, "analysis_config.json", "application/json"


# ---------------------------------------------------------------------------
# Result collection and display
# ---------------------------------------------------------------------------


def _collect_results(outdir: Path, tmpdir: Path) -> dict[str, Any]:
    """Read all analysis outputs into session-state-safe data before tmpdir is cleaned up."""
    results: dict[str, Any] = {}

    arch_path = outdir / "architecture_call.json"
    if arch_path.exists():
        results["arch"] = json.loads(arch_path.read_bytes())

    report_path = outdir / "report.txt"
    if report_path.exists():
        results["report_text"] = report_path.read_text(encoding="utf-8")

    table_files = [
        "endpoint_summary.csv",
        "fingerprint_evidence.csv",
        "diagnostics.csv",
        "sign_reversal.csv",
        "ec50_vs_m.csv",
        "mopt_vs_c.csv",
        "peak_metrics.csv",
        "final_response.csv",
        "delayed_protection.csv",
    ]
    results["tables"] = {}
    for fname in table_files:
        p = outdir / fname
        if p.exists():
            results["tables"][fname] = pd.read_csv(p)

    figure_files = [
        "endpoint_landscape.png",
        "ec50_vs_m.png",
        "mopt_vs_c.png",
        "dose_response_family.png",
        "evidence_summary.png",
        "timecourse_panel.png",
    ]
    results["figures"] = {}
    for fname in figure_files:
        p = outdir / fname
        if p.exists():
            results["figures"][fname] = p.read_bytes()

    results["all_files"] = {}
    for p in sorted(outdir.iterdir()):
        if p.is_file():
            results["all_files"][p.name] = p.read_bytes()

    zip_path = tmpdir / "mechanopharm_outputs.zip"
    _zip_directory(outdir, zip_path)
    results["zip_bytes"] = zip_path.read_bytes()

    return results


def _show_table_from_df(df: pd.DataFrame, title: str, *, height: int | None = None) -> None:
    st.markdown(f"### {title}")
    display_df = df.copy().where(pd.notnull(df), "")
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            display_df[col] = display_df[col].map(lambda x: round(x, 6) if x != "" else "")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=height)


def _display_architecture_call(arch: dict) -> None:
    call_value = arch.get("call", "NA")
    confidence = arch.get("confidence", "NA")
    warnings_list = arch.get("warnings", []) or []
    supporting = arch.get("supporting_evidence", []) or []
    counterpoints = arch.get("counterpoints", []) or []
    notes = arch.get("notes", []) or []

    label = _CALL_CONFIG.get(call_value, call_value)

    st.markdown(f"#### {label}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", str(confidence))
    m2.metric("Supporting evidence", len(supporting))
    m3.metric("Warnings", len(warnings_list))

    if supporting or counterpoints:
        ev_left, ev_right = st.columns(2)
        with ev_left:
            if supporting:
                st.markdown("**Supporting evidence**")
                for s in supporting:
                    st.markdown(f"- ✅ {s}")
        with ev_right:
            if counterpoints:
                st.markdown("**Counterpoints / not assessed**")
                for cp in counterpoints:
                    st.markdown(f"- ➖ {cp}")

    if notes:
        with st.expander("Interpretation notes", expanded=False):
            for n in notes:
                st.markdown(f"- {n}")

    if warnings_list:
        with st.expander(f"Warnings ({len(warnings_list)})", expanded=False):
            for i, w in enumerate(warnings_list, 1):
                st.warning(f"{i}. {w}")

    with st.expander("Architecture call JSON", expanded=False):
        st.json(arch)


def _display_results(results: dict[str, Any]) -> None:
    st.success("Analysis complete.")

    if "arch" in results:
        st.subheader("Architecture call")
        _display_architecture_call(results["arch"])

    if "report_text" in results:
        with st.expander("Raw text report", expanded=False):
            st.code(results["report_text"], language=None)

    # --- Structured tables ---
    st.subheader("Structured results")
    table_config = [
        ("endpoint_summary.csv", "Endpoint summary", 280),
        ("fingerprint_evidence.csv", "Fingerprint evidence", 320),
        ("diagnostics.csv", "Diagnostics", 360),
        ("sign_reversal.csv", "Sign reversal (c_rev)", 180),
        ("ec50_vs_m.csv", "EC50(m)", 220),
        ("mopt_vs_c.csv", "m*(c)", 220),
        ("peak_metrics.csv", "Peak metrics", 220),
        ("final_response.csv", "Final response", 220),
        ("delayed_protection.csv", "Delayed protection metrics", 220),
    ]
    for fname, title, height in table_config:
        if fname in results["tables"]:
            _show_table_from_df(results["tables"][fname], title, height=height)

    # --- Figures ---
    st.subheader("Figures")

    primary_figs = ["endpoint_landscape.png", "ec50_vs_m.png", "mopt_vs_c.png"]
    primary_bytes = [results["figures"][f] for f in primary_figs if f in results["figures"]]
    if primary_bytes:
        cols = st.columns(len(primary_bytes))
        for col, fname, img in zip(cols, [f for f in primary_figs if f in results["figures"]], primary_bytes):
            col.image(img, caption=fname, use_container_width=True)

    secondary_figs = ["dose_response_family.png", "evidence_summary.png"]
    secondary_present = [f for f in secondary_figs if f in results["figures"]]
    if secondary_present:
        cols2 = st.columns(len(secondary_present))
        for col, fname in zip(cols2, secondary_present):
            col.image(results["figures"][fname], caption=fname, use_container_width=True)

    if "timecourse_panel.png" in results["figures"]:
        tc_col, _ = st.columns(2)
        tc_col.image(results["figures"]["timecourse_panel.png"], caption="timecourse_panel.png", use_container_width=True)

    # --- Downloads ---
    st.subheader("Downloads")
    st.download_button(
        "Download all outputs as ZIP",
        data=results["zip_bytes"],
        file_name="mechanopharm_outputs.zip",
        mime="application/zip",
    )
    with st.expander("Download individual output files", expanded=False):
        for fname, fbytes in sorted(results["all_files"].items()):
            st.download_button(
                label=f"Download {fname}",
                data=fbytes,
                file_name=fname,
                mime="application/octet-stream",
                key=f"dl-{fname}",
            )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key, value in DEFAULT_ADVANCED.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title(APP_TITLE)
st.caption("Mechanopharmacology response-landscape inference toolkit — Streamlit MVP")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Input")
    input_mode = st.radio(
        "Input source",
        ["Use bundled demo data", "Direct input (table editor)", "Upload CSV"],
        index=0,
    )

    endpoint_file = None
    timecourse_file = None
    endpoint_edit_df: pd.DataFrame | None = None
    timecourse_edit_df: pd.DataFrame | None = None
    include_timecourse_direct = False

    if input_mode == "Upload CSV":
        endpoint_file = st.file_uploader("Endpoint CSV", type=["csv"])
        timecourse_file = st.file_uploader("Timecourse CSV (optional)", type=["csv"])
    elif input_mode == "Direct input (table editor)":
        st.caption(
            "Click cells to edit values. Use the '+' at the bottom-right to add rows; "
            "empty cells are dropped."
        )
        if st.session_state.get("_input_mode_prev") != "Direct input (table editor)":
            st.session_state["endpoint_editor_df"] = _default_endpoint_editor_df()
            st.session_state.pop("timecourse_editor_df", None)
        st.session_state["_input_mode_prev"] = "Direct input (table editor)"
        endpoint_edit_df = st.data_editor(
            st.session_state["endpoint_editor_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="endpoint_editor",
            column_config={
                "c": st.column_config.NumberColumn("c", format="%.6g"),
                "m": st.column_config.NumberColumn("m", format="%.6g"),
                "response": st.column_config.NumberColumn("response", format="%.6g"),
            },
        )
        include_timecourse_direct = st.checkbox("Also enter timecourse data", value=False)
        if include_timecourse_direct:
            if "timecourse_editor_df" not in st.session_state:
                st.session_state["timecourse_editor_df"] = _default_timecourse_editor_df()
            timecourse_edit_df = st.data_editor(
                st.session_state["timecourse_editor_df"],
                num_rows="dynamic",
                use_container_width=True,
                key="timecourse_editor",
                column_config={
                    "time": st.column_config.NumberColumn("time", format="%.6g"),
                    "c": st.column_config.NumberColumn("c", format="%.6g"),
                    "m": st.column_config.NumberColumn("m", format="%.6g"),
                    "value": st.column_config.NumberColumn("value", format="%.6g"),
                },
            )
    else:
        st.info("Using the demo endpoint and timecourse CSV files from examples/.")

    with st.expander("Advanced settings (assay metadata / thresholds)", expanded=False):
        config_file = st.file_uploader(
            "Config (YAML/JSON, optional)",
            type=["yaml", "yml", "json"],
            key="config_upload",
            help="Same schema as the CLI's --config flag. Uploaded values are applied as defaults to the widgets below.",
        )
        if config_file is not None and st.button("Apply config to controls", key="apply_config_btn"):
            try:
                cfg = _parse_config_bytes(config_file.name, config_file.getvalue())
                applied = _apply_config_to_state(cfg)
                if applied:
                    st.success("Config applied:\n- " + "\n- ".join(applied))
                else:
                    st.info("No applicable values were found in the config.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load config: {exc}")

        st.markdown("**Bootstrap**")
        n_boot = st.slider("Bootstrap resamples", min_value=10, max_value=500, step=10, key="n_boot")
        random_seed = st.number_input("Random seed", min_value=0, step=1, key="random_seed")

        st.markdown("**Assay metadata**")
        response_mode = st.selectbox("response_mode", RESPONSE_MODES, key="response_mode")
        readout_level = st.selectbox("readout_level", READOUT_LEVELS, key="readout_level")
        assay_family = st.text_input("assay_family", key="assay_family")
        normalization_mode = st.selectbox("normalization_mode", NORMALIZATION_MODES, key="normalization_mode")
        baseline_definition = st.selectbox("baseline_definition", BASELINE_DEFINITIONS, key="baseline_definition")

        st.markdown("**Thresholds**")
        ec50_min_dynamic_range = st.number_input(
            "ec50_min_dynamic_range", min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key="ec50_min_dynamic_range"
        )
        mopt_prominence_threshold = st.number_input(
            "mopt_prominence_threshold", min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key="mopt_prominence_threshold"
        )
        delayed_attenuation_threshold = st.number_input(
            "delayed_attenuation_threshold", min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key="delayed_attenuation_threshold"
        )

    # Config export
    st.markdown("---")
    st.markdown("### Citation")
    st.markdown("If you use this software, please cite the archived release:")
    st.markdown("[mechanopharm-infer v0.3.0](https://doi.org/10.5281/zenodo.19539760)")
    
    st.markdown("---")
    st.markdown("### Export config")
    cfg_content, cfg_filename, cfg_mime = _serialize_config(dict(st.session_state))
    st.download_button(
        "Download current settings",
        data=cfg_content,
        file_name=cfg_filename,
        mime=cfg_mime,
        help="Download the current advanced settings as a YAML/JSON file for the CLI or future uploads.",
    )

# ---------------------------------------------------------------------------
# Main area — intro text
# ---------------------------------------------------------------------------

st.markdown(
    """
This app runs the same core analysis as the command-line `mechanopharm-infer` workflow.
Choose an input source in the sidebar (bundled demo, direct table input, or CSV upload),
review the advanced settings, then click **Run analysis**.
"""
)

# ---------------------------------------------------------------------------
# Input preview and validation
# ---------------------------------------------------------------------------

st.subheader("Input preview")

endpoint_preview_df: pd.DataFrame | None = None
timecourse_preview_df: pd.DataFrame | None = None
validation_errors: list[str] = []

if input_mode == "Use bundled demo data":
    if DEFAULT_ENDPOINT.exists():
        endpoint_preview_df = pd.read_csv(DEFAULT_ENDPOINT)
    if DEFAULT_TIMECOURSE.exists():
        timecourse_preview_df = pd.read_csv(DEFAULT_TIMECOURSE)
elif input_mode == "Upload CSV":
    endpoint_preview_df = _read_csv_safely(endpoint_file)
    timecourse_preview_df = _read_csv_safely(timecourse_file)
    if endpoint_file is None:
        validation_errors.append("Endpoint CSV: no file uploaded.")
else:
    endpoint_preview_df = endpoint_edit_df
    if include_timecourse_direct:
        timecourse_preview_df = timecourse_edit_df

if endpoint_preview_df is not None:
    validation_errors.extend(_validate_dataframe(endpoint_preview_df, ENDPOINT_REQUIRED, label="Endpoint"))
if timecourse_preview_df is not None and not timecourse_preview_df.empty:
    validation_errors.extend(_validate_dataframe(timecourse_preview_df, TIMECOURSE_REQUIRED, label="Timecourse"))

prev_cols = st.columns(2)
with prev_cols[0]:
    st.markdown("**Endpoint**")
    if endpoint_preview_df is not None and not endpoint_preview_df.empty:
        st.caption(f"rows={len(endpoint_preview_df)} / cols={list(endpoint_preview_df.columns)}")
        st.dataframe(endpoint_preview_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("(no data)")
with prev_cols[1]:
    st.markdown("**Timecourse (optional)**")
    if timecourse_preview_df is not None and not timecourse_preview_df.empty:
        st.caption(f"rows={len(timecourse_preview_df)} / cols={list(timecourse_preview_df.columns)}")
        st.dataframe(timecourse_preview_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("(no timecourse)")

if validation_errors:
    for err in validation_errors:
        st.error(err)

# ---------------------------------------------------------------------------
# Run / Clear buttons
# ---------------------------------------------------------------------------

run_disabled = bool(validation_errors)
btn_cols = st.columns([3, 1])
run = btn_cols[0].button("Run analysis", type="primary", disabled=run_disabled)
if btn_cols[1].button("Clear results", disabled="results" not in st.session_state):
    del st.session_state["results"]
    st.rerun()

# ---------------------------------------------------------------------------
# Analysis execution
# ---------------------------------------------------------------------------

if run:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        endpoint_path = tmpdir / "endpoint.csv"
        timecourse_path: Path | None = None
        outdir = tmpdir / "outputs"

        if input_mode == "Use bundled demo data":
            endpoint_path.write_bytes(DEFAULT_ENDPOINT.read_bytes())
            if DEFAULT_TIMECOURSE.exists():
                timecourse_path = tmpdir / "timecourse.csv"
                timecourse_path.write_bytes(DEFAULT_TIMECOURSE.read_bytes())
        elif input_mode == "Upload CSV":
            _save_uploaded_file(endpoint_file, endpoint_path)
            if timecourse_file is not None:
                timecourse_path = tmpdir / "timecourse.csv"
                _save_uploaded_file(timecourse_file, timecourse_path)
        else:
            endpoint_preview_df.to_csv(endpoint_path, index=False)
            if (
                include_timecourse_direct
                and timecourse_preview_df is not None
                and not timecourse_preview_df.empty
            ):
                timecourse_path = tmpdir / "timecourse.csv"
                timecourse_preview_df.to_csv(timecourse_path, index=False)

        assay_metadata = {
            "response_mode": response_mode,
            "readout_level": readout_level,
            "assay_family": assay_family or "generic",
            "normalization_mode": normalization_mode,
            "baseline_definition": baseline_definition,
        }

        analyze_kwargs: dict[str, Any] = {
            "endpoint_path": str(endpoint_path),
            "timecourse_path": str(timecourse_path) if timecourse_path else None,
            "outdir": str(outdir),
            "n_boot": int(n_boot),
            "random_seed": int(random_seed),
        }
        unsupported: list[str] = []
        if _ANALYZE_SUPPORTS_ASSAY_METADATA:
            analyze_kwargs["assay_metadata"] = assay_metadata
        else:
            unsupported.append("assay_metadata")
        for name, value in (
            ("ec50_min_dynamic_range", float(ec50_min_dynamic_range)),
            ("mopt_prominence_threshold", float(mopt_prominence_threshold)),
            ("delayed_attenuation_threshold", float(delayed_attenuation_threshold)),
        ):
            if _ANALYZE_SUPPORTS_THRESHOLDS.get(name, False):
                analyze_kwargs[name] = value
            else:
                unsupported.append(name)
        if unsupported:
            st.warning(
                f"The installed mechanopharm-infer does not support: {unsupported}. "
                "They will be ignored for this run."
            )

        try:
            with st.spinner("Running mechanopharm-infer analysis..."):
                analyze(**analyze_kwargs)
        except Exception as exc:
            st.error("Analysis failed. Please check that the input data follows the expected schema.")
            st.exception(exc)
            st.stop()

        # Collect all outputs into session_state before tmpdir is removed.
        st.session_state["results"] = _collect_results(outdir, tmpdir)

# ---------------------------------------------------------------------------
# Display results (persisted in session_state)
# ---------------------------------------------------------------------------

if "results" in st.session_state:
    _display_results(st.session_state["results"])
elif run_disabled:
    st.info("Resolve the input errors above to enable **Run analysis**.")
else:
    st.info("Review your input and click **Run analysis**.")
