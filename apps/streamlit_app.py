from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow local execution from a source checkout without requiring installation first.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mechanopharm_infer.cli import _load_config, analyze
from mechanopharm_infer.types import (
    _ALLOWED_BASELINE_DEFINITIONS,
    _ALLOWED_NORMALIZATION_MODES,
    _ALLOWED_READOUT_LEVELS,
    _ALLOWED_RESPONSE_MODES,
)


APP_TITLE = "mechanopharm-infer"
DEFAULT_ENDPOINT = ROOT / "examples" / "demo_endpoint.csv"
DEFAULT_TIMECOURSE = ROOT / "examples" / "demo_timecourse.csv"

ENDPOINT_REQUIRED = ("c", "m", "response")
TIMECOURSE_REQUIRED = ("time", "c", "m", "value")

RESPONSE_MODES = sorted(_ALLOWED_RESPONSE_MODES)
NORMALIZATION_MODES = sorted(_ALLOWED_NORMALIZATION_MODES)
BASELINE_DEFINITIONS = sorted(_ALLOWED_BASELINE_DEFINITIONS)
READOUT_LEVELS = sorted(_ALLOWED_READOUT_LEVELS)

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


def _read_csv_safely(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        # `getvalue()` keeps the buffer reusable across reruns.
        from io import BytesIO

        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        st.error(f"CSV を読み込めませんでした: {exc}")
        return None


def _default_endpoint_editor_df() -> pd.DataFrame:
    if DEFAULT_ENDPOINT.exists():
        df = pd.read_csv(DEFAULT_ENDPOINT)
        return df[list(ENDPOINT_REQUIRED)].copy()
    return pd.DataFrame({col: pd.Series(dtype="float") for col in ENDPOINT_REQUIRED})


def _default_timecourse_editor_df() -> pd.DataFrame:
    if DEFAULT_TIMECOURSE.exists():
        df = pd.read_csv(DEFAULT_TIMECOURSE)
        return df[list(TIMECOURSE_REQUIRED)].copy()
    return pd.DataFrame({col: pd.Series(dtype="float") for col in TIMECOURSE_REQUIRED})


def _validate_dataframe(
    df: pd.DataFrame | None, required: tuple[str, ...], *, label: str
) -> list[str]:
    errors: list[str] = []
    if df is None or df.empty:
        errors.append(f"{label}: データが空です。")
        return errors
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{label}: 必須列が不足しています → {missing}")
        return errors
    for col in required:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().all():
            errors.append(f"{label}: 列 '{col}' に有効な数値がありません。")
        elif coerced.isna().any():
            n_bad = int(coerced.isna().sum())
            errors.append(
                f"{label}: 列 '{col}' に数値変換できないセルが {n_bad} 件あります。"
            )
    return errors


def _apply_config_to_state(config: dict[str, Any]) -> list[str]:
    """Push values from a config dict into st.session_state and report messages."""

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
    for key in (
        "response_mode",
        "readout_level",
        "assay_family",
        "normalization_mode",
        "baseline_definition",
    ):
        if key in md and md[key] is not None:
            st.session_state[key] = md[key]
            messages.append(f"assay_metadata.{key} = {md[key]}")

    thresholds = config.get("thresholds") or {}
    for key in (
        "ec50_min_dynamic_range",
        "mopt_prominence_threshold",
        "delayed_attenuation_threshold",
    ):
        if key in thresholds and thresholds[key] is not None:
            st.session_state[key] = float(thresholds[key])
            messages.append(f"thresholds.{key} = {float(thresholds[key])}")

    if config.get("endpoint") or config.get("timecourse"):
        messages.append(
            "(注) config の endpoint / timecourse パス指定はアプリでは無視されます。"
            " 入力欄から指定してください。"
        )
    return messages


# Initialise defaults for advanced widgets so that config upload can override them.
for key, value in DEFAULT_ADVANCED.items():
    st.session_state.setdefault(key, value)


st.title(APP_TITLE)
st.caption("Mechanopharmacology response-landscape inference toolkit — Streamlit MVP")

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
            "下の表をクリックして直接編集できます。行は右下の `+` で追加、空セルは削除されます。"
        )
        if "endpoint_editor_df" not in st.session_state:
            st.session_state["endpoint_editor_df"] = _default_endpoint_editor_df()
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
        include_timecourse_direct = st.checkbox(
            "タイムコースデータも入力する", value=False
        )
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
        st.info("examples/ のデモ用 endpoint / timecourse CSV を使用します。")

    with st.expander("詳細設定 (assay metadata / thresholds)", expanded=False):
        config_file = st.file_uploader(
            "Config (YAML/JSON, optional)",
            type=["yaml", "yml", "json"],
            key="config_upload",
            help="CLI の --config と同じ形式。読み込んだ値は下の各ウィジェットの初期値に反映されます。",
        )
        if config_file is not None and st.button(
            "Apply config to controls", key="apply_config_btn"
        ):
            try:
                tmp_cfg_path = Path(tempfile.gettempdir()) / config_file.name
                tmp_cfg_path.write_bytes(config_file.getvalue())
                cfg = _load_config(tmp_cfg_path)
                applied = _apply_config_to_state(cfg)
                if applied:
                    st.success("設定を反映しました:\n- " + "\n- ".join(applied))
                else:
                    st.info("反映できる値がありませんでした。")
                st.rerun()
            except Exception as exc:  # pragma: no cover - UI-level error handling
                st.error(f"Config の読み込みに失敗しました: {exc}")

        st.markdown("**Bootstrap**")
        n_boot = st.slider(
            "Bootstrap resamples",
            min_value=10,
            max_value=500,
            step=10,
            key="n_boot",
        )
        random_seed = st.number_input(
            "Random seed", min_value=0, step=1, key="random_seed"
        )

        st.markdown("**Assay metadata**")
        response_mode = st.selectbox(
            "response_mode", RESPONSE_MODES, key="response_mode"
        )
        readout_level = st.selectbox(
            "readout_level", READOUT_LEVELS, key="readout_level"
        )
        assay_family = st.text_input("assay_family", key="assay_family")
        normalization_mode = st.selectbox(
            "normalization_mode", NORMALIZATION_MODES, key="normalization_mode"
        )
        baseline_definition = st.selectbox(
            "baseline_definition", BASELINE_DEFINITIONS, key="baseline_definition"
        )

        st.markdown("**Thresholds**")
        ec50_min_dynamic_range = st.number_input(
            "ec50_min_dynamic_range",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.4f",
            key="ec50_min_dynamic_range",
        )
        mopt_prominence_threshold = st.number_input(
            "mopt_prominence_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.4f",
            key="mopt_prominence_threshold",
        )
        delayed_attenuation_threshold = st.number_input(
            "delayed_attenuation_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.4f",
            key="delayed_attenuation_threshold",
        )

    st.markdown("---")
    st.markdown("### Citation")
    st.markdown("If you use this software, please cite the archived release:")
    st.markdown(
        "[mechanopharm-infer v0.3.0](https://doi.org/10.5281/zenodo.19780165)"
    )

st.markdown(
    """
This app runs the same core analysis as the command-line `mechanopharm-infer` workflow.
入力方法を選んで（デモ / 直接入力 / CSV アップロード）詳細設定を確認したのち、
**Run analysis** を押してください。
"""
)

# ---- Input preview & validation ----
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
        validation_errors.append("Endpoint CSV: ファイルがアップロードされていません。")
else:  # Direct input
    endpoint_preview_df = endpoint_edit_df
    if include_timecourse_direct:
        timecourse_preview_df = timecourse_edit_df

if endpoint_preview_df is not None:
    validation_errors.extend(
        _validate_dataframe(endpoint_preview_df, ENDPOINT_REQUIRED, label="Endpoint")
    )
if timecourse_preview_df is not None and not timecourse_preview_df.empty:
    validation_errors.extend(
        _validate_dataframe(
            timecourse_preview_df, TIMECOURSE_REQUIRED, label="Timecourse"
        )
    )

prev_cols = st.columns(2)
with prev_cols[0]:
    st.markdown("**Endpoint**")
    if endpoint_preview_df is not None and not endpoint_preview_df.empty:
        st.caption(
            f"rows={len(endpoint_preview_df)} / cols={list(endpoint_preview_df.columns)}"
        )
        st.dataframe(
            endpoint_preview_df.head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("（データなし）")
with prev_cols[1]:
    st.markdown("**Timecourse (optional)**")
    if timecourse_preview_df is not None and not timecourse_preview_df.empty:
        st.caption(
            f"rows={len(timecourse_preview_df)} / cols={list(timecourse_preview_df.columns)}"
        )
        st.dataframe(
            timecourse_preview_df.head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("（タイムコースなし）")

if validation_errors:
    for err in validation_errors:
        st.error(err)

run_disabled = bool(validation_errors)
run = st.button("Run analysis", type="primary", disabled=run_disabled)

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
        else:  # Direct input
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

        try:
            with st.spinner("Running mechanopharm-infer analysis..."):
                analyze(
                    endpoint_path=str(endpoint_path),
                    timecourse_path=str(timecourse_path) if timecourse_path else None,
                    outdir=str(outdir),
                    n_boot=int(n_boot),
                    random_seed=int(random_seed),
                    assay_metadata=assay_metadata,
                    ec50_min_dynamic_range=float(ec50_min_dynamic_range),
                    mopt_prominence_threshold=float(mopt_prominence_threshold),
                    delayed_attenuation_threshold=float(delayed_attenuation_threshold),
                )
        except Exception as exc:  # pragma: no cover - UI-level error handling
            st.error(
                "Analysis failed. Please check that the input data follows the expected schema."
            )
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
    if run_disabled:
        st.info("入力エラーを解消すると **Run analysis** が押せるようになります。")
    else:
        st.info("入力を確認し、**Run analysis** を押してください。")
