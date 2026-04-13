from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_float(value: Any, default: float = float('nan')) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out


def endpoint_diagnostics(summary_df: pd.DataFrame, ec50_df: pd.DataFrame | None = None, mopt_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df is None or summary_df.empty:
        return pd.DataFrame([
            {
                'domain': 'endpoint',
                'item': 'dataset',
                'status': 'not_assessable',
                'severity': 'high',
                'message': 'Endpoint dataset is empty; endpoint fingerprints are not assessable.',
            }
        ])

    n_c = int(summary_df['c'].nunique()) if 'c' in summary_df.columns else 0
    n_m = int(summary_df['m'].nunique()) if 'm' in summary_df.columns else 0
    min_n = int(summary_df['n'].min()) if 'n' in summary_df.columns and not summary_df.empty else 0
    dyn = float(summary_df['response_mean'].max() - summary_df['response_mean'].min()) if 'response_mean' in summary_df.columns and not summary_df.empty else 0.0

    rows.append({
        'domain': 'endpoint', 'item': 'mechanics_levels',
        'status': 'ok' if n_m >= 4 else ('warning' if n_m >= 3 else 'not_assessable'),
        'severity': 'low' if n_m >= 4 else ('medium' if n_m >= 3 else 'high'),
        'message': 'Interior optimum assessment benefits from at least 4 mechanics levels.' if n_m < 4 else 'Mechanics level count is adequate for interior-optimum screening.',
        'value': n_m,
    })
    rows.append({
        'domain': 'endpoint', 'item': 'concentration_levels',
        'status': 'ok' if n_c >= 4 else ('warning' if n_c >= 3 else 'not_assessable'),
        'severity': 'low' if n_c >= 4 else ('medium' if n_c >= 3 else 'high'),
        'message': 'EC50 and sign-reversal screening benefit from at least 4 concentration levels.' if n_c < 4 else 'Concentration level count is adequate for shift/reversal screening.',
        'value': n_c,
    })
    rows.append({
        'domain': 'endpoint', 'item': 'replicates',
        'status': 'ok' if min_n >= 2 else 'warning',
        'severity': 'low' if min_n >= 2 else 'medium',
        'message': 'Bootstrap reliability is limited when any endpoint condition has <2 replicates.' if min_n < 2 else 'Replicate count supports basic bootstrap stability checks.',
        'value': min_n,
    })
    rows.append({
        'domain': 'endpoint', 'item': 'dynamic_range',
        'status': 'ok' if dyn >= 0.1 else ('warning' if dyn >= 0.05 else 'not_assessable'),
        'severity': 'low' if dyn >= 0.1 else ('medium' if dyn >= 0.05 else 'high'),
        'message': 'Endpoint dynamic range is small; shift detection and EC50 estimation may be unstable.' if dyn < 0.1 else 'Endpoint dynamic range is adequate for first-pass fingerprint extraction.',
        'value': dyn,
    })

    if ec50_df is not None and not ec50_df.empty:
        reliable = int(pd.to_numeric(ec50_df.get('is_reliable', pd.Series(dtype=float)), errors='coerce').fillna(0).astype(bool).sum())
        boot_rel = _as_float(pd.to_numeric(ec50_df.get('ec50_bootstrap_reliability', pd.Series(dtype=float)), errors='coerce').dropna().mean())
        status = 'ok' if reliable >= 2 else ('warning' if reliable >= 1 else 'not_assessable')
        rows.append({
            'domain': 'endpoint', 'item': 'ec50_assessability',
            'status': status,
            'severity': 'low' if status == 'ok' else ('medium' if status == 'warning' else 'high'),
            'message': 'Too few mechanics slices support reliable EC50 estimation.' if reliable < 2 else 'Sufficient mechanics slices support EC50-based shift screening.',
            'value': reliable,
            'bootstrap_reliability': boot_rel,
        })

    if mopt_df is not None and not mopt_df.empty:
        interior_frac = _as_float(pd.to_numeric(mopt_df.get('interior_optimum_fraction', pd.Series(dtype=float)), errors='coerce').dropna().mean())
        boot_rel = _as_float(pd.to_numeric(mopt_df.get('mopt_bootstrap_reliability', pd.Series(dtype=float)), errors='coerce').dropna().mean())
        n_interior = int((mopt_df.get('is_interior', pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if 'is_interior' in mopt_df.columns else 0
        status = 'ok' if n_m >= 4 and n_interior >= 1 else ('warning' if n_m >= 3 else 'not_assessable')
        rows.append({
            'domain': 'endpoint', 'item': 'interior_optimum_assessability',
            'status': status,
            'severity': 'low' if status == 'ok' else ('medium' if status == 'warning' else 'high'),
            'message': 'Interior optimum claims are limited by sparse mechanics sampling.' if n_m < 4 else 'Mechanics sampling is sufficient for first-pass interior optimum screening.',
            'value': n_interior,
            'bootstrap_reliability': boot_rel,
            'interior_optimum_fraction': interior_frac,
        })

    return pd.DataFrame(rows)



def timecourse_diagnostics(summary_df: pd.DataFrame, peak_df: pd.DataFrame | None = None, delayed_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary_df is None or summary_df.empty:
        return pd.DataFrame([
            {
                'domain': 'timecourse',
                'item': 'dataset',
                'status': 'not_assessable',
                'severity': 'high',
                'message': 'No timecourse dataset provided; dynamic fingerprints are not assessable.',
            }
        ])

    counts = [int(sub['time'].nunique()) for _, sub in summary_df.groupby(['c', 'm'])]
    durations = [float(sub['time'].max() - sub['time'].min()) for _, sub in summary_df.groupby(['c', 'm'])]
    min_tp = min(counts) if counts else 0
    min_dur = min(durations) if durations else 0.0

    rows.append({
        'domain': 'timecourse', 'item': 'timepoints_per_condition',
        'status': 'ok' if min_tp >= 4 else ('warning' if min_tp >= 3 else 'not_assessable'),
        'severity': 'low' if min_tp >= 4 else ('medium' if min_tp >= 3 else 'high'),
        'message': 'Peak and delayed-protection screening benefits from at least 4 time points per condition.' if min_tp < 4 else 'Time-point sampling is adequate for first-pass dynamic screening.',
        'value': min_tp,
    })
    rows.append({
        'domain': 'timecourse', 'item': 'observation_window',
        'status': 'ok' if min_dur > 0 else 'not_assessable',
        'severity': 'low' if min_dur > 0 else 'high',
        'message': 'Observation window ends immediately; delayed-protection is not assessable.' if min_dur <= 0 else 'Observation window is non-zero for all conditions.',
        'value': min_dur,
    })

    if peak_df is not None and not peak_df.empty:
        clear_frac = _as_float(pd.to_numeric(peak_df.get('has_clear_peak', pd.Series(dtype=float)), errors='coerce').fillna(0).astype(bool).mean())
        rows.append({
            'domain': 'timecourse', 'item': 'peak_detectability',
            'status': 'ok' if clear_frac >= 0.5 else ('warning' if clear_frac > 0 else 'not_assessable'),
            'severity': 'low' if clear_frac >= 0.5 else ('medium' if clear_frac > 0 else 'high'),
            'message': 'Few conditions show a clear non-terminal peak; transient amplification may be weak or not assessable.' if clear_frac < 0.5 else 'A substantial fraction of conditions show a clear non-terminal peak.',
            'value': clear_frac,
        })

    if delayed_df is not None and not delayed_df.empty:
        boot_rel = _as_float(pd.to_numeric(delayed_df.get('delayed_bootstrap_reliability', pd.Series(dtype=float)), errors='coerce').dropna().mean())
        frac = _as_float(pd.to_numeric(delayed_df.get('delayed_protection_detected', pd.Series(dtype=float)), errors='coerce').fillna(0).astype(bool).mean())
        rows.append({
            'domain': 'timecourse', 'item': 'delayed_protection_assessability',
            'status': 'ok' if min_tp >= 4 and min_dur > 0 else ('warning' if min_tp >= 3 else 'not_assessable'),
            'severity': 'low' if min_tp >= 4 and min_dur > 0 else ('medium' if min_tp >= 3 else 'high'),
            'message': 'Dynamic window may be too sparse or too short for stable delayed-protection calls.' if not (min_tp >= 4 and min_dur > 0) else 'Timecourse coverage is adequate for first-pass delayed-protection screening.',
            'value': frac,
            'bootstrap_reliability': boot_rel,
        })

    return pd.DataFrame(rows)



def combine_diagnostics(endpoint_diag: pd.DataFrame | None = None, timecourse_diag: pd.DataFrame | None = None) -> pd.DataFrame:
    dfs = [df for df in [endpoint_diag, timecourse_diag] if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame(columns=['domain', 'item', 'status', 'severity', 'message'])
    return pd.concat(dfs, ignore_index=True, sort=False)



def diagnostics_messages(diag_df: pd.DataFrame, min_severity: str = 'medium') -> list[str]:
    if diag_df is None or diag_df.empty:
        return []
    sev_rank = {'low': 1, 'medium': 2, 'high': 3}
    threshold = sev_rank[min_severity]
    out: list[str] = []
    for _, row in diag_df.iterrows():
        sev = str(row.get('severity', 'low'))
        if sev_rank.get(sev, 0) >= threshold and str(row.get('status')) != 'ok':
            out.append(str(row.get('message')))
    return out
