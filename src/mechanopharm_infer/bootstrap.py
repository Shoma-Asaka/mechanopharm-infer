from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fingerprints import delayed_protection_metrics, ec50_vs_m, endpoint_final_response, find_mechanical_optima, peak_metrics_by_condition
from .preprocess import endpoint_to_grid, split_timecourses_by_condition, summarize_endpoint, summarize_timecourse


@dataclass(frozen=True)
class BootstrapConfig:
    n_boot: int = 200
    random_seed: int = 0
    ci_level: float = 0.95


def _ci_bounds(ci_level: float) -> tuple[float, float]:
    alpha = 1.0 - float(ci_level)
    return alpha / 2.0, 1.0 - alpha / 2.0


def _sample_condition_rows(df: pd.DataFrame, group_cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, sub in df.groupby(group_cols, dropna=False, sort=False):
        if len(sub) <= 1:
            parts.append(sub.copy())
            continue
        take = rng.integers(0, len(sub), size=len(sub))
        sampled = sub.iloc[take].copy()
        parts.append(sampled)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _aggregate_bootstrap(samples: list[pd.DataFrame], key_col: str, value_col: str, support_col: str | None = None, ci_level: float = 0.95) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame(columns=[key_col, f"{value_col}_ci_low", f"{value_col}_ci_high", f"{value_col}_bootstrap_median", "bootstrap_support_fraction", "n_boot_used"])
    stacked = pd.concat(samples, ignore_index=True)
    if stacked.empty:
        return pd.DataFrame(columns=[key_col, f"{value_col}_ci_low", f"{value_col}_ci_high", f"{value_col}_bootstrap_median", "bootstrap_support_fraction", "n_boot_used"])
    q_low, q_high = _ci_bounds(ci_level)
    rows = []
    for key, sub in stacked.groupby(key_col, dropna=False, sort=True):
        vals = pd.to_numeric(sub[value_col], errors='coerce').dropna().to_numpy(dtype=float)
        support = pd.to_numeric(sub[support_col], errors='coerce').dropna().to_numpy(dtype=float) if support_col and support_col in sub.columns else np.ones(len(sub), dtype=float)
        rows.append({
            key_col: key,
            f"{value_col}_ci_low": float(np.quantile(vals, q_low)) if len(vals) else float('nan'),
            f"{value_col}_ci_high": float(np.quantile(vals, q_high)) if len(vals) else float('nan'),
            f"{value_col}_bootstrap_median": float(np.median(vals)) if len(vals) else float('nan'),
            "bootstrap_support_fraction": float(np.mean(support)) if len(support) else float('nan'),
            "n_boot_used": int(len(sub)),
        })
    return pd.DataFrame(rows)


def bootstrap_ec50_vs_m(endpoint_df: pd.DataFrame, *, n_boot: int = 200, random_seed: int = 0, ci_level: float = 0.95, min_dynamic_range: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    samples: list[pd.DataFrame] = []
    for b in range(int(n_boot)):
        sampled = _sample_condition_rows(endpoint_df, ['c', 'm'], rng)
        summary = summarize_endpoint(sampled)
        c_grid, m_grid, response = endpoint_to_grid(summary)
        fp = ec50_vs_m(c_grid, m_grid, response, min_dynamic_range=min_dynamic_range).copy()
        fp['boot_id'] = b
        fp['supported'] = fp.get('is_reliable', False).astype(float)
        samples.append(fp[['boot_id', 'm', 'ec50', 'supported']])
    agg = _aggregate_bootstrap(samples, key_col='m', value_col='ec50', support_col='supported', ci_level=ci_level)
    agg = agg.rename(columns={'bootstrap_support_fraction': 'ec50_bootstrap_reliability'})
    return agg


def bootstrap_mopt(endpoint_df: pd.DataFrame, *, n_boot: int = 200, random_seed: int = 0, ci_level: float = 0.95, prominence_threshold: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    samples: list[pd.DataFrame] = []
    for b in range(int(n_boot)):
        sampled = _sample_condition_rows(endpoint_df, ['c', 'm'], rng)
        summary = summarize_endpoint(sampled)
        c_grid, m_grid, response = endpoint_to_grid(summary)
        fp = find_mechanical_optima(c_grid, m_grid, response, prominence_threshold=prominence_threshold).copy()
        fp['boot_id'] = b
        fp['supported'] = fp.get('is_reliable', False).astype(float)
        fp['interior_supported'] = fp.get('is_interior', False).astype(float) * fp['supported']
        samples.append(fp[['boot_id', 'c', 'm_opt', 'supported', 'interior_supported']])
    agg = _aggregate_bootstrap(samples, key_col='c', value_col='m_opt', support_col='supported', ci_level=ci_level)
    agg = agg.rename(columns={'bootstrap_support_fraction': 'mopt_bootstrap_reliability'})
    stacked = pd.concat(samples, ignore_index=True)
    interior = stacked.groupby('c', dropna=False, sort=True)['interior_supported'].mean().reset_index().rename(columns={'interior_supported': 'interior_optimum_fraction'})
    return agg.merge(interior, on='c', how='left')


def bootstrap_delayed_protection(timecourse_df: pd.DataFrame, *, n_boot: int = 200, random_seed: int = 0, ci_level: float = 0.95, attenuation_threshold: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    samples: list[pd.DataFrame] = []
    for b in range(int(n_boot)):
        sampled = _sample_condition_rows(timecourse_df, ['c', 'm', 'time'], rng)
        summary = summarize_timecourse(sampled)
        by_condition = split_timecourses_by_condition(summary)
        peak_df = peak_metrics_by_condition(by_condition)
        final_df = endpoint_final_response(by_condition)
        delayed_df = delayed_protection_metrics(peak_df, final_df, attenuation_threshold=attenuation_threshold).copy()
        delayed_df['boot_id'] = b
        delayed_df['supported'] = delayed_df.get('is_reliable', False).astype(float)
        delayed_df['delayed_support'] = delayed_df.get('delayed_protection_detected', False).astype(float) * delayed_df['supported']
        samples.append(delayed_df[['boot_id', 'c', 'm', 'attenuation', 'supported', 'delayed_support']])
    if not samples:
        return pd.DataFrame(columns=['c', 'm'])
    stacked = pd.concat(samples, ignore_index=True)
    q_low, q_high = _ci_bounds(ci_level)
    rows = []
    for (c, m), sub in stacked.groupby(['c', 'm'], dropna=False, sort=True):
        vals = pd.to_numeric(sub['attenuation'], errors='coerce').dropna().to_numpy(dtype=float)
        support = pd.to_numeric(sub['supported'], errors='coerce').dropna().to_numpy(dtype=float)
        dp = pd.to_numeric(sub['delayed_support'], errors='coerce').dropna().to_numpy(dtype=float)
        rows.append({
            'c': c,
            'm': m,
            'attenuation_ci_low': float(np.quantile(vals, q_low)) if len(vals) else float('nan'),
            'attenuation_ci_high': float(np.quantile(vals, q_high)) if len(vals) else float('nan'),
            'attenuation_bootstrap_median': float(np.median(vals)) if len(vals) else float('nan'),
            'delayed_bootstrap_reliability': float(np.mean(support)) if len(support) else float('nan'),
            'delayed_protection_fraction': float(np.mean(dp)) if len(dp) else float('nan'),
            'n_boot_used': int(len(sub)),
        })
    return pd.DataFrame(rows)
