"""Lightweight plotting helpers.

The intent is to provide quick visual checks of the response landscape and of
extracted fingerprints.  None of these plots are meant to be publication
quality on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_endpoint_landscape(c_grid, m_grid, response, savepath=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mesh = ax.pcolormesh(c_grid, m_grid, response, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Response E(c, m)")
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Mechanical descriptor m")
    ax.set_title("Endpoint response landscape")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_ec50_vs_m(ec50_df: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(ec50_df["m"], ec50_df["ec50"], marker="o")
    if {"ec50_ci_low", "ec50_ci_high"}.issubset(ec50_df.columns):
        ax.fill_between(ec50_df["m"], ec50_df["ec50_ci_low"], ec50_df["ec50_ci_high"], alpha=0.2)
    ax.set_xlabel("Mechanical descriptor m")
    ax.set_ylabel("EC50")
    ax.set_title("EC50(m): mechanically shifted dose response")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_mopt_vs_c(mopt_df: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(mopt_df["c"], mopt_df["m_opt"], marker="o", label="m*(c)")
    if "m_opt_grid" in mopt_df.columns:
        ax.plot(mopt_df["c"], mopt_df["m_opt_grid"], marker="x", linestyle=":", alpha=0.5, label="grid max")
    if {"m_opt_ci_low", "m_opt_ci_high"}.issubset(mopt_df.columns):
        ax.fill_between(mopt_df["c"], mopt_df["m_opt_ci_low"], mopt_df["m_opt_ci_high"], alpha=0.2)
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Optimal mechanical condition m*")
    ax.set_title("m*(c): optimal mechanical condition")
    if "m_opt_grid" in mopt_df.columns:
        ax.legend(frameon=False, fontsize=8)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_dose_response_family(endpoint_summary: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    summary = endpoint_summary.sort_values(["m", "c"])
    for m_value, sub in summary.groupby("m"):
        ax.plot(sub["c"], sub["response_mean"], marker="o", label=f"m={m_value:g}")
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Response")
    ax.set_title("Dose-response family by mechanics")
    if summary["m"].nunique() <= 8:
        ax.legend(frameon=False, fontsize=8)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_timecourse_panel(
    timecourse_summary: pd.DataFrame,
    max_conditions: int = 6,
    savepath: str | None = None,
):
    """Plot representative timecourse trajectories.

    Accepts both the canonical schema (``time`` / ``value_mean``) and the
    legacy short-form (``t`` / ``response_mean``) used by some demo notebooks.
    """

    time_col = "time" if "time" in timecourse_summary.columns else "t"
    value_col = "value_mean" if "value_mean" in timecourse_summary.columns else "response_mean"
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    order = (
        timecourse_summary[["c", "m"]]
        .drop_duplicates()
        .sort_values(["m", "c"])
        .head(max_conditions)
    )
    for row in order.itertuples(index=False):
        sub = timecourse_summary[
            (timecourse_summary["c"] == row.c) & (timecourse_summary["m"] == row.m)
        ].sort_values(time_col)
        ax.plot(sub[time_col], sub[value_col], marker="o", label=f"c={row.c:g}, m={row.m:g}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Response")
    ax.set_title("Representative time courses")
    if len(order) <= 6:
        ax.legend(frameon=False, fontsize=7)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_evidence_summary(evidence_df: pd.DataFrame, savepath: str | None = None):
    rank = {"none": 0, "weak": 1, "moderate": 2, "strong": 3, "not_assessable": -1}
    labels = evidence_df["fingerprint"].astype(str).tolist()
    values = [rank.get(v, -1) for v in evidence_df["evidence_strength"].astype(str)]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.55 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.set_xticks([-1, 0, 1, 2, 3], ["NA", "none", "weak", "moderate", "strong"])
    ax.set_xlabel("Evidence strength")
    ax.set_title("Fingerprint evidence summary")
    ax.set_xlim(-1.2, 3.2)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_c_rev_diagnostic(
    reversal: dict,
    c_grid: np.ndarray | None = None,
    mean_slopes: np.ndarray | None = None,
    savepath: str | None = None,
):
    """Plot the linear ``s(c) = a + b * c`` fit used to estimate ``c_rev``.

    If only ``reversal`` is provided, only the estimate is annotated.
    """

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if c_grid is not None and mean_slopes is not None:
        ax.plot(c_grid, mean_slopes, "o", label=r"mean $\partial E/\partial m$")
        a = reversal.get("delta_lambda_proxy", float("nan"))
        b = reversal.get("delta_mu_proxy", float("nan"))
        if np.isfinite(a) and np.isfinite(b):
            xs = np.linspace(float(np.min(c_grid)), float(np.max(c_grid)), 64)
            ax.plot(xs, a + b * xs, "-", label=r"fit $a + b c$")
    c_rev = reversal.get("c_rev_estimate", float("nan"))
    if np.isfinite(c_rev):
        ax.axvline(c_rev, color="red", linestyle="--", label=f"c_rev ≈ {c_rev:.3g}")
    ax.axhline(0.0, color="grey", linewidth=0.7)
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Mean mechanical slope")
    ax.set_title("c_rev diagnostic")
    ax.legend(frameon=False, fontsize=8)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_benchmark_summary(benchmark_df: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    status_counts = benchmark_df["matched_expected"].value_counts().reindex([True, False], fill_value=0)
    ax.bar(["matched", "mismatched"], status_counts.values)
    ax.set_ylabel("Count")
    ax.set_title("Synthetic benchmark summary")
    for idx, value in enumerate(status_counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax
