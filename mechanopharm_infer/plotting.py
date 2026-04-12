from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_endpoint_landscape(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mesh = ax.pcolormesh(c_grid, m_grid, response, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Response")
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Mechanical condition m")
    ax.set_title("Endpoint response landscape")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_ec50_vs_m(ec50_df: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(ec50_df["m"], ec50_df["ec50"], marker="o")
    ax.set_xlabel("Mechanical condition m")
    ax.set_ylabel("EC50")
    ax.set_title("EC50(m)")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def plot_mopt_vs_c(mopt_df: pd.DataFrame, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(mopt_df["c"], mopt_df["m_opt"], marker="o")
    ax.set_xlabel("Concentration c")
    ax.set_ylabel("Optimal mechanical condition m*")
    ax.set_title("m*(c)")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax
