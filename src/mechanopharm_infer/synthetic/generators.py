from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _BaseConfig:
    concentrations: tuple[float, ...]
    mechanics: tuple[float, ...]
    n_replicates: int
    noise_sd: float
    random_seed: int
    dataset_id: str
    system: str
    assay: str


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _endpoint_frame(values: np.ndarray, concentrations: np.ndarray, mechanics: np.ndarray, n_replicates: int, noise_sd: float, seed: int, dataset_id: str, system: str, assay: str) -> pd.DataFrame:
    rng = _rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for i_m, m in enumerate(mechanics):
        for i_c, c in enumerate(concentrations):
            mu = float(values[i_m, i_c])
            for rep in range(n_replicates):
                y = float(np.clip(mu + rng.normal(0.0, noise_sd), 0.0, 1.0))
                rows.append({
                    'dataset_id': dataset_id,
                    'system': system,
                    'assay': assay,
                    'c': float(c),
                    'm': float(m),
                    'response': y,
                    'replicate': rep + 1,
                })
    return pd.DataFrame(rows)


def _timecourse_frame(values: np.ndarray, times: np.ndarray, concentrations: np.ndarray, mechanics: np.ndarray, n_replicates: int, noise_sd: float, seed: int, dataset_id: str, system: str, assay: str) -> pd.DataFrame:
    rng = _rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for i_c, c in enumerate(concentrations):
        for i_m, m in enumerate(mechanics):
            curve = values[i_c, i_m, :]
            for rep in range(n_replicates):
                noisy = np.clip(curve + rng.normal(0.0, noise_sd, size=len(times)), 0.0, 1.2)
                for t, y in zip(times, noisy, strict=True):
                    rows.append({
                        'dataset_id': dataset_id,
                        'system': system,
                        'assay': assay,
                        'c': float(c),
                        'm': float(m),
                        'time': float(t),
                        'value': float(y),
                        'replicate': rep + 1,
                    })
    return pd.DataFrame(rows)


def generate_two_state_endpoint(*, concentrations: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0), mechanics: list[float] | tuple[float, ...] = (-0.5, 0.0, 0.5), n_replicates: int = 4, noise_sd: float = 0.02, random_seed: int = 0, dataset_id: str = 'synthetic_two_state_endpoint', system: str = 'synthetic_two_state', assay: str = 'generic_effect') -> pd.DataFrame:
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    shift = 0.45 * m[:, None]
    coupling = 0.2 * m[:, None] * c[None, :]
    logits = 4.0 * (c[None, :] - 0.9 + shift + coupling)
    values = _sigmoid(logits)
    return _endpoint_frame(values, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


def generate_protected_state_endpoint(*, concentrations: list[float] | tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0), mechanics: list[float] | tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6), n_replicates: int = 4, noise_sd: float = 0.02, random_seed: int = 0, dataset_id: str = 'synthetic_protected_endpoint', system: str = 'synthetic_protected', assay: str = 'generic_effect') -> pd.DataFrame:
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    activation = _sigmoid(4.0 * (c[None, :] - 0.8 + 0.45 * m[:, None]))
    m_star = 0.35 + 0.45 * c
    protection = 0.38 * np.maximum(m[:, None] - m_star[None, :], 0.0) ** 1.25
    values = np.clip(activation - protection, 0.0, 1.0)
    return _endpoint_frame(values, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


def generate_two_state_timecourse(*, times: list[float] | tuple[float, ...] = (0, 1, 2, 4, 8, 12), concentrations: list[float] | tuple[float, ...] = (0.25, 0.75, 1.25), mechanics: list[float] | tuple[float, ...] = (-0.5, 0.0, 0.5), n_replicates: int = 4, noise_sd: float = 0.015, random_seed: int = 0, dataset_id: str = 'synthetic_two_state_timecourse', system: str = 'synthetic_two_state', assay: str = 'generic_signal') -> pd.DataFrame:
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    eq = _sigmoid(3.2 * (c[:, None] - 0.8 + 0.5 * m[None, :]))
    tau = 1.6 + 0.2 * (m[None, :] - m.min())
    values = eq[:, :, None] * (1.0 - np.exp(-t[None, None, :] / tau[:, :, None]))
    return _timecourse_frame(values, t, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


def generate_protected_state_timecourse(*, times: list[float] | tuple[float, ...] = (0, 1, 2, 4, 8, 12), concentrations: list[float] | tuple[float, ...] = (0.25, 0.75, 1.25), mechanics: list[float] | tuple[float, ...] = (0.0, 0.4, 0.8, 1.2, 1.6), n_replicates: int = 4, noise_sd: float = 0.015, random_seed: int = 0, dataset_id: str = 'synthetic_protected_timecourse', system: str = 'synthetic_protected', assay: str = 'generic_signal') -> pd.DataFrame:
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    peak_amp = _sigmoid(3.5 * (c[:, None] - 0.7 + 0.4 * m[None, :]))
    einf = np.clip(peak_amp - (0.18 + 0.10 * np.maximum(m[None, :] - (0.3 + 0.45 * c[:, None]), 0.0)), 0.0, 1.0)
    rise = 1.0 - np.exp(-t[None, None, :] / 1.3)
    decay_strength = 0.04 + 0.08 * np.maximum(m[None, :] - (0.3 + 0.45 * c[:, None]), 0.0)
    transient = peak_amp[:, :, None] * rise * np.exp(-decay_strength[:, :, None] * t[None, None, :])
    relax = einf[:, :, None] * (1.0 - np.exp(-t[None, None, :] / 4.5))
    values = np.maximum(transient, relax)
    return _timecourse_frame(values, t, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)
