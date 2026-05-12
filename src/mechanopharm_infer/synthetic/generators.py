"""Synthetic dataset generators built directly from the theory.

Two architectures are exposed.

Two-state
---------
Following Eq. (DeltaG) and Eq. (pstar-explicit) of the theory, the active-state
occupancy at fixed ``(c, m)`` is

    p1*(c, m) = 1 / (1 + exp(beta * Delta_G(c, m))),
    Delta_G(c, m) = Delta_G0 - Delta_alpha * c - Delta_lambda * m
                    - Delta_mu * c * m + 0.5 * kappa * m**2.

The quadratic ``kappa`` term is the lowest-order curvature in ``m`` that the
strict two-state theory needs to support an interior optimum
``m*(c) = (Delta_lambda + Delta_mu * c)/kappa`` (Eq. mstar-2state).  By
default ``kappa = 0`` so the two-state synthetic dataset deliberately does
*not* exhibit interior optimality.  Set a positive ``kappa`` to test the
quadratic two-state variant.

Three-state (protected branch)
------------------------------
Following Eq. (p1starR) and the log-bias parametrization of Eqs. (R01log,
R12log), the responsive fraction at fixed ``(c, m)`` is

    p1*(c, m) = 1 / (1 + R01(c, m)^{-1} + R12(c, m)),

with log-biases

    ln R01(c, m) = beta * (A0 + B0 * c + Lambda0 * m + M0 * c * m),
    ln R12(c, m) = beta * (A1 + B1 * c + Lambda1 * m + M1 * c * m).

Time-resolved curves use the composite approximation

    p1(t) ~ p1_qs * (1 - exp(-t / tau_fast)) * exp(-t / tau_esc)

of Eq. (p1approx), with ``tau_fast`` and ``tau_esc`` recovered from the
local-detailed-balance rates.

All parameters are tunable.  The defaults reproduce the qualitative
signatures used by ``run_synthetic_benchmark``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoStateParams:
    """State-bias parameters of the minimal two-state model.

    All values are dimensionless; ``beta`` is folded into the coefficients.
    """

    delta_g0: float = 3.2  # Delta_G0 (Eq. DeltaG)
    delta_alpha: float = 4.0  # purely chemical bias
    delta_lambda: float = 1.6  # purely mechanical bias
    delta_mu: float = 0.8  # mechanochemical coupling
    kappa: float = 0.0  # quadratic mechanical curvature; > 0 enables m*(c)


@dataclass(frozen=True)
class ProtectedStateParams:
    """Log-bias coefficients of the minimal three-state protected model.

    Conventions follow Eqs. (R01log, R12log) of the theory.  ``beta`` is
    absorbed into the coefficients.
    """

    a0: float = -2.0  # A0
    b0: float = 4.0  # B0
    lambda0: float = 1.2  # Lambda0
    m0: float = 0.4  # M0
    a1: float = -3.0  # A1
    b1: float = 0.5  # B1
    lambda1: float = 2.4  # Lambda1
    m1: float = 1.1  # M1


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _two_state_occupancy(
    c: np.ndarray, m: np.ndarray, params: TwoStateParams
) -> np.ndarray:
    """Return ``p1*(c, m)`` for the two-state model on a ``(m, c)`` grid."""

    cc = c[None, :]
    mm = m[:, None]
    delta_g = (
        params.delta_g0
        - params.delta_alpha * cc
        - params.delta_lambda * mm
        - params.delta_mu * cc * mm
        + 0.5 * params.kappa * mm * mm
    )
    return 1.0 / (1.0 + np.exp(delta_g))


def _three_state_occupancy(
    c: np.ndarray, m: np.ndarray, params: ProtectedStateParams
) -> np.ndarray:
    """Return ``p1*(c, m)`` for the three-state protected model."""

    cc = c[None, :]
    mm = m[:, None]
    log_r01 = params.a0 + params.b0 * cc + params.lambda0 * mm + params.m0 * cc * mm
    log_r12 = params.a1 + params.b1 * cc + params.lambda1 * mm + params.m1 * cc * mm
    inv_r01 = np.exp(-log_r01)
    r12 = np.exp(log_r12)
    return 1.0 / (1.0 + inv_r01 + r12)


def _endpoint_frame(
    values: np.ndarray,
    concentrations: np.ndarray,
    mechanics: np.ndarray,
    n_replicates: int,
    noise_sd: float,
    seed: int,
    dataset_id: str,
    system: str,
    assay: str,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for i_m, m in enumerate(mechanics):
        for i_c, c in enumerate(concentrations):
            mu = float(values[i_m, i_c])
            for rep in range(n_replicates):
                y = float(np.clip(mu + rng.normal(0.0, noise_sd), 0.0, 1.0))
                rows.append({
                    "dataset_id": dataset_id,
                    "system": system,
                    "assay": assay,
                    "c": float(c),
                    "m": float(m),
                    "response": y,
                    "replicate": rep + 1,
                })
    return pd.DataFrame(rows)


def _timecourse_frame(
    values: np.ndarray,
    times: np.ndarray,
    concentrations: np.ndarray,
    mechanics: np.ndarray,
    n_replicates: int,
    noise_sd: float,
    seed: int,
    dataset_id: str,
    system: str,
    assay: str,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for i_c, c in enumerate(concentrations):
        for i_m, m in enumerate(mechanics):
            curve = values[i_c, i_m, :]
            for rep in range(n_replicates):
                noisy = np.clip(curve + rng.normal(0.0, noise_sd, size=len(times)), 0.0, 1.2)
                for t, y in zip(times, noisy, strict=True):
                    rows.append({
                        "dataset_id": dataset_id,
                        "system": system,
                        "assay": assay,
                        "c": float(c),
                        "m": float(m),
                        "time": float(t),
                        "value": float(y),
                        "replicate": rep + 1,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Two-state endpoint and timecourse
# ---------------------------------------------------------------------------


def generate_two_state_endpoint(
    *,
    concentrations: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    mechanics: Sequence[float] = (-0.5, 0.0, 0.5),
    n_replicates: int = 4,
    noise_sd: float = 0.02,
    random_seed: int = 0,
    params: TwoStateParams | None = None,
    dataset_id: str = "synthetic_two_state_endpoint",
    system: str = "synthetic_two_state",
    assay: str = "generic_effect",
) -> pd.DataFrame:
    """Generate an endpoint dataset from the two-state model.

    The endpoint response is identified with ``p1*(c, m)`` (Eq. Ssimple-2state).
    Default parameters produce a mechanically shifted dose--response family
    together with a finite reversal concentration
    ``c_rev = -Delta_lambda / Delta_mu`` (Eq. crev).
    """

    p = params or TwoStateParams()
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    values = _two_state_occupancy(c, m, p)
    return _endpoint_frame(values, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


def generate_two_state_timecourse(
    *,
    times: Sequence[float] = (0, 1, 2, 4, 8, 12),
    concentrations: Sequence[float] = (0.25, 0.75, 1.25),
    mechanics: Sequence[float] = (-0.5, 0.0, 0.5),
    n_replicates: int = 4,
    noise_sd: float = 0.015,
    random_seed: int = 0,
    params: TwoStateParams | None = None,
    tau: float = 1.6,
    dataset_id: str = "synthetic_two_state_timecourse",
    system: str = "synthetic_two_state",
    assay: str = "generic_signal",
) -> pd.DataFrame:
    """Generate a timecourse dataset from the two-state model.

    The transient is a single exponential rise toward ``p1*(c, m)``.  The
    two-state architecture admits only monotone time courses (Eq. relax-p);
    no transient peak or delayed protection arises generically.
    """

    p = params or TwoStateParams()
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    # ``c`` along axis 0 to match the timecourse builder layout.
    eq = _two_state_occupancy(c=c, m=m, params=p).T  # (n_c, n_m)
    values = eq[:, :, None] * (1.0 - np.exp(-t[None, None, :] / float(tau)))
    return _timecourse_frame(values, t, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


# ---------------------------------------------------------------------------
# Three-state protected endpoint and timecourse
# ---------------------------------------------------------------------------


def generate_protected_state_endpoint(
    *,
    concentrations: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
    mechanics: Sequence[float] = (0.0, 0.4, 0.8, 1.2, 1.6),
    n_replicates: int = 4,
    noise_sd: float = 0.02,
    random_seed: int = 0,
    params: ProtectedStateParams | None = None,
    dataset_id: str = "synthetic_protected_endpoint",
    system: str = "synthetic_protected",
    assay: str = "generic_effect",
) -> pd.DataFrame:
    """Generate an endpoint dataset from the three-state protected model.

    The endpoint response is identified with the responsive-state occupancy
    ``p1*(c, m) = 1 / (1 + R01^-1 + R12)`` (Eq. p1starR).  Default parameters
    yield an interior optimum ``m*(c)`` whose location moves with ``c`` (Eq.
    mstar-opt).
    """

    p = params or ProtectedStateParams()
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)
    values = np.clip(_three_state_occupancy(c, m, p), 0.0, 1.0)
    return _endpoint_frame(values, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)


def generate_protected_state_timecourse(
    *,
    times: Sequence[float] = (0, 1, 2, 4, 8, 12),
    concentrations: Sequence[float] = (0.25, 0.75, 1.25),
    mechanics: Sequence[float] = (0.0, 0.4, 0.8, 1.2, 1.6),
    n_replicates: int = 4,
    noise_sd: float = 0.015,
    random_seed: int = 0,
    params: ProtectedStateParams | None = None,
    tau_fast: float = 1.3,
    tau_esc_base: float = 12.0,
    tau_esc_load_sensitivity: float = 8.0,
    dataset_id: str = "synthetic_protected_timecourse",
    system: str = "synthetic_protected",
    assay: str = "generic_signal",
) -> pd.DataFrame:
    """Generate a timecourse dataset from the three-state protected model.

    Uses the composite approximation of Eq. (p1approx),

        p1(t) ~ p1_qs * (1 - exp(-t / tau_fast)) * exp(-t / tau_esc),

    where ``p1_qs`` is taken from the responsive-state occupancy of Eq.
    (p1starR), and the escape timescale ``tau_esc`` shortens with increasing
    mechanical load and concentration (so that protection sets in faster under
    stronger combined challenge).  This produces a transient peak whose
    amplitude, timing, and late-time attenuation depend systematically on
    ``(c, m)``.
    """

    p = params or ProtectedStateParams()
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(mechanics, dtype=float)

    p1_qs = _three_state_occupancy(c=c, m=m, params=p).T  # (n_c, n_m)
    cc = c[:, None]
    mm = m[None, :]
    # tau_esc depends on the combined load, longer when mechanics is mild.
    load = np.clip(mm + 0.5 * cc * mm, 0.0, None)
    tau_esc = tau_esc_base / (1.0 + tau_esc_load_sensitivity * load / (1.0 + load))
    tau_esc = np.clip(tau_esc, 0.5, None)
    rise = 1.0 - np.exp(-t[None, None, :] / float(tau_fast))
    decay = np.exp(-t[None, None, :] / tau_esc[:, :, None])
    transient = p1_qs[:, :, None] * rise * decay
    values = np.clip(transient, 0.0, 1.0)
    return _timecourse_frame(values, t, c, m, n_replicates, noise_sd, random_seed, dataset_id, system, assay)
