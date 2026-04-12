from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "benchmarks" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


# Prefer an installed/reference package, then fall back to uploaded source files.
try:
    from mechanopharm_minimal.models import TwoStateModel, ThreeStateProtectionModel
except Exception:
    fallback_root = Path("/mnt/data")
    if str(fallback_root) not in sys.path:
        sys.path.insert(0, str(fallback_root))
    from models import TwoStateModel, ThreeStateProtectionModel


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def add_gaussian_noise(
    values: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.0,
) -> np.ndarray:
    noisy = np.asarray(values, dtype=float) + rng.normal(0.0, sigma, size=np.shape(values))
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi = np.inf if clip_max is None else clip_max
        noisy = np.clip(noisy, lo, hi)
    return noisy


def generate_two_state_endpoint_dataset(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    model: TwoStateModel | None = None,
    noise_sigma: float = 0.0,
    n_replicates: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    model = TwoStateModel() if model is None else model
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    C, M = np.meshgrid(c_grid, m_grid)
    response_clean = model.signal(C, M)

    for rep in range(1, n_replicates + 1):
        response = response_clean.copy()
        if noise_sigma > 0.0:
            response = add_gaussian_noise(response, sigma=noise_sigma, rng=rng)

        for i, m in enumerate(m_grid):
            for j, c in enumerate(c_grid):
                rows.append({
                    "c": float(c),
                    "m": float(m),
                    "response": float(response[i, j]),
                    "replicate": rep,
                })

    return pd.DataFrame(rows)


def generate_protected_state_endpoint_dataset(
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    model: ThreeStateProtectionModel | None = None,
    noise_sigma: float = 0.0,
    n_replicates: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    model = ThreeStateProtectionModel() if model is None else model
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    C, M = np.meshgrid(c_grid, m_grid)
    response_clean = model.responsive_fraction_steady(C, M)

    for rep in range(1, n_replicates + 1):
        response = response_clean.copy()
        if noise_sigma > 0.0:
            response = add_gaussian_noise(response, sigma=noise_sigma, rng=rng)

        for i, m in enumerate(m_grid):
            for j, c in enumerate(c_grid):
                rows.append({
                    "c": float(c),
                    "m": float(m),
                    "response": float(response[i, j]),
                    "replicate": rep,
                })

    return pd.DataFrame(rows)


def generate_protected_state_timecourse_dataset(
    t_grid: np.ndarray,
    c_grid: np.ndarray,
    m_grid: np.ndarray,
    model: ThreeStateProtectionModel | None = None,
    noise_sigma: float = 0.0,
    n_replicates: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    model = ThreeStateProtectionModel() if model is None else model
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for rep in range(1, n_replicates + 1):
        for c in c_grid:
            for m in m_grid:
                y = model.responsive_fraction_timecourse(np.asarray(t_grid, dtype=float), float(c), float(m))
                if noise_sigma > 0.0:
                    y = add_gaussian_noise(y, sigma=noise_sigma, rng=rng)
                for t, value in zip(t_grid, y):
                    rows.append({
                        "time": float(t),
                        "c": float(c),
                        "m": float(m),
                        "value": float(value),
                        "replicate": rep,
                    })

    return pd.DataFrame(rows)


def generate_clean_benchmark_bundle(outdir: str | Path = OUTDIR) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    c_grid = np.linspace(0.0, 2.0, 9)
    m_grid_two = np.linspace(-1.0, 1.0, 9)
    m_grid_three = np.linspace(0.0, 2.0, 9)
    t_grid = np.linspace(0.0, 20.0, 41)

    two_endpoint = generate_two_state_endpoint_dataset(c_grid, m_grid_two, noise_sigma=0.0, n_replicates=1, seed=1)
    three_endpoint = generate_protected_state_endpoint_dataset(c_grid, m_grid_three, noise_sigma=0.0, n_replicates=1, seed=2)
    three_timecourse = generate_protected_state_timecourse_dataset(
        t_grid=t_grid,
        c_grid=np.array([0.5, 1.0, 1.5]),
        m_grid=np.array([0.3, 0.8, 1.3]),
        noise_sigma=0.0,
        n_replicates=1,
        seed=3,
    )

    two_path = outdir / "two_state_endpoint_clean.csv"
    three_endpoint_path = outdir / "protected_state_endpoint_clean.csv"
    three_timecourse_path = outdir / "protected_state_timecourse_clean.csv"

    save_csv(two_path, two_endpoint)
    save_csv(three_endpoint_path, three_endpoint)
    save_csv(three_timecourse_path, three_timecourse)

    return {
        "two_state_endpoint": two_path,
        "protected_state_endpoint": three_endpoint_path,
        "protected_state_timecourse": three_timecourse_path,
    }


def generate_noisy_benchmark_bundle(
    outdir: str | Path = OUTDIR,
    noise_sigma_endpoint: float = 0.05,
    noise_sigma_timecourse: float = 0.05,
    n_replicates: int = 3,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    c_grid = np.linspace(0.0, 2.0, 9)
    m_grid_two = np.linspace(-1.0, 1.0, 9)
    m_grid_three = np.linspace(0.0, 2.0, 9)
    t_grid = np.linspace(0.0, 20.0, 41)

    two_endpoint = generate_two_state_endpoint_dataset(c_grid, m_grid_two, noise_sigma=noise_sigma_endpoint, n_replicates=n_replicates, seed=11)
    three_endpoint = generate_protected_state_endpoint_dataset(c_grid, m_grid_three, noise_sigma=noise_sigma_endpoint, n_replicates=n_replicates, seed=12)
    three_timecourse = generate_protected_state_timecourse_dataset(
        t_grid=t_grid,
        c_grid=np.array([0.5, 1.0, 1.5]),
        m_grid=np.array([0.3, 0.8, 1.3]),
        noise_sigma=noise_sigma_timecourse,
        n_replicates=n_replicates,
        seed=13,
    )

    two_path = outdir / "two_state_endpoint_noisy.csv"
    three_endpoint_path = outdir / "protected_state_endpoint_noisy.csv"
    three_timecourse_path = outdir / "protected_state_timecourse_noisy.csv"

    save_csv(two_path, two_endpoint)
    save_csv(three_endpoint_path, three_endpoint)
    save_csv(three_timecourse_path, three_timecourse)

    return {
        "two_state_endpoint": two_path,
        "protected_state_endpoint": three_endpoint_path,
        "protected_state_timecourse": three_timecourse_path,
    }


def main() -> None:
    clean = generate_clean_benchmark_bundle()
    noisy = generate_noisy_benchmark_bundle()

    print("Generated clean benchmark bundle:")
    for k, v in clean.items():
        print(f"  {k}: {v}")

    print("\nGenerated noisy benchmark bundle:")
    for k, v in noisy.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
