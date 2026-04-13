import pandas as pd

from mechanopharm_infer.bootstrap import bootstrap_delayed_protection, bootstrap_ec50_vs_m, bootstrap_mopt


def _endpoint_df():
    rows = []
    for rep in [1, 2, 3]:
        for m, shift in [(0.0, 0.0), (0.5, 0.2), (1.0, 0.4)]:
            for c, y in [(0.1, 0.15 + shift), (0.5, 0.5 + shift), (1.0, 0.85 + shift)]:
                rows.append({'c': c, 'm': m, 'response': y + 0.01 * rep, 'replicate': rep})
    return pd.DataFrame(rows)


def _timecourse_df():
    rows = []
    for rep in [1, 2, 3]:
        for t, y in [(0.0, 0.0), (1.0, 0.8), (2.0, 0.35)]:
            rows.append({'time': t, 'c': 1.0, 'm': 0.5, 'value': y + 0.01 * rep, 'replicate': rep})
    return pd.DataFrame(rows)


def test_bootstrap_ec50_vs_m_returns_ci_columns():
    out = bootstrap_ec50_vs_m(_endpoint_df(), n_boot=25, random_seed=1)
    assert {'m', 'ec50_ci_low', 'ec50_ci_high', 'ec50_bootstrap_median', 'ec50_bootstrap_reliability', 'n_boot_used'}.issubset(out.columns)
    assert len(out) == 3


def test_bootstrap_mopt_returns_interior_fraction():
    out = bootstrap_mopt(_endpoint_df(), n_boot=25, random_seed=1)
    assert {'c', 'm_opt_ci_low', 'm_opt_ci_high', 'm_opt_bootstrap_median', 'mopt_bootstrap_reliability', 'interior_optimum_fraction'}.issubset(out.columns)
    assert len(out) == 3


def test_bootstrap_delayed_protection_returns_fraction():
    out = bootstrap_delayed_protection(_timecourse_df(), n_boot=20, random_seed=1)
    assert {'c', 'm', 'attenuation_ci_low', 'attenuation_ci_high', 'delayed_bootstrap_reliability', 'delayed_protection_fraction'}.issubset(out.columns)
    assert len(out) == 1
