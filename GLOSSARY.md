# Glossary

This glossary fixes the technical vocabulary used by `mechanopharm-infer` so
that it remains consistent with the theory paper.

Whenever a name appears in the code base, in CSV outputs, or in
`architecture_call.json`, it follows the definitions below.

## 1. Inputs and response

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| c | `c` | Dimensionless chemical concentration `c = C / C_0`. |
| m | `m` | Coarse-grained mechanical descriptor. Assay-dependent (stiffness, tension, strain, pressure, ...). |
| t | `time` / `t_peak` / `t_inf` | Time elapsed since stimulation. |
| S | proximal readout (`response` / `value` when `readout_level="proximal"`) | Primary signaling output `S = sum_i epsilon_i(m) p_i`. |
| E | phenotypic readout (`response` / `value` when `readout_level="phenotypic"`) | Downstream functional outcome (viability, growth arrest, etc.). |

The reduced two- and three-state state vectors are denoted
`p_0, p_1, p_2`.  In code:

* state 0 = `less_responsive` macrostate (baseline)
* state 1 = `responsive` macrostate (signaling-competent)
* state 2 = `protected` macrostate (adaptive / response-reducing)

## 2. State-bias and admissibility

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| Delta_G | (implicit) `TwoStateParams.delta_g0`, `delta_alpha`, `delta_lambda`, `delta_mu`, `kappa` | Effective state-bias difference of the two-state model (Eq. DeltaG). |
| Delta_alpha | `delta_alpha` | Purely chemical bias coefficient. |
| Delta_lambda | `delta_lambda`, `delta_lambda_proxy` | Purely mechanical bias coefficient.  The "proxy" form is recovered up to scale from response-landscape data. |
| Delta_mu | `delta_mu`, `delta_mu_proxy` | Mechanochemical cross term (cm coupling).  Non-zero `Delta_mu` is the minimal condition for reversal of mechanical sensitivity. |
| kappa | `kappa` | Quadratic curvature in m (enables an interior m\* in the strict two-state model). |
| local detailed balance | imposed by construction in the generators | `ln(k_ij / k_ji) = -beta * (G_j - G_i)`; see Section 2.3 of the theory paper. |

## 3. Fingerprints (theory section 5 of the manuscript)

| Theory symbol | Code field / file | Notes |
|---------------|-------------------|-------|
| `EC50(m)` | `ec50_vs_m.csv`, column `ec50` | Mechanically conditioned half-effective concentration; Eq. c12. |
| `c_rev = -Delta_lambda / Delta_mu` | `sign_reversal.csv`, `fingerprint_values.c_rev.estimate` | Reversal concentration; Eq. crev.  Regression-based estimate with bootstrap CI. |
| `m*(c)` | `mopt_vs_c.csv`, column `m_opt` | Optimal mechanical condition at fixed c; Eq. mstar-2state / mstar-opt.  Parabolic sub-grid refinement when interior. |
| `E_peak(c, m)` | `peak_metrics.csv`, column `e_peak` | Peak amplitude of the proximal signal over time. |
| `t_peak(c, m)` | `peak_metrics.csv`, column `t_peak` | Time at which the peak is reached; Eq. tpeak-rates. |
| `E_inf(c, m)` | `final_response.csv`, column `e_inf` | Long-time / endpoint response. |
| delayed protection | `delayed_protection.csv`, column `attenuation` | `E_peak - E_inf`; flagged via `delayed_protection_detected`. |

## 4. Architecture classes

| Code label | Meaning | Required minimal architecture |
|------------|---------|------------------------------|
| `two_state_supported` | Mechanically shifted dose response and/or concentration-dependent sign reversal are detected, but no protected-state signatures. | Minimal two-state model with mechanochemical coupling (Section 3 of the theory paper). |
| `protected_state_suggested` | At least one of {interior optimum, moving optimum, transient peak, delayed protection} is reliably detected. | Minimal three-state protected extension (Section 4 of the theory paper). |
| `inconclusive` | Endpoint QC failed or no signature is reliably assessable. | Architecture cannot be inferred from the supplied data. |

## 5. Evidence strength

`evidence_strength` values are ranked as

```
not_assessable < none < weak < moderate < strong
```

Each fingerprint emits its own `evidence_strength`; the architecture call is
made by combining flags through the scoring rule in
`discriminate.discriminate_architecture` (the scoring rule itself is
documented inline in that function).

## 6. Why these definitions matter

The fingerprint names are deliberately stable, theory-anchored, and
machine-readable.  CSV columns, JSON payload keys, and figure axis labels all
re-use the same vocabulary so that downstream users can quote the symbol
(`m*(c)`, `c_rev`, `E_peak`) without ambiguity in publications and figures.
