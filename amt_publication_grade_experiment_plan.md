# Algorithmic Mesoscope Thesis (AMT)
## Publication-Grade Experimental Plan

---

# 1. Objective

To rigorously test whether structural breaks correspond to a collapse in mesoscopic compression anisotropy across symbolic projections of time-series data.

This document defines falsifiable hypotheses, robustness criteria, statistical validation standards, control experiments, and modular test harness architecture.

---

# 2. Operational Definitions

Given a time series partitioned into two segments S0 and S1:

Let a transduction lens L map real-valued segments into symbolic binary tapes T0 and T1.

Let G be a generator class (e.g., cellular automata rules).

Define per-segment compression density:

    rho_i = min_{g in G} K(T_i | g) / |T_i|

Define mesoscopic anisotropy:

    A = |rho_0 - rho_1|

Primary Thesis Claim:

    Structural breaks correspond to statistically significant collapse in A.

---

# 3. Pre-Registered Hypotheses

H1 — Anisotropy Collapse

    E[A | Break] < E[A | No Break]

H2 — Predictive Separability

    Divergence-based classifier yields ROC AUC > 0.60 on held-out test data.

H3 — Lens Robustness

    H1 holds across at least three distinct symbolic lenses.

H4 — Generator Robustness

    H1 holds under full ECA generator class (rules 0–255).

H5 — Generator Instability

    P(g0 != g1 | Break) > P(g0 != g1 | No Break)

Failure of H1–H4 falsifies the AMT claim for this dataset.

---

# 4. Dataset Protocol

Repository:
    algoplexity/computational-phase-transitions-data

Data Split:
    Train: 70%
    Validation: 10%
    Test: 20%

Requirements:
- Stratified by label
- No test leakage
- Hyperparameters fixed using validation only

---

# 5. Transduction Lenses (Robustness Axis 1)

Minimum Required:

L1 — Delta-Sign
    t = 1 if delta x >= 0 else 0

L2 — Mean Threshold
    t = 1 if x > mean(x) else 0

L3 — Rolling Z-Score Threshold
    t = 1 if z_t > 0 else 0

Optional (Stronger Test):
- Quantile binarization
- Volatility sign encoding

AMT must not depend on a single projection.

---

# 6. Generator Classes (Robustness Axis 2)

G1 — Minimal Exploratory Basis (9 rules)
G2 — Full Elementary Cellular Automata (0–255)
G3 — Random Boolean Rule Ensemble (sanity check)

Effect must persist under G2 to support structural claim.

---

# 7. Per-Sample Metrics

For each sample:

- rho_0
- rho_1
- A = |rho_0 - rho_1|
- best_rule_0
- best_rule_1
- rule_switch = 1 if best_rule_0 != best_rule_1

---

# 8. Statistical Validation

## 8.1 Effect Size

Report:
- Mean(A) per class
- Mean difference
- Cohen’s d
- 95% bootstrap confidence interval (10,000 resamples)

Reject H1 if CI strongly overlaps zero.

---

## 8.2 ROC Evaluation (Test Set Only)

Compute:
- ROC AUC
- 95% bootstrap CI
- Compare to random baseline (0.5)

Use DeLong test for AUC comparison when applicable.

---

## 8.3 Robustness Matrix

| Lens | Generator | Mean Δ | AUC | H1 Pass? |
|------|-----------|--------|-----|----------|
| L1   | G1        |        |     |          |
| L1   | G2        |        |     |          |
| L2   | G2        |        |     |          |
| L3   | G2        |        |     |          |

Broad consistency required for structural claim.

---

# 9. Control Experiments

## 9.1 Label Shuffle Test

Randomly permute labels.

Expected result:
    AUC approximately 0.5

If not, pipeline contains leakage or bias.

---

## 9.2 Segment Swap Test

Randomly swap partitions within sample.

If anisotropy signal persists, effect is segmentation artifact.

---

## 9.3 Length Confound Test

Regress A against segment length difference.

If divergence explained by length imbalance, reject structural interpretation.

---

# 10. Computational Plan

Full ECA sweep:
- 256 rules per segment
- Vectorized bit operations
- Early pruning when residual exceeds current minimum

No GPU required.
CPU cluster sufficient.

---

# 11. Reproducible Test Harness Architecture

Suggested modular structure:

    data_loader.py
    splits.py
    lenses.py
    generators.py
    anisotropy.py
    statistics.py
    controls.py
    experiment_runner.py

Outputs:
- Per-sample metrics CSV
- ROC curves
- Bootstrap distributions
- Robustness heatmap

---

# 12. Minimal Publication-Grade Claim

If validated:

    Structural breaks correspond to collapse in mesoscopic compression anisotropy across symbolic projections, robust to generator basis and lens selection.

If robustness fails:

    AMT is not supported on this dataset.

---

# 13. Explicit Falsification Conditions

AMT fails if:
- Mean divergence difference negligible
- AUC < 0.55 under full ECA
- Effect disappears under alternate lenses
- Label shuffle produces comparable separation

These conditions must be reported transparently.

---

End of Experimental Plan

