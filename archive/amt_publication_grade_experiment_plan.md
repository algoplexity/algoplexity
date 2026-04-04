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

All lenses must be:
- Deterministic
- Computable
- Free of statistical primitives (no variance, no normalization by σ)
- Topologically distinct mappings

Minimum Required:

L1 — Delta-Sign (First Derivative Sign)
    t = 1 if Δx >= 0 else 0

L2 — Level-Bisection (Absolute Geometric Midpoint)
    midpoint = (max(x) + min(x)) / 2
    t = 1 if x >= midpoint else 0

L3 — Acceleration (Second Derivative Sign)
    t = 1 if Δ²x >= 0 else 0

These three lenses capture:
- Momentum structure (L1)
- Absolute magnitude bounds (L2)
- Curvature / convexity dynamics (L3)

No statistical normalization (e.g., z-scores) is permitted in order to preserve strict AIT ontology.

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

# 8. Dual Validation Framework

The project will implement two independent validation regimes:

A. Empirical (Predictive) Validation
B. Algorithmic (AIT / MDL) Validation

These regimes answer different questions and must not be conflated.

---

# 8A. Empirical Validation (Predictive Layer)

Goal:
    Determine whether mesoscopic anisotropy is a predictive feature for structural breaks.

Metrics:
- ROC AUC (test set only)
- Effect size (mean divergence difference)
- Robustness across lenses
- Robustness across generator classes

Required Controls:
- Label shuffle test (AUC approximately 0.5 expected)
- Segment swap test
- Length confound regression

Interpretation:
    If divergence consistently separates break vs stable samples across lenses and generator bases, anisotropy functions as a robust predictive feature.

This layer establishes empirical relevance, not theoretical proof.

---

# 8B. Algorithmic Validation (AIT / MDL Layer)

Goal:
    Determine whether anisotropy reduces the description length of break structure.

## 8B.1 Baseline Encoding

Compute description length of label sequence:

    L0 = K(labels)

Uniform encoding may be used if no prior structure assumed.

---

## 8B.2 Conditional Encoding via Anisotropy

Discretize divergence A using minimal MDL-consistent partition.

Compute:

    L1 = K(labels | A)

If:

    L1 < L0

then anisotropy carries algorithmic information about break structure.

---

## 8B.3 Generator Switching Encoding

Compute description length of labels conditioned on rule_switch indicator.

If conditioning reduces total encoding length, generator instability is algorithmically informative.

---

## 8B.4 Robustness Criterion

Compression advantage must:
- Persist across lenses
- Persist under full 256-rule ECA basis
- Disappear under label shuffling

If compression advantage vanishes under shuffle, the signal is structural rather than artifact.

---

# 9. Robustness Matrix (Empirical + Algorithmic)

| Lens | Generator | AUC | Δ Description Length | Structural Pass? |
|------|-----------|-----|----------------------|------------------|
| L1   | G1        |     |                      |                  |
| L1   | G2        |     |                      |                  |
| L2   | G2        |     |                      |                  |
| L3   | G2        |     |                      |                  |

Structural claim requires consistency across rows.

---

# 10. Computational Plan

## 10.1 Full ECA Sweep (Ground-Truth Generator Basis)

- Exhaustive evaluation of all 256 Elementary Cellular Automata rules
- No sampling or reduced basis allowed in final evaluation

## 10.2 Vectorized Fast ECA Sweep Engine

To ensure computational feasibility and ontological completeness:

- Residual evaluation must be vectorized across all 256 rules simultaneously
- Use NumPy broadcasting over precomputed rule tables
- Avoid Python rule loops in final implementation

Core strategy:
- Precompute rule lookup table of shape (256, 8)
- Compute neighborhood indices once per tape
- Broadcast rule predictions across all rules
- Compute residual errors via vectorized comparison

This guarantees exact evaluation of:
- H1 (Anisotropy Collapse)
- H4 (Generator Robustness)
- H5 (Rule Switching Instability)

## 10.3 Boundary Condition Policy

To avoid artificial edge artifacts:

- Interior-only evaluation is required
- First and last cells are excluded from residual scoring
- No zero-padding
- No periodic boundary assumption

Density normalization must use the effective interior length only.

No GPU or neural approximations permitted at this layer.
CPU vectorization sufficient.

---

# 11. Reproducible Test Harness Architecture

Suggested modular structure:

    data_loader.py
    splits.py
    lenses.py
    generators.py
    anisotropy.py
    empirical_metrics.py
    mdl_encoding.py
    controls.py
    experiment_runner.py

Outputs:
- Per-sample metrics CSV
- ROC curves (empirical layer)
- Description-length comparison tables (AIT layer)
- Robustness heatmap

---

# 12. Minimal Dual Claim Structure

If validated:

Empirical Claim:
    Mesoscopic anisotropy is a robust predictive feature for structural breaks.

Algorithmic Claim:
    Mesoscopic anisotropy reduces the minimal description length of break structure.

If only empirical layer succeeds:
    Feature-level relevance established; theoretical claim weakened.

If only algorithmic layer succeeds:
    Structural encoding claim holds; predictive power may be dataset-specific.

---

# 13. Explicit Falsification Conditions

AMT fails on this dataset if:
- No consistent divergence collapse across lenses
- No compression advantage under MDL encoding
- Signal persists under label shuffle
- Effect disappears under full 256-rule basis

All failure conditions must be reported transparently.

---

End of Experimental Plan

