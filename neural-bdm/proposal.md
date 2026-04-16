# 📄 Project Proposal

## **Neural BDM under AMAS Constraints: Toward a Best-Available Operational Proxy for Kolmogorov Complexity**

---

## 1. Executive Summary

We propose a constrained framework for developing and validating **Neural BDM**, a neural estimator of algorithmic complexity, as a **best-available operational proxy** for Kolmogorov Complexity.

Rather than attempting to approximate true Kolmogorov Complexity (which is uncomputable), we redefine the problem as:

> Constructing a *measurement system* whose outputs are stable under a formally defined set of structure-preserving transformations.

To achieve this, we introduce an **Admissibility Constraint System (AMAS)** that defines which transformations should not alter measured complexity, and we train Neural BDM within a **controlled computational universe of Elementary Cellular Automata (ECA)** spanning the full spectrum of algorithmic regimes.

The result is a **constrained, testable, and invariant-aware estimator of algorithmic structure**, rather than an unconstrained predictor.

---

## 2. Motivation

Current approaches to measuring complexity in sequential systems rely on:

* entropy-based metrics
* compression algorithms (e.g. LZ-family)
* representation-dependent neural predictors

These methods fail in key ways:

* sensitivity to encoding and representation
* inability to distinguish structure vs randomness
* poor generalisation across dynamical regimes
* instability under distribution shifts

This leads to a fundamental problem:

> We do not currently have a **representation-invariant operational measure of algorithmic structure**.

---

## 3. Key Hypothesis

We hypothesise that:

> A neural estimator trained over a complete computational spectrum and constrained by admissibility invariance can serve as the most reliable operational proxy for algorithmic complexity currently achievable.

---

## 4. Core Idea

We redefine Neural BDM not as an estimator of truth, but as:

> **A structure-preserving measurement operator validated by invariance constraints**

Formally:

[
\Phi_{\theta}(S) := K_{\text{Neural BDM}}(S)
]

where Φ is valid only if it satisfies AMAS constraints:

[
\Phi(T(S)) \approx \Phi(S), \quad \forall T \in \mathcal{A}
]

---

## 5. AMAS (Admissibility Constraint System)

We define a set of admissible transformations:

### 5.1 Representation transformations

* binary ↔ symbolic encoding
* permutation of state labels
* embedding reparameterisation

### 5.2 Temporal transformations

* time scaling
* subsequence extraction
* window shifting

### 5.3 Perturbation transformations

* bounded noise injection
* partial masking
* stochastic dropout of observations

---

## 6. Dataset Design: Computational Spectrum Coverage

We construct a training and evaluation dataset based on **Elementary Cellular Automata (ECA)** covering:

* Wolfram Class I (order)
* Class II (periodic structure)
* Class III (chaos)
* Class IV (edge of chaos / complex structure)

Each system is sampled to ensure:

* full coverage of rule space
* stratified complexity distribution
* inclusion of near-critical regimes (edge of chaos emphasis)

This ensures the estimator is exposed to the full range of algorithmic behaviours.

---

## 7. Training Objective

Neural BDM is trained using a dual objective:

### 7.1 Predictive loss (compression surrogate)

[
\mathcal{L}*{pred} = -\log P*{\theta}(S)
]

### 7.2 Invariance regularisation (AMAS constraint)

[
\mathcal{L}*{inv} = \mathbb{E}*{T \sim \mathcal{A}} \left[ \left| \Phi(S) - \Phi(T(S)) \right| \right]
]

### Combined objective:

[
\mathcal{L} = \mathcal{L}*{pred} + \lambda \mathcal{L}*{inv}
]

---

## 8. Evaluation Framework

We evaluate Neural BDM across four axes:

### 8.1 Invariance robustness

* stability under AMAS transformations
* encoding independence

### 8.2 Complexity ordering consistency

* monotonicity across ECA classes
* correlation with known proxies (LZ, entropy, Lyapunov)

### 8.3 Structural sensitivity

* ability to distinguish:

  * same statistics, different generators
  * different statistics, same generator

### 8.4 Regime transition detection

* sensitivity to phase transitions (edge of chaos)
* detection of structural breaks in dynamics

---

## 9. Key Claim (Reframed Correctly)

We do NOT claim:

* exact estimation of Kolmogorov complexity
* universality
* unbiasedness

We DO claim:

> Neural BDM is the most expressive and empirically constrained operational proxy for algorithmic complexity, validated through invariance under a formally defined admissibility system.

---

## 10. Expected Contributions

### 10.1 Theoretical

* formalisation of admissibility constraints for complexity measurement
* reframing of neural compression as invariant structure estimation

### 10.2 Methodological

* dataset construction over full computational spectrum
* invariance-driven training objective

### 10.3 Empirical

* robustness benchmarks across ECA regimes
* comparison with classical compression metrics

---

## 11. Significance

This work reframes algorithmic complexity estimation from:

> “approximating an uncomputable quantity”

to:

> “constructing a constrained measurement system with invariant behaviour under admissible transformations”

This shift aligns complexity estimation with principles seen in physical measurement systems:

* invariance under transformation
* operational reproducibility
* controlled observational regimes

---

## 12. Closing Statement

Neural BDM is not proposed as a final estimator of truth, but as:

> **a constrained, invariant-aware measurement operator that represents the strongest currently feasible operational proxy for algorithmic complexity.**

---

# 🧠 Critical insight (important)

This proposal succeeds because it:

* removes the “bias/unbiased” argument entirely
* replaces it with **invariance as the correctness criterion**
* turns Neural BDM from a claim → a *testable instrument*

---

# 🚀 If you want next step

We can now move into one of these high-impact directions:

### 1. Formal paper version (NeurIPS / ICLR style)

Turn this into:

* abstract
* related work positioning
* theorem-style sections

### 2. Exact AMAS formalisation

Define:

* transformation group structure
* invariance proofs (or empirical invariance conditions)

### 3. Training system design

* architecture (TRM / transformer variant)
* loss scheduling
* curriculum over ECA classes


