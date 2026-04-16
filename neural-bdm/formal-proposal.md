# 📄 Unified Formal Proposal (Version 2: Fully Integrated System)

## **Title**

**AMAS-Constrained Neural BDM with Recursive Reasoning Models: A Unified Framework for Representation-Invariant Algorithmic Complexity and Structural Intelligence Evaluation**

---

# 1. Unified Hypothesis

We hypothesise that:

> A representation-invariant operational proxy for algorithmic complexity can be constructed by combining:
>
> * a recursive compression model (Neural BDM),
> * a recursive reasoning architecture (TRM),
> * a structured complexity curriculum (ECA / edge-of-chaos regime),
> * and an admissibility constraint system (AMAS),

such that the resulting system behaves consistently under structure-preserving transformations and distinguishes algorithmic structure from statistical noise.

Formally:

[
\Phi_{\theta}(S) := K_{\text{Neural BDM + TRM}}(S)
]

is a valid operational complexity measure iff:

[
\forall T \in \mathcal{A}, \quad |\Phi_{\theta}(S) - \Phi_{\theta}(T(S))| < \epsilon
]

---

# 2. Foundational Theoretical Pillars

This work integrates four established research streams:

---

## 2.1 Edge-of-Chaos Computational Regimes

Empirical studies on Elementary Cellular Automata demonstrate:

* low-complexity systems → trivial dynamics
* high-complexity systems → chaotic noise
* intermediate regimes → maximal generalisation (“edge of chaos”)

This regime is consistently associated with:

> maximal learnable structure and optimal generalisation capacity

**Implication:**
We construct training distributions that explicitly span and emphasise this regime to maximise exposure to compressible-but-nontrivial structure.

---

## 2.2 Transformer Learning of Algorithmic Dynamics (ECA systems)

Prior work shows that:

* transformers can infer latent update rules from state sequences
* models trained on ECA dynamics generalise beyond memorisation
* rule complexity correlates with representational richness
* deeper models improve multi-step consistency but remain limited in long-horizon planning

**Implication:**
Neural networks can approximate **latent generative rules**, suggesting compressibility is learnable.

---

## 2.3 Recursive Compression as a Model of Intelligence (SuperARC principle)

Prior AIT-inspired work establishes that:

> intelligence correlates with the ability to compress observations into executable generative models

Key insight:

* compression ≈ abstraction
* abstraction ≈ prediction
* prediction ≈ planning

However:

* current LLMs fail under algorithmic generalisation stress
* they collapse to memorisation under high-complexity regimes

**Implication:**
Compression alone is insufficient without structural constraints.

---

## 2.4 Recursive Reasoning via Tiny Models (TRM framework)

Tiny Recursive Models demonstrate that:

* reasoning ability does not scale monotonically with parameter count
* recursive refinement is more important than depth
* small models with iterative latent updates outperform large static architectures on structured reasoning tasks

Key mechanism:

[
(y, z)*{t+1} = f*{\theta}(x, y_t, z_t)
]

**Implication:**
Recursive refinement provides a **computational mechanism for iterative structure extraction**, complementary to compression-based estimation.

---

# 3. Core System Architecture

We propose a three-layer system:

---

## 3.1 Layer 1 — Computational Universe (ECA / Structured Dynamics)

A controlled dataset spanning:

* Wolfram Class I–IV systems
* edge-of-chaos transition regimes
* full rule space coverage

This defines a **computationally complete dynamical environment** for learning structural complexity.

---

## 3.2 Layer 2 — Recursive Reasoning Model (TRM)

A compact model performing iterative refinement:

[
(y_{t+1}, z_{t+1}) = \text{TRM}(x, y_t, z_t)
]

Where:

* ( y ): solution state
* ( z ): latent reasoning state

This layer provides:

* multi-step structural inference
* implicit simulation of generative rules
* correction of local prediction errors

---

## 3.3 Layer 3 — Neural Compression Estimator (Neural BDM)

We define:

[
\Phi_{\theta}(S) := \mathcal{L}_{\text{recursive compression}}(S)
]

This acts as:

> an operational proxy for algorithmic structure via learned description length

---

# 4. AMAS Constraint System (Critical Integration Layer)

We define a transformation set ( \mathcal{A} ):

* representation transformations
* temporal transformations
* perturbation transformations

---

## 4.1 Admissibility Condition

A valid complexity estimator must satisfy:

[
\Phi(S) \approx \Phi(T(S)), \quad \forall T \in \mathcal{A}
]

This enforces:

> structure-invariance as the defining property of validity

---

## 4.2 Role of AMAS in the Unified System

AMAS acts as:

* a **filter on measurement validity**
* a **test for estimator robustness**
* a **detector of spurious structure**

It replaces the notion of:

> “unbiased estimator”

with:

> “invariant-admissible estimator”

---

# 5. Unified Training Objective

We define a joint objective over TRM + Neural BDM:

---

## 5.1 Predictive / Compression Loss

[
\mathcal{L}*{pred} = -\log P*{\theta}(S)
]

---

## 5.2 Recursive Reasoning Loss (TRM)

[
\mathcal{L}*{reason} = \sum*{t} ||y_t - y^*_t||
]

---

## 5.3 AMAS Invariance Constraint

[
\mathcal{L}*{inv} = \mathbb{E}*{T \sim \mathcal{A}} \left[ |\Phi(S) - \Phi(T(S))| \right]
]

---

## 5.4 Total Objective

[
\mathcal{L} = \mathcal{L}*{pred} + \alpha \mathcal{L}*{reason} + \lambda \mathcal{L}_{inv}
]

---

# 6. Core Scientific Claim (Corrected Form)

We explicitly DO NOT claim:

* that Neural BDM computes true Kolmogorov complexity
* that LLMs approximate universal intelligence
* that edge-of-chaos guarantees optimal learning
* that the system is unbiased in a classical statistical sense

---

We DO claim:

> A recursive compression + reasoning system trained over a computationally complete dynamical spectrum and constrained by AMAS invariance yields the most expressive and structurally stable operational proxy for algorithmic complexity currently achievable.

---

# 7. Key Integration Insight (This is the real contribution)

This framework unifies three previously separate ideas:

---

## 7.1 TRM contributes:

> iterative reasoning / refinement dynamics

## 7.2 Neural BDM contributes:

> compression-based structural measurement

## 7.3 AMAS contributes:

> validity constraints over representations

---

## 7.4 Resulting object:

> A **constraint-based measurement system for algorithmic structure in dynamical systems**

---

# 8. Why Edge-of-Chaos Matters Here (Correct Role)

Edge-of-chaos is NOT used to prove unbiasedness.

It is used to ensure:

> the training distribution contains maximal identifiable structure for compression and reasoning systems to exploit

So it functions as:

* a **data regime design principle**
* not a theoretical justification of estimator correctness

---

# 9. Final Interpretation

This system reframes algorithmic intelligence measurement from:

> heuristic compression + heuristic reasoning

to:

> a constrained, recursive, invariance-validated measurement system operating over a computationally complete dynamical universe

---

# 🧠 Critical correction (important honesty layer)

This is now internally consistent, but:

### Still NOT established truths:

* Neural BDM is not proven unbiased
* TRM is not proven optimal reasoning architecture
* AMAS is not a known mathematical standard yet
* edge-of-chaos is not a guarantee of optimality

---

### What *is* defensible:

> You are constructing a **structured experimental and theoretical framework for evaluating algorithmic complexity estimators under invariance constraints**

That is:

* publishable as a framework paper
* testable
* falsifiable
* extensible

---

# 🚀 If you want next step

We should now decide which direction to lock in:

### 1. “Paper mode”

Turn this into NeurIPS-style:

* theorem statements
* experimental protocol
* baseline comparisons

### 2. “System mode”

Define:

* exact AMAS algebra
* TRM implementation spec
* training pipeline

### 3. “Falsification mode” (recommended)

Design:

> the set of experiments that could actually break this system

That’s what will make it real science rather than a strong architecture proposal.
