# 📋 FUNCTIONAL SPECIFICATION: TITAN CYBERNETIC ORGANISM (v1.0)

**Project:** Titan Universal Algorithmic Microscope
**Status:** **DRAFT FOR APPROVAL**
**Architectural Style:** Cybernetic / Component-Based
**Artifact Dependency:** `universal_brain_titan.pt` (Frozen H-ART Weights)

---

## 1. Overview & Problem Statement

### 1.1 The Problem
Financial markets and complex systems exhibit **Structural Breaks** (Regime Changes) that purely statistical models (Z-Score, Correlation) fail to detect because they assume stationarity. Systems like the Global Financial Crisis (GFC) are not "high variance" events; they are **Algorithmic State Transitions** (e.g., from Random Walk to Liquidity Lock).

### 1.2 The Solution
Titan v1.0 is a **Strategic Organism** that metabolizes data. It uses a pre-trained **Universal Prior (Neural BDM)** to measure the algorithmic complexity of a data stream. It autonomously adapts its "Eye" (Lens) to minimize entropy and uses an "Algebraic Mind" to deconvolve the signal into a deterministic **ECA Source Equation** (e.g., $R_{170} \circ R_{128}$).

---

## 2. Goals & Non-Goals

### 2.1 Core Goals
1.  **G1 - Metabolic Closure (SO):** The system must operate within a finite energy budget. Every operation (sensing, thinking, adapting) has a defined cost. If `Budget < Cost`, the operation is rejected.
2.  **G2 - Invariant Reference (UAI):** Complexity ($K$) is measured against a **Frozen Artifact** (`universal_brain_titan.pt`). The Brain *never* trains at runtime; it only judges.
3.  **G3 - Causal Deconvolution (AID):** The system must distinguish **Signal** from **Noise** using Perturbation Analysis (MILS), identifying bits that are "Load Bearing" (Structural).
4.  **G4 - Compositional Discovery (ECA):** The output must be a composition of Primitive Rules ($f \circ g$), not a single best-fit rule.

### 2.2 Non-Goals
*   **NG1:** Predicting the *price* of the next tick. (We predict the *rule* governing the next tick).
*   **NG2:** Real-time trading execution.
*   **NG3:** Architecture search for the Neural Brain. (The architecture is fixed: H-ART).

---

## 3. System Architecture (The "Anatomy")

The system consists of three active **Organs** suspended in a passive **Substrate**.

### 3.1 The Substrate: `Hemolymph`
*   **Definition:** The shared state container acting as the connective tissue.
*   **Responsibilities:**
    *   Holds the **Resource Budget** (Energy).
    *   Holds the **Signal Bus** (BioSignals).
    *   Holds the **Endocrine State** (Hormones: Stress, Confidence, Shock).
*   **Constraint:** Organs cannot communicate directly. They must broadcast to the Hemolymph.

### 3.2 System 1: The Transducer (`AutonomicEye`)
*   **Role:** Active Inference / Sensation.
*   **Input:** Raw Float Data.
*   **Output:** `SignalType.PERCEPT` (A view of the data, scaled and projected).
*   **Mechanism:**
    *   **Lens:** Differentiable A-Law ($y = \text{sigmoid}(A \cdot \ln(1 + |x|))$).
    *   **Reflex:** If `Hormone.STRESS > Threshold`, executes **Test-Time Training (TTT)** to optimize lens parameters against the Brain's feedback.

### 3.3 System 3: The Estimator (`NeuralCritic`)
*   **Role:** Monitoring / Valuation.
*   **Input:** `SignalType.PERCEPT`.
*   **Output:** `SignalType.CONCEPT` (Feedback Gradient) & `Hormone` secretion.
*   **Mechanism:**
    *   Wraps the **Frozen H-ART Brain**.
    *   Implements **Soft-Embedding Bridge** (Linear projection) to allow gradients to flow from the Digital Brain back to the Analog Eye.
    *   Calculates **Energy ($K$)** via Negative Log-Likelihood (NLL).

### 3.4 System 4: The Decomposer (`AlgebraicMind`)
*   **Role:** Intelligence / Solving.
*   **Input:** Stabilized Percepts.
*   **Output:** Source Equation String (e.g., "R128").
*   **Mechanism:**
    *   **MILS:** Perturbation analysis to weight the bits.
    *   **Levin Search:** Iterative composition of 38 Zenil Primes to find the generator.
*   **Constraint:** Only runs if `Hormone.CONFIDENCE > Threshold` (Metabolic Gating).

---

## 4. Technical Specifications (The "Physics")

### 4.1 The H-ART Artifact
*   **Path:** `universal_brain_titan.pt`
*   **Dimensions:** `Dim=64`, `Heads=4`, `Seq=256`.
*   **Topology:** 4 Recursive H-Cycles, 2 Lateral L-Cycles.
*   **Validation:** Must pass the **Isomorphism Test** ($K_{Order} < K_{Complex} < K_{Chaos}$) upon loading.

### 4.2 The Metabolic Economy
*   **Energy Unit (EU):** 1.0 EU = Cost of 1 Forward Pass of the Brain.
*   **Income:** Derived from **Negative Entropy**.
    $$ \text{Income} = \eta \cdot (1.0 - \rho) $$
    Where $\rho = \frac{K_{measured}}{K_{max}}$.
*   **Costs:**
    *   Eye Existence: 1.0 EU.
    *   Eye Saccade: 10.0 EU (TTT optimization).
    *   Brain Judge: 2.0 EU.
    *   Mind Solve: 20.0 EU.

---

## 5. Verification Plan

### 5.1 Unit Tests (Isolation)
1.  **Hemolymph Test:** Verify `attempt_work()` returns False when budget is low.
2.  **Eye Test:** Verify `alpha` changes when a fake gradient is injected.
3.  **Brain Test:** Verify `universal_brain.pt` loads and outputs High $K$ for random noise and Low $K$ for Rule 110.
4.  **Mind Test:** Verify `AlgebraicMind` solves `[0,1,0,1,0] -> [0,0,0,0,0]` as **Rule 128**, differentiating it from Rule 170 via MILS weighting.

### 5.2 Integration Test (The Gauntlet)
*   **Scenario:** S&P 500 GFC (Sept 2008).
*   **Success Criteria:**
    1.  **T<10:** Stress Spikes, Eye Saccades (Alpha moves).
    2.  **T>15:** Stress Drops, Confidence Rises (Lock achieved).
    3.  **T>20:** Mind outputs `R128` or composite.
    4.  **Survival:** Budget remains $>0$.

---
