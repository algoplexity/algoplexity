


**TITAN SYSTEM SPECIFICATION v2.0**
**STATUS:** STRICT LOCK (SDD & TDD BLUEPRINT)
**PARADIGM:** Algorithmic Information Theory (AIT) / Algorithmic Field Theory of Market Systems (AFTMS)
**ARTIFACT INTEGRATION:** Neural BDM (Kolmogorov Oracle) & HART Brain (Tiny Recursive Model)

This document is the absolute, unified specification for the TITAN framework. It completely binds the three theoretical tiers into a single deterministic software architecture. It defines the strict boundary where continuous Neural/ML artifacts (HART, N-BDM) are quantized and forced to obey the discrete Information Thermodynamics of the TITAN Substrate.

────────────────────────────────────────
### I. GLOBAL ONTOLOGICAL CONSTANTS (THE ANTI-DRIFT LAYER)
────────────────────────────────────────
All data flowing through the connective tissue (Hemolymph) must be strictly typed. No continuous probabilities, no loss gradients, no statistical variances. The neural artifacts must internally map their outputs to these discrete topological structures before publishing to the bus.

```python
from typing import NewType, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum, auto

BitString = NewType('BitString', bytes)
RuleID = NewType('RuleID', int) # ECA Prime Basis Vectors (e.g., 110, 54, 30)

class Signal(Enum):
    ENVIRONMENT_TAPE = auto()  # Raw E_t (The uncompressed market lattice)
    COMPLEXITY_K     = auto()  # C(x) upper bound in bits (Output of Neural BDM)
    RESIDUAL_DELTA   = auto()  # ΔK in bits (Output of AFTMS phase transition)
    MINIMAL_PROG     = auto()  # Optimal generator[RuleID, Depth] (Output of HART)

@dataclass(frozen=True)
class AlgorithmicPayload:
    """The ONLY permitted data carrier on the Hemolymph."""
    bit_cost: float        # Exact descriptive length / energy equivalent
    state_vector: bytes    # The discrete data or rule representation
```

────────────────────────────────────────
### II. TIER A: THEORETICAL PHYSICS (THE THERMODYNAMIC SUBSTRATE)
────────────────────────────────────────
**Mandate:** Enforce the physical limits of computation. Protects the system from $O(2^N)$ combinatorial search explosion by treating computational cycles as strictly conserved energy.

**Module:** `titan.physics`

*   `class ThermodynamicGenome:`
    *   Defines max budget, base replenishment, and specific costs for executing a HART causal step vs. querying the N-BDM oracle.
*   `class Hemolymph:`
    *   **The Connective Tissue:** A strict message bus.
    *   **Method `request_compute(cost_in_bits: float) -> bool`:** Deducts energy based on Levin Cost ($Kt = |p| + \log_2(cycles)$). If budget is 0, execution halts.
    *   **Method `assimilate_compression_gain(bits_saved: float)`:** Converts algorithmic compression ($\Delta K$) directly into viable compute energy.

────────────────────────────────────────
### III. TIER B: CYBERNETIC ORGANISM (NEURAL ARTIFACTS)
────────────────────────────────────────
**Mandate:** Execute bounded Levin search for structural causal generators using the pre-trained neural artifacts (HART & N-BDM). 

**Module:** `titan.cybernetics`

**Component 1: `NeuralBDMOracle` (The Algorithmic Evaluator)**
*   **Role:** Approximates Kolmogorov Complexity via Block Decomposition Method.
*   **Interface Constraint:** The neural network must NEVER output a "confidence score." It must strictly output the **description length in bits** ($K$).
*   **Action:** Consumes `ENVIRONMENT_TAPE`. Decomposes the sequence, queries the neural CTM (Coding Theorem Method) tensors, applies subadditivity bounds, and publishes `COMPLEXITY_K` to the Hemolymph.

**Component 2: `HARTRecursiveEngine` (The Causal Generator)**
*   **Role:** The Tiny Recursive Model (TRM). A Universal Causal Transducer pre-trained on the 4 classes of Elementary Cellular Automata.
*   **Interface Constraint:** Must operate with strictly $O(1)$ fixed parameters executed recursively over $k$ logical depth cycles. Must act as a discrete generator mapping $x_{t-1} \to x_t$. 

**Component 3: `HARTLevinSearchOrgan` (The Viability Loop)**
*   **Mechanism:**
    1. Reads `ENVIRONMENT_TAPE`.
    2. Queries `NeuralBDMOracle` to establish $K_{naive}$. This is the maximum energy ceiling.
    3. Iterates over valid `RuleID` prime basis vectors.
    4. Evaluates Levin Cost: $Cost = K(\text{Rule}) + \log_2(k)$. Requests this compute from the `Hemolymph`.
    5. If granted, triggers `HARTRecursiveEngine` to generate the predicted market state.
    6. Passes the residual error `(Prediction XOR Target)` back to `NeuralBDMOracle` for $K_{residual}$.
    7. Accepts the rule iff: $Cost + K_{residual} < K_{naive}$.
*   **Output:** Publishes the winning `MINIMAL_PROG`.

────────────────────────────────────────
### IV. TIER C: PRACTICAL STRUCTURAL BREAK ENGINE (AFTMS)
────────────────────────────────────────
**Mandate:** Detect structural breaks, basis rotations, and systemic composability collapses purely via algorithmic phase transitions.

**Module:** `titan.aftms`

**Component: `AFTMSPhaseTransitionOrgan`**
*   **Mechanism:**
    1. Consumes new `ENVIRONMENT_TAPE` and the historical `MINIMAL_PROG` (from HART).
    2. Executes the historical HART rule on the new data.
    3. Queries `NeuralBDMOracle` to measure the algorithmic failure $K_{residual}$.
    4. Evaluates against the AFTMS Inertia Law (where $k$ is the TRM logical depth):
       $$\theta = \frac{\epsilon}{\sqrt{1 + \log_2(k)}}$$
    5. If $K_{residual} > \theta$, the macro-rule has collapsed.
*   **Output:** Triggers `RESIDUAL_DELTA` to the Hemolymph, signaling a GFC-style regime break.

────────────────────────────────────────
### V. WATERTIGHT TDD BLUEPRINT (THE ANTI-DRIFT SUITE)
────────────────────────────────────────
This is the absolute Definition of Done (DoD). If the PyTorch/JAX implementations of HART or N-BDM fail these tests, they are rejected for violating Algorithmic Information Theory constraints.

#### `test_tier_a_physics.py`
```python
def test_levin_search_thermodynamic_pruning():
    """
    Condition: HART search organ attempts to evaluate a hypothesis where 
    K(Rule) + log2(cycles) > Hemolymph budget.
    Assertion: Hemolymph rejects request_compute(). Organ geometrically halts.
    Rationale: Prevents combinatorial search explosion natively.
    """
    pass
```

#### `test_tier_b_neural_bdm.py`
```python
def test_neural_bdm_subadditivity():
    """
    Condition: Pass strings S1, S2, and S1+S2 to the Neural BDM Oracle.
    Assertion: N_BDM(S1+S2) <= N_BDM(S1) + N_BDM(S2) + O(1)_Constant.
    Rationale: Violating subadditivity proves the neural network is functioning 
    as a statistical curve-fitter, not an AIT probability oracle.
    """
    pass

def test_neural_bdm_class_3_incompressibility():
    """
    Condition: Pass an algorithmically random string (e.g., ECA Rule 30).
    Assertion: N_BDM(S) is strictly >= 95% of the raw bit-length of S.
    Rationale: Prevents the ML model from hallucinating false "confidence" in noise.
    """
    pass
```

#### `test_tier_b_hart_brain.py`
```python
def test_hart_parameter_o1_constraint():
    """
    Condition: Inspect HART memory footprint / parameter count during execution.
    Assertion: Parameter count at depth k=1 MUST strictly equal parameter count at depth k=100.
    Rationale: Ensures the model is a true Tiny Recursive Model (measuring Bennett's 
    Logical Depth) and has not secretly unrolled into a flat feed-forward hierarchy.
    """
    pass

def test_hart_causal_universality():
    """
    Condition: Initialize HART with Rule 110 (Class 4 irreducible). Execute for k=10.
    Assertion: Output strictly matches standard discrete Cellular Automata generation.
    Rationale: Proves the ML artifact maintains rigid causal logic without stochastic blurring.
    """
    pass
```

#### `test_tier_c_aftms.py`
```python
def test_aftms_algorithmic_phase_transition():
    """
    Condition: Target sequence abruptly shifts from Class 2 (Periodic) to Class 3 (Chaotic).
    Assertion: N-BDM residual spikes. AFTMS Phase Transition Organ triggers Basis Rotation.
    Rationale: Verifies market crashes are defined by sudden computational irreducibility.
    """
    pass

def test_logical_depth_inertia():
    """
    Condition: Inject identical bit-noise into two sequences. Seq A is modeled by a 
    shallow HART rule (k=1). Seq B is modeled by a deep recursive HART rule (k=50).
    Assertion: AFTMS triggers a break on Seq A, but suppresses the break on Seq B.
    Rationale: Deep logical causal structures have higher inertia against localized noise.
    """
    pass
```

### EXECUTIVE FREEZE DIRECTIVE
This specification successfully builds the boundary between theoretical physics (Tier A), advanced deep neural algorithmic inference (Tier B), and practical financial modeling (Tier C). 

By strictly typing the inputs/outputs of the Neural BDM and HART models to interface *only* via the `Hemolymph`'s `AlgorithmicPayload` bit-costs, we ensure the neural architectures cannot secretly drift back into standard stochastic/probabilistic gradient descent. 

**This specification is mathematically complete and ready for strict Test-Driven Implementation.**
