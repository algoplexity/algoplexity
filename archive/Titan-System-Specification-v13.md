


# TITAN SYSTEM SPECIFICATION v3.1

## STATUS: STRICT LOCK (SDD & TDD BLUEPRINT)

## PARADIGM: Algorithmic Information Theory (AIT) / Algorithmic Field Theory of Market Systems (AFTMS)

## ARTIFACT INTEGRATION: Neural BDM (Kolmogorov Oracle) & HART Brain (Tiny Recursive Model)

## EXTENSION: Causal Decomposition & Prime Rule Composition (The Minimal Generating Substrate)

This document formally patches the TITAN architecture, deprecating the ontologically ambiguous "Symbolic Source Equation" construct. It locks the generalized hypothesis class ($H_{COMP}$) strictly to the **Boolean Composition of Prime Elementary Cellular Automata**, as defined by the Riedel-Zenil causal decomposition theorem.

All stochasticity, continuous mathematics, and arbitrary algebraic grammars are strictly prohibited. The environment is generable strictly through the sequential computational composition of prime filters and shifters.

────────────────────────────────────────

### I. GLOBAL ONTOLOGICAL CONSTANTS (ANTI-DRIFT LAYER)

────────────────────────────────────────

The semantic data structures must mathematically enforce the compositional grammar. Arbitrary AST operators (ADD, SUB, SHIFT) are excised.

```python
from typing import NewType, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

BitString = NewType('BitString', bytes)
ECA_ID = NewType('ECA_ID', int)             # [0, 255]
PrimeRuleID = NewType('PrimeRuleID', int)   # Restricted to the 38 minimal primes

class Signal(Enum):
    ENVIRONMENT_TAPE = auto()
    COMPLEXITY_K     = auto()
    RESIDUAL_DELTA   = auto()
    MINIMAL_PROG     = auto()

@dataclass(frozen=True)
class CompositionalGenerator:
    """
    Replaces GeneratorEncoding. 
    Strictly enforces rule generation via sequential causal composition.
    Example: Rule 110 =[170, 15, 118]
    """
    composition_chain: Tuple[PrimeRuleID, ...] 
    logical_depth: int
    
@dataclass(frozen=True)
class AlgorithmicPayload:
    bit_cost: float
    state_vector: bytes
```

**Constraint:** The hypothesis space is restricted. A valid causal generator $G$ is strictly defined as $G = P_n \circ P_{n-1} \circ \dots \circ P_1$, where $P_i \in \{38\ Minimal\ Primes\}$. 

────────────────────────────────────────

### II. TIER A: THEORETICAL PHYSICS (THERMODYNAMIC SUBSTRATE)

────────────────────────────────────────

The Levin Cost ($Kt$) definition is formally updated to penalize the structural complexity ($|p|$) of compositional chains.

*   $K(P_i) = 8 \text{ bits}$ (Physical cost of one prime instruction)
*   $|p| = \text{length}(composition\_chain) \times 8 \text{ bits}$

**Updated Levin Bound Equation:**
$Kt = (|p|) + \log_2(k) + K(x|p)$

A composite rule (e.g., $170 \circ 15 \circ 118$) has a baseline structural cost of 24 bits. It will only be selected over a single prime rule if its ability to compress the residual $K(x|p)$ via the Neural Oracle exceeds the 16-bit physical penalty of requiring two additional compositional steps.

────────────────────────────────────────

### III. TIER B: CYBERNETIC ORGANISM (CAUSAL COMPOSITION SPACE)

────────────────────────────────────────

**Mandate:** Perform Bounded Levin Search over the combinatorial space of Prime ECA compositions.

**Component 1: NeuralBDMOracle (The Isomorphic Mapper)**

*   **Constraint:** The pre-trained `universal_brain_titan.pt` physically encodes the Universal Distribution over the 256 ECA discrete dynamics.
*   **Mechanism:** To evaluate a `CompositionalGenerator`, the system must:
    1.  Compute the extensional Boolean truth table of the `composition_chain`.
    2.  If the composition maps to a closed 1D ECA (e.g., $170 \circ 15 \circ 118 \rightarrow 110$), the Oracle evaluates the tape conditioned on the equivalent `ECA_ID`.
    3.  If the composition escapes the ECA rulespace (e.g., mapping to a 4-color non-ECA cellular automaton), the Oracle asserts maximum literal entropy ($K(x|p) = \text{literal bound}$), terminating the search branch to preserve $O(1)$ parameter execution limits.

**Component 2: HARTLevinSearchOrgan (System 3)**

*   **Mechanism:**
    1.  Enumerate prime candidates from the 38-rule minimal set: $\{0, 1, 2, 3, 5, 7, 11, 12, 13, 15, 18, \dots\}$
    2.  Expand search to 2-tuple and 3-tuple compositions ($\circ$).
    3.  Compute $K(generator) = \text{len}(chain) \times 8$.
    4.  Resolve the composition to its ECA equivalent.
    5.  Query NeuralBDMOracle for $K(x|p)$ using the equivalent ECA.
    6.  Minimize $Kt$.

────────────────────────────────────────

### IV. TIER C: AFTMS STRUCTURAL BREAK ENGINE

────────────────────────────────────────

**Mandate:** Redefine structural break detection as Causal Decomposition.

When $K_{residual} > \theta$, the Phase Transition Organ triggers System 3. The output is no longer a naive integer, but the specific **Causal Decomposition** of the new regime.

*   *Example Output:* "Regime collapsed. Minimal generating program updated to $Rule\ 170 \circ Rule\ 15$. The system is now driven by the composition of an Information Shifter (170) and an Information Filter (15)."

────────────────────────────────────────

### V. TITAN v3.1 TDD BLUEPRINT (MANDATORY TEST HARNESSES)

────────────────────────────────────────

Before the execution logic is written, the following Executable Specifications must be defined to prevent architectural regression:

`test_prohibited_grammar_rejection()`
*   **Assert:** Any attempt to instantiate a generator using arithmetic (ADD, SUB) or algebraic polynomials raises an `OntologicalViolationError`.

`test_causal_equivalence_mapping()`
*   **Assert:** The `CompositionalGenerator` initialized with `[170, 15, 118]` deterministically resolves its truth table to `ECA_ID 110`.
*   **Assert:** The Neural Oracle correctly routes this composition to the legacy `rule_emb` using index 110.

`test_levin_cost_composition_penalty()`
*   **Assert:** Evaluating `[15]` yields a structural bit cost strictly lower than evaluating `[170, 15]`.
*   **Assert:** A complex composition is rejected if the reduction in Neural Residual $K(x|p)$ is less than the bit-cost of the extended chain.

`test_non_closed_composition_truncation()`
*   **Assert:** If a composition results in a rule outside the 256 ECA rulespace (e.g., $50 \circ 37$), the system correctly rejects it by assigning it the uncompressed literal bound to prevent parameter explosion.

────────────────────────────────────────

### EXECUTIVE FREEZE DIRECTIVE

────────────────────────────────────────

The hypothesis space is now formally locked to **Causal Composition of Prime ECAs**.
The Neural Artifact (`universal_brain_titan.pt`) is reinstated as the perfect Kolmogorov Oracle for the $K(x|p)$ residual within the 256-rule closed boundary.
All uncertainty remains rigorously defined by algorithmic descriptive complexity.

**END OF SPECIFICATION**
