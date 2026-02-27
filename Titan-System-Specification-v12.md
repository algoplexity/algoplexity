# TITAN SYSTEM SPECIFICATION v3.0

## STATUS: STRICT LOCK (SDD & TDD BLUEPRINT)

## PARADIGM: Algorithmic Information Theory (AIT) / Algorithmic Field Theory of Market Systems (AFTMS)

## ARTIFACT INTEGRATION: Neural BDM (Kolmogorov Oracle) & HART Brain (Tiny Recursive Model)

## EXTENSION: Symbolic Source Equation Discovery (Rule Primality Generalization)

This document defines the unified deterministic architecture of the TITAN framework after expansion of the generator space from finite ECA basis rules to minimal executable source equations. The thermodynamic substrate remains unchanged. The hypothesis class is generalized while preserving strict AIT constraints.

────────────────────────────────────────

I. GLOBAL ONTOLOGICAL CONSTANTS (ANTI-DRIFT LAYER)

────────────────────────────────────────

All data flowing through the Hemolymph must remain strictly typed. No probabilities, gradients, variances, or confidence scores are permitted.

```python
from typing import NewType, Union
from dataclasses import dataclass
from enum import Enum, auto

BitString = NewType('BitString', bytes)
RuleID = NewType('RuleID', int)              # Legacy ECA basis
EquationID = NewType('EquationID', int)     # Symbolic grammar index

class Signal(Enum):
    ENVIRONMENT_TAPE = auto()
    COMPLEXITY_K     = auto()
    RESIDUAL_DELTA   = auto()
    MINIMAL_PROG     = auto()

@dataclass(frozen=True)
class GeneratorEncoding:
    """Unified representation of executable generators."""
    serialized_form: bytes      # Bit-level encoding of rule or equation
    logical_depth: int          # k

Generator = GeneratorEncoding

@dataclass(frozen=True)
class AlgorithmicPayload:
    bit_cost: float
    state_vector: bytes
```

Constraint: Every generator (ECA rule or symbolic equation) must serialize into a BitString whose length defines its descriptive complexity upper bound.

────────────────────────────────────────

II. TIER A: THEORETICAL PHYSICS (THERMODYNAMIC SUBSTRATE)

────────────────────────────────────────

Unchanged in structure.

Mandate: Enforce conserved computation via Levin cost accounting.

Levin Cost definition:

Kt = |p| + log2(t)

Where |p| is the bit-length of the serialized generator and t is execution depth.

`Hemolymph.request_compute(cost_in_bits: float) -> bool`
Deducts energy strictly according to Levin cost.

`Hemolymph.assimilate_compression_gain(bits_saved: float)`
Converts compression gain directly into available compute.

No modification required at Tier A.

────────────────────────────────────────

III. TIER B: CYBERNETIC ORGANISM (GENERALIZED GENERATOR SPACE)

────────────────────────────────────────

Mandate: Perform bounded Levin search over an expanded hypothesis class including symbolic source equations.

Module: titan.cybernetics

Component 1: NeuralBDMOracle

Role: Approximate Kolmogorov Complexity via Block Decomposition Method.

Constraint:

* Must output description length in bits only.
* Must obey subadditivity bounds.
* Must respect incompressibility for Class 3 sequences.

Output: COMPLEXITY_K

Component 2: HARTRecursiveEngine

Role: Universal Causal Transducer with O(1) parameter constraint.

Extension:

* Must execute both:
  (a) Finite ECA transition rules
  (b) Serialized symbolic source equations

Constraint:

* Parameter count invariant across logical depth.
* Execution strictly recursive.
* Deterministic state mapping x_{t-1} -> x_t.

Component 3: HARTLevinSearchOrgan

Generalized Mechanism:

1. Read ENVIRONMENT_TAPE.
2. Query NeuralBDMOracle to obtain K_naive.
3. Enumerate candidate generators from:
   a. Finite ECA prime basis
   b. Symbolic equation grammar space
4. For each candidate:
   Cost = K(generator) + log2(k)
5. Request compute from Hemolymph.
6. If granted:
   Execute generator for depth k.
7. Compute residual = Prediction XOR Target.
8. Query NeuralBDMOracle for K_residual.
9. Accept generator iff:

   K(generator) + K_residual < K_naive

Output: MINIMAL_PROG (serialized source equation or ECA rule)

Acceptance is purely compression-based. Symbolic equations win only if shorter under AIT accounting.

────────────────────────────────────────

IV. TIER C: AFTMS STRUCTURAL BREAK ENGINE

────────────────────────────────────────

Unchanged in thermodynamic principle.

Component: AFTMSPhaseTransitionOrgan

Mechanism:

1. Consume new ENVIRONMENT_TAPE.

2. Execute historical MINIMAL_PROG.

3. Compute K_residual via NeuralBDMOracle.

4. Apply inertia law:

   theta = epsilon / sqrt(1 + log2(k))

5. If K_residual > theta:
   Trigger RESIDUAL_DELTA.

Symbolic generators are treated identically to ECA rules under this law.

────────────────────────────────────────

V. UPDATED TDD BLUEPRINT

────────────────────────────────────────

Existing tests remain mandatory.

Additions:

`test_symbolic_generator_serialization()`
Ensure every discovered equation serializes deterministically to BitString.

`test_symbolic_generator_compression()`
Assert:
K(generator) + K(residual) < K_naive

`test_generator_space_equivalence()`
If a symbolic equation is algebraically equivalent to an ECA rule,
the shorter encoding must dominate under Levin cost.

`test_parameter_invariance_symbolic_mode()`
Ensure HART parameter count remains O(1) when executing symbolic equations.

────────────────────────────────────────

EXECUTIVE FREEZE DIRECTIVE

────────────────────────────────────────

Tier A unchanged.
Tier C unchanged.
Tier B generalized to include symbolic executable source equations.

Neural artifacts remain quantized at the Hemolymph boundary.

No stochastic semantics introduced.
No thermodynamic relaxation permitted.

TITAN v3.0 preserves deterministic AIT substrate while enabling discovery of minimal source equations under strict Levin-bounded search.
