# amas-core/README.md

## Algorithmic Mesoscope Admissibility System (AMAS) — System Contract

---

## 1. System identity

AMAS defines a bounded computational universe in which all valid structures and processes are constrained by:

- ontological admissibility  
- structural invariants  
- measurement under bounded observers  
- computational transformation rules  
- dynamical evolution laws  

The system is not a model, pipeline, or framework.  
It is a **formal admissibility system over structure and time**.

---

## 2. Primitive decomposition

The system is defined by five coupled domains:

$$
\mathcal{S} = (\Omega, \mathcal{I}, \mathcal{M}, \mathcal{C}, \mathcal{D})
$$

Where:

- **Ω (Ontology)**: admissible entities and relations  
- **𝕀 (Invariants)**: constraints that must hold under all transformations  
- **𝕄 (Measurement)**: observer-bounded projection of structure  
- **𝕔 (Computation)**: admissible transformation rules over representations  
- **𝔻 (Dynamics)**: admissible state evolution laws over time  

No valid system element exists outside this tuple.

---

## 3. Core admissibility principle

All valid constructs must satisfy:

> A construct is valid iff it is representable as a trajectory in the AMAS admissibility space under constraints (Ω, 𝕀, 𝕄, 𝕔, 𝔻).

Invalid constructs include:

- unbounded state evolution  
- unconstrained optimization processes  
- observer-independent measurement claims  
- implicit or untracked control laws  

---

## 3.1 Trajectory definition

A trajectory is:

> a sequence of admissible states whose transitions are governed exclusively by 𝔻 under constraints (Ω, 𝕀, 𝕄, 𝕔)

No trajectory may be defined outside 𝔻.

No external process may redefine trajectory rules.

---

## 4. Observer constraint (AID coupling)

All measurement and interpretation is relative to a bounded observer class 𝒪.

Properties:

- observers are fixed by admissibility constraints  
- observers do not evolve through runtime learning  
- observer class is not inferable from data alone  

Measurement is defined only as:

> projection of system state under 𝒪 with bounded resolution and compression capacity  

---

## 5. Dynamics rule (core addition)

System evolution is governed by:

> 𝔻 defines all admissible state transitions.

Constraints:

- no external module may define transition rules  
- no system component may alter 𝔻 at runtime  
- feedback influences state, not laws  

Cybernetics is defined here as:

> constraint structure on allowable trajectories, not a control mechanism  

---

## 5.1 Hard boundary axiom (CRITICAL)

𝕔 (Computation) is strictly separated from 𝔻 (Dynamics).

Formally:

> No computation, inference process, or composition of 𝕔 is permitted to induce, approximate, modify, or reconstruct elements of 𝔻.

Implications:

- computation cannot learn or infer new dynamics and apply them
- learning systems cannot modify transition laws indirectly
- feedback loops cannot evolve system dynamics

Violation invalidates AMAS consistency.

---

## 6. Structural separation law

The following separation is mandatory:

- Ω defines what exists  
- 𝕀 defines what is consistent  
- 𝕄 defines what is observable  
- 𝕔 defines what is computable  
- 𝔻 defines what is evolvable  

No cross-definition is permitted.

---

## 6.1 Hierarchical dominance rule

In case of conflict:

1. 𝕀 (Invariants) dominate all layers  
2. 𝔻 (Dynamics) governs all temporal evolution  
3. 𝕄 (Measurement) is observer-relative but invariant-respecting  
4. 𝕔 (Computation) must operate within (𝕀, 𝔻)  
5. Ω (Ontology) defines base existence only  

No lower layer may override a higher layer.

---

## 7. System closure rule

All valid AMAS instantiations must:

- be fully derivable from this contract  
- not introduce external primitives  
- not extend the admissibility space  

Anything outside this contract is non-systemic.

---

## 8. Role of projections and systems

External modules (systems/, inference/, projections/, validation/) are:

> embeddings of AMAS into specific domains

They may:

- instantiate Ω elements  
- apply 𝕔 transformations  
- simulate 𝔻 trajectories  

They may NOT:

- redefine invariants  
- alter observer class  
- modify dynamics laws  
- introduce new admissibility rules  

---

## 9. Cybernetic interpretation

AMAS generalizes cybernetics:

- classical cybernetics: control via feedback and optimization  
- AMAS cybernetics: evolution constrained by admissibility laws  

Control is replaced by:

> selection over admissible trajectories within 𝔻

Selection does not imply optimization.

No objective function is required or assumed.

---

## 10. Final statement

AMAS is:

> a bounded, observer-relative, invariant-constrained dynamical system over computational representations.

It defines:

- what can exist  
- what can be observed  
- what can be computed  
- how it evolves  

Nothing else is system-valid.

