# Working Notes to migrate from static type system → bounded dynamical epistemic system.

---

projection becomes constraint-preserving re-encoding between admissible representations

---

# 0. Target object

New system name:

> **Algorithmic Mesoscope Admissibility System (AMAS)**

Core meaning:

* not a model
* not a pipeline
* not a type system

It is:

> a bounded observer-dependent system of admissible structure + dynamics + measurement

---

# 1. First principle change (critical)

Old core assumption:

* structure defines validity

New core assumption:

* **structure + dynamics define admissible trajectories of computation under bounded observation**

This single shift forces the entire rewrite.

---

# 2. New layer ontology (final architecture)

Replace current mental model with:

```text
AMAS CORE
├── 1 Ontology            (what exists)
├── 2 Theory              (what is true)
├── 3 Measurement         (what is observable/compressible)
├── 4 Computation         (how representations transform)
├── 5 Invariants          (what must never change)
├── 6 Dynamics            (how admissible states evolve)
├── META                  (how system is validated + governed)
```

Everything else becomes **projection/execution layers**:

```text
OUTSIDE CORE
├── systems/              (instantiations like CIO CPS)
├── inference/            (runtime reasoning engines)
├── projections/          (theory → instantiation mappings)
├── validation/           (empirical falsification layer)
```

---

# 3. The real conceptual upgrade

You are not upgrading structure.

You are introducing:

## A bounded epistemic manifold

Formally:

* states = admissible configurations
* dynamics = admissible transitions
* observers = bounded projections of state
* measurement = compression relative to observer

So the system becomes:

> a geometry of computable knowledge evolution

---

# 4. Required global rewrite strategy

This is not incremental. It is 4-phase migration.

---

## PHASE 1 — Canonical redefinition (META reset)

### Update:

* `meta/repository-structure.md`
* `meta/stack-governance.md`

### Replace:

* “type system”
* “static invariants”
* “system architecture”

### With:

* “admissibility system over trajectories”

Add rule:

> Dynamics is a first-class constraint layer equal in authority to invariants.

---

## PHASE 2 — Core reorganization

Add:

```text
6-dynamics/
    admissible-trajectories.md
    feedback-constraints.md
    evolution-laws.md
```

Move cybernetics here entirely.

---

## PHASE 3 — Remove control semantics from systems

In `systems/cio-cps/`:

Replace all:

* controller
* policy
* optimization
* feedback loop (control meaning)

With:

* admissible transition selector
* divergence-to-action mapping
* state evolution interface

Key constraint:

> systems cannot define temporal laws

---

## PHASE 4 — Introduce AMAS root contract

Create:

```text
invariant-structure-core/README.md
```

Rename conceptually to:

> AMAS System Contract

It must define:

* ontology
* invariants
* dynamics
* measurement
* computation
* closure rule

---

# 5. Cybernetics repositioning (critical correction)

Old incorrect placement:

* inside systems as control logic

New correct placement:

```text
core/6-dynamics/
```

Definition:

> cybernetics = constraint on admissible state evolution, not optimization of state

This is the single most important conceptual fix.

---

# 6. Key invariant to enforce globally

Add to `meta/structure-constraints.md`:

> No component may define or modify a state transition rule outside `6-dynamics`.

This prevents:

* systems inventing control loops
* inference engines altering system laws
* pipelines embedding optimization

---

# 7. AMAS formal interpretation

The system is now:

[
\mathcal{S} = (\Omega, \mathcal{I}, \mathcal{M}, \mathcal{C}, \mathcal{D})
]

Where:

* ( \Omega ): ontology
* ( \mathcal{I} ): invariants
* ( \mathcal{M} ): measurement (bounded observer projections)
* ( \mathcal{C} ): computation rules
* ( \mathcal{D} ): dynamics (admissible transitions)

---

# 8. What AMAS is NOT

Explicit exclusions:

* not a learning system
* not an optimization system
* not a control system
* not a neural architecture

These may exist inside projections, but not in core semantics.

---

# 9. Migration consequence for CIO CPS

CIO CPS becomes:

> a projection of AMAS dynamics into a physical sensing–actuation substrate

Not:

* a cyber-physical controller

This resolves your earlier tension completely.

---

# 10. Execution order for repo overhaul

Strict order:

1. meta layer rewrite (governance + stack)
2. core structure add dynamics
3. system layer refactor (remove control semantics)
4. root contract creation
5. reclassification of cybernetics into dynamics
6. update CIO projection alignment

---

# 11. Final state

After overhaul:

* core defines **admissible reality**
* systems instantiate **valid configurations**
* inference explores **within constraints**
* dynamics governs **all evolution**
* cybernetics becomes **law, not controller**

---

This completes the transition:

> from static type system → Algorithmic Mesoscope Admissibility System (AMAS)
