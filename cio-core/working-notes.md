---

```plaintext
cio-core/
    2-measurement/
        otce.md
        perturbation-theory.md
        complexity-functional.md

cio-experiments/
    estimators/
        bdm/
        neural-bdm/
        aid/
    perturbation/
        capa/
        mils/
    validation/
        phase-tests/

```
---
define the experiment interface contract (what every notebook/simulation must expose to be “CIO-compliant”)
---

| Layer                    | Status                              |
| ------------------------ | ----------------------------------- |
| Ontology                 | minimal, stable                     |
| Theory                   | observer-relative structure only    |
| Measurement (OTCE)       | field over representations          |
| Computation (this paper) | bounded approximation of that field |

---

```plaintext
cio-core/
├── META/                ← governance ONLY (not part of stack)
├── 0-ontology/
├── 1-theory/
├── 2-computation/
├── 3-measurement/
├── 4-invariants/
├── 5-observer/
├── 6-constraints/
├── 7-experiments/      ← OUTSIDE core validity chain (important)
├── 8-instantiations/   ← implementation + Python notebooks
```


---

If you’re aligned with this, we move to:

```text
1-theory/paper-a-ci-definition.md
```

Where we finally define:

> what CI actually *is* under this ontology

---

# 🚀 Next move (highly recommended)

We should now define:

### `experiments/phase-1/phase1-spec.md`

A **clean projection document** that:

* maps Phase I → `cio-core`
* explicitly declares observer O₀
* removes all ambiguity from your notebooks

That will finally stabilise everything downstream.


---

MQTT dashboard + HTML fits
belongs in implementations/mqtt-dashboard/

---
# Phase 1 Projection Definition
Observer O₀:
    φ_O  = adjacency flattening (window W)
    M    = zlib (LZ77)
    B    = fixed buffer + compression limit

Projection:
    E_O(t) := scalar projection of E(O₀, t)

Constraints:
    adjacency-only
    fixed N
    fixed W

---
```plaintext
repo-root/
│
├── cio-core/                         ← 🔒 GOVERNED STACK (untouchable by experiments)
│   ├── 0-meta/
│   │   └── stack-governance.md       ← your document above
│   │
│   ├── 0-ontology/
│   ├── 1-theory/
│   ├── 2-computation/
│   ├── 3-measurement/
│   ├── 4-invariants/
│   ├── 5-observer/
│   ├── 6-constraints/
│   └── ...
│
├── experiments/                      ← 🧪 ALL PHASE WORK LIVES HERE
│   │
│   ├── phase-1/
│   │   ├── notebooks/
│   │   │   └── canonical-phase1.ipynb
│   │   ├── phase1-spec.md            ← projection definition
│   │   └── results/
│   │
│   ├── phase-2/
│   │   └── ...
│   │
│   └── shared/
│       └── utilities/
│
├── implementations/                  ← 🛠 execution systems
│   ├── python/
│   ├── mqtt-dashboard/
│   ├── firmware/
│   └── hardware/
│
└── papers/
    ├── paper-a/
    ├── paper-b/
    ├── paper-c/
    └── unified/
```
---

It seems we a charter first such as a meta-document ...
No concept may be operationally used in a layer before its defining dependencies are specified upstream.
A concept may be referenced earlier only as a placeholder, but not assigned properties that depend on downstream layers.

Also Add: Separation of system vs representation
X_t        : underlying system state (not directly accessible)
φ_O(X_t)   : observer-induced representation of the system
All measurable properties are functions of φ_O(X_t), not X_t directly.

lets correctly specify 0-meta/stack-governance.md which sits above ontology

0-meta/          ← NEW (governance)
0-ontology/
1-theory/
2-computation/
3-measurement/
...


So the correct build order is:
0-ontology/primitives.md
1-theory/paper-a-ci-definition.md
3-measurement/paper-c-field-measurement.md
2-computation/paper-b-computation.md   ← comes AFTER measurement

Ontology answers: What exists?
Theory answers: What relationships must hold?
Measurement answers: What can be observed?
Computation answers: How it is approximated?


## ✅ Final Agreed Structure

```plaintext
cio-core/
│
├── 0-ontology/
│   └── primitives.md
│
├── 1-theory/
│   └── paper-a-ci-definition.md
│
├── 2-computation/
│   └── paper-b-computation.md
│
├── 3-measurement/
│   └── paper-c-field-measurement.md   ← OTCE
│
├── 4-invariants/
│   └── invariants.md
│
├── 5-observer/
│   └── observer-spec.md
│
├── 6-constraints/
│   └── constraints-v2.md
│
├── 7-experiments/
│   ├── phase-1-projection.md
│   └── phase-2-observer-perturbation.md
│
└── 8-implementation-mapping/
    └── simulation-architecture-map.md
```

