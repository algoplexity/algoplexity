
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

