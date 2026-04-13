# 2. Estimators

## Location

```
systems/cio-cps/estimators/
```

## Role

Estimators compute:

[
C_i(x_t^{(i)})
]

They are:

* functional approximators
* signal processors
* feature extractors

They are NOT:

* measurement definitions
* structural objects
* inference systems

---

## Internal structure (allowed)

Each estimator module:

```
compression/
entropy/
perturbation/
statistical/
neural/
```

Each contains:

```
forward.py
config.json
model_weights/
```

---

## Constraints

Estimators MUST:

* operate only on (x_t^{(i)})
* produce scalar/vector outputs
* be independent of other estimators

Estimators MUST NOT:

* define (\mathcal{K}_O)
* define invariants
* compute Δ-alignment
* perform cross-observer reasoning

---

# 3. Critical separation rule

## Observers:

produce representation

## Estimators:

transform representation → numeric signal

## Inference layer:

interprets Δ structure across estimators

## Validation layer:

decides whether interpretation is true

---

# 4. Why this matters in your system

Your previous CIO-CPS design mixed:

* observer semantics
* estimator meaning
* inference logic
* system visualization

Invariant-structure-core enforces:

> observers and estimators are **pure data production and transformation layers only**

---

# 5. Minimal mental model

* Observer = “how data is seen”
* Estimator = “how data is compressed or summarized”
* Inference = “how change across summaries is interpreted”
* Validation = “whether interpretation is real”

---

# 6. Placement summary

```
systems/cio-cps/
    observers/     → φ_O implementations (representation)
    estimators/    → C_i functions (compression / features)
```

Everything else happens downstream.

---

This completes the missing structural placement for both components.
