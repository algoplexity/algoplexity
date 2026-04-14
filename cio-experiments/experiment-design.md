# EXPERIMENT DESIGN SPECIFICATION

## 1. Purpose

To measure how different interaction rules affect:

> the distribution of admissible structure under a fixed constraint filter.

This is a comparative study framework, not a model.

---

## 2. Experimental Object

Each experiment is defined over:

```id="e1"
E = (G, V, P)
```

Where:

* G = generator family
* V = validation contract (fixed, imported)
* P = perturbation regime

---

## 3. Generator Families (Independent Variable)

All experiments must select from a fixed set:

### 3.1 Null System

* no interaction
* independent noise
* baseline entropy reference

---

### 3.2 Global Coupling System

* shared centroid interaction
* weak alignment dynamics
* low-dimensional compression tendency

---

### 3.3 Local Interaction Graph System

* adjacency-based coupling
* no global variable
* emergent coordination only via local rules

---

### 3.4 Shocked Variants (Required for CIO claim)

Each generator must also be evaluated under:

* impulse perturbation at t = tₛ
* noise amplification regime
* partial node disruption

---

## 4. Experimental Matrix

Each run must sweep:

### 4.1 System Type

* Null
* Global
* Local

### 4.2 System Size

* N ∈ {5, 10, 25, 50}

### 4.3 Time Horizon

* T ∈ {20, 50, 100}

### 4.4 Noise Level

* σ ∈ {low, medium, high}

### 4.5 Perturbation Strength

* η ∈ {0, 1σ, 3σ, 10σ}

---

## 5. Measurement Protocol (Fixed Interface)

For each configuration:

1. Generate M samples (e.g. M = 10,000)
2. Pass each trajectory through validation/
3. Compute:

```id="e2"
accept_rate = ACCEPT / (ACCEPT + REJECT)
```

No other statistics are computed here.

---

## 6. Primary Scientific Quantity

The only valid derived signal is:

### Acceptance Differential

```id="e3"
ΔA = A(system_i) - A(system_null)
```

Where:

* A = acceptance rate under fixed validation
* system_null = independent baseline

---

## 7. CIO Signal Definition (Strict)

A CIO effect is only claimed if:

### Condition 1 — Separation

```id="e4"
A(local_interaction) >> A(null)
```

AND

```id="e5"
A(local_interaction) > A(global_coupling)
```

---

### Condition 2 — Robustness under shock

```id="e6"
A(local_shocked) - A(local_unshocked) > A(null_shocked) - A(null_unshocked)
```

---

### Condition 3 — Scaling stability

Signal must persist across:

* N increase
* T increase
* σ increase

---

## 8. Forbidden Operations

This layer must NOT:

* redefine validation rules
* modify system dynamics
* adjust compression thresholds
* introduce learned parameters
* optimize generators
* compute gradients or scores for adaptation

This preserves causal separation.

---

## 9. Output Format (Experimental Report)

Each experiment produces:

```json
{
  "generator_type": "local_graph",
  "N": 25,
  "T": 50,
  "sigma": 0.1,
  "shock": 3.0,
  "accept_rate": 0.184,
  "baseline_rate": 0.012,
  "delta": 0.172
}
```

No interpretation is included.

---

## 10. Scientific Interpretation Boundary

This layer explicitly does NOT claim:

* intelligence
* emergence
* cognition
* agency

It only produces:

> structured differences in admissibility rates across controlled interaction systems

---

## 11. What this enables (implicitly)

Once this layer is operational, you can:

* plot phase transitions in acceptance rate
* identify regime shifts under coupling strength
* measure robustness of structure under perturbation
* compare interaction topologies quantitatively

---

