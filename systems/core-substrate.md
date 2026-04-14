## systems/core-substrate.md

This is the execution layer. It defines only one thing:

> how structured multi-entity behavior is physically generated.

No metrics. No compression. No evaluation. No learning.

---

# CORE SUBSTRATE SPECIFICATION

## 1. Purpose

This module defines a minimal multi-entity dynamical system capable of producing:

* independent trajectories
* coupled trajectories
* graph-structured interaction trajectories
* perturbation response trajectories

It is intentionally agnostic to any notion of “intelligence” or “quality”.

---

## 2. State Space Definition

Each system instance consists of:

* N agents
* T time steps
* D-dimensional state vector per agent

Formally:

```
X[t, i] ∈ ℝ^D
t ∈ [0, T]
i ∈ [1, N]
```

Default:

* N = variable (experiment-controlled)
* T = fixed (e.g. 20–100)
* D = 2 or 3 (spatial embedding)

---

## 3. Initialization (Identity Layer)

Each agent is initialized as:

```
X[0, i] ~ Normal(0, σ₀)
```

No global structure is assumed.

Identity is purely index-based:

* agent i is only identity marker
* no memory beyond state vector

---

## 4. Interaction Model (Dynamics Layer)

The system supports interchangeable interaction kernels.

### 4.1 Independent Mode (Null System)

```
X[t+1, i] = X[t, i] + ε
ε ~ Normal(0, σ)
```

No coupling.

---

### 4.2 Global Coupling Mode

```
C[t] = mean(X[t, :])
X[t+1, i] = X[t, i] + α (C[t] - X[t, i]) + ε
```

All agents weakly converge to shared centroid.

---

### 4.3 Local Graph Interaction Mode (Core CIO Mode)

Let:

* G = adjacency matrix (fixed or random geometric graph)
* N(i) = neighbors of i

```
X[t+1, i] =
    X[t, i]
    + α * mean(X[t, j] - X[t, i]) for j ∈ N(i)
    + ε
```

This is the **primary collective intelligence substrate**.

No global variable exists.

---

### 4.4 Perturbation Operator (Shock Injection)

At defined timestep t = tₛ:

```
X[tₛ, i] += η
η ~ HeavyTailedDistribution (e.g. Laplace or Gaussian σ >> baseline)
```

Optional:

* localized shock (subset of agents)
* global shock (all agents)

---

## 5. Execution Contract (Closure)

The system must obey:

* no state-dependent parameter updates
* no learning
* no adaptation of α, σ, or graph structure during runtime
* interaction rules are fixed per run

This ensures:

> dynamics are generated, not optimized

---

## 6. Output Definition (Representation Trace)

The only emitted object:

```
trajectory = X[0:T, 1:N, :]
```

No metadata is required.

No labeling is performed here.

---

## 7. System Variants (Experiment Hooks)

The experiment layer selects:

* interaction mode (independent / global / local)
* graph structure type:

  * fully connected
  * random Erdos-Renyi
  * geometric k-NN graph
* perturbation timing
* noise level σ

---

## 8. Non-Goals (Explicit Exclusions)

This system does NOT:

* measure complexity
* compute compression
* infer intelligence
* evaluate performance
* optimize structure
* maintain memory of past runs

---

## 9. Boundary Statement

This subsystem is only responsible for:

> generating structured multi-agent trajectories under fixed interaction rules

All interpretation is strictly external.

---



> `validation/admissibility-contract.md`

That is where CIO becomes falsifiable.
