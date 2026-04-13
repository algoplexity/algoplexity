# 📄 CIO-CPS agents/spec.md (v1.0 — invariant-structure compliant)

## 1. Purpose

This document defines the **orchestration layer of CIO-CPS**.

It specifies:

* execution ordering
* data routing rules
* scheduling constraints
* process coordination at runtime

It does NOT define:

* structure
* measurement meaning
* estimator semantics
* inference or interpretation

---

## 2. Role of Agents

Agents are **pure execution coordinators**.

They operate on:

* observer outputs (x_t^{(i)})
* estimator outputs (C_i(x_t^{(i)}))

Agents do NOT:

* interpret outputs
* aggregate meaning
* compute structure
* derive conclusions

---

## 3. System Position

Agents sit between:

```
Observers → Agents → Estimators → Outputs
```

But agents do not transform semantics across layers.

They only control flow.

---

## 4. Agent Types

### 4.1 Observer Scheduler

Responsibilities:

* trigger observer sampling
* enforce sampling cadence
* manage buffer timing

Input:

* system clock
* observer configuration

Output:

* activation signals for observers

Constraint:

* no modification of observer logic

---

### 4.2 Estimator Dispatcher

Responsibilities:

* route (x_t^{(i)}) to assigned estimators
* trigger computation calls
* manage execution queue

Input:

* observer streams

Output:

* estimator execution events

Constraint:

* no interpretation of estimator output

---

### 4.3 Stream Router

Responsibilities:

* route outputs to dashboard/logging
* maintain stream integrity
* ensure timestamp consistency

Constraint:

* no merging or aggregation logic
* no cross-observer comparison

---

## 5. Execution Model

Agents operate in a **deterministic orchestration loop**:

```
t →
    schedule observers
    collect x_t^{(i)}
    dispatch to estimators
    collect C_i(x_t^{(i)})
    forward outputs
```

No additional logic exists.

---

## 6. Data Handling Constraints

Agents MAY:

* buffer data
* queue tasks
* synchronize timestamps
* route streams

Agents MUST NOT:

* compute Δ signals
* compute alignment
* compare observers
* aggregate estimator outputs
* infer structural transitions

---

## 7. Cross-Observer Constraint

Agents are explicitly forbidden from:

* aligning observer outputs
* merging representations
* computing similarity
* constructing joint state

Each observer stream remains independent.

---

## 8. Determinism Requirement

Agent behavior must be:

* reproducible given identical inputs
* independent of estimator outputs
* invariant under observer permutation

---

## 9. Failure Modes (Invalid Agent Behavior)

The following are invalid:

* any heuristic that ranks observers
* any aggregation of estimator outputs
* any filtering based on “signal quality”
* any feedback loop influencing observers or estimators

---

## 10. Relationship to Other Layers

Agents:

* depend on observers and estimators (system layer)
* do NOT access:

  * invariants
  * measurement functionals
  * projection definitions
  * inference logic

Agents are strictly **downstream executors of architecture only**.

---

## 11. Physical Implementation Mapping

If implemented in hardware:

Allowed:

* scheduling microcontroller
* message bus (MQTT, ROS, etc.)
* task queue system

Not allowed:

* embedded interpretation logic
* visual feedback influencing agent decisions
* feedback loops affecting observer behavior

---

## 12. Role in CIO-CPS Stack

Agents define:

> how computation is executed

Not:

> what computation means

---

## 13. Summary

Agents provide:

* execution scheduling
* routing
* orchestration
* system coordination at runtime

Agents do NOT provide:

* inference
* measurement interpretation
* structural reasoning
* cross-observer synthesis

---


