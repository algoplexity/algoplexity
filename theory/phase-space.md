# Phase Space Model

The pair $(E_O, K_{joint})$ — and the extended triple $(E_O, K_{joint}, C_{auto})$ — define a phase space of collective dynamics. All quantities are defined in [core-definitions.md](core-definitions.md).

---

## 1. Primary Phase Space: (E_O, K_joint)

| Regime | Condition | Meaning |
| :--- | :--- | :--- |
| 🟢 Coordinated Intelligence | $E_O > 0,\; K_{joint} > 0$ | Structured, compressible interactions; emergent intelligence |
| 🟡 Independent Complexity | $E_O \approx 0,\; K_{joint} > 0$ | Active but uncoordinated agents; no emergent structure |
| 🔴 Fragmented / Incompressible | $E_O < 0,\; K_{joint} > 0$ | Chaos, adversarial dynamics, or observer mis-specification |
| ⚪ Trivial / Degenerate | $E_O \approx 0,\; K_{joint} \approx 0$ | Static or collapsed system; no structure |

```
          ↑  K_joint (Generative Complexity)
          |
🔴 Fragmented / Incompressible
          |      (E_O < 0, K > 0)
          |
----------+----------------------→  E_O (Coordination Energy)
          |
🟢 Coordinated Intelligence
    (E_O > 0, K > 0)
          |
⚪ Trivial / Degenerate
    (E_O ≈ 0, K ≈ 0)
```

---

## 2. Cost of Autonomy (C_auto)

Define the **Cost of Autonomy** as the total complexity of individual agent behaviors:

$$
C_{auto} = \sum_{i=1}^{n} K\!\left(O(A_i)\right)
$$

**Relationship to joint complexity:**

$$
K_{joint} = C_{auto} + E_O
$$

> Collective intelligence emerges when **autonomy is structured into compressible joint behavior** — not by suppressing individual complexity, but by organizing it.

| Value | Meaning |
| :--- | :--- |
| High $C_{auto}$ | Agents act independently; high expressive variability |
| Low $C_{auto}$ | Agents constrained; low expressive capacity |

---

## 3. Extended Phase Space: (E_O, C_auto)

| Regime | $E_O$ | $C_{auto}$ | Label |
| :--- | :--- | :--- | :--- |
| 🟢 Sweet Spot | Low (coordinated) | High (diverse) | Collective Intelligence |
| 🟡 Chaos | High (fragmented) | High (diverse) | Fragmentation |
| ⚪ Trivial Order | Low (coordinated) | Low (constrained) | Over-Constrained |
| 🔴 Dysfunction | High (fragmented) | Low (constrained) | Collapse |

```
          ↑  C_auto (Autonomy Cost)
          |
🟢 SWEET SPOT        🟡 CHAOS
          |
----------+----------------------→  E_O (Coordination Energy)
          |
⚪ TRIVIAL ORDER      🔴 DYSFUNCTION
```

**Insight:** The system must **structure autonomy** to achieve compressible coordination. Maximizing one without the other is ineffective.

---

## 4. Dynamics

Time-evolving trajectories trace paths through the phase space:

$$
t \mapsto \bigl(E_O(t),\; K_{joint}(t)\bigr)
$$

| Trajectory | Interpretation |
| :--- | :--- |
| $E_O(t)$ decreasing | Increasing coordination |
| $K_{joint}(t)$ increasing | Richer, more complex dynamics |
| $E_O(t)$ increasing | Fragmentation / coherence loss |
| $K_{joint}(t)$ decreasing | Collapse or trivialization |

---

## 5. Cybernetic Control Objective

The general control objective for a cybernetic regulation layer is:

$$
\min_{u(t)}\; E_O(t) \quad \text{subject to} \quad K_{joint}(t) > \varepsilon,\quad C_{auto}(t) > \delta
$$

> Maintain coordination without collapsing system dynamics or suppressing agent diversity.

The full feedback architecture is:

$$
\text{Agents} \;\to\; O(\cdot) \;\to\; (E_O,\, K_{joint},\, C_{auto}) \;\to\; \text{Control} \;\to\; \text{Agents}
$$

See [../implementation/measurement-pipeline.md](../implementation/measurement-pipeline.md) for the operational pipeline that computes these quantities in real time.
