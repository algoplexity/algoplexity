## cio-core: Full Content & Implementation Guide

### 1. Purpose & Philosophy
The `cio-core` directory is the canonical source for the design, theory, and prototype implementation details for the Collective Intelligence Observatory (CIO) system. It aims to be both a knowledge base and a developer’s toolkit for building, experimenting, and measuring collective intelligence systems.

---

### 2. Updated Folder Structure

- **meta/**: Stack governance, general project philosophy, integration guidelines.
- **0-ontology/**: Formal definitions of concepts—primitives, nodes, interactions, graphs, agents.
- **1-theory/**: Mathematical foundations and the main CI definition paper.
- **experiments/**: Sample experiments, protocols, and parameter sets for reproducible research.
- **observer/**: Details on observer roles and code for system/data “watchers” that interpret simulation results, run metric validation, and check constraints.
- **invariants/**: Documents specifying what remains unchanged (e.g., N constant, encoding structure) and rules for valid experiments/models.
- **measurements/**: Specifications and possibly code for metric computation (E_O, E_dir, etc.), data logging, and time-series output.
- **computation/**: Concrete scripts, functions, or pseudocode for running CIO simulations and performing core calculations.

---

### 3. What Each Folder Contains & How To Use

#### **meta/**
- **Read first** for project policies, stack overview, and integration rationale.

#### **0-ontology/**
- **Defines key data models** (e.g., nodes, adjacency matrices, graph windows).
- **Implement**: Map these directly to your Python classes and data structures.

#### **1-theory/**
- **Presents** formal models, key theorems, and the algorithmic backbone of your implementation.
- **Implement**: Translate formulas into Python and verify congruence with these definitions.

#### **experiments/**
- **Contains** concrete, step-by-step experiment plans and parameter files.
    - Example: preset simulation runs with node numbers, noise settings, perturbation events.
- **Implement**: Use as template for Jupyter/Colab experiments—parameters/settings can be copy–pasted into scripts for reproducibility.

#### **observer/**
- **Describes and may implement** the validation layer (“observer” logic).
    - Ensures collected data meets criteria, triggers checks for phase transitions, validates time and structure invariants.
- **Implement**: Add Python observer classes/functions that validate outputs, halt on constraint violations, and record checkpoints.

#### **invariants/**
- **Defines** what must never change during a valid experiment:
    - Number of nodes (N), window size (W), matrix dimensions, time step size, etc.
- **Implement**: Write runtime assertions/tests in your simulation to ensure invariants are never broken—essential for scientific demos.

#### **measurements/**
- **Documents and scripts** for all computed metrics (E_O, E_dir, state classification), measurement procedures, and data output policies.
- **Implement**: Directly referenced in Colab Python code for metric evaluation; describes how and when to emit logs/output to frontend/UI.

#### **computation/**
- **Includes code snippets and procedures** for the core simulation process:
    - The “main loop” for updating node state, computing metrics, maintaining buffers.
- **Implement**: This is your literal implementation reference; shape your simulation’s main() and supporting functions based on these scripts.

---

### 4. Implementation Roadmap for Colab + HTML UI Demo

#### **a) Read in this order:**
1. `meta/` → context, architecture, stack
2. `0-ontology/` → data structures to code
3. `1-theory/` → equations, formal simulation design
4. `experiments/` → ready-made runs to replicate
5. `observer/` → all runtime validation logic
6. `invariants/` → what to assert/check in every run
7. `measurements/` → how to compute and output metrics
8. `computation/` → core simulation loop/code

#### **b) Implementation steps:**
- **Model your data** using ontology and invariants (Python classes/DFs).
- **Script your computation** (core loop, metrics) by transcribing from computation/ and theory/.
- **Build experiments** by parameterizing your code based on experiments/.
- **Add runtime checks** for invariants and observer logic.
- **Output time series/data** in the document format from measurements/.
- **Visualize** via Colab plots and export for HTML/Javascript dashboard.

---

### 5. How it All Connects on Demo Day

- Your Colab notebook will:
    - Use ontology and invariants to set up robust, traceable simulations.
    - Run experiments as described; observer logic will catch “invalid science.”
    - Output metrics as time series (CSV/json), matched to measurement protocols.
    - Core simulation code is simple to audit/modify thanks to computation/ snippets.
- Your HTML UI will:
    - Display uploaded (or live streamed) simulation results and metrics.
    - You can highlight experiment conditions, violations (from observer/), and show conformity to invariants.

---

## Table: Folder and Implementation Role

| Folder           | Main Content                   | How You Use in Implementation          |
|------------------|-------------------------------|----------------------------------------|
| meta/            | Stack/design docs              | Architectural overview & rationale     |
| 0-ontology/      | Formal models                  | Classes, struct defs, typing           |
| 1-theory/        | Mathematical basis             | Numeric method, code translation       |
| experiments/     | Protocols, configs             | Setup/jinja templates for runs         |
| observer/        | Validation logic/scripts       | Runtime checks, logging, exceptions    |
| invariants/      | Rules/constraints              | Assert, test, and monitor invariants   |
| measurements/    | Metric definitions, logs       | Output metrics, time series, CSV/JSON  |
| computation/     | Scripts/code/algorithms        | Direct code reference for main loop    |

---

## Final Guidance

- View the structure as a **pipeline**: ontology → code → invariants → computation → observer/measurements → experiments → results/visualization.
- **No matter where you start, always check invariants and observer code**—this guarantees Demo Day runs are scientifically sound and reproducible.
- You’ll mainly copy from computation/ and theory/ but the investigator’s logic (observer/), and experiment configs, are equally crucial for a credible demo.

---

Let me know if you want a starter Python or HTML file tailored to this architecture!
