Here is the consolidated documentation suite. These two documents represent the **practical implementation** (Master's) and the **theoretical expansion** (PhD) of your research program.

They are aligned with the **ANU CYBN8001** requirements and the **Algoplexity Horizon 3** roadmap, incorporating the corrected citations for Taylor/Page (Brookings), Battiston (Nature), and Behrouz (NeurIPS/arXiv).

***

# PART 1: The CPS Project Proposal
**Submission for ANU Course CYBN8001 (Cyber-Physical Systems)**

## **Project Title:** The Civic Resonator
**Subtitle:** Engineering a Simplicial Gating Interface for Collective Intelligence

**Student:** Yeu Wen Mak
**Date:** January 14, 2026
**Theme:** The Civic Nervous System (Horizon 3)

---

### 1. Executive Summary
The **Civic Resonator** is a tabletop cyber-physical instrument designed to solve the "Room vs. Map" disconnect identified by the **Brookings Institution** [1]. It acts as a physical "thermostat" for group coherence.

By synthesizing **Battiston’s Network Physics** [2] with **Behrouz’s Nested Learning** [3], the device uses a novel **"Simultaneity Gate"** to filter out linear social noise. It only provides positive haptic feedback when a group interacts within a specific topological threshold (a Simplicial Complex), thereby physically reinforcing the non-linear dynamics required for complex consensus.

### 2. Theoretical Framework
*   **The Societal Problem:** Taylor & Page (2025) argue that policymakers are "Design-Minded" (focused on the room) while systems are "Model-Minded" (focused on the map). We lack infrastructure to translate local human intent into systemic signal [1].
*   **The Scientific Mechanism:** Battiston et al. (2025) prove that social cohesion does not spread via pairwise edges (dyadic), but via **Simplicial Complexes** (triangles/tetrahedrons). Influence requires "simultaneous exposure" [2].
*   **The Engineering Solution:** We must build a sensor that ignores $A \to B$ interactions and only triggers on $A+B+C$ interactions.

### 3. Technical Implementation (The "Making")

The device is a **Social-Cyber-Physical Loop** comprising three layers:

#### **Layer A: The Simplicial Sensor (Input)**
*   **Hardware:** 3x Capacitive Touch Points (Copper) + Microphone Array (ESP32-S3).
*   **The Logic:** The system uses a **Non-Linear Gating Function**.
    *   *Rule:* If Touch $T_A, T_B, T_C$ occur within $\Delta t < 50ms$, then `Input = 1`. Else `Input = 0`.
    *   *Goal:* To physically filter out "polite turn-taking" (linear) and detect "collaborative synchronization" (non-linear).

#### **Layer B: The Nested GNCA Processor (Model)**
*   **Architecture:** The firmware runs a **Graph Neural Cellular Automaton (GNCA)** inspired by the **Nested Learning** framework [3].
    *   **Inner Loop (Fast):** The "Reflex" responds to immediate touch simultaneity (Simplicial Gate).
    *   **Outer Loop (Slow):** The "Entropic Decay" cools the system over time. If interaction stops, the system reverts to Chaos (Rule 60).
    *   **Phase Transition:** When `Simplicial Input > Entropic Decay`, the system snaps to a **Soliton State** (Rule 110)—a self-sustaining coherent pattern.

#### **Layer C: The Ambient Actuator (Output)**
*   **Hardware:** Neopixel Ring (60 LED) + Haptic Motor.
*   **Feedback:**
    *   *State 0 (Entropy):* Dim, flickering light (Visualizing the "Entropy of Thought" [1]).
    *   *State 1 (Coherence):* Bright, breathing "Standing Wave" + 40Hz resonant hum.

### 4. Project Plan & Learning Outcomes

| Phase | Activity | CYBN8001 Outcome |
| :--- | :--- | :--- |
| **Weeks 1-4** | **Breadboarding the Gate:** Building the circuit that validates the "50ms window" logic. | *"Interrogate separate components"* (Sensors) |
| **Weeks 5-8** | **Coding the GNCA:** Implementing the Nested Learning loop on ESP32. | *"Technological Constellations"* (Code + Sociology) |
| **Week 9** | **Fabrication:** 3D printing the "Stone" chassis (Concrete/PLA hybrid). | *"Making & Building"* |
| **Week 10** | **The Mock Summit:** Testing the device using the **Nelson Wetlands Protocol** (Green Energy vs. Ecology scenario). | *"Work Integrated Learning"* |

### 5. References
**[1] Taylor, J., & Page, S. E. (2025).** *AI is changing the physics of collective intelligence.* The Brookings Institution.
**[2] Battiston, F., et al. (2025).** *Higher-order interactions shape collective human behavior.* Nature Human Behaviour.
**[3] Behrouz, A., et al. (2025).** *Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS / arXiv.

***

# PART 2: The PhD Research Proposal
**Horizon 3 of the Algoplexity Research Program**

## **Research Title:** Fractal Cybernetics
**Subtitle:** Scaling Collective Intelligence via Simplicial Renormalization and Nested Architectures

**Candidate:** Yeu Wen Mak
**Program:** PhD in Cybernetics / Complex Systems
**Context:** The Algoplexity Roadmap (Horizon 3: The Civic Nervous System)

---

### 1. Abstract
Current democratic and corporate governance systems suffer from a scaling problem: "Smart local actions can end up misaligned with system-wide synergies" [1]. This research proposes **Fractal Cybernetics**, a framework for engineering the **"Civic Nervous System."**

By integrating **Algorithmic Information Dynamics (AID)** with **Simplicial Topology** [2] and **Nested Learning Architectures** [3], this thesis seeks to define the physical laws of "Renormalization"—how local consensus (The Room) can be mathematically scaled to global intelligence (The Model) without losing signal fidelity.

### 2. Research Question
**Primary:** *How can Cyber-Physical Systems (CPS) induce and stabilize Higher-Order Interactions to facilitate phase transitions from social entropy to collective intelligence?*

**Secondary:**
*   Can a **Simplicial Gating Mechanism** physically filter "Groupthink" (Monoculture) from "Coherence" (Algorithmic Compression)?
*   Does **Nested Feedback** (Fast/Slow loops) prevent the "Lyapunov Instability" characteristic of polarization and market panics?

### 3. Theoretical Trinity

#### **I. The Physics: Algorithmic Information Dynamics**
Building on **Hutter** and **Zenil**, we define "Collective Intelligence" not as voting, but as **Compression**. A group is "intelligent" if it can compress the complexity of its environment into a viable policy.
*   *Metric:* We quantify "Systemic Failure" as **Global Coherence Loss**, occurring when Environmental Drift ($\Lambda$) exceeds the Collective Update Rate ($\eta$).

#### **II. The Topology: Simplicial Complexes**
We reject the standard network model (Edges). Following **Battiston (2025)**, we model the social graph as a **Simplicial Complex**.
*   *Hypothesis:* Information diffusion is linear, but **Behavior Change is Topological**. It requires the "Simultaneity Gate" (The Triangle) to overcome social inertia.

#### **III. The Computation: Nested Learning (Titans)**
We reject static Deep Learning models. Following **Behrouz (2025)**, we implement **Nested Architectures**.
*   *Mechanism:* The system is modeled as "Optimization inside Optimization."
    *   *Inner Loop:* The **Civic Resonator** (The Device) optimizing local human affect.
    *   *Outer Loop:* The **Civic Nervous System** (The Network) optimizing global resource allocation.

### 4. Methodology: The "Nelson Protocol"
The research will be validated through a series of "Room + Model" experiments, simulating high-stakes negotiation environments (e.g., The Nelson Wetlands Green Energy Conflict).

*   **Control Group:** Standard negotiation (Whiteboard + Conversation).
*   **Experimental Group:** Cyber-Physical negotiation (Using the **Civic Resonator**).
*   **Measurement:** We will measure the **Lyapunov Exponent** of the conversation flow. Does the CPS force the group dynamics to converge (Stable Attractor) or diverge (Chaos)?

### 5. Contribution to Knowledge
This thesis moves Cybernetics from **"First-Order"** (Steering the System) and **"Second-Order"** (Steering the Steer-er) to **"Higher-Order Cybernetics"**:
> *The engineering of environments that physically necessitate topological synchronization for actuation.*

### 6. Select Bibliography
1.  **Behrouz, A., et al. (2025).** *Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS.
2.  **Battiston, F., et al. (2025).** *Higher-order interactions shape collective human behavior.* Nature Human Behaviour.
3.  **Taylor, J., & Page, S. E. (2025).** *AI is changing the physics of collective intelligence.* The Brookings Institution.
4.  **Williams, C. F. (2025).** *Strategy as Ontology: A Quantum–Complex–Entropic–Adaptive Framework.* SSRN.
5.  **Wolfram, S. (2002).** *A New Kind of Science.* (Foundations of Class 4 Automata).
