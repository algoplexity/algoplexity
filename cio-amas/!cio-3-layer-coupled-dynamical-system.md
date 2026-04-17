Good — this is the point where it stops being a “system description” and becomes an actual coupled dynamical model. I’ll formalise it cleanly so you can drop it into a paper without hand-waving.

---

# 1. Core idea: a 3-layer closed-loop CPS

You have three interacting layers:

### (I) Interaction Graph (latent structure)

[
G(t) = (V, E(t), W(t))
]

### (II) Perceptual Field (continuous embedding / actuator surface)

[
s(x,t) = D(G(t), x)
]

### (III) Human Behavioural State (collective dynamics)

[
H(t) = {h_i(t)}_{i=1}^N
]

These are not separate modules — they are **mutually coupled dynamical subsystems**.

---

# 2. Full coupled dynamical system

We define the system as:

---

## (A) Graph dynamics (topological evolution)

[
G(t+1) = \mathcal{F}_G\big(G(t), H(t), \eta_G(t)\big)
]

Where:

* (H(t)) = human positions / actions (explicit coupling)
* (\eta_G(t)) = stochastic topology noise (Mode C, etc.)
* (\mathcal{F}_G) includes:

  * edge rewiring
  * weight updates
  * interaction sensitivity (PARS influence)

---

## (B) Field generation (graph → perceptual embedding)

[
s(x,t) = \mathcal{D}\big(G(t)\big)
]

But importantly this is not static — it is a **field functional**:

[
s(x,t) = \sum_{i \in V} \phi(x - x_i(t)) \cdot \psi_i(G(t))
]

Where:

* (\phi(\cdot)): spatial kernel (diffusion / LED mapping)
* (\psi_i(G)): node salience (function of PARS, centrality, etc.)

So:

* Coherence → radial density
* PARS variance → angular deformation
* AMAS → temporal discontinuity operator

---

## (C) Human behavioural dynamics (field-driven stochastic control system)

Each agent evolves as:

[
h_i(t+1) = \mathcal{F}_H\big(h_i(t), \nabla s(h_i(t),t), \xi_i(t)\big)
]

Where:

* (\nabla s) = perceptual gradient (attention + motion bias)
* (\xi_i(t)) = intrinsic behavioural noise
* (\mathcal{F}_H) encodes:

  * movement attraction/repulsion
  * alignment effects
  * social coupling mediated through perception

---

# 3. The closed-loop system (single equation view)

Substituting all dependencies:

[
\boxed{
\begin{aligned}
G(t+1) &= \mathcal{F}_G(G(t), H(t), \eta_G) \
H(t+1) &= \mathcal{F}_H\big(H(t), \nabla \mathcal{D}(G(t)), \xi_H\big)
\end{aligned}
}
]

This is a **bidirectionally coupled graph–field–agent dynamical system**.

---

# 4. Key derived objects (your “C, PARS, AMAS” live here)

These are **observables**, not state variables:

### Coherence

[
C(t) = \mathcal{C}(G(t))
]

Typically:

* spectral gap of Laplacian
* or normalized clustering strength

---

### PARS (variance of node sensitivity)

[
\text{PARS}(t) = \mathrm{Var}\big(\rho_i(G(t))\big)
]

Where (\rho_i) = local response sensitivity / centrality derivative.

---

### AMAS (regime transition functional)

[
\text{AMAS}(t) = \left|\frac{d}{dt}\Phi(G(t))\right|
]

Where:

* (\Phi(G)) = structural potential (e.g., entropy, energy, or graph Hamiltonian)
* AMAS is a **temporal discontinuity detector**

---

# 5. Stability conditions (this is the critical part)

You now define **three stability regimes**:

---

## (I) Structural stability of graph dynamics

The graph is stable if:

[
|\mathcal{F}_G(G,H) - G| < \epsilon_G
]

and more importantly:

[
\lambda_2(L(G(t))) > \delta
]

Where:

* (L(G)) = graph Laplacian
* (\lambda_2) = algebraic connectivity

### Interpretation:

> the interaction network remains connected and non-degenerate

---

## (II) Perceptual field stability

The field is stable if it is Lipschitz continuous in space:

[
|\nabla s(x,t)| < K_s
]

and temporally smooth:

[
|s(x,t+1) - s(x,t)| < \epsilon_s
]

### Interpretation:

> no perceptual “white noise collapse” or visual aliasing

---

## (III) Behavioural boundedness (human dynamics do not diverge)

[
|H(t)| < B_H
]

and:

[
\mathbb{E}[|h_i(t+1) - h_i(t)|] < \gamma
]

### Interpretation:

> humans do not enter runaway oscillation / chaotic dispersion

---

# 6. Global CPS stability condition (the real result)

The full system is stable if the coupled Lyapunov functional exists:

[
\exists V(G,H) \quad \text{s.t.} \quad \frac{dV}{dt} \le 0
]

Where:

[
V = \alpha \cdot \mathcal{H}(G) + \beta \cdot |H|^2 + \gamma \cdot \int |\nabla s|^2 dx
]

Interpretation:

* graph entropy is bounded
* human motion energy is bounded
* field roughness is bounded

---

# 7. What this gives you conceptually (important)

You now have a formally defined system where:

### 1. Graph is not data

It is a **dynamical substrate**

### 2. Field is not visualization

It is a **control embedding operator**

### 3. Behaviour is not output

It is a **state variable inside the system**

---

# 8. The key insight you now have (this is the real milestone)

You have constructed a system of the form:

[
\textbf{Graph} ;\longleftrightarrow; \textbf{Field} ;\longleftrightarrow; \textbf{Human Dynamics}
]

Which is mathematically equivalent to:

> a **tri-layer coupled nonlinear dynamical system with feedback across representational, physical, and behavioural manifolds**

---

If you want the next step, I can:

* derive the **linearisation around equilibrium (Jacobian block structure)**
* or show how AMAS becomes a **bifurcation detector formally (not just heuristically)**
* or compress this into a **2-page paper-ready “System Model” section**

That’s where this becomes publishable rather than just well-structured.
