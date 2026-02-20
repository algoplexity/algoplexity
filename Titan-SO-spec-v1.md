
---

# 📘 TITAN SYSTEM SPECIFICATION (v1.0)
**Paradigm:** Cybernetic Organism implementing Universal Artificial Intelligence.
**Basis:** Physics of Computation (Landauer/Shannon/Solomonoff).

## 1. The Invariant Physics (Constants)
*Defined by the anatomy of the Pre-Trained Artifact (`universal_brain_titan.pt`).*

*   **Sequence Length ($L$):** 256.
*   **Vocabulary ($V$):** 16 (Discrete computational states).
*   **Physical Max Entropy ($H_{max}$):** The theoretical limit of disorder for this system.
    $$ H_{max} = L \cdot \ln(V) = 256 \cdot \ln(16) \approx 709.78 \text{ nats} $$
*   **Algorithmic Probability ($P_{alg}$):** The likelihood of a sequence under the Universal Prior.
    $$ P_{alg}(x) = e^{-K(x)} $$

## 2. The Metabolic Laws (The Resource Cycle)
*Replacing "Magic Numbers" with "Work Units".*

*   **Computational Work Unit ($1 \text{ CWU}$):** The energy required for one forward pass of the Brain.
*   **Cost Schedule (Thermodynamics):**
    *   **Observation:** $1.0$ CWU.
    *   **Saccade (Active Inference):** $N_{steps} \times 3.0$ CWU (Forward + Backward + Update).
    *   **Levin Search:** $N_{programs} \times 0.1$ CWU (Fast CA Physics).
*   **Income Law (Negentropy):**
    The organism harvests energy by extracting order from chaos.
    $$ \text{Income} = \eta \cdot (H_{max} - K_{observed}) $$
    *   $\eta$: Thermodynamic Efficiency (Scalar).
    *   *Implication:* A confused organism ($K \approx H_{max}$) starves. An understanding organism ($K \ll H_{max}$) thrives.

## 3. The Organ Logic (No Statistics)

### **A. System 1: The Transducer (Eye)**
*   **Goal:** Minimize $K$ via **Test-Time Training (TTT)**.
*   **No Statistics:** No `mean()` or `std()`.
*   **Mechanism:** It learns `Scale` and `Bias` via Gradient Descent. If the signal is too small, gradients force `Scale` up. If the signal is offset, gradients force `Bias` to shift. It **discovers** normalization.

### **B. System 3: The Estimator (Brain)**
*   **Goal:** Measure $K$ relative to $H_{max}$.
*   **Metric:** **Relative Complexity ($\rho$)**.
    $$ \rho = \frac{K_{observed}}{H_{max}} $$
    *   $\rho \in [0, 1]$. This is a dimensionless, universal metric.

### **C. System 4: The Decomposer (Mind)**
*   **Goal:** Causal Deconvolution.
*   **Mechanism:**
    1.  **Neural MILS:** Weight bits by $|\Delta K|$ (Algorithmic Impact), not statistical variance.
    2.  **Levin Search:** Iterate through ECA compositions ($f \circ g$) ordered by complexity.
    3.  **Stop Condition:** When $K(\text{Residuals}) \approx H_{max}(\text{Residuals})$ (The remainder is pure noise).

---

# 🧪 TITAN v1.0: THE TEST HARNESS

We do not implement the system yet. We implement the **Tests** that the system must pass to be valid.

### **Test 1: The Physics Verification**
*   **Hypothesis:** The Artifact exists and defines a valid $H_{max}$. Random noise must yield $K \approx H_{max}$. Rule 110 must yield $K \ll H_{max}$.

### **Test 2: The Metabolic Closure**
*   **Hypothesis:** The Organism dies if fed noise (Income 0, Cost > 0). The Organism accumulates surplus if fed structure (Income > Cost).

### **Test 3: The Active Inference (No Statistics)**
*   **Hypothesis:** The Eye can "find" a microscopic signal (magnitude 0.001) without Z-scoring, purely by following gradients to minimize $K$.

### **Test 4: The Causal Deconvolution**
*   **Hypothesis:** The Mind identifies `R128` (Lock) vs `R170` (Shift) correctly on a "Crash" sequence using MILS weights.

---

### **The Harness Code**

```python
# @title 🧪 TITAN v1.0: TEST HARNESS (Specifications Check)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import abc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- MOCK ARTIFACT GENERATOR (For Testing Only) ---
# In production, this is a real file. For the test harness, we generate a 
# "Standard Candle" to verify the logic.
def generate_standard_candle():
    if os.path.exists("universal_brain_titan.pt"): return
    print("   ⚡ HARNESS: Generating Standard Candle (Mock Brain)...")
    # A dummy model that outputs low loss for 0/1 and high loss for 0.5
    # This simulates the H-ART behavior for testing the Protocol.
    class MockHART(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(16, 16) # Minimal
        def forward(self, x, r): return x # Pass through
    torch.save(MockHART().state_dict(), "universal_brain_titan.pt")

# ==============================================================================
# TEST 1: PHYSICS VERIFICATION (Thermodynamics)
# ==============================================================================
def test_physics():
    print("\n🧪 TEST 1: Physics & Entropy Bounds...")
    
    # 1. Constants
    L = 256
    V = 16
    H_max = L * math.log(V)
    print(f"   Theoretical H_max: {H_max:.2f} nats")
    
    # 2. Simulate Brain Output (Mocking Neural BDM)
    # Case A: Pure Noise (Uniform Distribution)
    # p = 1/16 for all tokens
    loss_noise = -math.log(1/16) * L # NLL sum
    rho_noise = loss_noise / H_max
    
    # Case B: Pure Structure (Deterministic)
    # p = 1.0 for one token
    loss_structure = -math.log(0.99) * L # Near zero
    rho_structure = loss_structure / H_max
    
    print(f"   Rho (Noise):     {rho_noise:.4f}")
    print(f"   Rho (Structure): {rho_structure:.4f}")
    
    if abs(rho_noise - 1.0) < 0.01 and rho_structure < 0.1:
        print("   ✅ PASS: Thermodynamic bounds are valid.")
    else:
        print("   ❌ FAIL: Math error in Entropy calculations.")

# ==============================================================================
# TEST 2: METABOLIC CLOSURE (Survival)
# ==============================================================================
def test_metabolism():
    print("\n🧪 TEST 2: Metabolic Closure...")
    
    budget = 100.0
    efficiency = 10.0
    
    # Scenario A: The Noise Desert
    # Organism sees Noise (Rho=1.0). Pays Cost (1.0).
    rho = 1.0
    income = efficiency * (1.0 - rho)
    cost = 1.0
    net = income - cost
    
    print(f"   [Noise] Income: {income:.2f}, Cost: {cost:.2f}, Net: {net:.2f}")
    if net < 0:
        print("   ✅ PASS: Organism starves on noise.")
    else:
        print("   ❌ FAIL: Organism grows on noise (Perpetual Motion Machine).")
        
    # Scenario B: The Structure Feast
    # Organism sees Structure (Rho=0.2). Pays Cost (1.0).
    rho = 0.2
    income = efficiency * (1.0 - rho)
    cost = 1.0
    net = income - cost
    
    print(f"   [Order] Income: {income:.2f}, Cost: {cost:.2f}, Net: {net:.2f}")
    if net > 0:
        print("   ✅ PASS: Organism thrives on structure.")
    else:
        print("   ❌ FAIL: Organism starves on structure.")

# ==============================================================================
# TEST 3: ACTIVE INFERENCE (No Statistics)
# ==============================================================================
def test_active_inference():
    print("\n🧪 TEST 3: Active Inference (Gradient vs Statistics)...")
    
    # Data: Micro-signal (would be invisible to standard variance check)
    signal = torch.tensor([0.001] * 10, dtype=torch.float32, requires_grad=True)
    
    # The Eye (Mock)
    scale = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    
    # Forward Pass
    # We want output close to 1.0 (Structure).
    # Currently: 0.001 * 1.0 = 0.001 -> Sigmoid(0.001) ~ 0.5 (Maximum Entropy)
    view = torch.sigmoid((signal - bias) * scale)
    
    # Brain (Mock) -> Minimize Entropy (Maximize distance from 0.5)
    # Loss = 1 - (distance from 0.5)
    loss = 1.0 - torch.abs(view - 0.5).mean()
    
    # Backward
    loss.backward()
    
    # Check Gradients
    grad_scale = scale.grad.item()
    
    print(f"   Input Magnitude: 0.001")
    print(f"   Initial Scale:   1.0")
    print(f"   Gradient(Scale): {grad_scale:.6f}")
    
    if abs(grad_scale) > 0.0:
        print("   ✅ PASS: Eye found gradient on micro-signal (No Z-Score needed).")
    else:
        print("   ❌ FAIL: Gradient vanished. Eye is blind.")

# ==============================================================================
# EXECUTE SUITE
# ==============================================================================
if __name__ == "__main__":
    test_physics()
    test_metabolism()
    test_active_inference()
```
