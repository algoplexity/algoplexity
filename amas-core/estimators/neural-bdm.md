## 📄 AMAS / amas-core / estimators / neural-bdm.md

---

## 1. Purpose

Neural BDM is defined as:

> an external estimator over AMAS projection space

It is NOT part of AMAS dynamics.

---

## 2. Embedding Definition

Let:

* S ∈ AMAS
* φ(S) ∈ X (projection space)

Neural BDM is:

```
Bθ : X → ℝⁿ
```

or optionally:

```
Bθ(S) := Bθ(φ(S))
```

---

## 3. Admissibility Constraint

Neural BDM must satisfy:

### 3.1 Projection dependence only

```
Bθ cannot access S directly
```

---

### 3.2 Non-inversion constraint

```
∄ Bθ⁻¹ reconstructing AMAS invariants
```

---

### 3.3 Non-morphism constraint

```
Bθ is NOT a morphism in AMAS sense
```

It does NOT preserve structure.

It only samples projection structure.

---

## 4. No Geometric Closure Rule (critical)

Neural BDM MUST NOT assume:

* manifold structure
* gradients as geometry
* latent continuity
* metric completeness

Unless explicitly encoded in φ

---

## 5. CIO Coupling Rule (safe form)

CIO may only use:

```
C_i(Bθ(φ(S)))
```

NOT:

* Bθ gradients
* internal parameter structure
* induced geometry over outputs

---

## 6. Interpretation Boundary

Neural BDM outputs:

* are not invariants
* are not structure
* are not AMAS objects

They are:

> coordinate-free projection signatures

---

# 🔥 5. Answer to your core question

> “Have we contaminated AMAS-core by inserting CIO interface?”

### Final answer:

### ❌ No contamination of core occurred

### ⚠️ But boundary discipline was previously underspecified

### ✅ Now it is explicitly sealed via projection-interface.md

---

# 🧠 6. Deep structural insight (important)

You now have 3 clean layers:

---

## 🟦 AMAS-core

> defines what *exists*

(invariants, dynamics, morphisms)

---

## 🟨 Projection Interface (NEW FIX)

> defines what can be *seen*

(φ only, no reconstruction)

---

## 🟥 CIO / Neural BDM

> defines what can be *measured externally*

(observers over φ only)

---

# 🚀 7. Result: system is now safe to extend

Now you can safely proceed to:

* Neural BDM full estimator suite
* CIO-AMAS coupling contracts
* measurement layer redesign
* observer registry expansion

without risking AMAS-core contamination.

---


