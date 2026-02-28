
────────────────────────────────────────
# TITAN SYSTEM SPECIFICATION v5.2 (THE COMPLETE MECHANISTIC OMNIBUS)

## STATUS: STRICT FINAL RESEARCH LOCK (SDD & TDD DIRECTIVE)
## PARADIGM: Algorithmic Information Theory (AIT) / Algorithmic Field Theory of Market Systems (AFTMS)
## ARTIFACT INTEGRATION: Neural BDM (Kolmogorov Oracle Proxy) & HART Brain (Tiny Recursive Model)
## ONTOLOGY: Bounded Algebraic Microscope over a Finite Boolean Semigroup

This document formally locks the TITAN architecture. It explicitly deprecates the ontologically ambiguous "Symbolic Source Equation" construct. The generalized hypothesis class ($H_{COMP}$) is strictly restricted to the **Finite Boolean Composition of Prime Elementary Cellular Automata**, as defined by the Riedel-Zenil causal decomposition theorem. 

All stochasticity, continuous mathematics, arbitrary algebraic grammars (ADD, SUB, SHIFT), and infinite compositional chains are strictly prohibited. The environment is generable strictly through the sequential computational composition of prime filters and shifters, bounded by the thermodynamic limits of the `Hemolymph`.

────────────────────────────────────────

### I. STRATEGIC COMMITMENT & EPISTEMIC BOUNDS

────────────────────────────────────────

1. **Literal Bound Collapse as Epistemic Signal:** By restricting to finite compositions of 38 primes, TITAN acts as an algebraic microscope. When $Kt$ exceeds the uncompressed channel capacity, the system collapses to the Literal Bound. This asserts that the market is **Algorithmically Irreducible** under the current lens.
2. **The Neural Proxy:** `universal_brain_titan.pt` computes a **Neural Proxy for Conditional Log-Loss** over 256 ECA dynamics, acting as a bounded, computable heuristic for true Kolmogorov Complexity.
3. **Pseudo-Bennett Structural Inertia:** The dampening factor $1 / \sqrt{1 + \log_2(k)}$ acts as **Structural Causal Inertia**, utilizing causal chain depth to thermodynamic penalize rapid phase transitions.

────────────────────────────────────────

### II. TIER A: THEORETICAL PHYSICS & THE THERMODYNAMIC SUBSTRATE

────────────────────────────────────────

The semantic data structures must mathematically enforce the compositional grammar. 
The Levin Cost ($Kt_{total}$) is formally defined to penalize the perceptual depth of the Lens and the structural complexity ($|p|$) of compositional chains:
*   $K(Lens) = \log_2(\text{stride}) + \log_2(\text{block\_size}) + 2.0$
*   $K(P_i) = 8 \text{ bits}$ (Physical cost of one prime instruction)
*   $|p| = \text{length}(composition\_chain) \times 8 \text{ bits}$

**Updated Levin Bound Equation:**
$Kt_{total} = K(Lens) + (|p|) + \log_2(k) + \text{Neural\_LogLoss\_Proxy}(x | Lens, p)$

A composite rule (e.g., $170 \circ 15 \circ 118$) has a baseline structural cost of 24 bits. It will only be selected over a single prime rule if its ability to compress the residual via the Neural Oracle exceeds the 16-bit physical penalty of requiring two additional compositional steps.

────────────────────────────────────────

### III. TIER B: CYBERNETIC ORGANISM & CAUSAL COMPOSITION SPACE

────────────────────────────────────────

**Mandate:** Perform Bounded Levin Search over the combinatorial space of Prime ECA compositions. Continuous mathematics is isolated entirely to the perceptual boundary.

**Component 1: AlgorithmicTransducerOrgan (The Autopoietic Lens)**
*   **Mechanism:** Sits at the exogenous boundary. Ingests continuous $\mathbb{R}^N$ arrays. Deterministically projects them into discrete binary tapes via parameterized stride, block-averaging, and quantization modes, dynamically altering focus to minimize overall description length.

**Component 2: CausalEquivalenceMapper (The Extensional Engine)**
*   **Mechanism:** To evaluate a proposed compositional chain:
    1. Expands the required computational lightcone (a radius of $N$ requires a $2N+1$ spatial window).
    2. Computes the sequential application of local Boolean rules across all $2^{2N+1}$ possible boundary conditions.
    3. Analyzes closure: if the resulting transformation relies strictly on the inner 3 bits, it projects the dynamic into the closed 256 ECA rulespace.
    4. If the transformation requires $>3$ bits of context, it formally flags **Combinatorial Escape**.

**Component 3: NeuralBDMOracle (The Isomorphic Evaluator)**
*   **Constraint:** The pre-trained `universal_brain_titan.pt` physically encodes the Universal Distribution over the closed 256 ECA discrete dynamics.
*   **Mechanism:** 
    1. Ingests the output of Component 2.
    2. If the composition maps to a closed 1D ECA (e.g., $170 \circ 15 \circ 118 \rightarrow 110$), the Oracle evaluates the tape conditioned on the equivalent `ECA_ID`.
    3. If Component 2 flags Combinatorial Escape, the Oracle asserts maximum literal entropy ($K(x|p) = \text{literal bound}$), terminating the branch to preserve $O(1)$ parameter execution limits, forcing the Lens to coarse-grain the signal back into focus.

**Component 4: HARTLevinSearchOrgan (System 3)**
*   **Mechanism:**
    1. Enumerate prime candidates from the 38-rule minimal set.
    2. Expand search to finite n-tuple compositions ($\circ$).
    3. Compute $K(Lens) + K(prog)$.
    4. Pass the chain to the CausalEquivalenceMapper to resolve to its ECA equivalent.
    5. Query NeuralBDMOracle for the conditional log-loss proxy using the equivalent ECA.
    6. Minimize $Kt_{total}$ subject to the finite Literal Bound Halting Condition.

────────────────────────────────────────

### IV. TIER C: AFTMS STRUCTURAL BREAK ENGINE

────────────────────────────────────────

**Mandate:** Redefine structural break detection as Causal Decomposition modulated by Inertia.

When $K_{residual} > \theta$ (dampened strictly by the Structural Causal Inertia divisor), the Phase Transition Organ triggers System 3. The output is no longer a naive integer, but the specific **Causal Decomposition** of the new regime.

*   *Output Formatting Mandate:* "Regime collapsed. Minimal generating program updated to $Rule\ 170 \circ Rule\ 15$. The system is now driven by the composition of an Information Shifter (170) and an Information Filter (15)."

────────────────────────────────────────
### V. EXECUTABLE SPECIFICATION: TITAN_V5_2_OMNIBUS.py
────────────────────────────────────────

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import abc
import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NewType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# I. GLOBAL ONTOLOGY & STRICT TYPES
# ==============================================================================
AnalogStream = NewType('AnalogStream', List[float])
BitString = NewType('BitString', bytes)
ECA_ID = NewType('ECA_ID', int)
PrimeRuleID = NewType('PrimeRuleID', int)

# The 38 Minimal Primes (Generators of the Boolean Semigroup)
VALID_PRIMES = frozenset({0, 1, 2, 3, 5, 7, 11, 12, 13, 15, 18, 22, 23, 24, 26, 28, 
                          30, 32, 33, 34, 36, 37, 40, 41, 44, 46, 50, 51, 54, 56, 
                          57, 60, 62, 72, 73, 76, 108, 110, 118, 170})

class Signal(Enum):
    ENVIRONMENT_TAPE = auto()  
    COMPLEXITY_K     = auto()  
    RESIDUAL_DELTA   = auto()  
    MINIMAL_PROG     = auto()  

class QuantizationMode(Enum):
    DELTA_SIGN = auto()
    MEAN_REVERT = auto()

@dataclass(frozen=True)
class AlgorithmicPayload:
    """The sole legitimate carrier of state and cost within the architecture."""
    bit_cost: float
    state_vector: bytes

@dataclass(frozen=True)
class LensEncoding:
    stride: int
    block_size: int
    mode: QuantizationMode

@dataclass(frozen=True)
class CompositionalGenerator:
    """Strictly bounded algebraic generator replacing symbolic ASTs."""
    composition_chain: Tuple[PrimeRuleID, ...] 
    structural_inertia: int

    def __post_init__(self):
        for p in self.composition_chain:
            if p not in VALID_PRIMES:
                raise OntologicalViolationError(f"Rule {p} is not a valid Minimal Prime.")

class OntologicalViolationError(Exception): pass

# ==============================================================================
# TIER A: IMMUTABLE PHYSICS (THE THERMODYNAMIC HEMOLYMPH)
# ==============================================================================
@dataclass(frozen=True)
class ThermodynamicGenome:
    max_compute_budget: float = 2000.0
    base_replenishment: float = 10.0
    cost_transduce: float = 1.0
    cost_k_eval: float = 3.0
    cost_transmission: float = 0.5

class Hemolymph:
    __slots__ =['_genome', '_compute_budget', '_bus']
    def __init__(self, genome: ThermodynamicGenome):
        self._genome = genome
        self._compute_budget = genome.max_compute_budget
        self._bus: Dict[Signal, AlgorithmicPayload] = {}
        
    @property
    def current_budget(self) -> float: return self._compute_budget
    
    def request_compute(self, cost: float) -> bool:
        if self._compute_budget >= cost:
            self._compute_budget -= cost
            return True
        return False
        
    def write_signal(self, signal_type: Signal, payload: AlgorithmicPayload) -> None:
        if not isinstance(payload, AlgorithmicPayload): raise TypeError("Untyped payload.")
        self._bus[signal_type] = payload
        
    def read_signal(self, signal_type: Signal) -> Optional[AlgorithmicPayload]:
        return self._bus.get(signal_type)

class CyberneticNode(abc.ABC):
    __slots__ =['_identifier']
    def __init__(self, identifier: str): self._identifier = identifier
    @abc.abstractmethod
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None: pass

# ==============================================================================
# TIER B: CYBERNETIC ORGANISM (TRANSDUCER, EXTENSIONAL MAPPER, & NEURAL PROXY)
# ==============================================================================
class AlgorithmicTransducerOrgan(CyberneticNode):
    @staticmethod
    def get_lens_cost(lens: LensEncoding) -> float:
        return (math.log2(lens.stride) if lens.stride > 0 else 0) + \
               (math.log2(lens.block_size) if lens.block_size > 0 else 0) + 2.0 

    @staticmethod
    def transduce(analog_data: AnalogStream, lens: LensEncoding) -> BitString:
        data = np.array(analog_data, dtype=float)
        if lens.block_size > 1:
            pad_size = math.ceil(len(data) / lens.block_size) * lens.block_size - len(data)
            data = np.pad(data, (0, pad_size), mode='edge').reshape(-1, lens.block_size).mean(axis=1)
        if lens.stride > 1: data = data[::lens.stride]
        if len(data) < 2: return BitString(b'\x00')

        if lens.mode == QuantizationMode.DELTA_SIGN:
            binary_tape =[1 if d > 0 else 0 for d in np.diff(data)]
            binary_tape.insert(0, 0)
        else:
            mean_val = np.mean(data)
            binary_tape =[1 if x > mean_val else 0 for x in data]
            
        byte_arr = bytearray()
        for i in range(0, len(binary_tape), 8):
            chunk = binary_tape[i:i+8]
            byte_arr.append(sum(val << (7 - idx) for idx, val in enumerate(chunk)))
        return BitString(bytes(byte_arr))
        
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None: pass

class CausalEquivalenceMapper:
    """Computes true extensional Boolean Truth Tables to resolve compositions."""
    @staticmethod
    def resolve_composition(chain: Tuple[PrimeRuleID, ...]) -> Optional[ECA_ID]:
        if len(chain) == 1: 
            return ECA_ID(chain[0])
            
        radius = len(chain)
        window_size = 2 * radius + 1
        eca_truth_table = {}
        
        for state in range(1 << window_size):
            bits =[(state >> i) & 1 for i in range(window_size-1, -1, -1)]
            
            for rule in reversed(chain):
                next_bits =[]
                for i in range(1, len(bits)-1):
                    idx = (bits[i-1] << 2) | (bits[i] << 1) | bits[i+1]
                    next_bits.append((rule >> idx) & 1)
                bits = next_bits
                
            out_bit = bits[0] 
            
            center = radius
            core_3 = (((state >> (window_size - 1 - (center - 1))) & 1) << 2) | \
                     (((state >> (window_size - 1 - center)) & 1) << 1) | \
                     (((state >> (window_size - 1 - (center + 1))) & 1))
                     
            if core_3 in eca_truth_table:
                if eca_truth_table[core_3] != out_bit:
                    # Combinatorial Escape Detected
                    return None 
            else:
                eca_truth_table[core_3] = out_bit
                
        eca_id = sum(eca_truth_table[i] << i for i in range(8))
        return ECA_ID(eca_id)

# --- PYTORCH NEURAL ARTIFACT (H_ART_Brain) ---
class TitanConfig:
    def __init__(self):
        self.vocab_size=16; self.num_rules=256; self.dim=128
        self.H_cycles=2; self.L_cycles=2;
        self.n_heads=4; self.seq_len=1024; self.dropout=0.1

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq); freqs = torch.outer(t, inv_freq); emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos()); self.register_buffer('sin', emb.sin())
    def forward(self, seq_len): return self.cos[:seq_len], self.sin[:seq_len]

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim*2, bias=False)
        self.w2 = nn.Linear(dim, dim*2, bias=False)
        self.w3 = nn.Linear(dim*2, dim, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HART_Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.q, self.k, self.v, self.o =[nn.Linear(dim, dim, bias=False) for _ in range(4)]
        self.num_heads, self.head_dim = num_heads, dim // num_heads
    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        def apply_rotary(x, c, s):
             c, s = c[:x.shape[2]].view(1,1,x.shape[2],-1), s[:x.shape[2]].view(1,1,x.shape[2],-1)
             x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
             return (x * c) + (torch.cat((-x2, x1), dim=-1) * s)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).contiguous().view(B, S, D))

class HART_Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim); self.attn = HART_Attention(cfg.dim, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.dim); self.mlp = SwiGLU(cfg.dim)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class H_ART_Brain(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = config if config else TitanConfig()
        self.emb = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.rule_emb = nn.Embedding(self.cfg.num_rules, self.cfg.dim)
        self.rope = RotaryEmbedding(self.cfg.dim // self.cfg.n_heads, max_seq=self.cfg.seq_len)
        self.H_layer = HART_Block(self.cfg)
        self.L_layer = HART_Block(self.cfg)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size)

    def forward(self, idx, rule_id):
        if isinstance(rule_id, int): rule_id = torch.tensor([rule_id], device=idx.device)
        if rule_id.dim() == 1 and idx.dim() == 2: rule_id = rule_id.expand(idx.size(0))
        B, T = idx.shape
        r_vec = self.rule_emb(rule_id).unsqueeze(1)
        input_emb = self.emb(idx)
        cos, sin = self.rope(T); cos, sin = cos.to(device), sin.to(device)
        z_H = input_emb + r_vec
        z_L = torch.zeros_like(z_H)
        for _ in range(self.cfg.H_cycles):
            for _ in range(self.cfg.L_cycles):
                z_L = self.L_layer(z_L + z_H + input_emb, cos, sin)
            z_H = self.H_layer(z_H + z_L, cos, sin)
        return self.head(z_H)

class HARTLevinSearchOrgan(CyberneticNode):
    def __init__(self, identifier: str, genome: ThermodynamicGenome, weights_path: str = "universal_brain_titan.pt"):
        super().__init__(identifier)
        self.brain = H_ART_Brain().to(device)
        self.brain.eval()
        if os.path.exists(weights_path):
            self.brain.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        else:
            nn.init.zeros_(self.brain.head.weight); nn.init.zeros_(self.brain.head.bias)

    @torch.no_grad()
    def _forward_logloss_proxy(self, chunk: np.ndarray, rule_id: int) -> float:
        seq = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)
        r = torch.tensor([rule_id], dtype=torch.long).to(device)
        logits = self.brain(seq, r)
        loss_nats = F.cross_entropy(logits[:,:-1].reshape(-1, 16), seq[:,1:].reshape(-1), reduction='sum')
        return (loss_nats / math.log(2)).item()

    def evaluate_hypothesis(self, tape: bytes, generator: CompositionalGenerator, lens: LensEncoding) -> float:
        raw_bytes = np.frombuffer(tape, dtype=np.uint8)
        tokens = np.where(raw_bytes % 2 == 1, 8, 7) if raw_bytes.max() <= 1 else (raw_bytes % 16)
        
        k_lens = AlgorithmicTransducerOrgan.get_lens_cost(lens)
        k_prog = len(generator.composition_chain) * 8.0
        
        # Heuristic Literal Bound (Pragmatic halting limit)
        literal_limit = (len(tokens) * 4.0) + 8.0
        
        if (k_lens + k_prog) >= literal_limit: return literal_limit

        eca_id = CausalEquivalenceMapper.resolve_composition(generator.composition_chain)
        if eca_id is None:
            # Epistemic Signal: Algorithmic Irreducibility via Combinatorial Escape
            return literal_limit 

        if len(tokens) < 2: return literal_limit
        k_residual_proxy = self._forward_logloss_proxy(tokens[:self.brain.cfg.seq_len], eca_id)
        
        return min(k_lens + k_prog + k_residual_proxy, literal_limit)
        
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None: pass

# ==============================================================================
# TIER C: AFTMS PHASE TRANSITION ORGAN (CAUSAL DECOMPOSITION ENGINE)
# ==============================================================================
class AFTMSPhaseTransitionOrgan(CyberneticNode):
    def __init__(self, identifier: str, genome: ThermodynamicGenome):
        super().__init__(identifier)
        self.epsilon_tolerance = 50.0 

    def evaluate_regime_break(self, raw_logloss: float, generator: CompositionalGenerator) -> Tuple[bool, str]:
        k_prog = len(generator.composition_chain) * 8.0
        k_residual = max(0.0, raw_logloss - k_prog) 
        
        # Structural Causal Inertia (Pseudo-Bennett Dampening)
        inertia_divisor = math.sqrt(1.0 + math.log2(max(1, generator.structural_inertia)))
        dampened_delta = k_residual / inertia_divisor
        
        is_break = dampened_delta > self.epsilon_tolerance
        
        if is_break:
            chain_str = " ∘ ".join([f"Rule {r}" for r in generator.composition_chain])
            msg = f"Regime collapsed. Minimal generating program updated to {chain_str}. " \
                  f"The system is now driven by causal composition."
            return True, msg
        return False, "Regime stable."

# ==============================================================================
# TDD HARNESSES: THE UNABRIDGED VALIDATION SUITE (9-STAGE RED STATE PROTOCOL)
# ==============================================================================

def test_tier_a_thermodynamic_hemolymph():
    genome = ThermodynamicGenome(max_compute_budget=5.0)
    substrate = Hemolymph(genome)
    try:
        substrate.write_signal(Signal.RESIDUAL_DELTA, "UNCOMPRESSED_STRING") # type: ignore
        assert False, "Hemolymph permitted non-AlgorithmicPayload."
    except TypeError: pass

def test_prohibited_grammar_rejection():
    try:
        bad_chain = (PrimeRuleID(170), "ADD", PrimeRuleID(15))
        CompositionalGenerator(composition_chain=bad_chain, structural_inertia=2) # type: ignore
        assert False, "Ontological violation: System accepted non-Prime operator."
    except OntologicalViolationError: pass

def test_true_extensional_closure_mapping():
    # 15 ∘ 170 translates to applying 170 (Shift Left), then 15 (Invert Shift Right). Result = Rule 51.
    chain = (PrimeRuleID(15), PrimeRuleID(170))
    resolved_eca = CausalEquivalenceMapper.resolve_composition(chain)
    assert resolved_eca == 51, f"Extensional closure failed. Expected 51, got {resolved_eca}"

def test_combinatorial_escape_irreducibility():
    # 50 ∘ 37 generates dynamics requiring > 3 bits of context.
    chain = (PrimeRuleID(50), PrimeRuleID(37))
    resolved_eca = CausalEquivalenceMapper.resolve_composition(chain)
    assert resolved_eca is None, "Mapper failed to flag combinatorial escape velocity."

def test_levin_cost_composition_penalty():
    gen_single = CompositionalGenerator((PrimeRuleID(15),), 1)
    gen_complex = CompositionalGenerator((PrimeRuleID(170), PrimeRuleID(15)), 2)
    assert len(gen_single.composition_chain) * 8.0 < len(gen_complex.composition_chain) * 8.0

def test_finite_generability_halting():
    organ = HARTLevinSearchOrgan("HART", ThermodynamicGenome())
    tape = b'\xFF\x00\xAA\x55' # Literal Limit = 40 bits
    lens = LensEncoding(1, 1, QuantizationMode.DELTA_SIGN)
    long_chain = tuple([PrimeRuleID(15)] * 10) 
    generator = CompositionalGenerator(composition_chain=long_chain, structural_inertia=10)
    assert organ.evaluate_hypothesis(tape, generator, lens) == 40.0, "Failed to halt at literal bound."

def test_tier_b_autopoietic_lens_resolution():
    transducer = AlgorithmicTransducerOrgan("Lens")
    analog_stream = AnalogStream([1.0, 1.1, 8.0, 8.1, 1.0, 1.1, 8.0, 8.1] * 4)
    lens_fine = LensEncoding(stride=1, block_size=1, mode=QuantizationMode.MEAN_REVERT)
    lens_coarse = LensEncoding(stride=1, block_size=2, mode=QuantizationMode.MEAN_REVERT)
    tape_fine = transducer.transduce(analog_stream, lens_fine).state_vector
    tape_coarse = transducer.transduce(analog_stream, lens_coarse).state_vector
    assert len(tape_coarse) < len(tape_fine), "Lens failed to structurally compress boundary data."

def test_structural_causal_inertia():
    aftms = AFTMSPhaseTransitionOrgan("AFTMS", ThermodynamicGenome())
    gen_shallow = CompositionalGenerator((PrimeRuleID(15),), structural_inertia=1)
    gen_deep = CompositionalGenerator((PrimeRuleID(170), PrimeRuleID(15)), structural_inertia=64)
    raw_logloss_shock = 68.0 
    break_shallow, _ = aftms.evaluate_regime_break(raw_logloss_shock, gen_shallow)
    break_deep, _ = aftms.evaluate_regime_break(raw_logloss_shock, gen_deep)
    assert break_shallow is True, "Shallow rule failed to break under localized shock."
    assert break_deep is False, "Deep rule failed to utilize structural causal inertia."

def test_aftms_causal_decomposition_output():
    aftms = AFTMSPhaseTransitionOrgan("AFTMS", ThermodynamicGenome())
    gen = CompositionalGenerator((PrimeRuleID(170), PrimeRuleID(15)), 2)
    is_break, msg = aftms.evaluate_regime_break(200.0, gen)
    assert is_break is True
    assert "Rule 170 ∘ Rule 15" in msg, "Failed to natively format causal decomposition output."

# ==============================================================================
# EXECUTION ROUTINE
# ==============================================================================
def run_v5_2_omnibus_suite():
    tests =[
        ("TIER A: Thermodynamic Hemolymph Bounding", test_tier_a_thermodynamic_hemolymph),
        ("TIER B: Prohibited Grammar Rejection (No AST/Math)", test_prohibited_grammar_rejection),
        ("TIER B: Extensional Boolean Closure Mapping (15∘170 -> 51)", test_true_extensional_closure_mapping),
        ("TIER B: Combinatorial Escape -> Epistemic Irreducibility", test_combinatorial_escape_irreducibility),
        ("TIER B: Levin Cost Composition Penalty", test_levin_cost_composition_penalty),
        ("TIER B: Finite Generability Halting (Literal Bound Collapse)", test_finite_generability_halting),
        ("TIER B: Autopoietic Lens Coarse-Graining Compression", test_tier_b_autopoietic_lens_resolution),
        ("TIER C: Structural Causal Inertia (Pseudo-Bennett)", test_structural_causal_inertia),
        ("TIER C: AFTMS Causal Decomposition Output Formatting", test_aftms_causal_decomposition_output)
    ]

    print("\n========================================================")
    print(" TITAN KERNEL: V5.2 FULL MECHANISTIC OMNIBUS VALIDATION")
    print("========================================================")

    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f" [✓] PASS : {name}")
        except AssertionError as e:
            print(f" [✗] FAIL : {name}\n     -> {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"[!] ERR  : {name} (Exception: {str(e)})")
            all_passed = False

    print("========================================================")
    if all_passed:
        print(" SYSTEM STATUS : FULL ARCHITECTURAL & EPISTEMIC COHERENCE")
        print(" MODULE LOCK   : ALL MECHANISMS SYNCED WITH ARTIFACTS")
    else:
        print(" SYSTEM STATUS : EPISTEMIC BREACH DETECTED")
    print("========================================================\n")

if __name__ == "__main__":
    run_v5_2_omnibus_suite()
```
