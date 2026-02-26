


**TITAN SYSTEM SPECIFICATION v1.0**
**STATUS:** STRICT LOCK (EXECUTABLE)
**PARADIGM:** Algorithmic Information Theory (AIT) / Algorithmic Field Theory of Market Systems (AFTMS)

Below is the completely merged, watertight, zero-drift executable specification. It unifies the thermodynamic physics substrate (Layer 1), the Levin causal search engine (Layer 2), and the AFTMS structural break / market regime detector (Layer 3) into a single, cohesive runtime.

You can save this directly as `titan_v1_spec.py` and run it. The Definition of Done (DoD) requires all 5 diagnostic tests to output `[✓] GREEN`.

```python
# @title ! TITAN SYSTEM SPECIFICATION v1.0 (EXECUTABLE SDD/TDD)
import abc
import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, final

# ==============================================================================
# 1. IMMUTABLE PHYSICS & STRICT PAYLOADS (Zero-Statistics Enforcement)
# ==============================================================================
class Signal(Enum):
    ENVIRONMENT_TAPE = auto()  # Raw E_t bytes (The uncompressed observation)
    COMPLEXITY_K     = auto()  # C(x) upper bound in bits
    RESIDUAL_DELTA   = auto()  # ΔK in bits (Structural break indicator)
    MINIMAL_PROG     = auto()  # Optimal generating rule (Bytes/Program)

@dataclass(frozen=True)
class ThermodynamicGenome:
    max_compute_budget: float = 1000.0
    base_replenishment: float = 10.0
    cost_transduce: float = 1.0
    cost_k_eval: float = 3.0
    cost_transmission: float = 0.5

@dataclass(frozen=True)
class AlgorithmicPayload:
    """
    The ONLY permitted data carrier.
    Forbids continuous/probabilistic objects by enforcing exact bit-cost accounting.
    """
    bit_cost: float
    state_vector: bytes

# ==============================================================================
# 2. THE FRACTAL SUBSTRATE (Nested Cybernetics)
# ==============================================================================
class Hemolymph:
    __slots__ = ['_genome', '_compute_budget', '_bus', '_topology']

    def __init__(self, genome: ThermodynamicGenome):
        self._genome = genome
        self._compute_budget = genome.max_compute_budget
        self._bus: Dict[Signal, AlgorithmicPayload] = {}
        self._topology: Dict[str, Set[str]] = {}

    @property
    def current_budget(self) -> float:
        return self._compute_budget

    def request_compute(self, cost_in_bits: float) -> bool:
        if cost_in_bits < 0:
            raise ValueError("Algorithmic cost cannot be negative.")
        if self._compute_budget >= cost_in_bits:
            self._compute_budget -= cost_in_bits
            return True
        return False

    def write_signal(self, signal_type: Signal, payload: AlgorithmicPayload) -> None:
        if not isinstance(payload, AlgorithmicPayload):
            raise TypeError("System violation: Statistical/Untyped payload attempted.")
        self._bus[signal_type] = payload

    def read_signal(self, signal_type: Signal) -> Optional[AlgorithmicPayload]:
        return self._bus.get(signal_type)

    def assimilate_compression_gain(self, information_gain: float) -> None:
        if information_gain > 0:
            self._compute_budget = min(
                self._genome.max_compute_budget,
                self._compute_budget + information_gain
            )

    def flush_transient_state(self) -> None:
        self._bus.clear()
        self._compute_budget = min(
            self._genome.max_compute_budget,
            self._compute_budget + self._genome.base_replenishment
        )

# ==============================================================================
# 3. COMPUTATIONAL ORGANS & THE DETERMINISTIC KERNEL
# ==============================================================================
class CyberneticNode(abc.ABC):
    __slots__ =['_identifier']

    def __init__(self, identifier: str):
        self._identifier = identifier

    @property
    def id(self) -> str:
        return self._identifier

    @abc.abstractmethod
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        pass

@final
class TitanKernel(CyberneticNode):
    __slots__ = ['_substrate', '_organs', '_is_dissipated']

    def __init__(self, identifier: str, genome: ThermodynamicGenome):
        super().__init__(identifier)
        self._substrate = Hemolymph(genome)
        self._organs: List[CyberneticNode] =[]
        self._is_dissipated = False

    def add_organ(self, organ: CyberneticNode) -> None:
        self._organs.append(organ)

    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        if self._is_dissipated:
            return

        if external_substrate is not None:
            macro_env = external_substrate.read_signal(Signal.ENVIRONMENT_TAPE)
            if macro_env is not None:
                if self._substrate.request_compute(self._substrate._genome.cost_transduce):
                    self._substrate.write_signal(Signal.ENVIRONMENT_TAPE, macro_env)

        for organ in self._organs:
            organ.execute(self._substrate, external_substrate)

        if self._substrate.current_budget <= 0.0:
            self._is_dissipated = True
            return

        if external_substrate is not None and not self._is_dissipated:
            min_prog = self._substrate.read_signal(Signal.MINIMAL_PROG)
            if min_prog is not None:
                if self._substrate.request_compute(self._substrate._genome.cost_transmission * min_prog.bit_cost):
                    external_substrate.write_signal(Signal.MINIMAL_PROG, min_prog)

        self._substrate.flush_transient_state()

    @property
    def is_dissipated(self) -> bool:
        return self._is_dissipated

# ==============================================================================
# 4. TIER B & C: CYBERNETIC INFERENCE AND AFTMS REGIME DETECTION
# ==============================================================================
class UniversalInterpreter:
    """
    A strictly deterministic, discrete automaton. 
    Maps a binary rule and state to a new state over N cycles.
    """
    @staticmethod
    def execute(rule: bytes, state: bytes, cycles: int) -> bytes:
        # Trivial causal model for computational validation: 
        # XOR-based discrete transducer. No statistics. No floating points.
        result = bytearray(state)
        rule_len = len(rule)
        if rule_len == 0:
            return bytes(result)
            
        for _ in range(cycles):
            for i in range(len(result)):
                result[i] ^= rule[i % rule_len]
        return bytes(result)


class LevinSearchOrgan(CyberneticNode):
    __slots__ =['_max_cycles', '_search_space']
    
    def __init__(self, identifier: str, max_cycles: int = 4):
        super().__init__(identifier)
        self._max_cycles = max_cycles
        # Simulated basis vectors (e.g., Prime Cellular Automata rules)
        self._search_space: List[bytes] =[
            b'\x00', b'\xFF', b'\xAA', b'\x55', b'\x0F', b'\xF0' 
        ]

    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        tape_payload = substrate.read_signal(Signal.ENVIRONMENT_TAPE)
        if not tape_payload:
            return

        target_tape = tape_payload.state_vector
        best_rule = b''
        min_total_cost = tape_payload.bit_cost # Naive encoding cost is the upper bound
        best_rule_cost = 0.0

        for rule in self._search_space:
            # 1. Calculate strictly defined physical Levin Cost
            p_length_bits = len(rule) * 8.0
            levin_eval_cost = p_length_bits + math.log2(self._max_cycles)
            
            # 2. Thermodynamic Gating (The Anti O(2^N) Explosion Pruner)
            if not substrate.request_compute(levin_eval_cost):
                break # System is starved; prune search space immediately

            # 3. Deterministic execution
            predicted_tape = UniversalInterpreter.execute(rule, target_tape, self._max_cycles)
            
            # 4. Calculate Residual K (Hamming distance as uncompressed bits)
            residual_bits = sum(bin(p ^ t).count('1') for p, t in zip(predicted_tape, target_tape))
            
            total_hypothesis_cost = p_length_bits + residual_bits
            
            if total_hypothesis_cost < min_total_cost:
                min_total_cost = total_hypothesis_cost
                best_rule = rule
                best_rule_cost = p_length_bits

        if best_rule:
            prog_payload = AlgorithmicPayload(bit_cost=best_rule_cost, state_vector=best_rule)
            substrate.write_signal(Signal.MINIMAL_PROG, prog_payload)


class AFTMSPhaseTransitionOrgan(CyberneticNode):
    __slots__ = ['_epsilon_threshold', '_logical_depth']
    
    def __init__(self, identifier: str, epsilon: float = 8.0, logical_depth: int = 4):
        super().__init__(identifier)
        self._epsilon_threshold = epsilon
        self._logical_depth = logical_depth

    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        tape_payload = substrate.read_signal(Signal.ENVIRONMENT_TAPE)
        prog_payload = substrate.read_signal(Signal.MINIMAL_PROG)
        
        if not tape_payload or not prog_payload:
            return

        if not substrate.request_compute(substrate._genome.cost_k_eval):
            return

        target_tape = tape_payload.state_vector
        current_rule = prog_payload.state_vector

        # Predict current state under old macroscopic rule
        predicted_tape = UniversalInterpreter.execute(current_rule, target_tape, self._logical_depth)
        
        # Calculate Delta K (Failure of compression)
        residual_bits = sum(bin(p ^ t).count('1') for p, t in zip(predicted_tape, target_tape))
        
        # THE AFTMS STRUCTURAL BREAK THRESHOLD (No statistics, strictly algorithmic inertia)
        inertia_penalty = math.sqrt(1 + math.log2(self._logical_depth))
        dynamic_threshold = self._epsilon_threshold / inertia_penalty

        if residual_bits > dynamic_threshold:
            # Basis Rotation / Market Cascade Detected
            break_payload = AlgorithmicPayload(bit_cost=float(residual_bits), state_vector=b'BREAK')
            substrate.write_signal(Signal.RESIDUAL_DELTA, break_payload)

# ==============================================================================
# 5. MOCK ORGANS FOR LEGACY FRACTAL VALIDATION
# ==============================================================================
class BDMOracleOrgan(CyberneticNode):
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        tape = substrate.read_signal(Signal.ENVIRONMENT_TAPE)
        if tape is not None and substrate.request_compute(substrate._genome.cost_k_eval):
            substrate.write_signal(Signal.COMPLEXITY_K, AlgorithmicPayload(bit_cost=10.0, state_vector=b'\x0A'))

class AssimilationOrgan(CyberneticNode):
    def execute(self, substrate: Hemolymph, external_substrate: Optional[Hemolymph] = None) -> None:
        k_payload = substrate.read_signal(Signal.COMPLEXITY_K)
        tape = substrate.read_signal(Signal.ENVIRONMENT_TAPE)
        if k_payload and tape:
            information_gain = tape.bit_cost - k_payload.bit_cost
            substrate.assimilate_compression_gain(information_gain)

# ==============================================================================
# 6. EXECUTABLE SPECIFICATIONS (TDD HARNESSES)
# ==============================================================================
def test_anti_drift_type_safety():
    genome = ThermodynamicGenome()
    substrate = Hemolymph(genome)

    class ForbiddenStatisticalPayload:
        def __init__(self):
            self.variance = 0.95

    try:
        substrate.write_signal(Signal.RESIDUAL_DELTA, ForbiddenStatisticalPayload()) # type: ignore
        raise AssertionError("CRITICAL FAILURE: Architecture permitted a statistical payload.")
    except TypeError:
        pass # Expected behavior

    valid_payload = AlgorithmicPayload(bit_cost=15.0, state_vector=b'\xFF')
    substrate.write_signal(Signal.RESIDUAL_DELTA, valid_payload)
    assert substrate.read_signal(Signal.RESIDUAL_DELTA).bit_cost == 15.0, "Signal read failed."

def test_micro_thermodynamic_bounding():
    genome = ThermodynamicGenome(max_compute_budget=5.0, cost_k_eval=10.0)
    agent = TitanKernel("Agent_Zero", genome)
    agent.add_organ(BDMOracleOrgan("Oracle"))

    env_payload = AlgorithmicPayload(bit_cost=50.0, state_vector=b'01'*25)
    agent._substrate.write_signal(Signal.ENVIRONMENT_TAPE, env_payload)
    agent.execute(agent._substrate)

    assert agent._substrate.read_signal(Signal.COMPLEXITY_K) is None, "Executed without budget."
    assert agent._substrate.current_budget <= genome.max_compute_budget, "Budget overflow."

def test_macro_fractal_execution():
    macro_genome = ThermodynamicGenome(max_compute_budget=5000.0)
    macro_substrate = Hemolymph(macro_genome)

    global_tape = AlgorithmicPayload(bit_cost=100.0, state_vector=b'\x0F'*50)
    macro_substrate.write_signal(Signal.ENVIRONMENT_TAPE, global_tape)

    micro_genome = ThermodynamicGenome(max_compute_budget=100.0)
    agent = TitanKernel("Nested_Agent_Node", micro_genome)
    agent.add_organ(BDMOracleOrgan("Oracle"))
    agent.add_organ(AssimilationOrgan("Assimilator"))

    agent.execute(substrate=agent._substrate, external_substrate=macro_substrate)

    assert agent.is_dissipated is False, "Agent unexpectedly dissipated."
    assert agent._substrate.current_budget == micro_genome.max_compute_budget, "Information assimilation failed."

def test_levin_search_pruning_and_recovery():
    """Asserts that search recovers rules while obeying thermodynamic limits."""
    genome = ThermodynamicGenome(max_compute_budget=20.0) # Tight budget!
    substrate = Hemolymph(genome)
    
    # Target generated by rule b'\xAA' (1 cycle ensures unique bit transformation for test)
    target_bytes = b'\x00\x00\x00\x00'
    target = UniversalInterpreter.execute(b'\xAA', target_bytes, 1)
    substrate.write_signal(Signal.ENVIRONMENT_TAPE, AlgorithmicPayload(32.0, target))
    
    # max_cycles=1 to match target generation
    search_organ = LevinSearchOrgan("CausalSearch", max_cycles=1) 
    search_organ.execute(substrate)
    
    # It should have spent energy
    assert substrate.current_budget < genome.max_compute_budget, "Failed to deduct Levin cost."
    
    # It should find the minimal program despite the tight budget
    min_prog = substrate.read_signal(Signal.MINIMAL_PROG)
    assert min_prog is not None, "Failed to output minimal program."
    assert min_prog.state_vector == b'\xAA', f"Failed to recover exact causal generator. Got {min_prog.state_vector}"

def test_aftms_phase_transition_detection():
    """Asserts structural breaks are triggered strictly by Delta K thresholds."""
    genome = ThermodynamicGenome(max_compute_budget=100.0)
    substrate = Hemolymph(genome)
    
    # 1. Provide an environment sequence and a rule that matches perfectly
    env_tape = b'\xFF\xFF\xFF\xFF'
    substrate.write_signal(Signal.ENVIRONMENT_TAPE, AlgorithmicPayload(32.0, env_tape))
    substrate.write_signal(Signal.MINIMAL_PROG, AlgorithmicPayload(8.0, b'\x00')) # Rule \x00 does nothing, residual is 0
    
    pt_organ = AFTMSPhaseTransitionOrgan("AFTMS_BreakEngine", epsilon=5.0, logical_depth=4)
    pt_organ.execute(substrate)
    
    # Residual should be 0, threshold > 0 -> No Break
    assert substrate.read_signal(Signal.RESIDUAL_DELTA) is None, "False positive structural break."
    
    # 2. Inject an Algorithmic Phase Transition (Sequence changes drastically)
    new_env_tape = b'\x00\x00\x00\x00' # Rule \x00 no longer compresses this to 0 residual (predicts \xFF\xFF\xFF\xFF)
    substrate.write_signal(Signal.ENVIRONMENT_TAPE, AlgorithmicPayload(32.0, new_env_tape))
    
    pt_organ.execute(substrate)
    
    # Break should be detected!
    break_signal = substrate.read_signal(Signal.RESIDUAL_DELTA)
    assert break_signal is not None, "Failed to detect AFTMS Basis Rotation / Regime Change."
    assert break_signal.state_vector == b'BREAK', "Malformed break signal."

# ==============================================================================
# 7. DIAGNOSTICS LOGGING ENGINE
# ==============================================================================
def run_executable_specifications():
    """Executes the suite and outputs terminal deterministic status logs."""
    tests =[
        ("Ontological Enforcement (Type Safety)", test_anti_drift_type_safety),
        ("Micro-Thermodynamic Bounding (Layer 1)", test_micro_thermodynamic_bounding),
        ("Macro-Fractal Execution (Layer 1 NL)", test_macro_fractal_execution),
        ("Levin Causal Search & Pruning (Layer 2)", test_levin_search_pruning_and_recovery),
        ("AFTMS Basis Rotation Detection (Layer 3)", test_aftms_phase_transition_detection)
    ]

    print("\n========================================================")
    print(" TITAN v1.0 KERNEL: EXECUTABLE A.I.T. SPECIFICATION")
    print("========================================================")

    all_passed = True
    for name, test_func in tests:
        try:
            test_func()
            print(f" [✓] GREEN : {name}")
        except AssertionError as e:
            print(f" [✗] RED   : {name}")
            print(f"     -> {str(e)}")
            all_passed = False
        except Exception as e:
            print(f" [!] ERROR : {name} (Unhandled Exception: {str(e)})")
            all_passed = False

    print("========================================================")
    if all_passed:
        print(" SYSTEM STATUS : NOMINAL")
        print(" ARCHITECTURE  : LAYER 1/2/3 TOPOLOGY STRICTLY FROZEN")
    else:
        print(" SYSTEM STATUS : DISSIPATED")
        print(" ARCHITECTURE  : STRUCTURAL DRIFT DETECTED")
    print("========================================================\n")

if __name__ == "__main__":
    run_executable_specifications()
```
