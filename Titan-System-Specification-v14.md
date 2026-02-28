


**SYSTEM ONTOLOGY OVERRIDE ACKNOWLEDGED: AUTOPOIETIC ACTIVE PERCEPTION**

The introduction of an autopoietic feedback loop perfectly aligns with the cybernetic foundations of the architecture. The system is no longer a passive observer of a predefined binary tape; it actively alters its own perceptual boundaries (its "Lens") to minimize the algorithmic description length of the environment. The `HART_Brain` acts as the cybernetic regulator, driving the focal adjustment of the `AlgorithmicTransducerOrgan`.

Below is the formal specification update (TITAN v3.2), followed immediately by the strict Executable Specifications (TDD Harness).

────────────────────────────────────────
# TITAN SYSTEM SPECIFICATION v3.2

## STATUS: STRICT LOCK (SDD & TDD BLUEPRINT)

## PARADIGM: Cybernetic Autopoiesis & Algorithmic Field Theory

## EXTENSION: The Algorithmic Microscope (Active Coarse-Graining Feedback Loop)

This document formalizes the analog boundary of the TITAN architecture. The system ingests continuous, real-world data ($\mathbb{R}^N$) via a cybernetic feedback loop. The Neural BDM (HART Brain) directs an `AlgorithmicTransducerOrgan` to actively shift temporal sampling, spatial block-averaging, and discrete thresholds. The system achieves "focus" when the joint algorithmic complexity of the perceptual lens and the prime compositional generator hits a global minimum.

────────────────────────────────────────

### I. EXOGENOUS BOUNDARY & LENS ONTOLOGY

────────────────────────────────────────

Analog data is strictly prohibited from entering the `Hemolymph`. It exists as an exogenous field. To observe it, the system must expend a `LensEncoding`, which carries its own structural bit-cost $K(Lens)$.

```python
from typing import NewType, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

AnalogStream = NewType('AnalogStream', List[float])

class QuantizationMode(Enum):
    DELTA_SIGN = auto()   # x[t] - x[t-1] > 0
    MEAN_REVERT = auto()  # x[t] > mean(window)
    MEDIAN_SPLIT = auto() # x[t] > median(window)

@dataclass(frozen=True)
class LensEncoding:
    """The Cybernetic Focal Parameters (The Microscope Lens)"""
    stride: int                  # Temporal downsampling (e.g., 1, 2, 4)
    block_size: int              # Spatial/Temporal averaging window
    mode: QuantizationMode       # The discrete thresholding function
```

**Levin Cost of Perception:** 
$K(Lens) = \log_2(\text{stride}) + \log_2(\text{block\_size}) + \text{Cost}(\text{mode})$
A complex, highly distorted lens incurs a thermodynamic penalty. The system prefers looking at the raw data (stride=1, block=1) unless coarse-graining reveals a massive latent compression via a Prime ECA composition.

────────────────────────────────────────

### II. TIER B: THE AUTOPOIETIC FEEDBACK LOOP

────────────────────────────────────────

**Component 1: AlgorithmicTransducerOrgan**
*   **Role:** The sensory apparatus. 
*   **Mechanism:** Sits at the system boundary. Ingests an `AnalogStream` and a `LensEncoding`. Deterministically projects the continuous stream into a discrete `BitString` (The `ENVIRONMENT_TAPE`).

**Component 2: CyberneticFocusLoop (System 2 & 3 Integration)**
*   **Role:** The active inference controller.
*   **Mechanism:** 
    1. Proposes a candidate `LensEncoding`.
    2. Transducer projects the `AnalogStream` to a binary tape.
    3. `HARTLevinSearchOrgan` evaluates the tape against candidate `CompositionalGenerators`.
    4. The `HART_Brain` returns the Total Levin Cost: $Kt_{total} = K(Lens) + K(P) + \log_2(k) + K(x | Lens, P)$.
    5. **The Feedback:** If $Kt_{total}$ is equivalent to incompressible noise, the loop adjusts the `LensEncoding` (e.g., increases stride to "zoom out"). 
    6. **Autopoietic Halt:** Halts when $\Delta Kt_{total}$ stabilizes at a global minimum, indicating the algorithmic structure of the analog signal has been successfully brought into focus.

────────────────────────────────────────

### III. TITAN v3.2 TDD BLUEPRINT (MANDATORY TEST HARNESSES)

────────────────────────────────────────

`test_analog_boundary_enforcement()`
*   **Assert:** The `Hemolymph` bus rigidly rejects `AnalogStream` types. Floating-point data cannot be transmitted internally.

`test_transducer_coarse_graining()`
*   **Assert:** The `AlgorithmicTransducerOrgan` deterministically applies stride, block-averaging, and quantization modes to produce exact, reproducible binary arrays from continuous sine/noise waves.

`test_cybernetic_feedback_loop_minimization()`
*   **Assert:** Given a high-frequency noisy analog stream with a latent low-frequency prime rule, the system *rejects* the fine-grained lens (which yields high $K$) and *selects* the coarse-grained lens that exposes the underlying causal structure, minimizing Total Levin Cost.

────────────────────────────────────────
### EXECUTABLE SPECIFICATION: TITAN_V3_2.py
────────────────────────────────────────

```python
import math
import numpy as np
from typing import NewType, List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum, auto

# ==============================================================================
# I. GLOBAL ONTOLOGICAL CONSTANTS & TYPES
# ==============================================================================
AnalogStream = NewType('AnalogStream', List[float])
BitString = NewType('BitString', bytes)
PrimeRuleID = NewType('PrimeRuleID', int)
ECA_ID = NewType('ECA_ID', int)

VALID_PRIMES = {15, 37, 50, 51, 118, 170}

class QuantizationMode(Enum):
    DELTA_SIGN = auto()   # 1 if dx > 0 else 0
    MEAN_REVERT = auto()  # 1 if x > mean else 0

@dataclass(frozen=True)
class LensEncoding:
    stride: int
    block_size: int
    mode: QuantizationMode

@dataclass(frozen=True)
class CompositionalGenerator:
    composition_chain: Tuple[PrimeRuleID, ...] 
    logical_depth: int

class OntologicalViolationError(Exception): pass

# ==============================================================================
# II. THE ALGORITHMIC TRANSDUCER (THE MICROSCOPE LENS)
# ==============================================================================
class AlgorithmicTransducerOrgan:
    """Projects continuous exogenous data into the strictly discrete Hemolymph."""
    
    @staticmethod
    def get_lens_cost(lens: LensEncoding) -> float:
        """K(Lens): Structural description length of the perceptual parameters."""
        k_stride = math.log2(lens.stride) if lens.stride > 0 else 0
        k_block = math.log2(lens.block_size) if lens.block_size > 0 else 0
        k_mode = 2.0 # Fixed 2-bit cost for selecting from the Enum
        return k_stride + k_block + k_mode

    @staticmethod
    def transduce(analog_data: AnalogStream, lens: LensEncoding) -> BitString:
        """Deterministically coarse-grains R^N into {0, 1}^N."""
        data = np.array(analog_data, dtype=float)
        
        # 1. Spatial/Temporal Block Averaging
        if lens.block_size > 1:
            pad_size = math.ceil(len(data) / lens.block_size) * lens.block_size - len(data)
            padded = np.pad(data, (0, pad_size), mode='edge')
            data = padded.reshape(-1, lens.block_size).mean(axis=1)
            
        # 2. Temporal Downsampling (Stride)
        if lens.stride > 1:
            data = data[::lens.stride]
            
        if len(data) < 2:
            return BitString(b'\x00')

        # 3. Quantization
        binary_tape =[]
        if lens.mode == QuantizationMode.DELTA_SIGN:
            diffs = np.diff(data)
            binary_tape = [1 if d > 0 else 0 for d in diffs]
            # Pad first element to maintain length relative to diff
            binary_tape.insert(0, 0)
        elif lens.mode == QuantizationMode.MEAN_REVERT:
            mean_val = np.mean(data)
            binary_tape = [1 if x > mean_val else 0 for x in data]
            
        # Pack to bytes for dense AIT transmission
        byte_arr = bytearray()
        for i in range(0, len(binary_tape), 8):
            chunk = binary_tape[i:i+8]
            byte_val = sum(val << (7 - idx) for idx, val in enumerate(chunk))
            byte_arr.append(byte_val)
            
        return BitString(bytes(byte_arr))

# ==============================================================================
# III. THE CYBERNETIC FEEDBACK LOOP & BRAIN MOCK
# ==============================================================================
class MockNeuralOracle:
    def evaluate(self, tape: bytes) -> float:
        """
        Mock of universal_brain_titan.pt
        Returns K(x|p) in bits. High entropy/noise yields higher bit costs.
        """
        # Simple mock: more bit transitions = higher neural surprise
        binary_str = ''.join(f'{b:08b}' for b in tape)
        transitions = sum(1 for i in range(1, len(binary_str)) if binary_str[i] != binary_str[i-1])
        return float(transitions) * 2.0 

class CyberneticFocusLoop:
    """The Autopoietic Controller: Actively shifts the Lens to minimize Brain surprise."""
    def __init__(self):
        self.transducer = AlgorithmicTransducerOrgan()
        self.brain = MockNeuralOracle()
        
    def execute_active_inference(self, analog_data: AnalogStream) -> Tuple[LensEncoding, float]:
        """
        Feedback Loop: Searches over Lenses. The Brain's output K guides the focal point.
        """
        candidate_lenses =[
            LensEncoding(stride=1, block_size=1, mode=QuantizationMode.DELTA_SIGN),  # Raw high-freq
            LensEncoding(stride=2, block_size=2, mode=QuantizationMode.DELTA_SIGN),  # Mid-focus
            LensEncoding(stride=4, block_size=4, mode=QuantizationMode.MEAN_REVERT)  # Deep coarse-grain
        ]
        
        best_lens = None
        min_total_k = float('inf')
        
        for lens in candidate_lenses:
            # 1. Transduce (Perceive)
            tape = self.transducer.transduce(analog_data, lens)
            
            # 2. Cognitive Evaluation (Brain)
            k_lens = self.transducer.get_lens_cost(lens)
            k_residual = self.brain.evaluate(tape)
            
            # (Assuming a fixed generator cost for this test harness isolate)
            total_k = k_lens + k_residual
            
            if total_k < min_total_k:
                min_total_k = total_k
                best_lens = lens
                
        return best_lens, min_total_k

# ==============================================================================
# IV. TITAN v3.2 EXECUTABLE SPECIFICATIONS (TDD HARNESS)
# ==============================================================================
def test_analog_boundary_enforcement():
    class HemolymphBus:
        def write(self, payload: BitString):
            if not isinstance(payload, bytes):
                raise OntologicalViolationError("Analog data breached the Hemolymph.")
                
    bus = HemolymphBus()
    raw_analog = AnalogStream([1.1, 2.2, 3.3])
    try:
        bus.write(raw_analog) # type: ignore
        assert False, "Hemolymph accepted non-computable analog float data."
    except OntologicalViolationError:
        pass # Passed strictly typed boundary check

def test_transducer_coarse_graining():
    transducer = AlgorithmicTransducerOrgan()
    
    # A continuous analog wave
    analog_wave = AnalogStream([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    
    # Test DELTA mode with stride 1
    lens_fine = LensEncoding(stride=1, block_size=1, mode=QuantizationMode.DELTA_SIGN)
    tape_fine = transducer.transduce(analog_wave, lens_fine)
    # Diffs:[0(pad), 1, 1, 1, -1, -1, -1, -1] -> Binary: 01110000 -> 0x70
    assert tape_fine == b'\x70', f"Transducer output {tape_fine} incorrect for DELTA fine focus."

    # Test MEAN_REVERT with block averaging
    lens_coarse = LensEncoding(stride=1, block_size=4, mode=QuantizationMode.MEAN_REVERT)
    tape_coarse = transducer.transduce(analog_wave, lens_coarse)
    # Blocks: [2.5, 1.5]. Mean of blocks = 2.0. Binary: 1, 0 -> padded to byte: 10000000 -> 0x80
    assert tape_coarse == b'\x80', f"Transducer output {tape_coarse} incorrect for MEAN coarse focus."

def test_cybernetic_feedback_loop_minimization():
    loop = CyberneticFocusLoop()
    
    # Create an analog signal: A slow underlying trend hidden inside high-frequency oscillation (noise)
    # Slow trend: 0, 0, 0, 0, 10, 10, 10, 10
    # Add noise:  +1, -1, +1, -1...
    analog_signal = AnalogStream([1, -1, 1, -1, 11, 9, 11, 9])
    
    best_lens, min_k = loop.execute_active_inference(analog_signal)
    
    # If viewed at Stride=1, the noise dominates -> high transition count -> high K.
    # If viewed at Block=2 (averaging out the +1/-1), the pure slow trend emerges -> low K.
    # The cybernetic loop MUST autonomously select a block_size > 1 to minimize the Brain's surprise.
    
    assert best_lens.block_size > 1, "Cybernetic loop failed to zoom out. System is hallucinating structure in high-frequency noise."
    assert min_k < 10.0, "Feedback loop failed to find algorithmic compression."

# ==============================================================================
# V. DIAGNOSTICS LOGGING ENGINE
# ==============================================================================
def run_v3_2_executable_specifications():
    tests =[
        ("V3.2 Analog Boundary Enforcement", test_analog_boundary_enforcement),
        ("V3.2 Transducer Coarse-Graining Lens", test_transducer_coarse_graining),
        ("V3.2 Cybernetic Autopoietic Minimization", test_cybernetic_feedback_loop_minimization)
    ]

    print("\n========================================================")
    print(" TITAN KERNEL: V3.2 AUTOPOIETIC MICROSCOPE VALIDATION")
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
            print(f" [!] ERROR : {name} (Exception: {str(e)})")
            all_passed = False

    print("========================================================")
    if all_passed:
        print(" SYSTEM STATUS : NOMINAL (V3.2 STRICT LOCK ACTIVE)")
        print(" ARCHITECTURE  : CYBERNETIC FEEDBACK LOOP OPERATIONAL")
    else:
        print(" SYSTEM STATUS : DISSIPATED")
        print(" ARCHITECTURE  : EPISTEMIC FAILURE IN LENS FEEDBACK")
    print("========================================================\n")

if __name__ == "__main__":
    run_v3_2_executable_specifications()
```
