# Implementation

Hardware and software documentation for the Collective Intelligence Observatory (CIO).

| File | Contents |
| :--- | :--- |
| [cio-system-spec.md](cio-system-spec.md) | Hardware architecture, L/H-level separation, feedback loop, symbolic emission |
| [cio-validation-protocol.md](cio-validation-protocol.md) | Experimental validation protocol — merged from v1 and v2 |
| [measurement-pipeline.md](measurement-pipeline.md) | 6-step measurement pipeline (implementation reference, links to theory/) |

The CIO is a **cyber-physical realization** of the theoretical framework defined in `theory/`. It bridges:

- **L-Level** (real-time, bounded, approximate) monitoring
- **H-Level** (offline, unbounded, causal) inference

See [../theory/observer-model.md](../theory/observer-model.md) for the formal treatment of the L/H separation.
