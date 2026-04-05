# RDMP — MACYB Master v4
## Repeatable Development, Metadata, and Knowledge Base Protocol

**Owner:** algoplexity  
**Scope:**  
- `algoplexity/algoplexity`
- `algoplexity/ANU-MACYB-private`
- `algoplexity/ANU-MACYB-public`
- Zotero
- Drive
- AI coding assistants

**Purpose:**  
This protocol defines how MACYB operates as a **repeatable, iterative, LLM-assisted knowledge system** spanning theory, implementation, publication, and archival recordkeeping.

MACYB/CIO is a three-repo, living knowledge system with a stable folder grammar and explicit provenance rules.

---

# 0. Core principle

MACYB is a three-layer system:

- **A_Theory** — canonical concepts and formal theory
- **B_Computation** — private implementation and working knowledge base
- **C_Measurement_CIO** — public-safe curated mirror and communication layer

The system improves through repeated cycles of:
- ingesting sources,
- compiling knowledge,
- implementing theory,
- producing outputs,
- linting and repairing,
- publishing safe subsets,
- and preserving provenance.

# 0.1 CIO three-repo architecture

The CIO programme operates as a coordinated three-repository system:

- `algoplexity/algoplexity` — theory source and canonical conceptual frame
- `algoplexity/ANU-MACYB-private` — private working knowledge engine and implementation layer
- `algoplexity/ANU-MACYB-public` — curated public mirror and communication layer

These repositories should be treated as structurally aligned layers of one research programme. They are not independent projects.

The objective is to keep their organizational grammar broadly consistent so that theory, private working notes, and public-safe outputs can be maintained with low cognitive overhead over long time horizons.

# 0.2 CIO cross-repo folder standard

To reduce cognitive load and support long-term maintenance, the CIO programme should use a shared folder grammar across the three repositories wherever practical.

## Shared folder roles

- `00_MANIFESTS/` — registry, sync status, indexes, artifact inventory
- `01_THEORY/` — canonical theory statements, formal definitions, theory notes
- `02_KNOWLEDGE/` — living wiki-style compiled knowledge pages, concept pages, backlinks, summaries
- `03_WHITEPAPERS/` — longform explanatory documents, technical notes, public/private-facing whitepapers
- `04_PROJECTS/` — project-specific material for CIO subprojects
- `05_OUTPUTS/` — generated deliverables, exports, charts, tables, reusable summaries
- `06_ARCHIVE/` — historical, superseded, or preserved artefacts

## Repository intent

### `algoplexity/algoplexity`
Theory source and canonical conceptual frame.
- prioritize `01_THEORY/`
- preserve stable terminology
- archive prior theory versions in `06_ARCHIVE/`

### `ANU-MACYB-private`
Working knowledge engine and implementation layer.
- use the full folder grammar where useful
- keep internal notes, decision logs, and richer derivations
- maintain provenance and sync metadata

### `ANU-MACYB-public`
Curated public mirror and communication layer.
- mirror only safe and stable concepts
- keep naming aligned with the private repo where possible
- omit private or unstable derivations

## Alignment rule

The three repositories should be structurally aligned, but not content-identical.

This means:
- same conceptual map
- same folder grammar where practical
- different depth and safety by layer

## Knowledge lifecycle

Major artefacts should follow the lifecycle:

1. ingest
2. compile
3. derive
4. validate
5. publish
6. archive

## Divergence rule

Divergence is allowed only when intentional, such as:
- private-only reasoning
- public-safe omission
- staging lag
- experimental notes

Whenever divergence occurs, record:
- what diverged
- why
- whether temporary or permanent
- which repository is canonical for that slice

---

# 1. Systems of record

## 1.1 Zotero
Authoritative for:
- references
- metadata
- source notes
- PDFs / URLs
- literature organization
- provenance anchors

## 1.2 `ANU-MACYB-private`
Authoritative for:
- working markdown drafts
- private implementation
- internal architecture
- LLM-maintained compiled knowledge
- project notes
- decision logs
- pre-publication staging

## 1.3 Drive
Authoritative for:
- final submitted PDFs / Docs
- feedback files
- large or binary artefacts
- archived deliverables
- authoritative final snapshots

## 1.4 `ANU-MACYB-public`
Authoritative for:
- sanitized public documentation
- curated summaries
- public Pages content
- safe demonstrations
- shareable knowledge outputs

---

# 2. Repo roles

## 2.1 `algoplexity/algoplexity` — Theory source
Defines:
- foundational concepts
- formal theory
- research framing
- canonical terminology
- theory revisions
- slices of CI-Unified theory

## 2.2 `ANU-MACYB-private` — Working implementation and knowledge engine
Contains:
- implementation of selected theory slices
- working docs and code
- LLM-maintained compiled knowledge
- internal design notes
- simulation and experimentation
- decisions and provenance

## 2.3 `ANU-MACYB-public` — Curated public mirror
Contains:
- sanitized summaries
- public-facing docs
- Pages site content
- stable explanation of the selected theory slice
- public-safe demos and examples

---

# 3. Canonical source hierarchy

When content conflicts, use this order:

1. Direct user instruction
2. Repository reality
3. `algoplexity/algoplexity` theory
4. `ANU-MACYB-private` implementation
5. `ANU-MACYB-public` publication
6. Drive final deliverables

If theory changes, downstream implementation and public layers must be reviewed.  
If private implementation changes, public mirror must be reviewed.  
If public wording changes, it must be checked against private and theory sources.

---

# 4. Zotero metadata protocol

## 4.1 Zotero is mandatory
All external sources and major conceptual influences should be tracked in Zotero.

## 4.2 Collections
Suggested collections:
- MACYB
  - Theory
  - CPS1-individual
  - CPS2-group
  - Methods
  - Public-writing
  - Background reading

## 4.3 Required fields
For each item:
- Title
- Author(s)
- Year
- Source type
- DOI / URL
- Tags
- Relevance note
- Linked repo/doc
- Related theory or implementation slice

## 4.4 Citation rule
If a repo artefact depends on a source:
- capture it in Zotero
- reference it in the artefact
- keep the note traceable to the source

---

# 5. Knowledge base operating model

MACYB is also a living knowledge base maintained by LLMs.

## 5.1 Knowledge layers
Use these logical layers:

- **raw/** — source capture
- **compiled/** — LLM-structured wiki
- **derived/** — outputs and syntheses
- **validated/** — linted and repaired artefacts
- **public/** — sanitized mirror
- **archive/** — historical artefacts

## 5.2 Ingestion
Sources may include:
- papers
- articles
- repos
- datasets
- images
- clips
- notes
- transcripts

All should be captured with provenance.

## 5.3 Compilation
The LLM should:
- summarize sources
- create concept pages
- build backlinks
- maintain indices
- extract relationships
- note uncertainty
- propose new article candidates

## 5.4 Query loop
When asked a question, the LLM should:
1. inspect compiled knowledge
2. consult raw sources if needed
3. synthesize an answer
4. generate reusable output if useful
5. file the output back into the knowledge base

## 5.5 Output filing
Outputs such as memos, slides, diagrams, charts, and tables should be filed back into the knowledge system unless strictly ephemeral.

## 5.6 Linting
The LLM should periodically check for:
- inconsistent data
- duplicate pages
- missing metadata
- stale concepts
- orphaned pages
- drift in terminology
- contradiction between theory and implementation

---

## 5.7 CIO living knowledge base

The CIO programme is maintained as a living, versioned knowledge base.

LLMs may be used to:
- ingest sources
- compile structured markdown pages
- maintain indices and backlinks
- generate reusable summaries
- file outputs back into the repository system
- surface contradictions and stale content
- support iterative refinement over time

This is not an uncontrolled wiki. All compiled knowledge must preserve provenance, traceability, and reviewability.

---

# 6. Change propagation rules

## 6.1 Theory → Private
If theory changes:
- update dependent private docs
- review implementation terminology
- update provenance blocks
- mark public content for review

## 6.2 Private → Public
If private implementation changes:
- sanitize the public mirror
- remove unstable/private detail
- preserve conceptual alignment
- update public docs only if safe

## 6.3 Public → Review
If public content changes:
- verify it matches private source
- verify it matches theory source
- verify it adds no new meaning

---

# 7. Iteration protocol

## 7.1 Theory revision
Occurs when a definition, formalism, scope boundary, or concept name changes.

Required:
- update private implementation docs
- update Zotero notes if relevant
- update public mirror if affected

## 7.2 Implementation revision
Occurs when code, metrics, pipeline stages, or architecture changes.

Required:
- record in decision log
- link to upstream theory
- update public-safe materials if necessary

## 7.3 Publication revision
Occurs when a stable private artifact is sanitized for public release.

Required:
- verify safety
- verify traceability
- preserve conceptual fidelity
- avoid leaking private work

---

# 8. Traceability and provenance

Every major artefact should record:
- source theory
- implementation source
- public mirror target
- Zotero items
- sync status
- last reviewed date
- divergence notes

## Suggested metadata block
```markdown
## Metadata
- Source theory: algoplexity/algoplexity
- Implementation source: ANU-MACYB-private/projects/cps1-individual/...
- Public mirror: ANU-MACYB-public/projects/cps1-individual/...
- Zotero collection: MACYB / Theory
- Upstream source(s): ...
- Sync status: aligned | partial | diverged
- Last reviewed: YYYY-MM-DD
- Notes: ...
```

## 8.1 Cross-repo alignment

CIO content is expected to remain broadly aligned across:
- theory source
- private working repository
- public mirror

Exact wording may differ by layer, but the underlying concepts, folder grammar, and source lineage should remain traceable.

If a concept changes in the theory layer, downstream layers should be reviewed.
If private material changes, public-safe derivatives should be reviewed.
If public wording changes, it must not introduce new meaning beyond the upstream sources.

---

# 9. Terminology governance

## Canonical terms
These should be treated carefully:
- theory names
- metric names
- architecture layer names
- repo role names
- system-of-record labels

## Allowed drift
- explanatory prose
- public narrative wording
- local experimental labels
- implementation shorthand

## Frozen terms
If a cross-repo term changes:
- theory must be reviewed
- implementation must be reviewed
- public wording must be reviewed

---

# 10. Divergence handling

Divergence is allowed only when intentional.

## Examples
- implementation simplification
- staged publication lag
- public-safe omission
- experimental terminology

## Required response
If divergence exists:
- record why
- record whether temporary or permanent
- identify affected repos/files
- mark sync status

---

# 11. Folder expectations

## 11.1 `ANU-MACYB-private`
- working drafts
- internal architecture
- decision logs
- metadata manifests
- compiled knowledge
- outputs staged for review

## 11.2 `ANU-MACYB-public`
- curated docs
- blog posts
- Pages content
- safe examples
- stable public knowledge

## 11.3 Drive
- authoritative final files
- large artefacts
- submission exports
- feedback files

## 11.4 Zotero
- bibliographic metadata
- notes
- source links
- PDFs / attachments

## 11.5 CIO repository grammar

Where possible, the CIO repositories should use a shared conceptual folder grammar so that users can navigate the system consistently across theory, private, and public layers.

Recommended shared roles:
- `00_MANIFESTS/` — indexes, sync state, artefact registry
- `01_THEORY/` — canonical theory and formal definitions
- `02_KNOWLEDGE/` — compiled wiki-style knowledge pages
- `03_WHITEPAPERS/` — longform explanatory documents
- `04_PROJECTS/` — project-specific material
- `05_OUTPUTS/` — generated deliverables and exports
- `06_ARCHIVE/` — historical or superseded material

Individual repositories may omit layers they do not need, but the naming logic should remain stable where possible.

---

# 12. Publication gate

Before publishing anything public:
- no secrets
- no sensitive reflections
- no restricted course submissions
- no copyrighted redistribution
- no private-only design detail
- no unstable theory claims

---

# 13. AI-assisted development governance

AI coding assistants may be used, but outputs must:
- follow repo architecture
- preserve provenance
- be reviewed before commit
- avoid secrets and restricted data
- not bypass Zotero/reference capture
- not introduce unreviewed theory claims

---

# 14. Operating loop

The system should run as a repeating cycle:

1. ingest raw sources
2. capture metadata in Zotero
3. compile sources into markdown knowledge
4. update theory or implementation as needed
5. generate outputs
6. file outputs back into the knowledge base
7. lint and repair
8. publish sanitized mirrors
9. archive old artefacts
10. repeat

---

# 15. Actionable CPS project folder structure

Use the following structure for the CPS project so the RDMP is immediately actionable.

## Recommended project root

```text
Collective-Intelligence-Observatory/
│
├── README.md
│
├── 00_MANIFESTS/
│   ├── artefact_index.md
│   ├── theory_mapping.md
│   ├── experiment_registry.md
│   └── sync_status.md
│
├── 00_ARCHITECTURE/
│   ├── system_diagram.png
│   ├── data_flow.md
│   ├── interface_contract.md
│   ├── telemetry_schema.md
│   └── timing_contract.md
│
├── 01_TINKERCAD/
│   ├── circuits/
│   │   ├── node_IMU/
│   │   └── hub_mock/
│   ├── 3D_designs/
│   │   ├── node_enclosure/
│   │   └── hub_enclosure/
│   └── exports/
│       ├── wiring_diagrams/
│       ├── STL_files/
│       └── screenshots/
│
├── 02_WOKWI/
│   ├── node_sim/
│   ├── hub_sim/
│   └── multi_node_system/
│
├── 03_FIRMWARE/
│   ├── node/
│   ├── hub/
│   └── shared/
│
├── 04_REALTIME_ENGINE/
│   ├── graph_builder/
│   ├── metric_engine/
│   └── tests/
│
├── 05_ACTUATION/
│   ├── LED_mapping/
│   └── animations/
│
├── 06_EXPERIMENT/
│   ├── protocol/
│   ├── scripts/
│   ├── results/
│   └── run_logs/
│
├── 07_DEMO/
│   ├── demo_script.md
│   └── fallback_modes.md
│
├── 08_THEORY_ALIGNMENT/
│   ├── Paper_A_Theory.md
│   ├── Paper_B_Computation.md
│   ├── Paper_C_Measurement_CIO.md
│   └── theory_notes.md
│
├── 09_DATA/
│   ├── raw/
│   ├── processed/
│   ├── derived/
│   └── metadata/
│
├── 10_VALIDATION/
│   ├── unit_tests/
│   ├── system_tests/
│   └── benchmarks/
│
├── 11_LOGS/
│   ├── runtime/
│   └── experiment_runs/
│
└── 12_NOTES/
    ├── research_memos/
    ├── design_decisions/
    └── synthesis/
```

---

# 16. Recommended project-local file expectations

## 16.1 `README.md`
Should explain:
- project purpose
- repo role in MACYB
- current phase
- how to run simulation / firmware / tests
- what is theory alignment vs implementation

## 16.2 `00_MANIFESTS/`
Should track:
- artefact index
- theory-to-implementation mapping
- experiment registry
- sync status

## 16.3 `00_ARCHITECTURE/`
Should define:
- system diagram
- data flow
- interface contracts
- telemetry schema
- timing contracts

## 16.4 `06_EXPERIMENT/`
Should contain:
- protocols
- run scripts
- results
- logs

## 16.5 `08_THEORY_ALIGNMENT/`
Should contain:
- the theory slices relevant to this project
- local interpretation notes
- implementation mapping
- any public-safe wording derived from the theory

## 16.6 `09_DATA/`
Should distinguish:
- raw
- processed
- derived
- metadata

---

# 17. Short rule summary

> **Zotero records the sources.**  
> **Theory defines the concepts.**  
> **Private implements the slice.**  
> **Public curates the safe subset.**  
> **Drive stores the final deliverables.**  
> **LLMs compile and maintain the living knowledge base.**  
> **The CPS folder structure should make this workflow executable immediately.**

---

# 18. Maintenance rule

Update this RDMP if introducing:
- new repositories
- new storage systems
- Git LFS
- CI/CD pipelines
- DOI/Zenodo archival
- new public publication channels
- significant theory/implementation renaming

Otherwise, treat it as the current operating protocol.
