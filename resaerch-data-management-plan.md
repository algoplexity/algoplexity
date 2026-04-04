# RDMP — MACYB v2  
**Repeatable Development & Metadata Protocol**  
**Owner:** algoplexity  
**Scope:** `algoplexity/algoplexity`, `algoplexity/ANU-MACYB-private`, `algoplexity/ANU-MACYB-public`, Drive, Zotero  
**Purpose:** make theory → implementation → publication iterative, traceable, and maintainable as content evolves

---

## 0. Intent

This RDMP defines the operating protocol for maintaining a theory-to-implementation pipeline across three layers:

- **A_Theory** — `algoplexity/algoplexity`
- **B_Computation** — `algoplexity/ANU-MACYB-private`
- **C_Measurement_CIO** — `algoplexity/ANU-MACYB-public`

It also defines how **Zotero** captures metadata and how changes are propagated when content changes ripple across repos.

---

## 1. Systems of record

### Authoritative systems

- **Zotero**  
  Authoritative for:
  - references
  - bibliographic metadata
  - source notes
  - source links / PDFs
  - reading traces and literature annotations

- **GitHub private**  
  Authoritative for:
  - working Markdown drafts
  - private implementation notes
  - code
  - internal architecture
  - theory-to-implementation translation
  - pre-publication staging

- **Drive**  
  Authoritative for:
  - final submitted PDFs / Docs
  - feedback files
  - large or binary artefacts
  - exported deliverables
  - final snapshots for coursework or external submission

- **GitHub public + Pages**  
  Authoritative for:
  - sanitized public documentation
  - curated summaries
  - shareable explanations
  - stable public-facing project pages

---

## 2. Repo roles

### 2.1 `algoplexity/algoplexity` — Theory source
This repo defines:
- conceptual foundations
- formal definitions
- theoretical claims
- research framing
- canonical terminology
- theory revisions

### 2.2 `algoplexity/ANU-MACYB-private` — Implementation source
This repo defines:
- working implementation
- internal documentation
- architecture notes
- private drafts
- project-specific translation of theory into executable or operational form
- publication staging material

### 2.3 `algoplexity/ANU-MACYB-public` — Public-safe curated mirror
This repo defines:
- public-facing summaries
- sanitized docs
- GitHub Pages content
- stable and shareable project explanations
- curated samples or demos that do not leak private details

---

## 3. Canonical source hierarchy

When there is conflict, use this order:

1. **Direct user instruction**
2. **Repository reality**
3. **`algoplexity/algoplexity` theory**
4. **`ANU-MACYB-private` implementation**
5. **`ANU-MACYB-public` publication**
6. **Drive snapshots** for final submission artefacts

If public wording differs from private implementation, the private implementation wins.  
If private implementation differs from theory, the theory wins unless an intentional deviation is recorded.

---

## 4. Zotero metadata protocol

### 4.1 Zotero is mandatory for reference tracking
All substantive theory claims should be traceable to:
- a Zotero item
- a literature note
- or an internal theory note that cites relevant sources

### 4.2 Zotero collections
Use a structured collection scheme such as:

- MACYB
  - Theory
  - CPS1-individual
  - CPS2-group
  - Public-writing
  - Methods
  - Background reading

### 4.3 Required metadata fields
For each source item, capture:
- Title
- Author(s)
- Year
- Source type
- DOI / URL if available
- Tags
- Short relevance note
- Linked repo / doc
- Related theory slice or implementation slice

### 4.4 Citation rule
When a repo document uses an external source or important conceptual influence:
- log it in Zotero
- link it in the doc’s references section or notes
- preserve the Zotero item as the bibliographic source of record

### 4.5 Internal notes
If a concept is generated internally rather than taken from literature:
- mark it as **internal theory**
- still add a Zotero-style reference note if it influences the doc
- distinguish clearly between **external references** and **repo-authored theory**

---

## 5. Metadata block for major artefacts

Every major document, design note, or implementation doc should include a small provenance block.

### Suggested format
```markdown
## Metadata
- Source theory: algoplexity/algoplexity
- Implementation source: ANU-MACYB-private/projects/cps1-individual/...
- Public mirror: ANU-MACYB-public/projects/cps1-individual/...
- Zotero collection: MACYB / Theory
- Upstream theory slice: <name or file>
- Downstream public target: <name or file>
- Sync status: aligned | partial | diverged
- Last reviewed: YYYY-MM-DD
- Notes: <short note>
```

---

## 6. Change propagation rules

### 6.1 Theory → Private
If a theory concept changes in `algoplexity/algoplexity`, then:
- private docs depending on that concept must be reviewed
- terminology in implementation docs must be checked
- any affected public summaries must be queued for update

### 6.2 Private → Public
If implementation or wording changes in `ANU-MACYB-private`, then:
- public-safe wording must be reviewed
- unstable or private-only content must be removed
- public docs must not invent new meaning

### 6.3 Public → Review loop
If public docs change:
- verify consistency with the private artifact
- verify consistency with the theory source
- verify the change does not leak private material

Public edits must never become a second hidden theory source.

---

## 7. Iteration protocol

### 7.1 Theory revision
A theory revision occurs when:
- a definition changes
- a concept is renamed
- a scope boundary shifts
- a formal model is refined
- a new theory slice is added

Required actions:
- update private implementation docs
- update provenance blocks
- update Zotero notes where relevant
- mark public docs for review

### 7.2 Implementation revision
An implementation revision occurs when:
- metric definitions change
- pipeline stages change
- architecture changes
- a doc/code artifact changes meaning
- a private-to-public sanitization target changes

Required actions:
- record the change in decision logs
- link the change to theory source
- update public mirror if relevant

### 7.3 Public publication revision
A public revision occurs when:
- a private artifact has stabilized enough to share
- public wording has been sanitized
- sensitive details have been removed
- the public artifact is ready to mirror

Required actions:
- ensure it is derived from a vetted private source
- confirm no hidden private dependencies remain
- preserve traceability to the private source

---

## 8. Terminology governance

### 8.1 Canonical terms
The following kinds of terms should be treated as canonical and reviewed carefully:
- theory names
- metric names
- pipeline layer names
- repo role names
- system-of-record labels

### 8.2 Allowed drift
The following may evolve:
- explanatory wording
- public narrative phrasing
- implementation-specific labels
- local experimental names

### 8.3 Frozen terms
Once a term is used cross-repo, changing it requires:
- theory review
- implementation review
- public review
- provenance update

---

## 9. Sync checkpoints

A change is not considered complete until all relevant checkpoints are updated.

### Required checkpoints
- theory updated
- private implementation reviewed
- public mirror reviewed
- Zotero metadata updated
- Drive export recorded if applicable
- provenance block updated
- decision log updated if deviation exists

---

## 10. Divergence handling

Sometimes the repos will intentionally diverge.

### Allowed divergence
- implementation simplification
- experimental terminology
- public-safe omission
- staged publication lag

### Required handling
Whenever divergence is intentional:
- record it explicitly
- state why it exists
- mark whether it is temporary or permanent
- identify the repo(s) affected

---

## 11. Workflow summary

### Standard workflow
1. Update theory in `algoplexity/algoplexity`
2. Map the change into `ANU-MACYB-private`
3. Update or create supporting Zotero metadata
4. Verify implementation alignment
5. Sanitize and mirror to `ANU-MACYB-public`
6. Export final deliverables to Drive if required
7. Record sync status and divergence notes

---

## 12. Systems-specific folder expectations

### 12.1 `ANU-MACYB-private`
- working drafts
- decision logs
- internal architecture
- implementation notes
- project docs
- export notes
- Drive manifests

### 12.2 `ANU-MACYB-public`
- curated docs
- public blog posts
- stable summaries
- Pages content
- public samples only

### 12.3 Drive
- final PDFs
- final docs
- feedback files
- large outputs
- authoritative submitted artefacts

### 12.4 Zotero
- references
- notes
- PDF links
- bibliographic metadata
- source traces

---

## 13. Publication gate

Before anything goes public, confirm:
- no secrets
- no sensitive reflections
- no restricted submission material
- no copyrighted redistribution
- no private implementation details that should remain internal
- alignment with the current theory slice

---

## 14. AI-assisted development governance

AI coding assistants may be used, but outputs must:
- conform to repo architecture
- be reviewed before commit
- preserve provenance
- avoid introducing untracked concepts
- avoid bypassing Zotero/reference capture

If AI changes theory-adjacent wording, that wording must be reviewed like any other content change.

---

## 15. Short rule summary

> **Zotero records the sources.**  
> **Theory defines the concepts.**  
> **Private implements the selected slice.**  
> **Public curates the safe subset.**  
> **Drive stores the final deliverables.**  
> **Every change must be traced forward and backward.**

---
