# RDMP — MACYB v3  
## Knowledge Base Operating Model Extension

**Purpose:**  
This extension defines how MACYB operates as a **repeatable, self-improving knowledge system** using LLM-assisted ingestion, compilation, querying, linting, and re-ingestion.

It complements the existing RDMP rules for:
- private/public repo separation
- Drive as final deliverable store
- Zotero as reference system
- AI-assisted development governance

---

## 9. Knowledge Base Operating Model

### 9.1 Core principle

MACYB is not only a set of repositories. It is a **living knowledge base** in which:

- raw sources are ingested,
- LLMs compile them into structured markdown,
- outputs are re-filed into the system,
- and future reasoning improves from prior work.

The knowledge base grows by use.

---

## 9.2 Knowledge layers

The system should distinguish these layers:

### A. `raw/`
Source material ingested from external or internal origins.

Examples:
- articles
- PDFs
- papers
- repo snapshots
- screenshots / images
- datasets
- web clips
- transcripts
- notes imported from Zotero

Purpose:
- preserve original source material
- maintain provenance
- support re-compilation

### B. `compiled/`
LLM-maintained markdown knowledge base.

Contains:
- topic pages
- concept syntheses
- summaries
- backlinks
- structured comparisons
- source digests
- living indices

Purpose:
- convert raw sources into navigable knowledge
- maintain conceptual structure
- support Q&A and synthesis

### C. `derived/`
Outputs generated from compiled knowledge.

Examples:
- slides
- diagrams
- plots
- tables
- briefings
- blog drafts
- answer memos
- visual summaries

Purpose:
- produce reusable knowledge products
- support communication and exploration

### D. `linted/` or `validated/`
Cleaned and checked knowledge artefacts.

Contains:
- consistency fixes
- contradiction reports
- missing metadata reports
- concept drift notes
- deduplicated summaries

Purpose:
- improve integrity
- prevent drift
- support maintenance

### E. `public/`
Sanitized subset suitable for public publication.

Contains:
- shareable summaries
- public docs
- Pages content
- safe examples
- curated explanations

Purpose:
- publish without leaking private or unstable content

### F. `archive/`
Superseded or historical artefacts.

Purpose:
- preserve history
- prevent accidental overwriting
- support provenance and reconstruction

---

## 9.3 Ingestion rule

All incoming source material should be placed into `raw/` or a source-equivalent capture location before being compiled.

If a source enters via Zotero, a repo note, or a web clip:
- preserve the original reference
- store the raw artefact if available
- link it to a Zotero item
- record the source in the provenance block

---

## 9.4 LLM compilation rule

LLMs are responsible for compiling raw material into knowledge artefacts.

Compilation includes:
- summarizing sources
- grouping related material
- creating concept pages
- adding backlinks
- extracting relationships
- noting uncertainties
- generating indices and maps

The LLM should not merely rewrite text. It should **structure knowledge**.

---

## 9.5 Compilation standards

Compiled pages should:
- have stable names
- link to source material
- distinguish facts, hypotheses, and interpretations
- use concise markdown structure
- preserve traceability to raw sources
- be designed for future incremental updates

A compiled page should be easy for an LLM to read again later.

---

## 9.6 Query loop

The knowledge base should support iterative questioning.

When a query is asked, the LLM should:
1. inspect compiled pages first
2. consult raw material if needed
3. identify missing links or contradictions
4. answer the question
5. produce a new markdown artefact if the answer is reusable
6. file the artefact back into the knowledge base

This ensures that queries **add to the system**, rather than merely consuming it.

---

## 9.7 Output filing rule

All substantial outputs should be considered knowledge artefacts.

Examples:
- research memos
- summaries
- synthesis notes
- slide decks
- charts
- comparison tables
- generated explanations
- review findings

Unless ephemeral, outputs should be filed back into `derived/`, `compiled/`, or a project-specific knowledge area.

---

## 9.8 Linting and health checks

The LLM should periodically perform health checks on the knowledge base.

Checks may include:
- missing metadata
- duplicate pages
- inconsistent terminology
- contradictions
- stale concepts
- orphaned pages
- weak backlinks
- ungrounded claims
- drift between theory, implementation, and publication

Where possible, the LLM should:
- repair the issue
- propose the repair
- or create a follow-up task note

---

## 9.9 Canonical knowledge maintenance loop

The recurring maintenance cycle is:

1. ingest raw sources
2. compile into markdown
3. query and synthesize
4. generate outputs
5. file outputs back in
6. lint and repair
7. publish sanitized subset
8. repeat

This creates compounding knowledge value.

---

## 9.10 Relationship to Zotero

Zotero remains the reference backbone.

Zotero should be used for:
- bibliographic metadata
- source notes
- PDFs and links
- literature organization
- provenance anchors

The markdown knowledge base should reference Zotero items where possible.

Recommended practice:
- each concept page links to relevant Zotero items
- each raw capture includes a Zotero reference if available
- each synthesis page records its source basis

---

## 9.11 Relationship to repos

### `algoplexity/algoplexity`
Use as the theory source and conceptual anchor.

### `ANU-MACYB-private`
Use as the working knowledge engine:
- raw captures
- compiled wiki
- private synthesis
- internal outputs
- implementation notes
- knowledge maintenance

### `ANU-MACYB-public`
Use as the public-safe mirror of selected compiled knowledge:
- sanitized summaries
- public explanations
- curated outputs
- stable content only

### Drive
Use for final submissions and heavy artefacts.

---

## 9.12 Structural recommendation

To support this model, project folders may adopt a knowledge-base pattern such as:

```text
projects/<project>/
  raw/
  compiled/
  derived/
  validated/
  public/
  archive/
```

Or, if the repo remains more conventional, the same logic can be embedded under:
- `docs/`
- `notes/`
- `manifests/`
- `outputs/`
- `archive/`

The key is not the exact folder name, but the presence of the **knowledge lifecycle**.

---

## 9.13 Knowledge artefact metadata block

Every major knowledge artefact should record:

```markdown
## Metadata
- Source type: raw | compiled | derived | validated | public
- Upstream source(s): ...
- Zotero item(s): ...
- Related theory slice: ...
- Related implementation slice: ...
- Sync status: aligned | partial | diverged
- Last reviewed: YYYY-MM-DD
- Notes: ...
```

---

## 9.14 LLM stewardship principle

The LLM is not only a writer. It is a **steward of structure**.

It should:
- preserve hierarchy
- improve cross-linking
- maintain provenance
- surface missing connections
- keep the knowledge base navigable

Human intervention should focus on:
- strategic direction
- high-level review
- publication approval
- exception handling

---

## 9.15 Short version

> **Raw sources are ingested.**
> **LLMs compile them into a living wiki.**
> **Queries generate new durable knowledge.**
> **Outputs are re-filed.**
> **Linting improves integrity.**
> **Publication exposes only the safe subset.**

---

