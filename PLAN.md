# RAG Knowledge Bot — Plan

## Vision
An agent that reviews uploaded documents to answer questions in a interactive manner.
Agent cites sources and will state if its unable to find particular information from the documentation.

- 5-step pipeline: File Discovery → Change Detection → Retrieve → Generate → Return
- Vector store: ChromaDB (persistent, cosine similarity)
- Embeddings: SentenceTransformer `all-MiniLM-L6-v2`
- Incremental re-indexing via file/metadata tracking
- Built-in eval loop: ground-truth QA generation, LLM-as-judge, cosine similarity,
  reference precision/recall

## Goals for this build
- Primary purpose: **learning** — RAG internals, React, and system design tradeoffs —
  built as a portfolio-quality project, not just "make it work."
- Starting from scratch, no existing backend code.

## Tech stack decisions

| Layer | Choice | Why |
|---|---|---|
| RAG/backend service | **Python + FastAPI** | RAG ecosystem (sentence-transformers, ChromaDB, chunking) is Python-native. Fighting that in Node means reinventing or wrapping Python anyway. |
| Frontend | **React (TypeScript)** | Chat/upload UI, calls the FastAPI backend over HTTP. |
| Vector DB | **ChromaDB** (local, persistent) now; **Pinecone** later as an explicit "swap the vector store behind an interface" exercise | Chroma is free, zero-infra, and API-shape-compatible enough with Pinecone that swapping it later is a clean, deliberate lesson rather than upfront cost. |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` | Small, fast, local, no API cost, good enough quality for doc retrieval. |
| LLM | **OpenAI API** (chosen for cost) | Needed for generation + eval (LLM-as-judge, ground-truth generation). |
| File/index metadata tracking | Flat JSON or SQLite file | No need for a full RDBMS until we have real structured/relational needs. |
| Relational DB + ORM | **Deferred to Phase 5** — Postgres + SQLAlchemy (Python-side equivalent of Prisma) | Phases 0–4 don't need per-team config or user accounts. Introduce structured storage only once real schema requirements exist (team config in Phase 5, users in Phase 7), so the schema is designed from real needs instead of guessed upfront. |

## Phases

### Phase 0 — Foundations & repo setup
Project structure, Python env, choice of LLM API, a "hello world" script that calls
the LLM. Goal: prove the wiring works before any RAG complexity.

### Phase 1 — Minimal RAG pipeline (CLI only, no UI)
Load a folder of docs → chunk → embed (`all-MiniLM-L6-v2`) → store in ChromaDB → take
a question from the terminal → retrieve top-k → stuff into prompt → get an answer.
No citations yet, no incremental indexing. Goal: a working RAG loop end-to-end,
understood line by line.

### Phase 2 — Grounding & citations
Add source-tracking metadata (doc, chunk, score) so answers cite exactly where they
came from. Add the "don't answer if not in docs" guardrail.

### Phase 3 — Incremental indexing (change detection)
Track file hashes/timestamps so re-running only re-embeds changed files instead of
the whole corpus.

### Phase 4 — React frontend
Chat UI (upload/point-to-folder, ask question, see answer + citations) talking to a
FastAPI backend wrapping Phase 1–3 logic. Covers React state, async data fetching,
component structure.

### Phase 5 — Customization layer
Tone, answer format, citation requirements — configurable per team/project. This is
where Postgres + SQLAlchemy get introduced for per-team config storage.

### Phase 6 — Evaluation pipeline
Ground-truth QA generation, LLM-as-judge, cosine similarity, reference
precision/recall — closes the feedback loop shown in the system design diagram.

### Phase 7 (stretch)
Auth, multi-user, OneDrive/SharePoint connectors, deployment, Pinecone swap-in.

## Status
- [x] Concept + system design (diagrams)
- [x] Phased plan + tech stack decisions
- [x] Phase 0
- [x] Phase 1
- [x] Phase 2
- [x] Phase 3
- [x] Phase 4
- [ ] Phase 5
- [ ] Phase 6
- [ ] Phase 7
