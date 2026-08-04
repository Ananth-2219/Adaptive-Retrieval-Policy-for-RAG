# CLAUDE.md — Adaptive Retrieval Policy for Local RAG

This file is persistent context for Claude Code. Read it fully before making changes.
Keep it updated as decisions change — it is the source of truth for the project, not
any single conversation.

## What this project is

University course project. Title: **"Adaptive Retrieval Policy for Local
Retrieval-Augmented Generation: A Lightweight AI Framework for Hallucination
Reduction."**

Core idea: most RAG systems use one fixed retrieval method for every query. This
project classifies each incoming query's **intent** (factual / comparative /
definitional / multi-hop) and **routes** it to the retrieval pipeline best suited to
that intent (vector search, BM25, hybrid RRF, or iterative multi-hop retrieval),
reranks the retrieved chunks, then generates an answer with a locally-hosted LLM.
Everything runs fully local/open-source, no cloud APIs, no GPU training.

Evaluated on public QA datasets (HotpotQA, SQuAD) using: retrieval precision,
context relevance, answer correctness, hallucination rate, response latency.

## Who's building this

Three-person team, all non-coders, driving Claude Code in plain English. **Do not
assume the user can debug Python independently** — explain what you changed and why
in plain terms, keep changes small and reviewable, and prefer readable code over
clever code. When something breaks, explain the error in plain English before fixing
it, not just the fix.

Work split:
- **User (me)**: retrieval pipelines (vector, BM25, hybrid, multi-hop) — `src/retrieval/`
- **Teammate**: classifier + router + evaluation harness — `src/classifier/`,
  `src/router/`, `src/eval/`
- Generation (`src/generation/`) and reranker (`src/reranker/`) are shared/whoever
  gets there first — check git log before starting work on these.

## Hardware constraint — read this before picking any model

Development machine: **HP 15s laptop, Intel i5, 16GB RAM, Intel Iris Xe (no
dedicated GPU)**. This is a hard constraint on every model choice:

- Local LLM via Ollama must be a **small quantized model** — default to
  `llama3.2:3b-instruct` or `qwen2.5:3b-instruct` as the working model.
  `phi3.5` is a fallback if those underperform. Do not pull anything above ~7B
  params without checking with the user first — it may not run at usable speed on
  CPU-only inference.
- Embedding model via sentence-transformers: default to `all-MiniLM-L6-v2` (small,
  fast, CPU-friendly). Do not switch to a larger embedding model without flagging
  the RAM/latency tradeoff.
- Cross-encoder reranker: default to `cross-encoder/ms-marco-MiniLM-L-6-v2` for the
  same reason.
- Batch sizes, dataset subset sizes for dev/testing, and vector store index size
  should all assume no GPU. When experimenting, use a small dev subset of
  HotpotQA/SQuAD (e.g. 50-100 questions) and only run the full eval set
  intentionally, since latency evaluation is one of the five metrics and CPU-only
  runs are slow.

## Tech stack (decided — do not swap without discussion)

| Component | Tool |
|---|---|
| Local LLM | Ollama |
| Embeddings | sentence-transformers |
| Vector store | ChromaDB |
| Keyword retrieval | rank_bm25 |
| Hybrid fusion | Hand-rolled Reciprocal Rank Fusion (no library) |
| Reranker | cross-encoder (sentence-transformers CrossEncoder) |
| Query intent classifier | sklearn (trained classifier, not a prompted LLM) |
| Evaluation | RAGAS |
| Datasets | HotpotQA, SQuAD |

## The shared data contract

This is what keeps the two workstreams (retrieval vs. classifier/router/eval)
compatible. **Every retrieval pipeline must return the same shape**, regardless of
which method produced it, so the router and eval harness can treat them
interchangeably. Define this as a dataclass or pydantic model in
`src/utils/schemas.py` — do not let each pipeline invent its own return shape.

Retrieval pipeline output (list of these per query):
```python
{
    "chunk_id": str,
    "text": str,
    "score": float,          # method-specific relevance score, NOT normalized across methods
    "source_method": str,    # "vector" | "bm25" | "hybrid" | "multihop"
    "metadata": dict,        # doc_id, source dataset, any hop info for multi-hop
}
```

Router output:
```python
{
    "query": str,
    "predicted_intent": str,  # "factual" | "comparative" | "definitional" | "multihop"
    "confidence": float,
    "pipeline_used": str,
}
```

If either workstream needs to change this contract, update it here first, then in
`src/utils/schemas.py`, then message the other person — don't just change your own
pipeline's output shape silently.

## Folder structure
 
 ```
 adaptive-rag/
 ├── CLAUDE.md                  # this file — persistent project context
 ├── README.md
 ├── requirements.txt
 ├── .gitignore
 ├── data/
 │   ├── raw/                   # downloaded datasets — gitignored
 │   ├── processed/             # cleaned/chunked data — gitignored
 │   └── vector_store/          # ChromaDB persistence dir — gitignored
 ├── src/
 │   ├── retrieval/             # All retrieval pipelines
 │   │   ├── vector_search.py   # ChromaDB + sentence-transformers vector search
 │   │   ├── bm25_search.py     # rank_bm25 keyword retrieval
 │   │   ├── hybrid_rrf.py      # Reciprocal Rank Fusion of vector + BM25
 │   │   ├── multi_hop.py       # Multi-hop retrieval (LLM query decomposition)
 │   │   └── reranker.py        # Cross-encoder reranking
 │   ├── reranker/              # (shared — reranker.py also in retrieval/)
 │   ├── classifier/            # query intent classifier (train + inference) — Week 2
 │   ├── router/                # routes query -> retrieval pipeline based on classifier output — Week 2
 │   ├── generation/            # Ollama LLM call + prompt templates
 │   ├── eval/                  # RAGAS harness, metric scripts — Week 3
 │   ├── tests/                 # Unit tests for all src/ modules
 │   └── utils/                 # schemas.py (data contract), config loading, logging
 ├── configs/                   # yaml/json configs — model names, paths, hyperparams
 ├── scripts/                   # one-off runnable scripts (download data, build index, run eval, demo)
 │   └── demo_mvp.py            # MVP demo script — runs all pipelines
 ├── tests/                     # unit tests, one file per src/ module
 ├── notebooks/                 # exploratory only — nothing here should be required to run the pipeline
 ├── results/
 │   ├── logs/                  # run logs — gitignored
 │   └── metrics/               # eval output CSVs/JSON — committed, this is what goes in the report
 └── docs/                      # architecture notes, viva prep notes, decisions
 ```

## Git rules

- **Never commit**: `data/raw/`, `data/processed/`, `data/vector_store/`, model
  weights, `results/logs/`, `.venv/` — see `.gitignore`, don't fight it.
- **Always commit**: `results/metrics/` outputs (small JSON/CSV, needed for the
  report), `configs/`, all `src/`.
- One branch per component/person where practical. Small, frequent commits with
  plain-English messages over big infrequent ones — this makes it possible to see
  who broke what during the viva-prep crunch.
- If the data contract in `src/utils/schemas.py` changes, that commit message must
  say so explicitly, since it affects the other person's code.

## Current phase — Week 1
 
 Goal: get all four retrieval pipelines and the reranker working **independently**,
 each callable as a plain function that takes a query string and returns the
 contract shape above. Do not wire up the classifier/router yet — that's Week 2.
 Do not worry about evaluation metrics yet — that's Week 3. Use a small hand-picked
 set of ~10-20 test queries against a small chunked subset of one dataset (SQuAD is
 simpler to start with than HotpotQA) to sanity-check each pipeline manually before
 moving to the next one.
 
 Order suggested: vector search first (simplest, proves the ChromaDB + embeddings
 path end-to-end) → BM25 (no embeddings needed, fast to add) → hybrid RRF (combines
 the two above, so do it third) → multi-hop (most complex, needs the LLM in the loop
 for query decomposition, do it last) → reranker (bolts onto any of the above).
 
### Week 1 Status — COMPLETE
 
 All four retrieval pipelines and the reranker are implemented and tested.
 A demo script (`scripts/demo_mvp.py`) ties everything together for the MVP.
 
#### Implemented files:
 | File | Description |
 |---|---|
 | `src/utils/schemas.py` | Pydantic data contract models (`RetrievalChunk`, `RouterResult`) |
 | `src/retrieval/vector_search.py` | ChromaDB + sentence-transformers vector search pipeline |
 | `src/retrieval/bm25_search.py` | rank_bm25 keyword retrieval pipeline |
 | `src/retrieval/hybrid_rrf.py` | Reciprocal Rank Fusion combining vector + BM25 |
 | `src/retrieval/multi_hop.py` | Multi-hop retrieval (LLM query decomposition — mock for MVP) |
 | `src/retrieval/reranker.py` | Cross-encoder reranking pipeline |
 | `src/tests/test_retrieval_pipelines.py` | Unit tests for all retrieval pipelines |
 | `scripts/demo_mvp.py` | MVP demo script — runs all pipelines with a single command |
 
#### Next steps:
 - Week 2: Wire up classifier + router (`src/classifier/`, `src/router/`)
 - Week 3: Evaluation harness (`src/eval/`) with RAGAS metrics

## Conventions
 
 - Python 3.10+, type hints on function signatures, docstrings on every public
   function (this project will be read by a professor who doesn't have the codebase
   memorized).
 - No notebooks for anything that needs to run repeatably — notebooks are for
   one-off exploration only.
 - Config values (model names, chunk sizes, top-k, dataset paths) live in
   `configs/`, not hardcoded in `src/`.
 - Before starting a task that touches more than one file or more than one
   component (e.g. anything that touches both `src/retrieval/` and
   `src/utils/schemas.py`), use Plan Mode and confirm the plan with the user before
   writing code.
 - All retrieval pipelines must return `RetrievalChunk` objects matching the
   data contract in `src/utils/schemas.py`. Do not invent custom return shapes.
 - The reranker (`src/retrieval/reranker.py`) accepts a list of `RetrievalChunk`
   objects and returns them re-ranked by cross-encoder score.
 - The demo script (`scripts/demo_mvp.py`) is the entry point for the MVP
   demonstration — run it with `python scripts/demo_mvp.py`.
 - Tests are in `src/tests/test_retrieval_pipelines.py` and validate that
   each pipeline returns the correct number of results with the right
   `source_method` and `score` types.
