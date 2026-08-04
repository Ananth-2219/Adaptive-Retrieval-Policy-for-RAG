# Adaptive Retrieval Policy for Local RAG

Classifies query intent (factual / comparative / definitional / multi-hop) and
routes to the best-suited retrieval pipeline (vector / BM25 / hybrid / multi-hop),
reranks, and generates an answer with a locally-hosted LLM. See `CLAUDE.md` for
full project context, tech stack, folder layout, and current phase.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama separately (not a pip package): https://ollama.com/download
ollama pull llama3.2:3b-instruct
```

## Status

Week 1 in progress — building the four retrieval pipelines and reranker. See
`CLAUDE.md` → "Current phase" for details.
