# info290 — User Story RAG System

A RAG (Retrieval-Augmented Generation) system that ingests product research artifacts (user interviews, sprint retrospectives, PRDs, support tickets) and generates structured product outputs — pain points and user stories in Connextra format — grounded in real source material.

**Domain:** Databricks (data lakehouse platform)  
**Team:** Azad Jagtap, Vinay Thorat, Romit Kar, Arno Demarteau

---

## How it works

```
PM query: "What are the top pain points around MLflow?"
      ↓
Retrieval: top-k semantically relevant chunks from the knowledge base
      ↓
Generation: LLM reasons over retrieved context
      ↓
Output: pain points + user stories with source citations
```

---

## Project Structure

```
.
├── ingest.py                  # Step 1 — chunk, embed, and load into vector store
├── query.py                   # Step 2 — query the RAG system from the CLI
├── evaluate.py                # Step 3 — run the evaluation suite
├── requirements.txt
│
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # Parses .txt docs into (metadata, body) pairs
│   │   ├── chunker.py         # Section-aware chunking with configurable size/overlap
│   │   └── pipeline.py        # Orchestrates loading → parsing → chunking
│   ├── vectorstore/
│   │   └── store.py           # ChromaDB wrapper (embed, upsert, query)
│   ├── rag/
│   │   ├── retriever.py       # Retrieves top-k chunks and formats context block
│   │   ├── generator.py       # Prompt construction + structured LLM output
│   │   └── pipeline.py        # Full RAG query (retrieve → generate)
│   └── evaluation/
│       ├── metrics.py         # Retrieval and generation metrics (cheap + LLM judge)
│       └── evaluator.py       # Runs test suite and aggregates results
│
└── data/
    ├── knowledge_base/        # Synthetic Databricks research corpus
    │   ├── interviews/        # 15 user interview transcripts (INT-001 to INT-015)
    │   ├── retros/            # 10 sprint retrospectives (retro-001 to retro-010)
    │   ├── prds/              # 5 product requirement documents (prd-001 to prd-005)
    │   └── tickets/           # 15 support/feedback tickets (TKT-001 to TKT-015)
    ├── documentation/
    │   └── user stories/      # Mendeley user stories dataset (ground truth examples)
    └── eval/
        └── test_set.json      # 15 labeled test cases for evaluation
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Ingest the knowledge base

```bash
python ingest.py
```

This chunks all 45 documents, embeds them with `text-embedding-3-small`, and loads them into a local ChromaDB vector store (`chroma_db/`).

---

## Usage

### Query

```bash
# Basic query
python query.py "What are the top pain points around MLflow?"

# Restrict retrieval to a specific document type
python query.py "What did users say about Unity Catalog permissions?" --filter-type interview

# More context, cheaper model
python query.py "Cluster cost issues" --top-k 8 --model gpt-4o-mini

# Machine-readable output
python query.py "DLT pipeline failures" --json
```

**Query options:**

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--model` | gpt-4o | OpenAI model for generation |
| `--temperature` | 0.2 | Generation temperature |
| `--filter-type` | none | Restrict to `interview`, `retro`, `prd`, or `ticket` |
| `--json` | false | Output raw JSON instead of formatted text |

### Run the live demo (poster day)

A small FastAPI server (`app.py`) wraps the RAG pipeline and serves a single-page demo UI from `static/`.

```bash
# After ingest.py has populated chroma_db/ and OPENAI_API_KEY is set
uvicorn app:app --reload --port 8000
```

Then open <http://localhost:8000>. The page lets you:

- Pick a generation **variant** — V1 (gpt-4o-mini + baseline), V2 (gpt-4o-mini + enhanced), or V3 (gpt-4o + enhanced).
- Adjust **top-k** (3 / 5 / 10) and filter by **doc type** (interview, prd, retro, ticket).
- Click an example query chip or type your own.
- See the structured output rendered as cards: **summary**, **pain points** (with severity badges and source citations), **Connextra user stories** (with rationale + citations), and the **retrieved evidence chunks** (collapsible).

The same pipeline that backs the CLI (`query.py`) also backs the API endpoint (`POST /api/query`), so the demo always shows the real system.

### Evaluate

```bash
# Fast — cheap metrics only (no extra API cost)
python evaluate.py

# Full — includes LLM judge (faithfulness, relevance, INVEST)
python evaluate.py --llm-eval

# Save results for comparison
python evaluate.py --output results/run_default.json

# Run a subset of test cases
python evaluate.py --test-ids T01 T02 T05
```

**Evaluation options:**

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Retrieval depth |
| `--model` | gpt-4o | Generation model |
| `--llm-eval` | false | Run LLM judge metrics |
| `--judge-model` | gpt-4o-mini | Model for LLM judge calls |
| `--test-ids` | all | Restrict to specific test case IDs |
| `--output` | none | Save full results as JSON |

---

## Metrics

| Metric | Type | What it measures |
|---|---|---|
| Recall@k | Cheap | Fraction of expected source docs appearing in top-k |
| MRR | Cheap | Mean Reciprocal Rank of first correct source |
| Pain keyword recall | Cheap | Expected keywords present in generated pain points |
| Story format score | Cheap | Connextra compliance, named feature, benefit clause |
| Faithfulness | LLM judge | Answer grounded in retrieved context (1–5) |
| Relevance | LLM judge | Answer addresses the query (1–5) |
| INVEST score | LLM judge | User story quality across 6 INVEST criteria (6–18) |

---

## Hyperparameter Experiments

The ingestion and evaluation pipelines are designed for systematic comparison. To run a chunk-size experiment:

```bash
# Rebuild vector store with different chunk sizes
python ingest.py --chunk-size 200 --overlap 40 --reset
python evaluate.py --output results/chunk200.json

python ingest.py --chunk-size 400 --overlap 80 --reset
python evaluate.py --output results/chunk400.json

python ingest.py --chunk-size 600 --overlap 120 --reset
python evaluate.py --output results/chunk600.json
```

Compare retrieval Recall@k and MRR across runs to find the optimal chunk size for this corpus.

---

## Knowledge Base

The knowledge base is a synthetic Databricks product research corpus (45 documents, ~123 chunks) generated to simulate a PM's internal knowledge base:

| Type | Count | Content |
|---|---|---|
| User interviews | 15 | UX research sessions with data engineers, ML engineers, analysts, admins, data scientists |
| Sprint retros | 10 | Team retrospectives covering infrastructure, ML platform, governance, streaming, BI |
| PRDs | 5 | Product requirement docs for Databricks notebook, catalog, MLflow, Workflows, and cost features |
| Support tickets | 15 | Customer feedback, bug reports, and NPS verbatim responses |

All documents are domain-specific — referencing real Databricks concepts (Unity Catalog, MLflow, Delta Lake, DLT, Workflows, Mosaic AI, etc.).
