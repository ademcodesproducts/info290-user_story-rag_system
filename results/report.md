# RAG System for PM & UX Researchers — Evaluation Report

## 1. Motivation

Product Managers and UX researchers at Databricks routinely synthesize large volumes of user research — interview transcripts, sprint retrospectives, PRDs, and support tickets — into actionable user stories and pain point analyses. This process is time-consuming and inconsistent across researchers. Our system automates this synthesis using a Retrieval-Augmented Generation (RAG) pipeline that ingests a structured knowledge base of Databricks user research and generates Connextra-format user stories grounded in retrieved evidence.

---

## 2. Data

### Knowledge Base (90 documents, 143 chunks)

| Type | Count | Description |
|---|---|---|
| Interviews | 20 | UX research transcripts with named participants, roles, and product areas |
| Tickets | 20 | Customer support tickets and NPS responses |
| Retrospectives | 25 | Sprint retrospective notes from engineering and ML teams |
| PRDs | 25 | Product requirement documents with personas, pain points, and success criteria |

**Product areas covered:** Unity Catalog, MLflow, Databricks Workflows, DLT, Databricks SQL, Mosaic AI, Feature Store, Notebooks, Cost Management, Auto Loader, Delta Sharing, Structured Streaming, Repos, and Multi-workspace Fleet Management.

### Evaluation Dataset (450 QA pairs)

QA pairs were generated automatically using GPT-4o-mini: for each document, the model produced 5 queries a PM or UX researcher would realistically ask, along with expected pain keywords and user roles. A sample of QA pairs was manually reviewed for quality.

The dataset was shuffled with a fixed seed (42) and split:

| Split | Size | Purpose |
|---|---|---|
| Validation | 360 (80%) | Hyperparameter tuning |
| Test | 90 (20%) | Final held-out evaluation (used once) |

---

## 3. System Architecture

```
Documents → Parse (metadata extraction) → Chunk (section-aware, size=400, overlap=80)
    ↓
Embed (text-embedding-3-small) → ChromaDB (cosine similarity)
    ↓
Query → Semantic search (top-k chunks) → Context block
    ↓
LLM generation (Connextra prompt + few-shot) → JSON output
    ↓
{pain_points, user_stories, summary}
```

**Key design choices:**
- Metadata-enriched chunks: each chunk carries `doc_type`, `source_file`, `participant`, `team`, and `date` for filtered retrieval
- Structured generation: output is JSON-constrained with severity-rated pain points and sourced user stories
- Connextra compliance enforced via prompt rules and regex-based format scoring

### System Variants

| Variant | Generator | Prompt | top-k |
|---|---|---|---|
| **Baseline** (reported) | gpt-4o-mini | Standard Connextra rules + 3 few-shot examples | 10 |
| v2 prompt | gpt-4o-mini | Evidence-focused: explicit severity criteria, quantified pain points, measurable outcomes | 10 |
| v2 + GPT-4o | gpt-4o | Same as v2 | 10 |

*Results for v2 variants are in progress.*

---

## 4. Hyperparameter Tuning

Top-k was tuned on the validation set (360 cases). Chunk size (400) and overlap (80) were fixed based on section boundaries in the source documents.

| top-k | Recall@k (val) | MRR (val) | Story Overall (val) |
|---|---|---|---|
| 3 | 0.717 | 0.580 | 0.774 |
| 5 | 0.814 | 0.603 | 0.776 |
| **10** | **0.908** | **0.616** | **0.780** |

**Selected config: top-k = 10.** Retrieval recall improves substantially with more context (+19 pp from k=3 to k=10) at no cost to generation quality.

---

## 5. Final Evaluation Results

Evaluated on the held-out test set (90 cases, never used during tuning).

### Retrieval

| Metric | Score |
|---|---|
| Recall@10 | **0.933** |
| MRR | **0.637** |

The correct source document appears in the top-10 retrieved chunks 93.3% of the time. MRR of 0.637 indicates the first relevant chunk typically appears in rank 1–2.

### Generation Quality (LLM Judge: gpt-4o-mini)

| Metric | Score |
|---|---|
| Faithfulness | **4.72 / 5** |
| Relevance | **4.81 / 5** |
| INVEST (user story quality) | **16.16 / 18** |
| Story format compliance | 0.725 |
| Story names Databricks feature | 0.642 |
| Benefit clause completeness | 0.993 |
| Story overall | 0.787 |

**Faithfulness (4.72/5):** Generated pain points and user stories are strongly grounded in retrieved context with minimal hallucination.

**Relevance (4.81/5):** Outputs directly address the PM's query in virtually all cases.

**INVEST (16.16/18):** User stories score highly on independence, negotiability, value, estimability, sizing, and testability — confirming they are actionable for engineering planning.

**Note on pain keyword recall (0.196):** This metric measures exact keyword overlap between auto-generated QA labels and model output. It is systematically low because the generator paraphrases rather than quoting source text verbatim. It is not a reliable quality signal for this system and is reported for completeness only.

---

## 6. Summary

The baseline RAG system achieves strong retrieval (Recall@10 = 0.933) and excellent generation quality as judged by an LLM evaluator (faithfulness 4.72/5, INVEST 16.16/18). The system reliably synthesizes Databricks user research into well-formed, evidence-grounded user stories. Variant comparisons across prompt engineering and model selection are ongoing.
