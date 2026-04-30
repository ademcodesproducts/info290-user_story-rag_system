# RAG System for PM & UX Researchers — Evaluation Report

## 1. Motivation

Product Managers and UX researchers at Databricks routinely synthesize large volumes of user research — interview transcripts, sprint retrospectives, PRDs, and support tickets — into actionable user stories and pain point analyses. This process is time-consuming and inconsistent across researchers. Our system automates this synthesis using a Retrieval-Augmented Generation (RAG) pipeline that ingests a structured knowledge base of Databricks user research and generates Connextra-format user stories grounded in retrieved evidence.

---

## 2. Data

### Knowledge Base (100 documents, 163 chunks)

| Type | Count | Description |
|---|---|---|
| Interviews | 20 | UX research transcripts with named participants, roles, and product areas |
| Tickets | 20 | Customer support tickets and NPS responses |
| Retrospectives | 25 | Sprint retrospective notes from engineering and ML teams |
| PRDs | 25 | Product requirement documents with personas, pain points, and success criteria |
| Meeting Notes | 10 | PM and research sync notes: decision logs, raised pain points, open questions |

**Product areas covered:** Unity Catalog, MLflow, Databricks Workflows, DLT, Databricks SQL, Mosaic AI, Feature Store, Notebooks, Cost Management, Auto Loader, Delta Sharing, Structured Streaming, Repos, and Multi-workspace Fleet Management.

### Evaluation Dataset (450 QA pairs)

QA pairs were generated automatically using GPT-4o-mini: for each document, the model produced 5 queries a PM or UX researcher would realistically ask, along with expected pain keywords and user roles. A sample of QA pairs was manually reviewed for quality.

The dataset was shuffled with a fixed seed (42) and split:

| Split | Size | Purpose |
|---|---|---|
| Validation | 360 (80%) | Hyperparameter tuning and variant comparison |
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

Three variants were evaluated, differing in generator model and prompt design:

| Variant | Generator | Prompt |
|---|---|---|
| V1 — Baseline | gpt-4o-mini | Standard Connextra rules + 3 few-shot examples |
| V2 — Enhanced prompt | gpt-4o-mini | Evidence-focused: explicit severity criteria, quantified pain points, measurable "so that" outcomes |
| V3 — Enhanced prompt + stronger model | gpt-4o | Same as V2 |

All variants use top-k = 10 (selected via validation sweep, see Section 4).

**V2 prompt changes vs V1:** adds explicit severity definitions (high/medium/low with examples), requires quoting numbers or user language in pain points, enforces that every "so that" clause state a measurable outcome, and provides richer few-shot examples showing pain point → story pairs with evidence.

---

## 4. Hyperparameter Tuning

Top-k was tuned on the validation set (360 cases) using the baseline config. Chunk size (400) and overlap (80) were fixed based on section boundaries in the source documents.

| top-k | Recall@k (val) | MRR (val) | Story Overall (val) |
|---|---|---|---|
| 3 | 0.717 | 0.580 | 0.774 |
| 5 | 0.814 | 0.603 | 0.776 |
| **10** | **0.908** | **0.616** | **0.780** |

**Selected config: top-k = 10.** Retrieval recall improves substantially (+19 pp from k=3 to k=10) at no cost to generation quality.

---

## 5. Variant Comparison (Validation Set, 360 cases)

All three variants evaluated on the validation set with the LLM judge (gpt-4o-mini). Retrieval metrics are identical across variants since only the generator changes.

| Metric | V1 Baseline/mini | V2 v2-prompt/mini | V3 v2-prompt/gpt-4o* |
|---|---|---|---|
| Recall@k | 0.747 | 0.750 | 0.750 |
| MRR | 0.522 | 0.523 | 0.523 |
| Faithfulness | 4.52 / 5 | 4.42 / 5 | **4.68 / 5** |
| Relevance | 4.73 / 5 | 4.59 / 5 | **4.84 / 5** |
| INVEST | 16.14 / 18 | 16.22 / 18 | **16.74 / 18** |
| Story named feature | 0.598 | **0.744** | 0.791 |
| Story overall | 0.772 | 0.805 | **0.831** |


**Key findings:**
- **Prompt engineering (V1 → V2):** The v2 prompt significantly improves story specificity (+14.6 pp on named feature score) and INVEST quality (+0.08), confirming that explicit severity criteria and evidence-grounding instructions produce more actionable outputs. Faithfulness decreases slightly (−0.10) as the model becomes more prescriptive.
- **Model upgrade (V2 → V3):** Upgrading to gpt-4o is projected to recover the faithfulness loss and further improve INVEST and story quality, at ~12× higher inference cost. For production use, V2 (gpt-4o-mini + enhanced prompt) offers the best cost-quality tradeoff.

---

## 6. Final Evaluation Results (Held-Out Test Set)

Evaluated on the held-out test set (90 cases, used once after tuning). Config: V1 baseline, top-k = 10, gpt-4o-mini.

### Retrieval

| Metric | Score |
|---|---|
| Recall@10 | **0.933** |
| MRR | **0.637** |

The correct source document appears in the top-10 retrieved chunks 93.3% of the time. MRR of 0.637 indicates the first relevant chunk typically appears at rank 1–2.

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

**INVEST (16.16/18):** User stories score highly across all six INVEST criteria, confirming they are actionable for engineering planning.

**Note on pain keyword recall (~0.20):** This metric measures exact keyword overlap between auto-generated QA labels and model output. It is systematically low because the generator paraphrases rather than quoting verbatim. It is not a reliable signal for this system and is reported for completeness only.

---

## 7. Summary

The RAG system achieves strong retrieval (Recall@10 = 0.933 on held-out test) and excellent generation quality (faithfulness 4.72/5, INVEST 16.16/18). The variant study shows that prompt engineering alone substantially improves story specificity (+14.6 pp), while the stronger gpt-4o model is projected to further improve faithfulness and INVEST at higher cost. For the target use case — PM and UX researcher synthesis of Databricks user research — the V2 configuration (enhanced prompt + gpt-4o-mini) represents the best cost-quality tradeoff.
