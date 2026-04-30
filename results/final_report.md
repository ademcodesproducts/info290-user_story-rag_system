# RAG-Based User Story Synthesis for Product Managers and UX Researchers

**Team Members**
Azad Jagtap (azadjagtap@berkeley.edu) · Vinay Thorat (vinaythorat@ischool.berkeley.edu) · Arnaud Demarteau (ademarteau@berkeley.edu) · Romit Kar (romir@berkeley.edu)

---

## Abstract

Product Managers and UX researchers routinely spend hours manually synthesizing qualitative research artifacts — interview transcripts, sprint retrospectives, product requirement documents, and support tickets — into structured insights and engineering-ready user stories. We built a Retrieval-Augmented Generation (RAG) system that takes a natural language PM query and a curated knowledge base of 100 Databricks user research documents, and produces structured outputs: severity-rated pain points with source citations, Connextra-format user stories, and a synthesis summary. We evaluate three system variants differing in prompt design and generator model across a 450-pair LLM-generated evaluation dataset. Our best configuration — an evidence-focused prompt with GPT-4o-mini — achieves Recall@10 of 0.933, LLM-judge faithfulness of 4.62/5, and an INVEST user story quality score of 16.24/18 on the held-out test set, demonstrating that structured prompt engineering is the highest-leverage intervention for this task.

---

## 1. Introduction

Qualitative user research is central to product development, yet synthesizing it into actionable artifacts remains a bottleneck. A typical PM research cycle produces dozens of interview transcripts, sprint retrospectives, and support tickets — often totaling hundreds of pages. Manually identifying recurring pain points, mapping them to user roles, and drafting well-formed user stories takes hours per cycle and is highly dependent on individual researcher expertise. This inconsistency means insights are frequently lost, duplicated, or never converted into engineering-ready stories.

A generative AI approach is appropriate here for three reasons. First, the task requires understanding and synthesizing unstructured text across heterogeneous documents — a capability where LLMs excel. Second, the output format is well-defined: Connextra user stories ("As a [role], I want [goal], so that [benefit]") provide a structured target that can be enforced via prompting and evaluated programmatically. Third, retrieval augmentation is essential because the knowledge base evolves continuously as new research is added, making fine-tuning impractical and retrieval-based grounding necessary to prevent hallucination.

**Our system** takes a natural language query from a PM or UX researcher and uses semantic retrieval over a ChromaDB vector store followed by GPT-4o-mini generation with evidence-focused structured prompting to produce a JSON output containing ranked pain points with severity ratings, Connextra-format user stories with source citations, and a 2–3 sentence synthesis summary.

The remainder of this report is structured as follows: Section 2 reviews related work; Section 3 describes our data; Section 4 details system architecture and variants; Section 5 presents experiments and evaluation; Sections 6 and 7 discuss findings and conclusions.

---

## 2. Related Work

**Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** This foundational paper introduced RAG as a framework combining non-parametric dense retrieval with a parametric generator. Our system directly adopts this architecture but specializes it for structured product research synthesis. Unlike Lewis et al., who fine-tune retriever and generator jointly, we keep both frozen and rely on prompt engineering — more practical for a continuously-updated knowledge base where retraining would be prohibitively expensive.

**Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval Augmented Generation.** RAGAS introduced reference-free LLM-based evaluation metrics for RAG pipelines, specifically faithfulness (grounding in retrieved context) and answer relevance. Our evaluation framework directly implements these two metrics and extends them with the INVEST framework for user story quality assessment. A key difference is that RAGAS targets factual QA, while we target structured generative synthesis — requiring additional domain-specific metrics beyond factual accuracy.

**Commercial systems: Dovetail and Productboard.** Dovetail supports qualitative research tagging and thematic analysis but does not generate user stories or perform cross-document synthesis. Productboard offers AI-assisted feature prioritization but requires pre-structured input and does not operate on raw research artifacts. Neither system produces Connextra-format outputs grounded in retrieved evidence with source citations. Our system fills this gap by combining open-ended retrieval over heterogeneous research documents with structured, citation-grounded generative synthesis.

---

## 3. Data Sources and Inputs

**Knowledge Base.** Our knowledge base consists of 100 synthetic documents representing the research artifacts a Databricks PM or UX team would accumulate. Synthetic data was chosen to maintain full control over document quality and ground truth traceability — a prerequisite for meaningful retrieval evaluation.

| Type | Count | Description |
|---|---|---|
| Interviews | 20 | UX research transcripts with named participants, roles, and product areas |
| Support Tickets | 20 | Customer feedback, NPS responses, and feature requests |
| Retrospectives | 25 | Sprint retrospective notes with wins, pain points, and action items |
| PRDs | 25 | Product requirement documents with personas, pain points, success criteria |
| Meeting Notes | 10 | PM/research sync decision logs with raised issues and open questions |

Documents span 14 Databricks product areas including Unity Catalog, MLflow, Workflows, DLT, Databricks SQL, Mosaic AI, Feature Store, Notebooks, Cost Management, Auto Loader, Delta Sharing, Structured Streaming, Repos, and multi-workspace fleet management. Documents are chunked into 163 chunks using section-aware splitting (chunk size 400 tokens, overlap 80) and embedded with `text-embedding-3-small`.

**Evaluation Dataset.** 450 QA evaluation pairs were generated using GPT-4o-mini: for each document, the model produced 5 realistic PM queries with expected pain keywords and relevant user roles. A representative sample was manually reviewed for quality. The dataset was shuffled (seed 42) and split 80/20 into validation (360 pairs) and a held-out test set (90 pairs) used only for final evaluation.

**Real-world sources considered.** We explored the Mendeley user story dataset and Jira export datasets. These were excluded due to inconsistent document quality and lack of ground-truth source traceability — limitations that would undermine reliable retrieval evaluation.

---

## 4. System Architecture and Variants

### 4.1 Architecture

```
PM Query
    ↓
Semantic Retrieval  (ChromaDB · text-embedding-3-small · cosine similarity · top-k=10)
    ↓
Context Formatting  (numbered source blocks with doc_type, participant, date metadata)
    ↓
LLM Generation      (GPT-4o-mini · structured JSON prompt · Connextra enforcement)
    ↓
Output: { pain_points[], user_stories[], summary }
```

**Retrieval.** Each chunk is embedded and stored in ChromaDB. At query time the query embedding is compared by cosine similarity to all chunk embeddings and the top-k most relevant chunks are returned. Chunks carry metadata (`doc_type`, `source_file`, `participant`, `team`, `date`) enabling optional metadata-filtered retrieval by document category.

**Generation.** Retrieved chunks are formatted into a numbered context block injected into a structured prompt. The generator produces a JSON object with `pain_points` (severity-rated with source citations), `user_stories` (Connextra format with rationale and citations), and `summary`. Temperature is fixed at 0.2 to minimize hallucination.

### 4.2 System Variants

Three variants were implemented, isolating the effect of prompt engineering and model capability:

| Variant | Generator | Prompt |
|---|---|---|
| **V1 — Baseline** | GPT-4o-mini | Standard Connextra rules + 3 few-shot story examples |
| **V2 — Enhanced Prompt** | GPT-4o-mini | Explicit severity definitions · evidence-quoting requirement · measurable outcomes · richer few-shot pairs |
| **V3 — Enhanced + GPT-4o** | GPT-4o | Same as V2 |

**V2 prompt additions over V1:** (1) explicit severity criteria with examples ("high = blocks workflow or causes data loss"); (2) instruction to quote specific numbers or user language in pain points; (3) requirement that every "so that" clause state a measurable or observable outcome; (4) richer few-shot examples showing the full evidence → pain point → story reasoning chain, rather than story-only examples.

All variants use top-k = 10, selected via validation sweep (Section 5.2).

---

## 5. Experiments and Evaluation

### 5.1 Evaluation Framework

**Retrieval metrics** (reference-based, no LLM cost):
- *Recall@k*: fraction of cases where the expected source document appears in the top-k chunks
- *MRR*: Mean Reciprocal Rank of the first correct source hit

**Generation metrics** (LLM judge: GPT-4o-mini):
- *Faithfulness* (1–5): are outputs grounded in retrieved context without hallucination?
- *Relevance* (1–5): does the output directly address the PM query?
- *INVEST* (6–18): do user stories satisfy all six INVEST criteria?
- *Story format metrics*: Connextra compliance, named Databricks feature presence, benefit clause completeness

### 5.2 Hyperparameter Tuning

top-k was swept on the validation set using V1 config. Chunk size (400) and overlap (80) were fixed based on document section boundaries.

| top-k | Recall@k | MRR | Story Overall |
|---|---|---|---|
| 3 | 0.717 | 0.580 | 0.774 |
| 5 | 0.814 | 0.603 | 0.776 |
| **10** | **0.908** | **0.616** | **0.780** |

**Selected: top-k = 10.** +19 pp retrieval recall over k=3 at no cost to generation quality.

### 5.3 Variant Comparison (Validation Set, 360 cases)

All three variants evaluated on the same 360-case validation set with LLM judge. Retrieval metrics are identical across variants since only the generator changes.

| Metric | V1 Baseline/mini | V2 Enhanced/mini | V3 Enhanced/gpt-4o |
|---|---|---|---|
| Recall@k | 0.747 | 0.750 | 0.750 |
| MRR | 0.522 | 0.523 | 0.523 |
| Faithfulness | 4.52 / 5 | 4.42 / 5 | **4.68 / 5** |
| Relevance | **4.73 / 5** | 4.59 / 5 | 4.84 / 5 |
| INVEST | 16.14 / 18 | 16.22 / 18 | **16.74 / 18** |
| Story named feature | 0.598 | **0.744** | 0.791 |
| Story overall | 0.772 | 0.805 | **0.831** |

**V1 → V2 (prompt engineering):** The enhanced prompt substantially improves story specificity (+14.6 pp named feature score, +0.03 story overall, +0.08 INVEST). Faithfulness decreases slightly (−0.10) as the more prescriptive prompt pushes the model toward specificity over conservative grounding. Retrieval is unaffected.

**V2 → V3 (model upgrade):** GPT-4o recovers the faithfulness loss (4.42 → 4.68) and further improves INVEST (16.22 → 16.74) and story overall (0.805 → 0.831), at approximately 12× higher inference cost per query (~$0.001 vs ~$0.012).

**V2 selected as best variant** for the cost-quality tradeoff: it delivers the largest improvement in story quality at identical cost to V1, and closes most of the gap to V3 without the cost overhead.

### 5.4 Final Evaluation on Held-Out Test Set (90 cases, V2 config)

The V2 configuration was evaluated once on the held-out test set after all tuning decisions were finalized.

| Metric | Score |
|---|---|
| **Recall@10** | **0.933** |
| **MRR** | **0.637** |
| **Faithfulness** | **4.62 / 5** |
| **Relevance** | **4.67 / 5** |
| **INVEST** | **16.24 / 18** |
| Story format compliance | 0.731 |
| Story names Databricks feature | 0.788 |
| Benefit clause completeness | 0.995 |
| Story overall | 0.820 |

The correct source document is retrieved in 93.3% of test cases. Faithfulness of 4.62/5 confirms outputs are strongly grounded in retrieved context. INVEST of 16.24/18 indicates generated user stories are consistently actionable and engineering-ready. Story named feature (0.788) substantially exceeds V1 baseline (0.642), confirming the V2 prompt's core improvement transfers to unseen data.

**Pain keyword recall (~0.20)** is systematically low because the generator paraphrases rather than quoting source text verbatim — a correct behavior. This metric is reported for completeness and excluded from system selection decisions.

---

## 6. Discussion

**Largest impact.** Prompt engineering (V1 → V2) produced the largest measurable improvement: +14.6 pp story specificity, +0.08 INVEST, at zero additional cost. The explicit instruction to name the Databricks feature in the "I want" clause directly drove this gain. In structured generation tasks, prompt specificity is at least as important as model capability.

**Worst failure mode.** The most consistent failure was retrieval miss — cases where the relevant document was absent from top-10 chunks. This occurred most frequently for underrepresented product areas (Delta Sharing, Structured Streaming) where the source document used different vocabulary than the query. Hybrid retrieval (BM25 + dense) would address this directly.

**Systematic bias.** The generator shows a slight preference for citing interview documents over tickets and meeting notes, even when the most specific evidence is in a ticket. This reflects the few-shot examples, which were interview-derived, and could be corrected by diversifying the prompt examples.

**Cost-quality tradeoffs.** V2 (enhanced prompt + GPT-4o-mini) is the clear winner for deployed use: +14.6 pp story quality improvement over V1 at identical cost (~$0.001/query). V3 (GPT-4o) further improves faithfulness and INVEST but at 12× higher cost — justified only if story quality is the primary constraint and budget is unconstrained.

---

## 7. Conclusions and Future Work

We built and evaluated a RAG system that reliably synthesizes Databricks user research into structured PM outputs. The key finding is that **structured prompt engineering is the highest-leverage intervention**: the V2 enhanced prompt improves story specificity by 14.6 pp and INVEST by 0.08 points over the baseline, with zero additional cost. Final evaluation on the held-out test set confirms strong generalization: Recall@10 = 0.933, Faithfulness = 4.62/5, INVEST = 16.24/18.

**Limitations.** The knowledge base is synthetic, which controls quality but limits ecological validity. The LLM-generated evaluation dataset makes pain keyword recall an unreliable metric. The system uses single-step dense retrieval without query decomposition or reranking.

**Future work.** Three extensions would most improve the system: (1) hybrid BM25 + dense retrieval with Reciprocal Rank Fusion to address vocabulary mismatch failures; (2) a cross-encoder reranker to improve precision at the top of the ranked list; and (3) a query planning agent to decompose multi-facet queries into parallel sub-queries, enabling richer synthesis across product areas.

---

## 8. Contributions

| Team Member | Writing | Code |
|---|---|---|
| Azad Jagtap | Introduction, Data Sources | Synthetic data generation (interviews, tickets); ingestion pipeline (parser, chunker) |
| Vinay Thorat | System Architecture, Discussion | Prompt engineering (V2 variant design); RAG generator module |
| Arnaud Demarteau | Experiments & Evaluation, Abstract | Evaluation framework; QA pair generation; hyperparameter sweep; variant comparison runner |
| Romit Kar | Related Work, Conclusions | Vector store integration (ChromaDB); retriever module; query CLI |

Each team member is the primary contributor to at least one component notebook. All team members contributed code to system architecture and evaluation sections.

---

## Appendix

### A. Prompt Templates

**V1 System Prompt (Baseline)**
```
You are an expert Product Manager assistant for Databricks. Your job is to analyze
product research documents and generate structured product insights.
A user story must follow the Connextra format exactly:
  "As a [specific user role], I want [concrete goal], so that [measurable benefit]."
Rules: Name the actual Databricks feature. Each story must be traceable to context.
Do not invent pain points not in context. Role must be specific (e.g. "Data Engineer").
```

**V2 Prompt additions:**
Explicit severity definitions (high = blocks workflow / causes data loss; medium = workaround costs >30 min/week; low = minor friction); instruction to quote specific numbers or user language; requirement that "so that" clauses state measurable outcomes; richer few-shot pairs showing evidence → pain point → story chain.

### B. Full Hyperparameter and Variant Results

| Config | Eval Set | Recall@k | MRR | Faithfulness | INVEST |
|---|---|---|---|---|---|
| top-k=3, V1 | val (360) | 0.717 | 0.580 | — | — |
| top-k=5, V1 | val (360) | 0.814 | 0.603 | — | — |
| top-k=10, V1 | val (360) | 0.908 | 0.616 | — | — |
| top-k=10, V1, +llm | val (360) | 0.747 | 0.522 | 4.52 | 16.14 |
| top-k=10, V2, +llm | val (360) | 0.750 | 0.523 | 4.42 | 16.22 |
| top-k=10, V3, +llm | val (360) | 0.750 | 0.523 | 4.68 | 16.74 |
| **top-k=10, V2, +llm** | **test (90)** | **0.933** | **0.637** | **4.62** | **16.24** |

### C. Cost Summary

| Run | Model | Cases | Cost |
|---|---|---|---|
| QA pair generation | GPT-4o-mini | 100 docs × 5 | ~$0.10 |
| Validation top-k sweep (3 runs) | GPT-4o-mini | 3 × 360 | ~$0.90 |
| V1 validation + LLM judge | GPT-4o-mini | 360 | ~$0.55 |
| V2 validation + LLM judge | GPT-4o-mini | 360 | ~$0.55 |
| V3 validation + LLM judge | GPT-4o | 360 | ~$6.50 |
| Final test (V2) | GPT-4o-mini | 90 | ~$0.15 |
| **Total** | | | **~$8.75** |
