"""Quorum demo server.

Wraps the existing RAG pipeline behind a small FastAPI app and serves the
single-page demo UI in static/. Run it with:

    uvicorn app:app --reload --port 8000

Then open http://localhost:8000.
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag.pipeline import build_clients, run_query
from src.vectorstore.store import collection_stats

RESULTS_DIR = Path(__file__).parent / "results"

load_dotenv()

app = FastAPI(title="Quorum Demo", version="1.0")

VALID_DOC_TYPES = {"interview", "retro", "prd", "ticket", "meeting_note"}

# Variant → (model, prompt_variant). Mirrors the poster's V1/V2/V3.
VARIANTS = {
    "v1": {"label": "V1 · gpt-4o-mini + baseline", "model": "gpt-4o-mini", "prompt_variant": "baseline"},
    "v2": {"label": "V2 · gpt-4o-mini + enhanced", "model": "gpt-4o-mini", "prompt_variant": "v2"},
    "v3": {"label": "V3 · gpt-4o + enhanced", "model": "gpt-4o", "prompt_variant": "v2"},
}

EXAMPLE_QUERIES = [
    {
        "title": "Unity Catalog lineage",
        "query": "What are the top pain points around Unity Catalog lineage for data engineers?",
    },
    {
        "title": "MLflow experiment tracking",
        "query": "Where do MLflow users hit friction in their experiment-tracking workflow?",
    },
    {
        "title": "Workflow failure debugging",
        "query": "What slows data engineers down when debugging Workflow task failures?",
    },
    {
        "title": "Cost attribution & FinOps",
        "query": "What features do platform admins need to attribute Databricks spend to teams?",
    },
]

# Lazy singleton so we don't spin up Chroma / OpenAI clients until first query.
_collection = None
_openai_client = None


def _get_clients():
    global _collection, _openai_client
    if _collection is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY is not set. Add it to your .env file.",
            )
        _collection, _openai_client = build_clients(api_key)
    return _collection, _openai_client


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=600)
    variant: Literal["v1", "v2", "v3"] = "v2"
    top_k: int = Field(10, ge=1, le=20)
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    filter_type: Optional[str] = None


@app.get("/api/health")
def health():
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    stats = {}
    if api_key_present:
        try:
            collection, _ = _get_clients()
            stats = collection_stats(collection)
        except Exception as e:
            stats = {"error": str(e)}
    return {
        "ok": api_key_present,
        "api_key_configured": api_key_present,
        "variants": VARIANTS,
        "stats": stats,
    }


@app.get("/api/examples")
def examples():
    return {"examples": EXAMPLE_QUERIES}


@app.post("/api/query")
def api_query(req: QueryRequest):
    if req.filter_type and req.filter_type not in VALID_DOC_TYPES:
        raise HTTPException(400, f"filter_type must be one of {sorted(VALID_DOC_TYPES)}")

    variant = VARIANTS[req.variant]
    collection, openai_client = _get_clients()
    where = {"doc_type": req.filter_type} if req.filter_type else None

    result = run_query(
        query_text=req.query,
        collection=collection,
        openai_client=openai_client,
        top_k=req.top_k,
        model=variant["model"],
        temperature=req.temperature,
        prompt_variant=variant["prompt_variant"],
        where=where,
    )

    return {
        "query": result.query,
        "summary": result.summary,
        "pain_points": result.pain_points,
        "user_stories": result.user_stories,
        "model": result.model,
        "variant": req.variant,
        "variant_label": variant["label"],
        "top_k": result.top_k,
        "sources": [
            {
                "label": h["source_label"],
                "doc_type": h["metadata"].get("doc_type"),
                "source_file": h["metadata"].get("source_file"),
                "distance": round(h.get("distance", 0.0), 4),
                "snippet": (h["text"][:240] + "…") if len(h["text"]) > 240 else h["text"],
            }
            for h in result.hits
        ],
    }


# ── Benchmark / eval results endpoints ───────────────────────────────────────
# Reads the JSON files in results/ so the Benchmarks tab shows real numbers
# rather than hand-typed duplicates from the report.

_BENCHMARK_FILES = {
    "test_final": "test_final_topk10.json",
    "v1": "val_baseline_mini.json",
    "v2": "val_v2_mini.json",
    "k3": "val_topk3.json",
    "k5": "val_topk5.json",
    "k10": "val_topk10.json",
}


def _load_run(name: str) -> dict:
    fname = _BENCHMARK_FILES.get(name)
    if not fname:
        raise HTTPException(404, f"Unknown run '{name}'")
    path = RESULTS_DIR / fname
    if not path.exists():
        raise HTTPException(404, f"results/{fname} not found")
    with open(path) as f:
        return json.load(f)


def _aggregate_summary(name: str, label: str) -> dict:
    """Flat per-run summary used for comparison tables."""
    run = _load_run(name)
    agg = run.get("aggregate", {})
    cfg = run.get("config", {})
    return {
        "run": name,
        "label": label,
        "n_cases": agg.get("n_cases"),
        "config": {
            "model": cfg.get("model"),
            "top_k": cfg.get("top_k"),
            "eval_set": cfg.get("eval_set"),
        },
        "retrieval": agg.get("retrieval", {}),
        "generation": agg.get("generation", {}),
        "llm_judge": agg.get("llm_judge", {}),
    }


@app.get("/api/benchmarks")
def benchmarks():
    """Returns headline + comparison data for the Benchmarks tab."""
    summaries = []
    for key, label in [
        ("test_final", "Final · held-out test (n=90)"),
        ("v1", "V1 · gpt-4o-mini + baseline"),
        ("v2", "V2 · gpt-4o-mini + enhanced"),
        ("k3", "k = 3 · top-k ablation"),
        ("k5", "k = 5 · top-k ablation"),
        ("k10", "k = 10 · top-k ablation"),
    ]:
        try:
            summaries.append(_aggregate_summary(key, label))
        except HTTPException:
            continue

    by_run = {s["run"]: s for s in summaries}

    return {
        "headline": by_run.get("test_final"),
        "prompt_compare": [by_run.get("v1"), by_run.get("v2")],
        "topk_compare": [by_run.get("k3"), by_run.get("k5"), by_run.get("k10")],
        "all": summaries,
    }


@app.get("/api/benchmarks/cases")
def benchmark_cases(run: str = "test_final", limit: int = 200):
    """Returns per-case results for a single run (default: held-out test)."""
    data = _load_run(run)
    cases = data.get("cases", [])[:limit]

    def _flatten(case: dict) -> dict:
        story_format = case.get("story_format", {})
        faithfulness = case.get("faithfulness")
        relevance = case.get("relevance")
        invest = case.get("invest", {})
        if isinstance(faithfulness, dict):
            faith_score = faithfulness.get("score")
            faith_reason = faithfulness.get("reason")
        else:
            faith_score = faithfulness
            faith_reason = None
        if isinstance(relevance, dict):
            rel_score = relevance.get("score")
            rel_reason = relevance.get("reason")
        else:
            rel_score = relevance
            rel_reason = None
        invest_total = invest.get("total") if isinstance(invest, dict) else invest
        return {
            "id": case.get("test_id"),
            "query": case.get("query"),
            "recall_at_k": case.get("recall_at_k"),
            "mrr": case.get("mrr"),
            "pain_keyword_recall": case.get("pain_keyword_recall"),
            "story_overall": story_format.get("overall"),
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "relevance": rel_score,
            "relevance_reason": rel_reason,
            "invest_total": invest_total,
            "summary": case.get("summary", ""),
            "pain_points": case.get("pain_points", []),
            "user_stories": case.get("user_stories", []),
            "sources": case.get("sources", []),
        }

    return {
        "run": run,
        "config": data.get("config", {}),
        "aggregate": data.get("aggregate", {}),
        "n_cases": len(cases),
        "cases": [_flatten(c) for c in cases],
    }


# ── Static frontend ──────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
