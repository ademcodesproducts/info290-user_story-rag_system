import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from openai import OpenAI, RateLimitError
from json import JSONDecodeError

from src.evaluation.metrics import (
    faithfulness_score,
    invest_scores,
    mean_reciprocal_rank,
    pain_keyword_recall,
    relevance_score,
    retrieval_recall_at_k,
    user_story_format_score,
)
from src.rag.pipeline import RAGResult, run_query
from src.rag.retriever import format_context_block

DEFAULT_TEST_SET_PATH = Path("data/eval/test_set.json")


@dataclass
class TestCaseResult:
    test_id: str
    query: str
    rag_result: RAGResult | None = None
    recall_at_k: float = 0.0
    mrr: float = 0.0
    pain_keyword_recall: float = 0.0
    story_format: dict = field(default_factory=dict)
    faithfulness: dict = field(default_factory=dict)
    relevance: dict = field(default_factory=dict)
    invest: list = field(default_factory=list)


def result_from_dict(d: dict) -> "TestCaseResult":
    """Reconstruct a TestCaseResult from a serialized case dict (for resume)."""
    return TestCaseResult(
        test_id=d["test_id"],
        query=d["query"],
        recall_at_k=d.get("recall_at_k", 0.0),
        mrr=d.get("mrr", 0.0),
        pain_keyword_recall=d.get("pain_keyword_recall", 0.0),
        story_format=d.get("story_format", {}),
        faithfulness=d.get("faithfulness", {}),
        relevance=d.get("relevance", {}),
        invest=d.get("invest", []),
    )


def load_test_set(path: Path = DEFAULT_TEST_SET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


from typing import Generator


def run_evaluation(
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 5,
    model: str = "gpt-4o",
    llm_eval: bool = False,
    judge_model: str = "gpt-4o-mini",
    test_ids: list[str] | None = None,
    test_set_path: Path = DEFAULT_TEST_SET_PATH,
    prompt_variant: str = "baseline",
    skip_ids: set[str] | None = None,
) -> Generator[TestCaseResult, None, None]:
    test_cases = load_test_set(test_set_path)
    if test_ids:
        test_cases = [t for t in test_cases if t["id"] in test_ids]
    if skip_ids:
        test_cases = [t for t in test_cases if t["id"] not in skip_ids]
        print(f"  Resuming — skipping {len(skip_ids)} already-completed cases")

    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {tc['id']}: {tc['query'][:60]}...")

        for attempt in range(5):
            try:
                rag_result = run_query(
                    query_text=tc["query"],
                    collection=collection,
                    openai_client=openai_client,
                    top_k=top_k,
                    model=model,
                    prompt_variant=prompt_variant,
                )
                break
            except RateLimitError:
                wait = 2 ** attempt
                print(f"    rate limited — waiting {wait}s…")
                time.sleep(wait)
            except JSONDecodeError:
                print(f"    malformed JSON on attempt {attempt + 1}, retrying…")
                time.sleep(1)
        else:
            print(f"    SKIPPED {tc['id']} after 5 attempts")
            continue

        tc_result = TestCaseResult(
            test_id=tc["id"],
            query=tc["query"],
            rag_result=rag_result,
            recall_at_k=retrieval_recall_at_k(rag_result.hits, tc["expected_sources"]),
            mrr=mean_reciprocal_rank(rag_result.hits, tc["expected_sources"]),
            pain_keyword_recall=pain_keyword_recall(rag_result.pain_points, tc["expected_pain_keywords"]),
            story_format=user_story_format_score(rag_result.user_stories),
        )

        if llm_eval:
            context_block = format_context_block(rag_result.hits)
            tc_result.faithfulness = faithfulness_score(
                openai_client, context_block, rag_result.pain_points, rag_result.user_stories, model=judge_model,
            )
            tc_result.relevance = relevance_score(
                openai_client, tc["query"], rag_result.summary, model=judge_model,
            )
            if rag_result.user_stories:
                tc_result.invest = invest_scores(openai_client, rag_result.user_stories[:3], model=judge_model)

        yield tc_result


def aggregate_metrics(results: list[TestCaseResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def mean(vals):
        return round(sum(vals) / len(vals), 3)

    agg = {
        "n_cases": n,
        "retrieval": {
            "recall_at_k": mean([r.recall_at_k for r in results]),
            "mrr": mean([r.mrr for r in results]),
        },
        "generation": {
            "pain_keyword_recall": mean([r.pain_keyword_recall for r in results]),
            "story_format_compliance": mean([r.story_format.get("format_compliance", 0) for r in results]),
            "story_named_feature": mean([r.story_format.get("has_named_feature", 0) for r in results]),
            "story_benefit_detail": mean([r.story_format.get("benefit_detail", 0) for r in results]),
            "story_overall": mean([r.story_format.get("overall", 0) for r in results]),
        },
    }

    llm_results = [r for r in results if r.faithfulness]
    if llm_results:
        agg["llm_judge"] = {
            "faithfulness": mean([r.faithfulness.get("score", 0) for r in llm_results]),
            "relevance": mean([r.relevance.get("score", 0) for r in llm_results]),
        }
        invest_all = [s for r in llm_results for s in r.invest]
        if invest_all:
            agg["llm_judge"]["invest_mean_total"] = mean([s.get("total", 0) for s in invest_all])

    return agg


def format_report(results: list[TestCaseResult], agg: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 70)

    lines.append(f"\n{'ID':<6} {'Recall@k':>9} {'MRR':>7} {'KwRcl':>7} {'FmtScr':>8}  Query")
    lines.append("-" * 70)
    for r in results:
        lines.append(
            f"{r.test_id:<6} "
            f"{r.recall_at_k:>9.2f} "
            f"{r.mrr:>7.2f} "
            f"{r.pain_keyword_recall:>7.2f} "
            f"{r.story_format.get('overall', 0):>8.2f}  "
            f"{r.query[:40]}"
        )
        if r.faithfulness:
            lines.append(f"{'':6}  faithfulness={r.faithfulness.get('score', '?')}/5  relevance={r.relevance.get('score', '?')}/5")

    lines.append("\n" + "=" * 70)
    lines.append("AGGREGATE METRICS")
    lines.append("=" * 70)
    lines.append(f"  Cases evaluated       : {agg['n_cases']}")
    lines.append(f"\n  Retrieval")
    lines.append(f"    Recall@k            : {agg['retrieval']['recall_at_k']:.3f}")
    lines.append(f"    MRR                 : {agg['retrieval']['mrr']:.3f}")
    lines.append(f"\n  Generation")
    lines.append(f"    Pain keyword recall : {agg['generation']['pain_keyword_recall']:.3f}")
    lines.append(f"    Story format        : {agg['generation']['story_format_compliance']:.3f}")
    lines.append(f"    Story named feature : {agg['generation']['story_named_feature']:.3f}")
    lines.append(f"    Story benefit detail: {agg['generation']['story_benefit_detail']:.3f}")
    lines.append(f"    Story overall       : {agg['generation']['story_overall']:.3f}")

    if "llm_judge" in agg:
        lj = agg["llm_judge"]
        lines.append(f"\n  LLM Judge ({agg.get('judge_model', 'gpt-4o-mini')})")
        lines.append(f"    Faithfulness        : {lj.get('faithfulness', '?'):.2f} / 5")
        lines.append(f"    Relevance           : {lj.get('relevance', '?'):.2f} / 5")
        if "invest_mean_total" in lj:
            lines.append(f"    INVEST mean total   : {lj['invest_mean_total']:.2f} / 18")

    lines.append("=" * 70 + "\n")
    return "\n".join(lines)
