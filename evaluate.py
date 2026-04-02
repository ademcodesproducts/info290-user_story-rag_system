"""
Run the evaluation suite against the RAG system.

Usage:
    # Fast — cheap metrics only (no LLM judge calls)
    python evaluate.py

    # Full — includes faithfulness, relevance, and INVEST scoring
    python evaluate.py --llm-eval

    # Experiment: compare two chunk sizes
    python evaluate.py --chunk-size 300 --top-k 3
    python evaluate.py --chunk-size 600 --top-k 8

    # Run only specific test cases
    python evaluate.py --test-ids T01 T02 T05

    # Save results as JSON
    python evaluate.py --output results/eval_run_1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.evaluator import (
    aggregate_metrics,
    format_report,
    run_evaluation,
)
from src.vectorstore.store import get_collection

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Databricks RAG system.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Retrieval depth (default: 5)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Generation model (default: gpt-4o)")
    parser.add_argument("--llm-eval", action="store_true",
                        help="Run LLM judge metrics (faithfulness, relevance, INVEST)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        help="Model for LLM judge calls (default: gpt-4o-mini)")
    parser.add_argument("--test-ids", nargs="+", default=None,
                        help="Restrict to specific test IDs, e.g. --test-ids T01 T03")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results as JSON to this path")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("=" * 50)
    print("Databricks RAG — Evaluation Suite")
    print(f"  top_k      : {args.top_k}")
    print(f"  model      : {args.model}")
    print(f"  llm_eval   : {args.llm_eval}")
    if args.test_ids:
        print(f"  test_ids   : {args.test_ids}")
    print("=" * 50 + "\n")

    collection = get_collection(openai_api_key=api_key)
    openai_client = OpenAI(api_key=api_key)

    print("Running test cases...\n")
    results = run_evaluation(
        collection=collection,
        openai_client=openai_client,
        top_k=args.top_k,
        model=args.model,
        llm_eval=args.llm_eval,
        judge_model=args.judge_model,
        test_ids=args.test_ids,
    )

    agg = aggregate_metrics(results)
    print(format_report(results, agg))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = []
        for r in results:
            serializable.append({
                "test_id": r.test_id,
                "query": r.query,
                "recall_at_k": r.recall_at_k,
                "mrr": r.mrr,
                "pain_keyword_recall": r.pain_keyword_recall,
                "story_format": r.story_format,
                "faithfulness": r.faithfulness,
                "relevance": r.relevance,
                "invest": r.invest,
                "pain_points": r.rag_result.pain_points,
                "user_stories": r.rag_result.user_stories,
                "summary": r.rag_result.summary,
                "sources": [h["source_label"] for h in r.rag_result.hits],
            })

        output_path.write_text(
            json.dumps({"config": vars(args), "aggregate": agg, "cases": serializable},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
