import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.evaluator import aggregate_metrics, format_report, run_evaluation
from src.vectorstore.store import get_collection

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--llm-eval", action="store_true")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--test-ids", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    print(f"top_k={args.top_k} | model={args.model} | llm_eval={args.llm_eval}\n")

    collection = get_collection(openai_api_key=api_key)
    openai_client = OpenAI(api_key=api_key)

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

        serializable = [
            {
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
            }
            for r in results
        ]

        output_path.write_text(
            json.dumps({"config": vars(args), "aggregate": agg, "cases": serializable},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
