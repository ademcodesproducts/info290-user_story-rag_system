import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.evaluator import aggregate_metrics, format_report, result_from_dict, run_evaluation
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
    parser.add_argument(
        "--eval-set",
        type=str,
        default="data/eval/test_set.json",
        help="Path to evaluation set JSON (e.g. data/eval/qa_validation.json)",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="baseline",
        choices=["baseline", "v2"],
        help="Prompt variant: baseline or v2 (evidence-focused with severity definitions)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run — skips cases already saved in --output file",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    eval_set_path = Path(args.eval_set)
    if not eval_set_path.exists():
        print(f"Error: eval set not found: {eval_set_path}")
        sys.exit(1)

    print(f"top_k={args.top_k} | model={args.model} | prompt={args.prompt_variant} | llm_eval={args.llm_eval} | eval_set={eval_set_path.name}\n")

    # Load previously completed cases when resuming
    prior_cases: list[dict] = []
    skip_ids: set[str] = set()
    if args.resume and args.output:
        output_path = Path(args.output)
        if output_path.exists():
            prior_data = json.loads(output_path.read_text(encoding="utf-8"))
            prior_cases = prior_data.get("cases", [])
            skip_ids = {c["test_id"] for c in prior_cases}

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
        test_set_path=eval_set_path,
        prompt_variant=args.prompt_variant,
        skip_ids=skip_ids or None,
    )

    # Merge prior results (resume) with new ones for aggregate + save
    prior_results = [result_from_dict(c) for c in prior_cases]
    all_results = prior_results + results

    agg = aggregate_metrics(all_results)
    print(format_report(all_results, agg))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        new_cases = [
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
                "pain_points": r.rag_result.pain_points if r.rag_result else [],
                "user_stories": r.rag_result.user_stories if r.rag_result else [],
                "summary": r.rag_result.summary if r.rag_result else "",
                "sources": [h["source_label"] for h in r.rag_result.hits] if r.rag_result else [],
            }
            for r in results
        ]

        output_path.write_text(
            json.dumps(
                {"config": vars(args), "aggregate": agg, "cases": prior_cases + new_cases},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved to {output_path}  ({len(prior_cases + new_cases)} total cases)")


if __name__ == "__main__":
    main()
