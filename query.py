import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.rag.pipeline import build_clients, format_result, run_query

load_dotenv()

VALID_DOC_TYPES = ["interview", "retro", "prd", "ticket"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--filter-type", type=str, choices=VALID_DOC_TYPES, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    where = {"doc_type": args.filter_type} if args.filter_type else None

    print(f"Querying: \"{args.query}\"")
    print(f"top_k={args.top_k} | model={args.model} | filter={args.filter_type or 'none'}\n")

    collection, openai_client = build_clients(api_key)

    result = run_query(
        query_text=args.query,
        collection=collection,
        openai_client=openai_client,
        top_k=args.top_k,
        model=args.model,
        temperature=args.temperature,
        where=where,
    )

    if args.json:
        print(json.dumps({
            "query": result.query,
            "summary": result.summary,
            "pain_points": result.pain_points,
            "user_stories": result.user_stories,
            "sources": [h["source_label"] for h in result.hits],
        }, indent=2))
    else:
        print(format_result(result))


if __name__ == "__main__":
    main()
