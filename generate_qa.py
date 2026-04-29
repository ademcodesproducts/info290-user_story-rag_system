"""
Generate ~200 QA evaluation pairs from the knowledge base using GPT-4o-mini,
then split them 80/20 into validation and test sets.

Output files:
  data/eval/qa_pairs_all.json     — full generated set
  data/eval/qa_validation.json    — 80 % split (hyperparameter tuning)
  data/eval/qa_test.json          — 20 % split (final held-out evaluation)

Usage:
  python generate_qa.py
  python generate_qa.py --n-per-doc 5 --model gpt-4o-mini --seed 42
  python generate_qa.py --dry-run          # runs on first 3 docs only
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KB_ROOT = Path("data/knowledge_base")
EVAL_DIR = Path("data/eval")
DOC_SUBDIRS = ["interviews", "prds", "retros", "tickets"]

DOC_TYPE_MAP = {
    "interviews": "interview",
    "prds": "prd",
    "retros": "retro",
    "tickets": "ticket",
}

SYSTEM_PROMPT = (
    "You are building an evaluation dataset for a RAG system that helps Product Managers "
    "and UX researchers synthesize Databricks user research into user stories and pain points."
)

USER_PROMPT_TEMPLATE = """\
Given this document, generate {n} distinct QA evaluation specs. Each spec tests whether \
the RAG system can correctly retrieve this document and synthesize its key content.

For each spec, return:
- "query": A question a PM or UX researcher would realistically ask. Focus on pain points, \
user frustrations, unmet needs, or feature requests — NOT implementation details.
- "expected_pain_keywords": 4–7 lowercase keywords that a correct answer MUST mention in its \
pain points section. Be specific to this document (avoid generic words like "issue" or "problem").
- "expected_story_roles": 1–3 user role strings relevant to this document, chosen from: \
"Data Engineer", "ML Engineer", "Analytics Engineer", "Platform Admin", "Data Scientist", \
"BI Analyst", "SRE", "FinOps", "Engineering Manager", "Security Engineer".

Document: {filename}  (type: {doc_type})

---
{content}
---

Return ONLY valid JSON in this exact shape, no other text:
{{"pairs": [ <{n} objects with keys query, expected_pain_keywords, expected_story_roles> ]}}"""


def load_documents(dry_run: bool = False) -> list[dict]:
    docs = []
    for subdir in DOC_SUBDIRS:
        subpath = KB_ROOT / subdir
        if not subpath.exists():
            continue
        for fp in sorted(subpath.glob("*.txt")):
            docs.append(
                {
                    "filename": fp.name,
                    "doc_type": DOC_TYPE_MAP[subdir],
                    "content": fp.read_text(encoding="utf-8"),
                }
            )
    if dry_run:
        docs = docs[:3]
    return docs


def generate_pairs_for_doc(
    client: OpenAI,
    doc: dict,
    n: int,
    model: str,
    retries: int = 3,
) -> list[dict]:
    content = doc["content"][:4500]  # keep well within token budget
    prompt = USER_PROMPT_TEMPLATE.format(
        n=n,
        filename=doc["filename"],
        doc_type=doc["doc_type"],
        content=content,
    )

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)

            # Accept {"pairs": [...]} or bare list
            if isinstance(parsed, list):
                pairs = parsed
            else:
                pairs = next((v for v in parsed.values() if isinstance(v, list)), [])

            # Attach source file
            for p in pairs:
                p["expected_sources"] = [doc["filename"]]

            return pairs

        except Exception as exc:
            wait = 2 ** attempt
            print(f"    attempt {attempt + 1} failed ({exc}); retrying in {wait}s…")
            time.sleep(wait)

    print(f"    SKIPPED {doc['filename']} after {retries} failed attempts")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA pairs from knowledge base")
    parser.add_argument("--n-per-doc", type=int, default=5, help="QA pairs per document (default 5)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--val-ratio", type=float, default=0.8, help="Validation fraction (default 0.8)")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 3 docs only")
    args = parser.parse_args()

    random.seed(args.seed)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    docs = load_documents(dry_run=args.dry_run)
    print(f"Loaded {len(docs)} documents")
    if args.dry_run:
        print("DRY RUN — processing first 3 documents only\n")

    all_pairs: list[dict] = []
    for i, doc in enumerate(docs):
        print(f"  [{i + 1}/{len(docs)}] {doc['filename']} ({doc['doc_type']})…")
        pairs = generate_pairs_for_doc(client, doc, n=args.n_per_doc, model=args.model)
        all_pairs.extend(pairs)
        print(f"    → {len(pairs)} pairs  (running total: {len(all_pairs)})")

    if not all_pairs:
        print("No pairs generated — exiting.")
        return

    # Assign sequential IDs after shuffling
    random.shuffle(all_pairs)
    for idx, pair in enumerate(all_pairs):
        pair["id"] = f"Q{idx + 1:03d}"

    # 80 / 20 split
    split = int(len(all_pairs) * args.val_ratio)
    val_set = all_pairs[:split]
    test_set = all_pairs[split:]

    print(f"\nTotal: {len(all_pairs)} pairs  |  val: {len(val_set)}  |  test: {len(test_set)}")

    # Save
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for path, data, label in [
        (EVAL_DIR / "qa_pairs_all.json", all_pairs, "all"),
        (EVAL_DIR / "qa_validation.json", val_set, "validation"),
        (EVAL_DIR / "qa_test.json", test_set, "test"),
    ]:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {label}: {path}  ({len(data)} pairs)")


if __name__ == "__main__":
    main()
