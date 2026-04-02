"""
Evaluation metrics for the RAG system.

Two tiers:
  Cheap  — no API calls, run always
    - retrieval_recall_at_k   : did expected sources appear in top-k hits?
    - reciprocal_rank         : rank of the first expected source hit
    - pain_keyword_recall     : do expected keywords appear in generated pain points?
    - user_story_format_score : Connextra format compliance + basic INVEST checks

  LLM judge — costs tokens, run with --llm-eval flag
    - faithfulness_score      : is the answer grounded in the retrieved context?
    - relevance_score         : does the answer actually address the query?
    - invest_score            : INVEST criteria scoring per user story
"""

import re
from openai import OpenAI


# ---------------------------------------------------------------------------
# Cheap metrics (no API)
# ---------------------------------------------------------------------------

def retrieval_recall_at_k(hits: list[dict], expected_sources: list[str]) -> float:
    """
    Fraction of expected source files that appear in the retrieved hits.
    A hit matches if any expected source filename is a substring of the hit's source_file.
    """
    if not expected_sources:
        return 1.0
    retrieved_files = {h["metadata"].get("source_file", "") for h in hits}
    matched = sum(
        1 for exp in expected_sources
        if any(exp in rf for rf in retrieved_files)
    )
    return matched / len(expected_sources)


def mean_reciprocal_rank(hits: list[dict], expected_sources: list[str]) -> float:
    """
    MRR: 1/rank of the first hit that matches any expected source.
    Returns 0 if no match found.
    """
    for rank, hit in enumerate(hits, start=1):
        source_file = hit["metadata"].get("source_file", "")
        if any(exp in source_file for exp in expected_sources):
            return 1.0 / rank
    return 0.0


def pain_keyword_recall(pain_points: list[dict], expected_keywords: list[str]) -> float:
    """
    Fraction of expected keywords found (case-insensitive substring match)
    anywhere in the generated pain point descriptions.
    """
    if not expected_keywords:
        return 1.0
    combined_text = " ".join(pp.get("description", "").lower() for pp in pain_points)
    matched = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return matched / len(expected_keywords)


CONNEXTRA_PATTERN = re.compile(
    r"as\s+a\s+.+?,\s+i\s+want\s+.+?,\s+so\s+that\s+.+",
    re.IGNORECASE,
)

def user_story_format_score(user_stories: list[dict]) -> dict:
    """
    Checks each user story against:
      - Connextra format compliance  (As a..., I want..., so that...)
      - Specificity                  (mentions a named feature / tool)
      - Benefit clause               (so that clause has >5 words)

    Returns:
        {
          "format_compliance":  0.0–1.0  (fraction with correct format),
          "has_named_feature":  0.0–1.0  (fraction mentioning a specific tool),
          "benefit_detail":     0.0–1.0  (fraction with meaningful benefit clause),
          "overall":            0.0–1.0  (mean of the three),
          "per_story":          list of per-story detail dicts
        }
    """
    if not user_stories:
        return {
            "format_compliance": 0.0,
            "has_named_feature": 0.0,
            "benefit_detail": 0.0,
            "overall": 0.0,
            "per_story": [],
        }

    # Known Databricks features — presence of any = "specific"
    db_features = [
        "mlflow", "unity catalog", "delta", "workflow", "dlt", "delta live",
        "auto loader", "cluster", "notebook", "sql warehouse", "repos",
        "mosaic", "vector search", "feature store", "photon", "databricks",
    ]

    per_story = []
    for us in user_stories:
        story = us.get("story", "")
        story_lower = story.lower()

        fmt = bool(CONNEXTRA_PATTERN.match(story_lower))

        has_feature = any(f in story_lower for f in db_features)

        # Extract "so that ..." clause and count words
        so_that_match = re.search(r"so\s+that\s+(.+)", story, re.IGNORECASE)
        benefit_words = len(so_that_match.group(1).split()) if so_that_match else 0
        benefit_ok = benefit_words >= 6

        per_story.append({
            "story": story[:80],
            "format_ok": fmt,
            "has_named_feature": has_feature,
            "benefit_detail_ok": benefit_ok,
        })

    n = len(per_story)
    fmt_score = sum(s["format_ok"] for s in per_story) / n
    feature_score = sum(s["has_named_feature"] for s in per_story) / n
    benefit_score = sum(s["benefit_detail_ok"] for s in per_story) / n

    return {
        "format_compliance": round(fmt_score, 3),
        "has_named_feature": round(feature_score, 3),
        "benefit_detail": round(benefit_score, 3),
        "overall": round((fmt_score + feature_score + benefit_score) / 3, 3),
        "per_story": per_story,
    }


# ---------------------------------------------------------------------------
# LLM judge metrics
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating a RAG system's output for faithfulness.

CONTEXT (retrieved chunks):
{context}

GENERATED ANSWER:
Pain points: {pain_points}
User stories: {user_stories}

Score the faithfulness of the generated answer on a scale of 1–5:
  5 = All claims are directly supported by the context
  4 = Most claims supported, minor extrapolation
  3 = Some claims supported, some invented
  2 = Many claims not found in context
  1 = Answer is mostly hallucinated

Respond with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

RELEVANCE_PROMPT = """You are evaluating whether a RAG system's answer addresses the user's question.

QUESTION: {query}

ANSWER SUMMARY: {summary}

Score the relevance on a scale of 1–5:
  5 = Directly and completely answers the question
  4 = Mostly answers with minor gaps
  3 = Partially answers
  2 = Tangentially related
  1 = Does not answer the question

Respond with JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

INVEST_PROMPT = """You are evaluating a user story against the INVEST criteria.

USER STORY: {story}

Score each criterion 1 (poor) to 3 (good):
  I — Independent: does not depend on another story to make sense
  N — Negotiable: describes a goal, not an implementation detail
  V — Valuable: clear user benefit stated
  E — Estimable: specific enough to estimate effort
  S — Small: focused, not an epic
  T — Testable: has an implicit or explicit acceptance criterion

Respond with JSON:
{{
  "independent": <1-3>,
  "negotiable": <1-3>,
  "valuable": <1-3>,
  "estimable": <1-3>,
  "small": <1-3>,
  "testable": <1-3>,
  "total": <sum 6-18>
}}"""


def _llm_judge(client: OpenAI, prompt: str, model: str = "gpt-4o-mini") -> dict:
    import json as _json
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return _json.loads(response.choices[0].message.content)


def faithfulness_score(
    client: OpenAI,
    context_block: str,
    pain_points: list[dict],
    user_stories: list[dict],
    model: str = "gpt-4o-mini",
) -> dict:
    pp_text = "; ".join(p.get("description", "") for p in pain_points)
    us_text = "; ".join(u.get("story", "") for u in user_stories)
    prompt = FAITHFULNESS_PROMPT.format(
        context=context_block[:3000],
        pain_points=pp_text,
        user_stories=us_text,
    )
    return _llm_judge(client, prompt, model)


def relevance_score(
    client: OpenAI,
    query: str,
    summary: str,
    model: str = "gpt-4o-mini",
) -> dict:
    prompt = RELEVANCE_PROMPT.format(query=query, summary=summary)
    return _llm_judge(client, prompt, model)


def invest_scores(
    client: OpenAI,
    user_stories: list[dict],
    model: str = "gpt-4o-mini",
) -> list[dict]:
    results = []
    for us in user_stories:
        prompt = INVEST_PROMPT.format(story=us.get("story", ""))
        score = _llm_judge(client, prompt, model)
        score["story"] = us.get("story", "")[:80]
        results.append(score)
    return results
