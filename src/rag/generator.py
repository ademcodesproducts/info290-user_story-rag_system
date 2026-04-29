import json
import time
from openai import OpenAI


MODEL = "gpt-4o"

# ── Baseline prompt (v1) ──────────────────────────────────────────────────────

_FEW_SHOT_V1 = """
Example 1:
As a Data Engineer, I want to receive a plain-language summary of why a Workflow task
failed, so that I can identify and fix the root cause without manually searching through
thousands of lines of Spark logs.

Example 2:
As a Data Scientist, I want to pin key cell outputs (charts, metrics) to a persistent
sidebar, so that I can reference earlier results while continuing downstream analysis
without re-running expensive compute cells.

Example 3:
As a Platform Admin, I want to enforce mandatory cost-center tags on all cluster
provisioning, so that I can accurately attribute DBU spend to individual teams and
generate automated chargeback reports.
""".strip()

SYSTEM_PROMPT = f"""You are an expert Product Manager assistant for Databricks.
Your job is to analyze product research documents and generate structured product insights.

A user story must follow the Connextra format exactly:
  "As a [specific user role], I want [concrete goal], so that [measurable benefit]."

Rules for user stories:
- Be specific — name the actual Databricks feature (e.g. MLflow, Unity Catalog, Workflows)
- Each story must be directly traceable to the provided context
- Do not invent pain points not present in the context
- Role must be specific (e.g. "Data Engineer" not "user")

Examples of well-formed user stories:
{_FEW_SHOT_V1}
"""

QUERY_PROMPT_TEMPLATE = """
Using only the context below, answer the PM's question.

PM QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Respond with a JSON object using this exact schema:
{{
  "pain_points": [
    {{
      "description": "concise description of the pain point",
      "severity": "high | medium | low",
      "sources": ["[1]", "[2]"]
    }}
  ],
  "user_stories": [
    {{
      "story": "As a ..., I want ..., so that ...",
      "rationale": "one sentence explaining why this story addresses a real pain",
      "sources": ["[1]"]
    }}
  ],
  "summary": "2-3 sentence synthesis of the key findings"
}}

Return only valid JSON. No markdown, no extra text.
"""

# ── Enhanced prompt (v2) ─────────────────────────────────────────────────────

_FEW_SHOT_V2 = """
Example 1 — Interview evidence, high severity:
Pain: Data Engineers spend 3+ hours debugging DLT pipeline failures because error messages
show only internal stack traces with no indication of which source table or transformation caused the failure.
Story: As a Data Engineer, I want DLT error messages to pinpoint the exact failing transformation
and surface the upstream data issue, so that I can resolve pipeline failures in minutes instead of hours.

Example 2 — Ticket evidence, medium severity:
Pain: ML Engineers report that model endpoint cold starts exceed 45 seconds, violating
the 5-second SLA for production inference APIs, with no lower-cost warm standby option.
Story: As an ML Engineer, I want to configure a low-cost warm standby for Model Serving endpoints,
so that cold-start latency never exceeds my production SLA even at minimal traffic.

Example 3 — Retro evidence, high severity (quantified):
Pain: Platform Admins cannot attribute $120K/month in DBU spend to individual teams or projects
because Databricks lacks mandatory cost-center tagging at cluster provisioning time.
Story: As a Platform Admin, I want to enforce mandatory cost-center tags on all cluster and
SQL warehouse provisioning, so that I can generate accurate team-level chargeback reports
without manual data reconciliation.
""".strip()

SYSTEM_PROMPT_V2 = f"""You are a senior Product Manager at Databricks specializing in synthesizing
user research into actionable product insights.

SEVERITY DEFINITIONS — apply these consistently:
  high   = blocks a core workflow, causes data loss, or violates a compliance requirement
  medium = a workaround exists but costs significant time (>30 min/week) or erodes trust
  low    = friction that degrades experience but has a quick workaround

PAIN POINT RULES:
1. Include specific evidence from the context: quote numbers, time estimates, or direct user language
2. If the same pain appears in multiple sources, list all relevant source citations
3. Rank pain points high → medium → low within the output

USER STORY RULES:
1. Follow Connextra format exactly: "As a [role], I want [concrete feature], so that [measurable outcome]."
2. Name the specific Databricks feature (MLflow, Unity Catalog, DLT, Workflows, etc.)
3. The "so that" clause must state a measurable or observable outcome, not a vague benefit
4. Every story must be traceable to at least one pain point in the context
5. Role must be a real Databricks user persona (Data Engineer, ML Engineer, Analytics Engineer,
   Platform Admin, Data Scientist, BI Analyst, SRE, FinOps)

Examples of well-formed pain point → story pairs:
{_FEW_SHOT_V2}
"""

QUERY_PROMPT_TEMPLATE_V2 = """
Using ONLY the context below, answer the PM's research question.
Do not introduce information not present in the context.

PM QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Respond with a JSON object using this exact schema:
{{
  "pain_points": [
    {{
      "description": "specific pain point with evidence from the context (quote numbers or user language where available)",
      "severity": "high | medium | low",
      "sources": ["[1]", "[2]"]
    }}
  ],
  "user_stories": [
    {{
      "story": "As a [specific role], I want [concrete Databricks feature or capability], so that [measurable outcome].",
      "rationale": "one sentence linking this story directly to a pain point above",
      "sources": ["[1]"]
    }}
  ],
  "summary": "2-3 sentence synthesis: what is the core problem, who is most affected, and what is the highest-priority opportunity"
}}

Return only valid JSON. No markdown, no extra text.
"""

# ── Prompt registry ───────────────────────────────────────────────────────────

PROMPT_VARIANTS = {
    "baseline": (SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE),
    "v2": (SYSTEM_PROMPT_V2, QUERY_PROMPT_TEMPLATE_V2),
}


def generate(
    client: OpenAI,
    query_text: str,
    context_block: str,
    model: str = MODEL,
    temperature: float = 0.2,
    prompt_variant: str = "baseline",
) -> dict:
    system_prompt, query_template = PROMPT_VARIANTS.get(
        prompt_variant, PROMPT_VARIANTS["baseline"]
    )
    user_prompt = query_template.format(query=query_text, context=context_block)

    for attempt in range(3):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
            else:
                return {"pain_points": [], "user_stories": [], "summary": ""}

