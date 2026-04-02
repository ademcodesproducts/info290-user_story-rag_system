import json
from openai import OpenAI


MODEL = "gpt-4o"

FEW_SHOT_EXAMPLES = """
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
{FEW_SHOT_EXAMPLES}
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


def generate(
    client: OpenAI,
    query_text: str,
    context_block: str,
    model: str = MODEL,
    temperature: float = 0.2,
) -> dict:
    user_prompt = QUERY_PROMPT_TEMPLATE.format(query=query_text, context=context_block)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)
