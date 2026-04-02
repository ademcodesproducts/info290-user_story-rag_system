import os
from dataclasses import dataclass, field

import chromadb
from openai import OpenAI

from src.rag.retriever import format_context_block, retrieve
from src.rag.generator import generate, MODEL
from src.vectorstore.store import get_collection


@dataclass
class RAGResult:
    query: str
    hits: list[dict]
    pain_points: list[dict]
    user_stories: list[dict]
    summary: str
    model: str
    top_k: int


def run_query(
    query_text: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 5,
    model: str = MODEL,
    temperature: float = 0.2,
    where: dict | None = None,
) -> RAGResult:
    hits = retrieve(collection, query_text, top_k=top_k, where=where)
    context_block = format_context_block(hits)
    output = generate(
        client=openai_client,
        query_text=query_text,
        context_block=context_block,
        model=model,
        temperature=temperature,
    )
    return RAGResult(
        query=query_text,
        hits=hits,
        pain_points=output.get("pain_points", []),
        user_stories=output.get("user_stories", []),
        summary=output.get("summary", ""),
        model=model,
        top_k=top_k,
    )


def build_clients(api_key: str) -> tuple[chromadb.Collection, OpenAI]:
    collection = get_collection(openai_api_key=api_key)
    openai_client = OpenAI(api_key=api_key)
    return collection, openai_client


def format_result(result: RAGResult) -> str:
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"QUERY: {result.query}")
    lines.append(f"Retrieved {len(result.hits)} chunks | Model: {result.model} | top_k={result.top_k}")
    lines.append(f"{'='*60}\n")

    lines.append("SOURCES USED:")
    for i, hit in enumerate(result.hits, 1):
        lines.append(f"  [{i}] {hit['source_label']}  (distance: {hit['distance']:.3f})")

    lines.append(f"\nSUMMARY:\n{result.summary}")

    lines.append(f"\nPAIN POINTS ({len(result.pain_points)}):")
    for i, pp in enumerate(result.pain_points, 1):
        sources = ", ".join(pp.get("sources", []))
        lines.append(f"  {i}. [{pp.get('severity','').upper()}] {pp['description']}  {sources}")

    lines.append(f"\nUSER STORIES ({len(result.user_stories)}):")
    for i, us in enumerate(result.user_stories, 1):
        sources = ", ".join(us.get("sources", []))
        lines.append(f"\n  {i}. {us['story']}")
        lines.append(f"     Rationale: {us.get('rationale', '')}")
        lines.append(f"     Sources: {sources}")

    lines.append(f"\n{'='*60}")
    return "\n".join(lines)
