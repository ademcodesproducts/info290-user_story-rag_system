"""
Retrieval layer — fetches the most relevant chunks from ChromaDB
and formats them as numbered context blocks for the prompt.
"""

import chromadb

from src.vectorstore.store import query


# Source label builders per doc type
def _format_source_label(metadata: dict) -> str:
    doc_type = metadata.get("doc_type", "unknown")
    if doc_type == "interview":
        return (
            f"[Interview {metadata.get('interview_id', metadata.get('source_file'))} | "
            f"{metadata.get('participant', 'Unknown')} | "
            f"{metadata.get('product_area', '')}]"
        )
    if doc_type == "retro":
        return (
            f"[Retro {metadata.get('meeting_id', metadata.get('source_file'))} | "
            f"{metadata.get('team', 'Unknown Team')} | "
            f"{metadata.get('date', '')}]"
        )
    if doc_type == "prd":
        return (
            f"[PRD: {metadata.get('title', metadata.get('source_file'))}]"
        )
    if doc_type == "ticket":
        return (
            f"[Ticket {metadata.get('ticket_id', metadata.get('source_file'))} | "
            f"{metadata.get('role', '')} | "
            f"{metadata.get('product_area', '')}]"
        )
    return f"[{metadata.get('source_file', 'Unknown')}]"


def retrieve(
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieve top-k chunks and attach a formatted source label to each.

    Returns list of dicts with keys: text, metadata, distance, source_label.
    """
    hits = query(collection, query_text, top_k=top_k, where=where)
    for hit in hits:
        hit["source_label"] = _format_source_label(hit["metadata"])
    return hits


def format_context_block(hits: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context string for the prompt.

    Example:
        [1] [Interview INT-001 | Marcus Webb, Senior DE | DLT, Auto Loader]
        DLT error handling is rough. When a pipeline fails...

        [2] [Retro RETRO-003 | Analytics Engineering | March 05, 2024]
        DLT pipeline debugging is frustrating; error messages are cryptic...
    """
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit['source_label']}")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines).strip()
