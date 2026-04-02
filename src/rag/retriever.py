import chromadb

from src.vectorstore.store import query


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
        return f"[PRD: {metadata.get('title', metadata.get('source_file'))}]"
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
    hits = query(collection, query_text, top_k=top_k, where=where)
    for hit in hits:
        hit["source_label"] = _format_source_label(hit["metadata"])
    return hits


def format_context_block(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit['source_label']}")
        lines.append(hit["text"])
        lines.append("")
    return "\n".join(lines).strip()
