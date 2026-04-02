"""
Ingestion pipeline — orchestrates loading, parsing, and chunking
all documents in the knowledge base.

Usage:
    python -m src.ingestion.pipeline
    python -m src.ingestion.pipeline --chunk-size 300 --overlap 60
"""

import argparse
import json
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import Chunk, chunk_document


KB_DIR = Path("data/knowledge_base")
DOC_FOLDERS = ["interviews", "retros", "prds", "tickets"]


def load_all_documents(kb_dir: Path = KB_DIR) -> list[tuple[dict, str]]:
    """Walk all known subfolders and parse every .txt file."""
    documents = []
    for folder in DOC_FOLDERS:
        folder_path = kb_dir / folder
        if not folder_path.exists():
            print(f"[WARN] Folder not found: {folder_path}")
            continue
        for filepath in sorted(folder_path.glob("*.txt")):
            result = parse_document(filepath)
            if result:
                documents.append(result)
    return documents


def run_pipeline(
    chunk_size: int = 400,
    overlap: int = 80,
    kb_dir: Path = KB_DIR,
) -> list[Chunk]:
    """
    Full ingestion pipeline.

    Returns a flat list of Chunk objects ready for embedding.
    """
    print(f"\n{'='*50}")
    print(f"Ingestion Pipeline")
    print(f"  chunk_size : {chunk_size} tokens (approx)")
    print(f"  overlap    : {overlap} tokens (approx)")
    print(f"  kb_dir     : {kb_dir}")
    print(f"{'='*50}\n")

    documents = load_all_documents(kb_dir)
    print(f"Loaded {len(documents)} documents\n")

    all_chunks: list[Chunk] = []
    doc_type_counts: dict[str, int] = {}

    for metadata, body in documents:
        chunks = chunk_document(body, metadata, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        doc_type = metadata.get("doc_type", "unknown")
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + len(chunks)

    print("Chunks produced:")
    for doc_type, count in sorted(doc_type_counts.items()):
        print(f"  {doc_type:<12} {count} chunks")
    print(f"  {'TOTAL':<12} {len(all_chunks)} chunks\n")

    return all_chunks


def chunks_to_json(chunks: list[Chunk]) -> list[dict]:
    """Serialize chunks to a list of dicts (for inspection or downstream use)."""
    return [{"text": c.text, "metadata": c.metadata} for c in chunks]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Target chunk size in approx tokens (default: 400)")
    parser.add_argument("--overlap", type=int, default=80,
                        help="Overlap between chunks in approx tokens (default: 80)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save chunks as JSON (e.g. chunks.json)")
    args = parser.parse_args()

    chunks = run_pipeline(chunk_size=args.chunk_size, overlap=args.overlap)

    # Print a few sample chunks
    print("Sample chunks:")
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.metadata.get('source_file')} | chunk {chunk.metadata.get('chunk_index')} ---")
        print(chunk.text[:300])
        print("...")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(chunks_to_json(chunks), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\nSaved {len(chunks)} chunks to {output_path}")
