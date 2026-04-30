import argparse
import json
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import Chunk, chunk_document


KB_DIR = Path("data/knowledge_base")
DOC_FOLDERS = ["interviews", "retros", "prds", "tickets", "meeting_notes"]


def load_all_documents(kb_dir: Path = KB_DIR) -> list[tuple[dict, str]]:
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


def run_pipeline(chunk_size: int = 400, overlap: int = 80, kb_dir: Path = KB_DIR) -> list[Chunk]:
    print(f"\nchunk_size={chunk_size} | overlap={overlap} | kb_dir={kb_dir}\n")

    documents = load_all_documents(kb_dir)
    print(f"Loaded {len(documents)} documents\n")

    all_chunks: list[Chunk] = []
    doc_type_counts: dict[str, int] = {}

    for metadata, body in documents:
        chunks = chunk_document(body, metadata, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        doc_type = metadata.get("doc_type", "unknown")
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + len(chunks)

    for doc_type, count in sorted(doc_type_counts.items()):
        print(f"  {doc_type:<12} {count} chunks")
    print(f"  {'TOTAL':<12} {len(all_chunks)} chunks\n")

    return all_chunks


def chunks_to_json(chunks: list[Chunk]) -> list[dict]:
    return [{"text": c.text, "metadata": c.metadata} for c in chunks]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    chunks = run_pipeline(chunk_size=args.chunk_size, overlap=args.overlap)

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
