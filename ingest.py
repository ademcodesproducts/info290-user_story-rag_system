"""
Full ingestion run: load → chunk → embed → store in ChromaDB.

Usage:
    python ingest.py
    python ingest.py --chunk-size 300 --overlap 60
    python ingest.py --reset   # wipe and rebuild the vector store from scratch
"""

import argparse
import os

from dotenv import load_dotenv

from src.ingestion.pipeline import run_pipeline
from src.vectorstore.store import add_chunks, collection_stats, get_collection

load_dotenv()


def main(chunk_size: int, overlap: int, reset: bool) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    # Step 1 — chunk all documents
    chunks = run_pipeline(chunk_size=chunk_size, overlap=overlap)

    # Step 2 — connect to (or create) the vector store
    print("Connecting to ChromaDB...")
    collection = get_collection(openai_api_key=api_key, reset=reset)

    # Step 3 — embed and upsert
    print(f"Embedding and upserting {len(chunks)} chunks...\n")
    add_chunks(collection, chunks)

    # Summary
    stats = collection_stats(collection)
    print(f"\nDone. Collection '{stats['collection']}' now has {stats['total_chunks']} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB.")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Target chunk size in approx tokens (default: 400)")
    parser.add_argument("--overlap", type=int, default=80,
                        help="Overlap in approx tokens (default: 80)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete and rebuild the vector store from scratch")
    args = parser.parse_args()

    main(chunk_size=args.chunk_size, overlap=args.overlap, reset=args.reset)
