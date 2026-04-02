import argparse
import os

from dotenv import load_dotenv

from src.ingestion.pipeline import run_pipeline
from src.vectorstore.store import add_chunks, collection_stats, get_collection

load_dotenv()


def main(chunk_size: int, overlap: int, reset: bool) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to your .env file.")

    chunks = run_pipeline(chunk_size=chunk_size, overlap=overlap)

    print("Connecting to ChromaDB...")
    collection = get_collection(openai_api_key=api_key, reset=reset)

    print(f"Embedding and upserting {len(chunks)} chunks...\n")
    add_chunks(collection, chunks)

    stats = collection_stats(collection)
    print(f"\nDone. Collection '{stats['collection']}' now has {stats['total_chunks']} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    main(chunk_size=args.chunk_size, overlap=args.overlap, reset=args.reset)
