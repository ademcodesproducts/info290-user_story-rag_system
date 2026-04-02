"""
ChromaDB vector store wrapper.

Handles:
- Creating / loading a persistent ChromaDB collection
- Adding chunks with their embeddings and metadata
- Similarity search given a query string
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings

from src.ingestion.chunker import Chunk


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "databricks_kb"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_collection(
    openai_api_key: str,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Load or create the ChromaDB collection.

    Args:
        openai_api_key:  OpenAI API key for the embedding function.
        chroma_path:     Directory where ChromaDB persists data.
        collection_name: Name of the collection.
        reset:           If True, delete and recreate the collection.
    """
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=EMBEDDING_MODEL,
    )

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def add_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    """
    Embed and upsert all chunks into the collection.
    ChromaDB calls the embedding function automatically.

    Uses batches of 100 to stay within API rate limits.
    """
    BATCH_SIZE = 100
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        ids = [
            f"{c.metadata['source_file']}__chunk_{c.metadata['chunk_index']}"
            for c in batch
        ]
        documents = [c.text for c in batch]
        metadatas = [_sanitize_metadata(c.metadata) for c in batch]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        print(f"  Upserted chunks {i+1}–{min(i+BATCH_SIZE, total)} / {total}")


def query(
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        collection:  The ChromaDB collection to search.
        query_text:  The PM's question or prompt.
        top_k:       Number of results to return.
        where:       Optional metadata filter, e.g. {"doc_type": "retro"}.

    Returns:
        List of dicts with keys: text, metadata, distance.
    """
    kwargs = {
        "query_texts": [query_text],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})

    return hits


def collection_stats(collection: chromadb.Collection) -> dict:
    """Return basic stats about the collection."""
    count = collection.count()
    return {"collection": collection.name, "total_chunks": count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool values in metadata.
    Convert anything else to string.
    """
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized
