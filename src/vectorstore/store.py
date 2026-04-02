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
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    BATCH_SIZE = 100
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [f"{c.metadata['source_file']}__chunk_{c.metadata['chunk_index']}" for c in batch]
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
    kwargs = {
        "query_texts": [query_text],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_stats(collection: chromadb.Collection) -> dict:
    return {"collection": collection.name, "total_chunks": collection.count()}


def _sanitize_metadata(metadata: dict) -> dict:
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in metadata.items()
    }
