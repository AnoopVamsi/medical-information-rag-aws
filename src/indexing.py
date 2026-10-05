"""Store cached passages and embeddings in a local Chroma index."""

import hashlib
import json
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
INDEX = ROOT / "data/index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    chunks = json.loads(
        (CACHE / "chunks.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (CACHE / "embeddings_manifest.json").read_text(encoding="utf-8")
    )
    vectors = np.load(CACHE / "embeddings.npy", allow_pickle=False)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No chunks found. Run python -m src.chunking.")

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]

    # Verify that these embeddings belong to the current chunk contents.
    signature_data = {
        "model": MODEL_NAME,
        "normalize_embeddings": True,
        "chunk_ids": ids,
        "texts": texts,
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if manifest.get("signature") != signature:
        raise ValueError(
            "Embeddings are outdated. Run python -m src.embeddings."
        )

    if manifest.get("chunk_ids") != ids:
        raise ValueError("Embedding order does not match chunk order.")

    if len(set(ids)) != len(ids):
        raise ValueError("Chunk IDs must be unique.")

    if vectors.shape != (len(chunks), 384):
        raise ValueError("Unexpected embedding dimensions.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numbers.")

    required_fields = (
        "document_id",
        "title",
        "publisher",
        "source_url",
        "section",
        "page_reference_type",
    )
    for chunk in chunks:
        for field in required_fields:
            if not isinstance(chunk.get(field), str) or not chunk[field]:
                raise ValueError(f"Missing citation field: {field}")
        if not isinstance(chunk.get("page_number"), int):
            raise ValueError("Each chunk needs a page number.")

    # Chroma metadata accepts scalar values. Preserve all such fields.
    metadatas = [
        {
            key: value
            for key, value in chunk.items()
            if key != "text"
            and isinstance(value, (str, int, float, bool))
        }
        for chunk in chunks
    ]

    # Changed text or metadata gets a separate collection.
    # This prevents old passages remaining in the active index.
    index_signature = hashlib.sha256(
        json.dumps(
            {"embedding_signature": signature, "chunks": chunks},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    collection_name = f"diabetes_{index_signature[:24]}"

    client = chromadb.PersistentClient(path=str(INDEX))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Supplying embeddings avoids any additional embedding model call.
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=[chunk["text"] for chunk in chunks[start:end]],
            embeddings=vectors[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    if collection.count() != len(chunks):
        raise ValueError("Stored passage count does not match chunk count.")

    active_index = {
        "collection_name": collection_name,
        "model": MODEL_NAME,
        "embedding_signature": signature,
        "passage_count": collection.count(),
        "distance": "cosine",
    }
    (CACHE / "active_index.json").write_text(
        json.dumps(active_index, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Indexed passages: {collection.count()}")
    print(f"Collection: {collection_name}")
    print(f"Database: {INDEX}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"Indexing error: {exc}") from exc