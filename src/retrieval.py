"""Retrieve evidence from the local Chroma database."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_INDEX = ROOT / "data/cache/active_index.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model():
    """Load the cached model once per Python process."""
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(ROOT / ".cache/models"),
        local_files_only=True,
    )


def retrieve(question, top_k=3):
    """Return ranked passages; ranking does not prove sufficient evidence."""
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")
    if len(question) > 1000:
        raise ValueError("Questions must be 1,000 characters or fewer.")
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    if not ACTIVE_INDEX.exists():
        raise ValueError("Index is missing. Run python -m src.indexing.")

    active = json.loads(ACTIVE_INDEX.read_text(encoding="utf-8"))
    if active.get("model") != MODEL_NAME:
        raise ValueError("Index uses a different embedding model.")

    client = chromadb.PersistentClient(path=str(ROOT / "data/index"))
    collection = client.get_collection(
        name=active["collection_name"],
        embedding_function=None,
    )

    count = collection.count()
    if count == 0:
        return []

    model = load_model()
    token_count = len(
        model.tokenizer.encode(question, add_special_tokens=True)
    )
    if token_count > model.max_seq_length:
        raise ValueError("Question is too long for the embedding model.")

    vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=vector.tolist(),
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    return passages


def main():
    parser = argparse.ArgumentParser(description="Search local evidence.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    passages = retrieve(args.question, args.top_k)
    if not passages:
        print("No passages found.")
        return

    for rank, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]
        print(f"\nRESULT {rank}: {passage['chunk_id']}")
        print(f"Section: {metadata['section']}")
        print(f"Source: {metadata['title']}")
        print(f"Local PDF page: {metadata['page_number']}")
        print(f"URL: {metadata['source_url']}")
        print(f"Cosine distance: {passage['distance']:.4f}")
        print(f"Passage: {passage['text']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"Retrieval error: {exc}") from exc