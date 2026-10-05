"""Create and cache local embeddings for document chunks."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data/cache/chunks.json"
VECTORS_PATH = ROOT / "data/cache/embeddings.npy"
MANIFEST_PATH = ROOT / "data/cache/embeddings_manifest.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = ROOT / ".cache/models"


def main():
    if not CHUNKS_PATH.exists():
        raise ValueError("Chunks are missing. Run python -m src.chunking first.")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a nonempty list.")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a JSON object.")
        for field in ("chunk_id", "section", "text"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise ValueError(f"Each chunk needs a nonempty {field}.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs must be unique.")

    # Include the section heading to give each passage more context.
    texts = [
        f"{chunk['section']}\n{chunk['text']}"
        for chunk in chunks
    ]

    signature_data = {
        "model": MODEL_NAME,
        "normalize_embeddings": True,
        "chunk_ids": chunk_ids,
        "texts": texts,
    }
    signature = hashlib.sha256(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if VECTORS_PATH.exists() and MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            vectors = np.load(VECTORS_PATH, allow_pickle=False)
            if vectors.shape == (len(chunks), 384) and np.isfinite(vectors).all():
                print(f"Reused cached embeddings: {vectors.shape}")
                return

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        local_files_only=True,
    )

    # This model truncates inputs longer than its token limit.
    # Stop here rather than silently losing part of a passage.
    for chunk, text in zip(chunks, texts):
        token_count = len(
            model.tokenizer.encode(text, add_special_tokens=True)
        )
        if token_count > model.max_seq_length:
            raise ValueError(
                f"{chunk['chunk_id']} has {token_count} tokens; "
                f"the model limit is {model.max_seq_length}. "
                "Reduce the chunk size before continuing."
            )

    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all():
        raise ValueError("The model returned invalid embeddings.")

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors, allow_pickle=False)

    manifest = {
        "model": MODEL_NAME,
        "signature": signature,
        "chunk_ids": chunk_ids,
        "shape": list(vectors.shape),
        "normalize_embeddings": True,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved embeddings: {vectors.shape}")
    print(f"Vectors: {VECTORS_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Embedding error: {exc}") from exc