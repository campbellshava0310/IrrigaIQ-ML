"""Sentence embeddings for the agricultural knowledge base (retrieval for the assistant).

Runs locally with fastembed (ONNX, no GPU, no external API): BAAI/bge-small-en-v1.5, 384
dimensions. The model is downloaded once into a cache directory on first use.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
# bge models retrieve better when queries (not passages) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CACHE_DIR = Path(os.environ.get("IRRIGAIQ_EMBEDDINGS_CACHE", Path(__file__).resolve().parents[1] / ".cache" / "fastembed"))


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # imported lazily: loading ONNX is slow

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))


def embed(texts: list[str], *, query: bool = False) -> list[list[float]]:
    """Unit-length vectors, so cosine similarity is a dot product."""
    if not texts:
        return []
    inputs = [QUERY_PREFIX + t if query else t for t in texts]
    return [vector.tolist() for vector in _model().embed(inputs)]
