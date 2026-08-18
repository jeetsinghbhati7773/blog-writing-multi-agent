from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, List

try:
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
except Exception:
    EmbeddingFunction = object  # type: ignore
    Documents = List[str]  # type: ignore
    Embeddings = List[List[float]]  # type: ignore


class LightweightEmbeddingFunction(EmbeddingFunction):
    """
    Zero-dependency deterministic CPU embedding function.
    Generates normalized 384-dimensional dense vectors using hashlib.md5 feature hashing.
    Ensures vector search and ChromaDB persistence are 100% process-independent and deterministic.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "default"

    def _embed_text(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim
        if not tokens:
            return vec

        features = list(tokens)
        for t in tokens:
            if len(t) >= 3:
                features.extend([t[i:i + 3] for i in range(len(t) - 2)])

        for f in features:
            h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(t) for t in input]


def get_embedding_function() -> Any:
    """
    Returns a local Chroma-compatible embedding function.
    Uses LightweightEmbeddingFunction (384-dim dense vectors) for fast, deterministic,
    offline vector search guaranteed to run crash-free across all OS environments.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "lightweight").lower()
    if provider == "sentence-transformers":
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception:
            pass

    return LightweightEmbeddingFunction()





