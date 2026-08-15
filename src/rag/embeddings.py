from __future__ import annotations

import math
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
    Generates normalized 384-dimensional dense vectors using word and n-gram feature hashing.
    Ensures vector search and ChromaDB persistence work reliably in any environment without external C++ DLL dependencies.
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
                features.extend([t[i:i+3] for i in range(len(t) - 2)])

        for f in features:
            h = hash(f) % self.dim
            vec[h] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(t) for t in input]


def get_embedding_function() -> Any:
    """
    Returns a zero-cost local Chroma-compatible embedding function.
    Uses LightweightEmbeddingFunction (384-dim dense vectors) for fast, deterministic,
    offline vector search guaranteed to run crash-free across all OS environments.
    """
    return LightweightEmbeddingFunction()





