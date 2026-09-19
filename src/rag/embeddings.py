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
    Zero-dependency deterministic CPU embedding function (offline fallback).
    Generates normalized 384-dimensional dense vectors using hashlib.md5 feature hashing.
    Deterministic and process-independent. This is a LEXICAL (keyword-overlap) embedding,
    used only when real sentence-transformer embeddings are unavailable or explicitly
    disabled via EMBEDDING_PROVIDER=lightweight.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "lightweight-hash-384"

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


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Real semantic embedding function backed by the all-MiniLM-L6-v2 sentence-transformer.
    Produces normalized 384-dimensional dense vectors with genuine semantic similarity
    (unlike the lexical hash fallback). The model (~90MB) is downloaded and cached by
    sentence-transformers on first use, then loaded from disk on subsequent runs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._model.encode(
            list(input),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


def get_embedding_function() -> Any:
    """
    Returns the active Chroma-compatible embedding function.

    Default: real semantic all-MiniLM-L6-v2 sentence-transformer embeddings (384-dim).
    Set the EMBEDDING_PROVIDER environment variable to control this:
      - "sentence-transformers" / "minilm" (default): real semantic embeddings.
      - "lightweight" / "hash" / "offline": deterministic offline hash embeddings
        (no model download; useful for CI or air-gapped environments).

    If sentence-transformers (or the model) cannot be loaded, this automatically
    falls back to the lightweight function so the app still runs.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip().lower()

    if provider in ("lightweight", "hash", "offline", "default"):
        return LightweightEmbeddingFunction()

    try:
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception as e:  # pragma: no cover - depends on runtime environment
        print(
            "[embeddings] Could not load sentence-transformers 'all-MiniLM-L6-v2' "
            f"({e}); falling back to lightweight hash embeddings. Install "
            "sentence-transformers or set EMBEDDING_PROVIDER=lightweight to silence this."
        )
        return LightweightEmbeddingFunction()
