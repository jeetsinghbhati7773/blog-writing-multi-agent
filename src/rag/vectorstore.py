from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.embeddings import get_embedding_function
from src.paths import CHROMA_DIR


import math

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy is a transitive dep, but stay safe
    _HAS_NUMPY = False


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 > 0 and norm2 > 0:
        return float(dot / (norm1 * norm2))
    return 0.0


class ChromaVectorStore:
    """
    Manages persistent local vector storage for Document Chat (RAG).
    Path: chroma_db/
    Collection: documents

    Each chunk is tagged with the embedding model that produced its vector
    (``embedding_model``). Queries, counts, and listings only consider vectors
    created by the CURRENTLY active embedding function. This makes switching
    embedding models safe: for example, upgrading from the lightweight hash
    embeddings to real all-MiniLM-L6-v2 vectors won't compare incompatible
    vectors against each other (which would produce meaningless similarity
    scores). Stale vectors are simply ignored until their documents are
    re-indexed, at which point they are overwritten in place.
    """

    def __init__(self, db_path: Optional[str] = None, collection_name: str = "documents"):
        self.db_path = Path(db_path) if db_path else CHROMA_DIR
        self.db_path.mkdir(exist_ok=True, parents=True)
        self.collection_name = collection_name
        self.sqlite_path = self.db_path / f"{collection_name}.sqlite"
        self.embedding_fn = get_embedding_function()
        try:
            self.embedding_signature = str(self.embedding_fn.name())
        except Exception:
            self.embedding_signature = self.embedding_fn.__class__.__name__
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.sqlite_path))

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    file_type TEXT,
                    page INTEGER,
                    chunk_index INTEGER,
                    content TEXT,
                    metadata_json TEXT,
                    embedding_json TEXT,
                    embedding_model TEXT
                )
                """
            )
            # Migrate older databases created before the embedding_model column existed.
            existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()]
            if "embedding_model" not in existing_cols:
                conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT")
            conn.commit()

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds document chunks to the persistent vector store.
        Returns the number of added chunks.
        """
        if not chunks:
            return 0

        contents: List[str] = []
        records: List[tuple] = []

        for idx, chunk in enumerate(chunks):
            content = chunk.get("page_content", "").strip()
            if not content:
                continue

            meta = chunk.get("metadata", {})
            filename = str(meta.get("filename", "doc"))
            file_type = str(meta.get("file_type", "txt"))
            page = int(meta.get("page", 1))
            chunk_idx = int(meta.get("chunk_index", idx))

            hash_id = hashlib.md5(f"{filename}_{page}_{chunk_idx}_{content[:50]}".encode()).hexdigest()[:12]
            doc_id = f"{filename}_p{page}_c{chunk_idx}_{hash_id}"

            contents.append(content)
            records.append((
                doc_id,
                filename,
                file_type,
                page,
                chunk_idx,
                content,
                json.dumps(meta),
            ))

        if not records:
            return 0

        # Generate dense vector embeddings
        raw_embeddings = self.embedding_fn(contents)
        embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embeddings]

        with self._get_connection() as conn:
            for rec, emb in zip(records, embeddings):
                doc_id, fn, ft, pg, ci, cnt, meta_json = rec
                emb_json = json.dumps(emb)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (id, filename, file_type, page, chunk_index, content, metadata_json, embedding_json, embedding_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, fn, ft, pg, ci, cnt, meta_json, emb_json, self.embedding_signature)
                )
            conn.commit()

        return len(records)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs similarity search in the persistent vector store for query_text.
        Only vectors created by the active embedding model are considered.
        Returns top matching chunks with content, metadata, and distance.

        Scoring is vectorized with NumPy (a single matrix operation over all
        candidate vectors) when available, which is dramatically faster than a
        per-row Python loop as the collection grows. Falls back to pure Python
        if NumPy is unavailable.
        """
        if not query_text.strip() or self.count_chunks() == 0:
            return []

        raw_query_emb = self.embedding_fn([query_text])[0]
        query_emb = raw_query_emb.tolist() if hasattr(raw_query_emb, "tolist") else list(raw_query_emb)
        dim = len(query_emb)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT content, metadata_json, embedding_json FROM documents WHERE embedding_model = ?",
                (self.embedding_signature,),
            )
            rows = cursor.fetchall()

        # Materialize only well-formed, dimension-matching vectors.
        contents: List[str] = []
        metas: List[dict] = []
        embeddings: List[List[float]] = []
        for content, meta_json, emb_json in rows:
            emb = json.loads(emb_json) if emb_json else []
            if len(emb) != dim:
                continue
            contents.append(content)
            metas.append(json.loads(meta_json) if meta_json else {})
            embeddings.append(emb)

        if not embeddings:
            return []

        if _HAS_NUMPY:
            mat = np.asarray(embeddings, dtype=np.float32)          # (n, d)
            q = np.asarray(query_emb, dtype=np.float32)             # (d,)
            mat_norms = np.linalg.norm(mat, axis=1)
            q_norm = float(np.linalg.norm(q))
            denom = mat_norms * q_norm
            sims = np.zeros(len(embeddings), dtype=np.float32)
            nz = denom > 0
            sims[nz] = (mat[nz] @ q) / denom[nz]
            dists = np.clip(1.0 - sims, 0.0, None)

            k = min(n_results, dists.shape[0])
            # Partial selection of the k smallest distances, then sort just those k.
            top_idx = np.argpartition(dists, k - 1)[:k]
            top_idx = top_idx[np.argsort(dists[top_idx])]
            return [
                {"content": contents[i], "metadata": metas[i], "distance": float(dists[i])}
                for i in top_idx
            ]

        # Pure-Python fallback.
        scored = [
            (max(0.0, 1.0 - _cosine_similarity(query_emb, emb)), content, meta)
            for content, meta, emb in zip(contents, metas, embeddings)
        ]
        scored.sort(key=lambda x: x[0])
        return [
            {"content": content, "metadata": meta, "distance": dist}
            for dist, content, meta in scored[:n_results]
        ]

    def delete_document(self, filename: str) -> bool:
        """
        Deletes all chunks matching the given filename (across all embedding models).
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_documents(self) -> None:
        """
        Clears the entire document collection.
        """
        with self._get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS documents")
            conn.commit()
        self._init_db()

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lists unique documents indexed with the active embedding model, along with
        their file types and chunk counts.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT filename, file_type, page FROM documents WHERE embedding_model = ?",
                (self.embedding_signature,),
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        docs_summary: Dict[str, Dict[str, Any]] = {}
        for fn, ft, pg in rows:
            if fn not in docs_summary:
                docs_summary[fn] = {
                    "filename": fn,
                    "file_type": ft,
                    "chunks": 0,
                    "pages": set(),
                }
            docs_summary[fn]["chunks"] += 1
            if pg:
                docs_summary[fn]["pages"].add(pg)

        out = []
        for name, info in docs_summary.items():
            out.append({
                "filename": name,
                "file_type": info["file_type"],
                "chunks": info["chunks"],
                "page_count": len(info["pages"]) if info["pages"] else 1,
            })
        return sorted(out, key=lambda x: x["filename"])

    def count_chunks(self) -> int:
        """Counts chunks indexed with the active embedding model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE embedding_model = ?",
                (self.embedding_signature,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    def count_all_chunks(self) -> int:
        """Counts every stored chunk regardless of embedding model (diagnostics/migration)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            row = cursor.fetchone()
            return row[0] if row else 0
