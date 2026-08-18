from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.embeddings import get_embedding_function


import math


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
    """

    def __init__(self, db_path: str = "chroma_db", collection_name: str = "documents"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True, parents=True)
        self.collection_name = collection_name
        self.sqlite_path = self.db_path / f"{collection_name}.sqlite"
        self.embedding_fn = get_embedding_function()
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
                    embedding_json TEXT
                )
                """
            )
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
                    (id, filename, file_type, page, chunk_index, content, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc_id, fn, ft, pg, ci, cnt, meta_json, emb_json)
                )
            conn.commit()

        return len(records)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs similarity search in persistent vector store for query_text.
        Returns top matching chunks with content, metadata, and distance.
        """
        if not query_text.strip() or self.count_chunks() == 0:
            return []

        raw_query_emb = self.embedding_fn([query_text])[0]
        query_emb = raw_query_emb.tolist() if hasattr(raw_query_emb, "tolist") else list(raw_query_emb)


        with self._get_connection() as conn:
            cursor = conn.execute("SELECT content, metadata_json, embedding_json FROM documents")
            rows = cursor.fetchall()

        scored: List[tuple[float, str, dict]] = []
        for content, meta_json, emb_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            emb = json.loads(emb_json) if emb_json else []
            sim = _cosine_similarity(query_emb, emb)
            dist = max(0.0, 1.0 - sim)
            scored.append((dist, content, meta))

        scored.sort(key=lambda x: x[0])
        top_k = scored[:n_results]

        return [
            {
                "content": content,
                "metadata": meta,
                "distance": dist,
            }
            for dist, content, meta in top_k
        ]

    def delete_document(self, filename: str) -> bool:
        """
        Deletes all chunks matching the given filename.
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
        Lists all unique uploaded documents stored along with their file types and chunk counts.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT filename, file_type, page FROM documents")
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
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            row = cursor.fetchone()
            return row[0] if row else 0

