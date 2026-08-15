from __future__ import annotations

from typing import Any, Dict, List


def _recursive_split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively splits text on separators ('\n\n', '\n', ' ', '') to fit within chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", " ", ""]
    separator = ""
    for s in separators:
        if s in text:
            separator = s
            break

    splits = text.split(separator) if separator else list(text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for piece in splits:
        piece_len = len(piece) + (len(separator) if current_chunk else 0)
        if current_len + piece_len > chunk_size and current_chunk:
            chunk_str = separator.join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Retain overlap
            overlap_pieces: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) <= chunk_overlap:
                    overlap_pieces.insert(0, item)
                    overlap_len += len(item)
                else:
                    break
            current_chunk = overlap_pieces
            current_len = sum(len(x) for x in current_chunk) + (len(separator) * (len(current_chunk) - 1) if current_chunk else 0)

        current_chunk.append(piece)
        current_len += piece_len

    if current_chunk:
        chunk_str = separator.join(current_chunk).strip()
        if chunk_str:
            chunks.append(chunk_str)

    return chunks


def split_documents(
    docs: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Splits document objects into chunk objects while preserving metadata (filename, file_type, page).
    """
    chunks: List[Dict[str, Any]] = []

    for doc in docs:
        content = doc.get("page_content", "")
        meta = doc.get("metadata", {})
        if not content.strip():
            continue

        split_texts = _recursive_split_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, text_chunk in enumerate(split_texts):
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = i
            chunks.append({
                "page_content": text_chunk,
                "metadata": chunk_meta,
            })

    return chunks
