from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List


def load_document_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Parses raw bytes of an uploaded file into document chunk objects with page contents and metadata.
    Supported formats: .pdf, .txt, .md, .docx
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError(f"File '{filename}' is empty.")

    ext = Path(filename).suffix.lower()
    docs: List[Dict[str, Any]] = []

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            if not reader.pages:
                raise ValueError(f"PDF '{filename}' contains no readable pages.")

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "page_content": text.strip(),
                        "metadata": {
                            "filename": filename,
                            "file_type": "pdf",
                            "page": i + 1,
                        }
                    })
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Error parsing PDF file '{filename}': {e}")

    elif ext in (".txt", ".md"):
        try:
            file_type = "md" if ext == ".md" else "txt"
            text = file_bytes.decode("utf-8", errors="replace").strip()
            if not text:
                raise ValueError(f"File '{filename}' is empty.")
            docs.append({
                "page_content": text,
                "metadata": {
                    "filename": filename,
                    "file_type": file_type,
                    "page": 1,
                }
            })
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Error reading text file '{filename}': {e}")

    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
            if not full_text:
                raise ValueError(f"DOCX file '{filename}' contains no readable text.")
            docs.append({
                "page_content": full_text,
                "metadata": {
                    "filename": filename,
                    "file_type": "docx",
                    "page": 1,
                }
            })
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Error parsing DOCX file '{filename}': {e}")

    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .md, .docx.")

    if not docs:
        raise ValueError(f"No extractable text content found in file '{filename}'.")

    return docs


def load_document_file(filepath: Path | str) -> List[Dict[str, Any]]:
    """
    Utility to load a document directly from a filesystem path.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return load_document_bytes(path.read_bytes(), path.name)
