from __future__ import annotations

import os
from pathlib import Path

# Anchor all app data directories to the PROJECT ROOT (the parent of this
# ``src`` package) rather than the current working directory. Previously
# ``outputs/``, ``images/`` and ``chroma_db/`` were plain relative paths, so the
# app only behaved correctly when launched from the repo root. Resolving them
# against this file's location makes the app runnable from anywhere.
#
# Each location can also be overridden with an environment variable, which is
# handy for containers, read-only installs, or pointing at a shared data volume.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(env_var: str, default_name: str) -> Path:
    override = os.getenv(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / default_name


# Generated blog markdown is written here (and the "Past Articles" archive reads it).
OUTPUTS_DIR = _resolve("BLOG_OUTPUTS_DIR", "outputs")

# Generated images/diagrams are written under IMAGES_DIR/<topic_slug>/.
# NOTE: image links embedded in the markdown are stored relative to PROJECT_ROOT
# (e.g. "images/<slug>/file.png"), and the renderer resolves them against
# PROJECT_ROOT, so the default location keeps existing article files working.
IMAGES_DIR = _resolve("BLOG_IMAGES_DIR", "images")

# Persistent SQLite-backed vector store for Document Chat / RAG.
CHROMA_DIR = _resolve("BLOG_CHROMA_DIR", "chroma_db")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
