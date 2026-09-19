from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    """
    Bundles the generated markdown file along with any images into a single zip file.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    """
    Creates a zip archive of all generated images.
    """
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Any, config: Optional[Dict[str, Any]] = None) -> Iterator[Tuple[str, Any]]:
    """
    Streams graph progress step by step and yields the final accumulated state.

    The graph is executed with thread config for checkpointer support. We stream both
    per-node "updates" and full "values" snapshots.
    """
    try:
        final_state: Any = None
        for mode, chunk in graph_app.stream(inputs, config=config, stream_mode=["updates", "values"]):
            if mode == "updates":
                yield ("updates", chunk)
            elif mode == "values":
                final_state = chunk
        yield ("final", final_state if final_state is not None else {})
        return
    except Exception as e:
        logger.debug("Combined updates/values stream unavailable, falling back: %s", e)

    try:
        final_state = None
        for chunk in graph_app.stream(inputs, config=config, stream_mode="values"):
            final_state = chunk
            yield ("values", chunk)
        yield ("final", final_state if final_state is not None else {})
        return
    except Exception as e:
        logger.debug("Values-only stream unavailable, falling back to invoke: %s", e)

    out = graph_app.invoke(inputs, config=config)
    yield ("final", out)



def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


def list_past_blogs() -> List[Path]:
    """
    Returns saved .md files in the outputs directory, ordered newest first.
    """
    files: List[Path] = []
    outputs_dir = OUTPUTS_DIR
    if outputs_dir.exists() and outputs_dir.is_dir():
        files.extend([p for p in outputs_dir.glob("*.md") if p.is_file()])
    
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Extracts the first H1 header '# ' from markdown as the document title.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback
