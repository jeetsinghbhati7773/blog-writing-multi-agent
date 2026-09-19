from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple
import streamlit as st

from src.paths import PROJECT_ROOT

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def resolve_image_path(src: str) -> Path:
    """Resolve a local markdown image link to an absolute path.

    Relative links (e.g. ``images/<slug>/file.png``) are resolved against the
    PROJECT ROOT rather than the current working directory, so images render
    correctly no matter where the app was launched from.
    """
    src = src.strip().lstrip("./")
    p = Path(src)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def render_markdown_with_local_images(md: str):
    """
    Renders markdown text, resolving any local image links to absolute paths.

    Markdown is always rendered through Streamlit's Markdown parser (never
    wrapped in a raw HTML block), so headings, bold, lists and code fences
    display correctly for both freshly generated and saved/past articles.
    """
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        # Render as real Markdown. Wrapping the whole document in a raw
        # <div> with unsafe_allow_html=True turns it into an HTML block, so
        # Streamlit stops parsing the Markdown inside it and headings, bold,
        # lists and code fences show up as literal text. That is what made
        # saved / past articles "load but not display properly" when opened
        # from history. Rendering the string directly parses it as Markdown.
        st.markdown(md)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            try:
                st.image(src, caption=caption or (alt or None), use_container_width=True)
            except Exception:
                st.caption(f"🖼️ *Image URL:* [{alt or 'View Image'}]({src})")
        else:
            img_path = resolve_image_path(src)
            if img_path.exists():
                try:
                    st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
                except Exception:
                    st.info(f"🖼️ **Diagram / Image Placeholder:** {caption or alt or 'Technical Diagram'}")
            else:
                st.warning(f"Image artifact not found: `{src}` (searched `{img_path}`)")

        i += 1
