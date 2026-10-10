from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.paths import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def safe_slug(title: str, max_chars: int = 60) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip("_")
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
                yield ("values", chunk)
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


from datetime import date


def export_to_markdown_with_frontmatter(
    article: str,
    seo: Optional[Any] = None,
    social: Optional[Any] = None,
) -> str:
    """
    Generates Markdown with Hugo/Jekyll/Gatsby compatible YAML frontmatter (title, date, tags, description, slug).
    """
    title = extract_title_from_md(article, "Blog Post")
    today_str = date.today().isoformat()

    seo_dict = seo if isinstance(seo, dict) else seo.model_dump() if hasattr(seo, "model_dump") else {}
    social_dict = social if isinstance(social, dict) else social.model_dump() if hasattr(social, "model_dump") else {}

    meta_title = seo_dict.get("meta_title") or title
    description = seo_dict.get("meta_description") or (article.replace("#", "").strip()[:160] + "...")
    slug = seo_dict.get("suggested_slug") or safe_slug(title)

    tags_list = []
    if seo_dict.get("primary_keyword"):
        tags_list.append(seo_dict["primary_keyword"])
    if seo_dict.get("secondary_keywords"):
        tags_list.extend(seo_dict["secondary_keywords"][:5])
    if not tags_list:
        tags_list = ["blog", "article"]

    tags_formatted = ", ".join(f'"{t}"' for t in tags_list)

    frontmatter = (
        f"---\n"
        f'title: "{meta_title}"\n'
        f'date: "{today_str}"\n'
        f'description: "{description}"\n'
        f'slug: "{slug}"\n'
        f'tags: [{tags_formatted}]\n'
        f"---\n\n"
    )

    clean_article = article.strip()

    social_block = ""
    if social_dict:
        social_block = "\n\n---\n\n## 📢 Derivative Social Media Content\n\n"
        if social_dict.get("twitter_thread"):
            social_block += "### 🐦 Twitter/X Thread\n"
            for tweet in social_dict["twitter_thread"]:
                social_block += f"- {tweet}\n\n"
        if social_dict.get("linkedin_post"):
            social_block += f"### 💼 LinkedIn Post\n{social_dict['linkedin_post']}\n\n"
        if social_dict.get("newsletter_summary"):
            social_block += f"### 📧 Newsletter Summary\n{social_dict['newsletter_summary']}\n\n"

    return frontmatter + clean_article + social_block


def export_to_html(markdown_text: str) -> str:
    """
    Converts Markdown to standalone styled HTML using a clean typography CSS layout.
    """
    html_body = ""
    try:
        import markdown
        html_body = markdown.markdown(
            markdown_text,
            extensions=["extra", "codehilite", "tables", "fenced_code", "toc"]
        )
    except Exception:
        lines = markdown_text.splitlines()
        html_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                if in_code:
                    html_lines.append("</code></pre>")
                    in_code = False
                else:
                    lang = line[3:].strip()
                    html_lines.append(f'<pre><code class="language-{lang}">')
                    in_code = True
            elif in_code:
                html_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.strip() == "":
                html_lines.append("<br/>")
            else:
                html_lines.append(f"<p>{line}</p>")
        if in_code:
            html_lines.append("</code></pre>")
        html_body = "\n".join(html_lines)

    title = extract_title_from_md(markdown_text, "Article Export")

    styled_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --accent-color: #38bdf8;
            --border-color: #334155;
            --code-bg: #090d16;
        }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.7;
            padding: 2rem 1rem;
            margin: 0;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
        }}
        h1, h2, h3, h4 {{
            color: #ffffff;
            font-weight: 700;
            margin-top: 1.8em;
            margin-bottom: 0.8em;
            line-height: 1.3;
        }}
        h1 {{ font-size: 2.25rem; border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; margin-top: 0; color: var(--accent-color); }}
        h2 {{ font-size: 1.75rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem; }}
        h3 {{ font-size: 1.35rem; }}
        a {{ color: var(--accent-color); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{ font-family: 'Fira Code', monospace; background: var(--code-bg); padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9em; border: 1px solid var(--border-color); }}
        pre {{ background: var(--code-bg); padding: 1.2rem; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border-color); }}
        pre code {{ background: transparent; padding: 0; border: none; }}
        blockquote {{ border-left: 4px solid var(--accent-color); margin: 1.5em 0; padding-left: 1rem; color: var(--text-muted); font-style: italic; background: rgba(56, 189, 248, 0.05); padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 1.5rem 0; border: 1px solid var(--border-color); }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
        th, td {{ border: 1px solid var(--border-color); padding: 0.75rem; text-align: left; }}
        th {{ background: #0f172a; font-weight: 600; color: var(--accent-color); }}
        hr {{ border: 0; height: 1px; background: var(--border-color); margin: 2.5rem 0; }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""
    return styled_document


def export_to_json(state: Dict[str, Any]) -> str:
    """
    Serializes entire generation payload (Article, SEO metadata, Fact Check Report, Social Posts) as formatted JSON.
    """
    import json

    def default_serializer(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        if isinstance(obj, (Path, date)):
            return str(obj)
        return str(obj)

    clean_payload = {}
    for k, v in state.items():
        if k in (
            "topic",
            "audience",
            "tone",
            "plan",
            "seo_plan",
            "fact_check_report",
            "critic_evaluation",
            "social_posts",
            "merged_md",
            "final",
            "image_specs",
        ):
            clean_payload[k] = v

    return json.dumps(clean_payload, indent=2, default=default_serializer)

