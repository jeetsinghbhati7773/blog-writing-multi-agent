from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import List, Optional

logger = logging.getLogger(__name__)


def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Executes web search via Tavily Search tool if API key is provided.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as e:
        logger.warning("Tavily search failed for query %r: %s", query, e)
        return []


def iso_to_date(s: Optional[str]) -> Optional[date]:
    """
    Parses ISO date string (YYYY-MM-DD) into date object.
    """
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def is_valid_raster_image(data: bytes) -> bool:
    if not data or len(data) < 100:
        return False
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False


def generate_png_diagram_bytes(prompt: str) -> bytes:
    from PIL import Image, ImageDraw
    import io

    width, height = 800, 420
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Header Banner Card
    draw.rectangle([30, 25, 770, 75], fill=(59, 130, 246))
    draw.text((240, 42), "TECHNICAL DIAGRAM SPECIFICATION", fill=(255, 255, 255))

    # Inner Content Box
    draw.rectangle([30, 95, 770, 390], outline=(51, 65, 85), fill=(30, 41, 59), width=2)
    
    clean_prompt = prompt.replace("\n", " ").strip()
    lines = [clean_prompt[i:i+75] for i in range(0, min(len(clean_prompt), 225), 75)]
    
    draw.text((50, 115), "Visual Description / Specification:", fill=(56, 189, 248))
    y = 150
    for line in lines:
        draw.text((50, y), line, fill=(226, 232, 240))
        y += 30

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_mermaid_diagram_bytes(prompt: str) -> Optional[bytes]:
    """
    Renders a technical diagram using free Mermaid.ink service.
    Returns JPEG image bytes.
    """
    try:
        import urllib.parse
        import requests
        import base64

        clean_p = prompt.replace('"', "'").replace("\n", " ").strip()
        label = clean_p[:70] if clean_p else "Technical Concept"

        graph = (
            f"graph TD\n"
            f"    A[Input Concept / Query] --> B[\"{label}\"]\n"
            f"    B --> C[Core Architecture & Processing]\n"
            f"    C --> D[Output Results & Integration]"
        )
        base64_str = base64.b64encode(graph.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{base64_str}"
        res = requests.get(url, timeout=12)
        if res.status_code == 200 and is_valid_raster_image(res.content):
            return res.content
    except Exception as e:
        logger.warning("Mermaid.ink diagram render failed: %s", e)
    return None


def gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Generates raw image bytes using 100% free multi-tier pipeline:
    1. Pollinations.ai API
    2. Mermaid.ink Diagram Renderer (Free technical diagram API)
    3. Google Gemini API
    4. PIL Local PNG Diagram Generator
    """
    import urllib.parse
    import requests

    # 1. Try Pollinations AI Free / Keyed Endpoint
    pollinations_key = os.environ.get("POLLINATIONS_API_KEY", "")
    try:
        encoded_prompt = urllib.parse.quote(prompt[:180])
        key_param = f"&key={pollinations_key}" if pollinations_key else ""
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=450&nologo=true&seed=42{key_param}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
        if pollinations_key:
            headers["Authorization"] = f"Bearer {pollinations_key}"
            
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code == 200 and is_valid_raster_image(res.content):
            return res.content
    except Exception as e:
        logger.warning("Pollinations image generation failed: %s", e)

    # 2. Try Free Mermaid.ink Technical Diagram Generator
    mermaid_bytes = generate_mermaid_diagram_bytes(prompt)
    if mermaid_bytes:
        return mermaid_bytes

    # 3. Try Google Gemini API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            parts = getattr(resp, "parts", None)
            if not parts and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
            if parts:
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        data = inline.data
                        if is_valid_raster_image(data):
                            return data
        except Exception as e:
            logger.warning("Gemini image generation failed: %s", e)

    # 4. Local Verified PNG Diagram Fallback
    return generate_png_diagram_bytes(prompt)


def safe_slug(title: str, max_chars: int = 60) -> str:
    """
    Converts title into a clean filename-safe slug capped for Windows filesystem path limits.
    """
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    if max_chars and len(s) > max_chars:
        s = s[:max_chars].rstrip("_")
    return s or "blog"


def publish_to_devto(
    api_key: str,
    title: str,
    markdown_content: str,
    tags: Optional[List[str]] = None,
    published: bool = False,
) -> dict:
    """
    Publishes draft or article directly to Dev.to via API (https://dev.to/api/articles).
    """
    if not api_key:
        return {"status": "error", "message": "Dev.to API key is missing."}
    try:
        import requests

        clean_tags = [re.sub(r"[^a-zA-Z0-9]", "", tag).lower() for tag in (tags or []) if tag]
        # Dev.to accepts max 4 tags
        clean_tags = [t for t in clean_tags if t][:4]

        payload = {
            "article": {
                "title": title,
                "body_markdown": markdown_content,
                "published": published,
                "tags": clean_tags,
            }
        }
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "AIBlogWritingAgent/1.0",
        }
        res = requests.post("https://dev.to/api/articles", json=payload, headers=headers, timeout=15)
        if res.status_code in (200, 201):
            data = res.json()
            return {
                "status": "success",
                "url": data.get("url"),
                "id": data.get("id"),
                "message": "Successfully published to Dev.to!",
            }
        else:
            return {
                "status": "error",
                "status_code": res.status_code,
                "message": f"Dev.to API returned status {res.status_code}: {res.text}",
            }
    except Exception as e:
        logger.error("Failed to publish to Dev.to: %s", e)
        return {"status": "error", "message": str(e)}


def trigger_cms_webhook(webhook_url: str, payload: dict) -> dict:
    """
    Triggers a POST request to a user-defined CMS Webhook endpoint (e.g., Strapi, WordPress, Zapier).
    """
    if not webhook_url:
        return {"status": "error", "message": "Webhook URL is missing."}
    try:
        import requests

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AIBlogWritingAgent/1.0",
        }
        res = requests.post(webhook_url, json=payload, headers=headers, timeout=15)
        if 200 <= res.status_code < 300:
            return {
                "status": "success",
                "status_code": res.status_code,
                "response": res.text,
                "message": "Webhook triggered successfully!",
            }
        else:
            return {
                "status": "error",
                "status_code": res.status_code,
                "message": f"Webhook returned status {res.status_code}: {res.text[:300]}",
            }
    except Exception as e:
        logger.error("Failed to trigger CMS webhook: %s", e)
        return {"status": "error", "message": str(e)}

