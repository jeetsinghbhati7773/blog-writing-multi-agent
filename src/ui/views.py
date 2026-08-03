from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.ui.helpers import safe_slug, bundle_zip, images_zip, extract_title_from_md
from src.ui.renderer import render_markdown_with_local_images


def render_plan_tab(out: Dict[str, Any]):
    st.subheader("🧩 Blog Execution Plan")
    plan_obj = out.get("plan")
    if not plan_obj:
        st.info("No plan details found in the agent output.")
        return

    if hasattr(plan_obj, "model_dump"):
        plan_dict = plan_obj.model_dump()
    elif isinstance(plan_obj, dict):
        plan_dict = plan_obj
    else:
        plan_dict = json.loads(json.dumps(plan_obj, default=str))

    st.markdown(f"### **Title:** {plan_dict.get('blog_title', 'Untitled')}")
    cols = st.columns(3)
    cols[0].metric("Target Audience", str(plan_dict.get("audience", "General")))
    cols[1].metric("Content Tone", str(plan_dict.get("tone", "Technical")))
    cols[2].metric("Blog Format", str(plan_dict.get("blog_kind", "explainer")).upper())

    tasks = plan_dict.get("tasks", [])
    if tasks:
        st.markdown("#### **Section Breakdown & Requirements**")
        df = pd.DataFrame(
            [
                {
                    "ID": t.get("id"),
                    "Section Title": t.get("title"),
                    "Target Words": t.get("target_words"),
                    "Needs Research": "Yes" if t.get("requires_research") else "No",
                    "Citations": "Yes" if t.get("requires_citations") else "No",
                    "Code Snippet": "Yes" if t.get("requires_code") else "No",
                    "Tags": ", ".join(t.get("tags") or []),
                }
                for t in tasks
            ]
        ).sort_values("ID")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🔍 View Raw Task Specifications"):
            st.json(tasks)


def render_evidence_tab(out: Dict[str, Any]):
    st.subheader("🔎 Web Research Evidence & Sources")
    evidence = out.get("evidence") or []
    if not evidence:
        st.info("No web evidence gathered (e.g., closed-book mode or no search API key configured).")
        return

    rows = []
    for e in evidence:
        if hasattr(e, "model_dump"):
            e = e.model_dump()
        rows.append(
            {
                "Title": e.get("title"),
                "Published Date": e.get("published_at") or "Unknown",
                "Source Domain": e.get("source") or "Web",
                "URL": e.get("url"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📄 Detailed Evidence Snippets"):
        for item in evidence:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            st.markdown(f"**[{item.get('title')}]({item.get('url')})**")
            if item.get("snippet"):
                st.caption(f"> {item.get('snippet')}")
            st.divider()


def render_preview_tab(out: Dict[str, Any]):
    st.subheader("📝 Live Blog Markdown Preview")
    final_md = out.get("final") or ""
    if not final_md:
        st.warning("No generated markdown available to preview.")
        return

    render_markdown_with_local_images(final_md)

    plan_obj = out.get("plan")
    if hasattr(plan_obj, "blog_title"):
        blog_title = plan_obj.blog_title
    elif isinstance(plan_obj, dict):
        blog_title = plan_obj.get("blog_title", "blog")
    else:
        blog_title = extract_title_from_md(final_md, "blog")

    md_filename = f"{safe_slug(blog_title)}.md"

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Markdown (.md)",
            data=final_md.encode("utf-8"),
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        bundle = bundle_zip(final_md, md_filename, Path("images"))
        st.download_button(
            "📦 Download Complete Bundle (.zip)",
            data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )


def render_images_tab(out: Dict[str, Any]):
    st.subheader("🖼️ Generated Diagrams & Assets")
    specs = out.get("image_specs") or []
    images_dir = Path("images")

    if not specs and not images_dir.exists():
        st.info("No technical diagrams or images were generated for this post.")
        return

    if specs:
        st.markdown("#### **Image Specifications & Prompts**")
        st.json(specs)

    if images_dir.exists():
        files = [p for p in images_dir.iterdir() if p.is_file()]
        if not files:
            st.warning("Image directory exists but contains no image files.")
        else:
            st.markdown("#### **Generated Image Artifacts**")
            cols = st.columns(min(len(files), 3))
            for i, p in enumerate(sorted(files)):
                with cols[i % len(cols)]:
                    st.image(str(p), caption=p.name, use_container_width=True)

            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download All Images (.zip)",
                    data=z,
                    file_name="blog_images.zip",
                    mime="application/zip",
                )


def render_logs_tab(logs: List[str]):
    st.subheader("🧾 Agent Execution Logs")
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if logs:
        st.session_state["logs"].extend(logs)

    all_logs = "\n\n".join(st.session_state["logs"][-100:])
    st.text_area("Live Graph Stream Log", value=all_logs, height=500)
