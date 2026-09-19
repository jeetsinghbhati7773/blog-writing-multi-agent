from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.ui.components import (
    render_article_metrics,
    render_content_brief,
    render_workflow_progress,
)
from src.ui.helpers import (
    bundle_zip,
    extract_title_from_md,
    images_zip,
    list_past_blogs,
    read_md_file,
    safe_slug,
)
from src.ui.renderer import render_markdown_with_local_images
from src.rag.vectorstore import ChromaVectorStore
from src.rag.loader import load_document_bytes
from src.rag.splitter import split_documents
from src.paths import IMAGES_DIR


def render_plan_tab(out: Dict[str, Any]):
    st.subheader("🧩 Article Execution Plan")
    plan_obj = out.get("plan")
    if not plan_obj:
        st.info("No plan details found in the output.")
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
    cols[2].metric("Article Format", str(plan_dict.get("blog_kind", "explainer")).upper())

    tasks = plan_dict.get("tasks", [])
    if tasks:
        st.markdown("#### **Section Breakdown & Specifications**")
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
    st.subheader("🔎 Research Evidence & Citation Sources")
    evidence = out.get("evidence") or []
    if not evidence:
        st.info("No web evidence gathered for this run.")
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
    final_md = out.get("final") or ""
    if not final_md:
        st.warning("No generated markdown available to preview.")
        return

    plan_obj = out.get("plan")
    if hasattr(plan_obj, "blog_title"):
        blog_title = plan_obj.blog_title
    elif isinstance(plan_obj, dict):
        blog_title = plan_obj.get("blog_title", "article")
    else:
        blog_title = extract_title_from_md(final_md, "article")

    words_count = len(final_md.split())
    read_mins = max(1, round(words_count / 220))
    sources_count = len(out.get("evidence") or [])

    st.markdown(
        f"""
        <div class="article-meta">
            <strong>By AI Content Studio</strong> · {read_mins} min read · {sources_count} sources · {words_count:,} words
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_markdown_with_local_images(final_md)

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
        bundle = bundle_zip(final_md, md_filename, IMAGES_DIR)
        st.download_button(
            "📦 Download Complete Bundle (.zip)",
            data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )


def render_images_tab(out: Dict[str, Any]):
    st.subheader("🖼️ Diagrams & Visual Assets")
    specs = out.get("image_specs") or []
    images_dir = IMAGES_DIR

    if not specs and not images_dir.exists():
        st.info("No technical diagrams or images were generated for this post.")
        return

    if specs:
        st.markdown("#### **Image Specifications & Prompts**")
        st.json(specs)

    if images_dir.exists():
        files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.svg')
        ]
        if not files:
            st.warning("Image directory exists but contains no image files.")
        else:
            st.markdown("#### **Generated Visual Artifacts**")
            cols = st.columns(min(len(files), 3))
            for i, p in enumerate(sorted(files, key=lambda x: str(x))):
                rel_name = str(p.relative_to(images_dir))
                with cols[i % len(cols)]:
                    st.image(str(p), caption=rel_name, use_container_width=True)

            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download All Images (.zip)",
                    data=z,
                    file_name="article_images.zip",
                    mime="application/zip",
                )


def render_logs_tab(logs: List[str]):
    st.subheader("🧾 Technical Run Logs")
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    if logs:
        st.session_state["logs"].extend(logs)

    all_logs = "\n\n".join(st.session_state["logs"][-100:])
    st.text_area("Live Execution Stream Log", value=all_logs, height=500)


def render_article_workspace(out: Dict[str, Any], is_history: bool = False, blog_name: str = ""):
    """
    Renders workspace for an article.
    If viewing a past history blog, shows ONLY the Article preview and metrics banner.
    If viewing a newly generated article, shows full execution tabs.
    """
    if is_history:
        st.info(f"📂 **Viewing Library Article:** `{blog_name}`")
        render_article_metrics(out)
        st.divider()
        render_preview_tab(out)
    else:
        render_article_metrics(out)
        tab_article, tab_research, tab_plan, tab_visuals, tab_logs = st.tabs(
            ["📝 Article", "🔎 Research", "🧩 Plan", "🖼️ Visuals", "🧾 Run Details"]
        )
        with tab_article:
            render_preview_tab(out)
        with tab_research:
            render_evidence_tab(out)
        with tab_plan:
            render_plan_tab(out)
        with tab_visuals:
            render_images_tab(out)
        with tab_logs:
            render_logs_tab([])


def cb_back_to_archive():
    st.session_state["selected_past_article"] = None


def cb_open_past_article(filename: str):
    st.session_state["selected_past_article"] = filename
    st.session_state["app_mode"] = "📜 Past Articles"


def render_library_workspace():
    """
    Renders a dedicated SaaS page to view, read, and browse old/past articles.
    Supports dedicated full-screen article reading view with a back navigation button.
    """
    selected_file = st.session_state.get("selected_past_article")

    # Mode A: If a specific past article is selected, render dedicated Article Reader view
    if selected_file:
        past_files = list_past_blogs()
        target_path = next((p for p in past_files if p.name == selected_file), None)

        if target_path:
            col_back, col_space = st.columns([2.5, 5])
            with col_back:
                st.button(
                    "⬅️ Back to All Past Articles",
                    use_container_width=True,
                    type="secondary",
                    key="back_to_archive_btn",
                    on_click=cb_back_to_archive,
                )

            try:
                md_text = read_md_file(target_path)
                title = extract_title_from_md(md_text, target_path.stem)
            except Exception:
                md_text = ""
                title = target_path.stem

            st.info(f"📜 **Viewing Past Article:** `{title}` (`{target_path.name}`)")

            out = {
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "final": md_text,
            }
            render_article_metrics(out)
            st.divider()
            render_preview_tab(out)
            return

    # Mode B: If no article is selected, render the Past Articles Archive grid
    st.subheader("📜 Past Articles Archive")
    st.caption("Browse, read, and export your previously generated technical articles.")

    past_files = list_past_blogs()
    if not past_files:
        st.info("No saved articles found in the outputs archive.")
        return

    cols = st.columns(2)
    for idx, p in enumerate(past_files):
        try:
            md_text = read_md_file(p)
            title = extract_title_from_md(md_text, p.stem)
            words = len(md_text.split())
            read_time = max(1, round(words / 220))
            mtime = p.stat().st_mtime
            import datetime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime("%d %b %Y")
        except Exception:
            title = p.stem
            words = 0
            read_time = 1
            date_str = "Unknown"

        with cols[idx % 2]:
            with st.container():
                st.markdown(
                    f"""
                    <div class="library-card">
                        <h4 style="margin-bottom: 0.5rem; color: #F8FAFC;">{html.escape(title)}</h4>
                        <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 1rem;">
                            📅 {html.escape(str(date_str))} · ⏱️ {read_time} min read ({words:,} words)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "📖 Read Full Article",
                    key=f"lib_btn_{p.name}",
                    use_container_width=True,
                    on_click=cb_open_past_article,
                    args=(p.name,),
                )


def render_knowledge_base_workspace():
    """
    Renders dedicated Knowledge Base management view.
    """
    st.subheader("🗂 Knowledge Base Manager (ChromaDB)")
    st.caption("Upload and manage PDF, TXT, MD, and DOCX documents used for grounded RAG research.")

    vectorstore = ChromaVectorStore()

    uploaded_files = st.file_uploader(
        "Upload New Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown (.md), DOCX",
        key="kb_manager_uploader",
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Documents", type="primary", key="kb_process_btn"):
            progress_bar = st.progress(0, text="Initializing Knowledge Base Indexing Pipeline...")
            status_box = st.status("🚀 **Document Processing & Indexing Pipeline Active...**", expanded=True)

            total_chunks = 0
            total_pages = 0
            processed = 0
            num_files = len(uploaded_files)

            for file_idx, f in enumerate(uploaded_files):
                try:
                    # 1. File Reading & Text Extraction
                    status_box.write(f"📄 **Step 1/3 (Page & Text Extraction):** Reading file bytes and parsing pages from `{f.name}`...")
                    progress_bar.progress(int(((file_idx * 3) + 1) / (num_files * 3) * 100), text=f"Processing {f.name}: Extracting text...")
                    file_bytes = f.read()
                    docs = load_document_bytes(file_bytes, f.name)
                    file_page_count = len(docs)
                    total_pages += file_page_count
                    status_box.write(f"└─ ✅ Extracted **{file_page_count}** pages/sections from `{f.name}`")

                    # 2. Semantic Chunking
                    status_box.write(f"✂️ **Step 2/3 (Semantic Chunking):** Splitting document text into 1,000-character semantic chunks...")
                    progress_bar.progress(int(((file_idx * 3) + 2) / (num_files * 3) * 100), text=f"Processing {f.name}: Creating semantic chunks...")
                    chunks = split_documents(docs)
                    file_chunk_count = len(chunks)
                    status_box.write(f"└─ ✅ Generated **{file_chunk_count}** text chunks with metadata")

                    # 3. Embedding Generation & VectorDB Storage
                    status_box.write(f"🧠 **Step 3/3 (Vector Embedding & Ingestion):** Computing 384-dim dense embeddings (`all-MiniLM-L6-v2`) & saving to ChromaDB...")
                    progress_bar.progress(int(((file_idx * 3) + 3) / (num_files * 3) * 100), text=f"Processing {f.name}: Computing embeddings & saving to vector storage...")
                    added = vectorstore.add_documents(chunks)
                    total_chunks += added
                    processed += 1
                    status_box.write(f"└─ ✅ Ingested **{added}** vector embeddings into ChromaDB storage")
                    st.toast(f"✅ Indexed {f.name} ({added} chunks)", icon="📄")

                except Exception as e:
                    status_box.write(f"❌ **Error processing `{f.name}`:** {e}")
                    st.error(f"Failed to process '{f.name}': {e}")

            progress_bar.progress(100, text="Indexing complete!")
            status_box.update(
                label=f"🎉 **Knowledge Base Indexing Complete!** Processed {processed} file(s), {total_pages} total pages, and {total_chunks} vector chunks.",
                state="complete",
                expanded=True,
            )

            # Display visual summary metrics
            st.divider()
            cols = st.columns(4)
            cols[0].metric("Files Indexed", f"{processed}/{num_files}")
            cols[1].metric("Total Pages Extracted", f"{total_pages}")
            cols[2].metric("Semantic Chunks Created", f"{total_chunks}")
            cols[3].metric("Vector Embedding Model", "all-MiniLM-L6-v2")

    doc_list = vectorstore.list_documents()
    if doc_list:
        st.markdown("#### **Indexed Knowledge Base Files**")
        df = pd.DataFrame(doc_list)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear Entire Knowledge Base", type="secondary"):
            vectorstore.clear_documents()
            st.toast("Cleared ChromaDB vector storage!", icon="🧹")
            st.rerun()
    else:
        st.info("No documents currently indexed in ChromaDB knowledge base.")
