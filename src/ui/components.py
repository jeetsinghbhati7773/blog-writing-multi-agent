from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, Tuple
import streamlit as st


def render_system_status_sidebar():
    """
    Renders API system status at the bottom of the sidebar.
    """
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    groq_set = bool(os.getenv("GROQ_API_KEY"))
    tavily_set = bool(os.getenv("TAVILY_API_KEY"))
    pollinations_set = bool(os.getenv("POLLINATIONS_API_KEY"))
    google_set = bool(os.getenv("GOOGLE_API_KEY"))
    langsmith_set = (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY"))
    )

    all_ready = (groq_set or openai_set)
    label = "System Status ✅" if all_ready else "System Status ⚠️"

    with st.expander(f"⚙️ **{label}**", expanded=False):
        st.caption(f"- **LLM**: {'✅ Active' if (groq_set or openai_set) else '⚠️ Missing Key'}")
        st.caption(f"- **Search**: {'✅ Tavily' if tavily_set else 'ℹ️ Off'}")
        st.caption(f"- **Images**: {'✅ Active' if (pollinations_set or google_set) else 'ℹ️ Off'}")
        st.caption(f"- **Tracing**: {'✅ LangSmith' if langsmith_set else 'ℹ️ Off'}")
        st.caption("- **Vector DB**: ✅ ChromaDB")


def render_system_status_popover():
    """
    Renders API system status in a compact popover drawer.
    """
    render_system_status_sidebar()


def render_studio_header(title: str = "AI Content Studio", subtitle: str = "Generate research-backed technical content with multi-agent intelligence"):
    """
    Renders top studio header.
    """
    st.title(f"🚀 {title}")
    st.caption(subtitle)


def render_content_brief() -> Tuple[bool, Dict[str, Any]]:
    """
    Renders the structured Content Brief card on the home screen.
    Returns (run_clicked, brief_data_dict).
    """
    st.markdown("### ✨ **CONTENT BRIEF**")
    st.caption("Configure the parameters for your technical article below.")

    with st.container():
        topic = st.text_area(
            "Article Topic / Prompt",
            placeholder="e.g. Architecting High-Performance RAG Pipelines with Vector Search & Reranking",
            height=90,
            key="brief_topic",
        )

        col1, col2 = st.columns(2)
        with col1:
            audience = st.selectbox(
                "Target Audience",
                ["Software Engineers", "ML Engineers", "AI Researchers", "Product Managers", "Students", "General Technical Audience"],
                index=0,
                key="brief_audience",
            )
            tone = st.selectbox(
                "Content Tone",
                ["Expert & Practical", "Conversational Technical", "Analytical Deep-Dive", "Academic & Rigorous"],
                index=0,
                key="brief_tone",
            )
        with col2:
            format_kind = st.selectbox(
                "Article Format",
                ["System Design", "Tutorial", "Explainer", "Comparison", "Technical Deep-Dive"],
                index=0,
                key="brief_format",
            )
            word_count = st.selectbox(
                "Target Word Count",
                ["1200–1800 words", "1800–2500 words", "2500–3500 words"],
                index=1,
                key="brief_word_count",
            )

        col_date, col_kb = st.columns([1, 1])
        with col_date:
            as_of = st.date_input("As-of Target Date", value=date.today(), key="brief_as_of")
        with col_kb:
            st.write("")
            use_uploaded_docs = st.checkbox(
                "📚 Use Knowledge Base (ChromaDB)",
                value=True,
                help="Includes uploaded PDFs, TXT, MD, or DOCX documents as research context.",
                key="brief_use_kb",
            )

        # Construct comprehensive prompt internally
        full_topic_prompt = (
            f"Topic: {topic.strip()}\n"
            f"Target Audience: {audience}\n"
            f"Tone: {tone}\n"
            f"Format: {format_kind}\n"
            f"Target Length: {word_count}"
        )

        run_btn = st.button("🚀 Generate Technical Article", type="primary", use_container_width=True, key="brief_submit_btn")

        brief_data = {
            "topic_raw": topic.strip(),
            "full_prompt": full_topic_prompt,
            "audience": audience,
            "tone": tone,
            "format": format_kind,
            "word_count": word_count,
            "as_of": as_of,
            "use_uploaded_docs": use_uploaded_docs,
        }

        return run_btn, brief_data


def render_workflow_progress(current_node: str, state: Dict[str, Any]):
    """
    Renders visual workflow pipeline progress indicator during graph execution.
    """
    steps = [
        ("plan_node", "Plan"),
        ("research_router_node", "Research"),
        ("research_worker_node", "Evidence"),
        ("draft_node", "Writing"),
        ("image_gen_node", "Visuals"),
        ("assemble_node", "Complete"),
    ]

    completed_nodes = state.get("_completed_nodes", [])

    st.markdown("#### **WORKFLOW PIPELINE**")

    cols = st.columns(len(steps))
    for idx, (node_key, label) in enumerate(steps):
        if node_key == current_node:
            icon = "●"
            color = "#FF4B4B"
            status_text = f"**{label}** (In Progress)"
        elif node_key in completed_nodes or (current_node and idx < [s[0] for s in steps].index(current_node) if current_node in [s[0] for s in steps] else False):
            icon = "✓"
            color = "#10B981"
            status_text = f"**{label}**"
        else:
            icon = "○"
            color = "#94A3B8"
            status_text = f"{label}"

        with cols[idx]:
            st.markdown(f"<div style='text-align: center; color: {color}; font-size: 1.2rem;'>{icon}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: {color};'>{status_text}</div>", unsafe_allow_html=True)

    # Progress stats summary
    evidence_count = len(state.get("evidence", []) or [])
    sections_count = len(state.get("sections", []) or [])
    images_count = len(state.get("image_specs", []) or [])

    st.caption(f"📊 Gathering context: `{evidence_count}` sources collected · `{sections_count}` sections written · `{images_count}` visual assets generated")


def render_article_metrics(out: Dict[str, Any]):
    """
    Renders clean article metrics banner above generated markdown.
    """
    final_md = out.get("final") or ""
    words = len(final_md.split()) if final_md else 0
    evidence = out.get("evidence") or []
    sources_count = len(evidence)
    sections = out.get("sections") or []
    sections_count = len(sections) if sections else (final_md.count("\n## ") or 1)
    specs = out.get("image_specs") or []
    visuals_count = len(specs)

    cols = st.columns(4)
    cols[0].metric("WORDS", f"{words:,}")
    cols[1].metric("SOURCES", str(sources_count))
    cols[2].metric("SECTIONS", str(sections_count))
    cols[3].metric("VISUALS", str(visuals_count))
