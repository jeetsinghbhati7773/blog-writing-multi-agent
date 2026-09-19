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


def render_system_status_popover():
    """
    Renders API system status in a compact popover drawer.
    """
    render_system_status_sidebar()


def render_studio_header(title: str = "AI Blog Writing Agent", subtitle: str = "Generate research-backed technical content with multi-agent intelligence"):
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
            placeholder="e.g. How Retrieval-Augmented Generation Works under the hood",
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

        col_date, col_inst = st.columns([1, 1])
        with col_date:
            as_of = st.date_input("As-of Target Date", value=date.today(), key="brief_as_of")
        with col_inst:
            keywords = st.text_input(
                "Keywords (Optional)",
                placeholder="e.g. RAG, embeddings, vector database, retrieval",
                key="brief_keywords",
            )

        instructions = st.text_area(
            "Additional Instructions (Optional)",
            placeholder="e.g. Include practical code snippet or system diagram.",
            height=60,
            key="brief_instructions",
        )

        # Construct comprehensive prompt internally
        full_topic_prompt = (
            f"Topic: {topic.strip()}\n"
            f"Target Audience: {audience}\n"
            f"Tone: {tone}\n"
            f"Format: {format_kind}\n"
            f"Target Length: {word_count}\n"
            f"Keywords: {keywords.strip()}\n"
            f"Special Instructions: {instructions.strip()}"
        )

        run_btn = st.button("🚀 Generate Technical Article", type="primary", use_container_width=True, key="brief_submit_btn")

        brief_data = {
            "topic_raw": topic.strip(),
            "full_prompt": full_topic_prompt,
            "audience": audience,
            "tone": tone,
            "format": format_kind,
            "word_count": word_count,
            "keywords": keywords.strip(),
            "instructions": instructions.strip(),
            "as_of": as_of,
        }

        return run_btn, brief_data


def render_workflow_progress(current_node: str, state: Dict[str, Any]):
    """
    Renders visual workflow pipeline progress indicator during graph execution.
    """
    steps = [
        ("router", "Route"),
        ("research", "Research"),
        ("orchestrator", "Plan"),
        ("plan_approval", "🧑 Human Review"),
        ("worker", "Writing"),
        ("reducer", "Assemble"),
    ]

    completed_nodes = state.get("_completed_nodes", [])
    step_keys = [s[0] for s in steps]
    current_idx = step_keys.index(current_node) if current_node in step_keys else -1

    st.markdown("#### **WORKFLOW PIPELINE**")

    cols = st.columns(len(steps))
    for idx, (node_key, label) in enumerate(steps):
        is_current = node_key == current_node
        is_done = (node_key in completed_nodes) or (current_idx != -1 and idx < current_idx)

        if is_current:
            icon = "●"
            color = "#FF4B4B"
            status_text = f"**{label}** (In Progress)"
        elif is_done:
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

    evidence_count = len(state.get("evidence", []) or [])
    sections_count = len(state.get("sections", []) or [])
    images_count = len(state.get("image_specs", []) or [])

    st.caption(f"📊 Gathering context: `{evidence_count}` sources collected · `{sections_count}` sections written · `{images_count}` visual assets generated")


def render_plan_approval_card(interrupt_value: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Renders Human-in-the-Loop (HITL) Plan Approval UI card.
    Returns (action, feedback) when user acts, else (None, None).
    """
    st.warning("### 🧑 **PLAN APPROVAL REQUIRED**", icon="✋")
    st.info("The multi-agent pipeline has generated the article plan. Please review and approve the outline before section generation begins.")

    plan_data = interrupt_value.get("plan", {})
    if hasattr(plan_data, "model_dump"):
        plan_dict = plan_data.model_dump()
    elif isinstance(plan_data, dict):
        plan_dict = plan_data
    else:
        plan_dict = {}

    plan_version = interrupt_value.get("plan_version", 1)
    topic = interrupt_value.get("topic", "N/A")
    evidence_items = interrupt_value.get("evidence", [])

    st.markdown(f"#### 📄 **Article Plan (Version {plan_version})**")
    st.markdown(f"**Title:** `{plan_dict.get('blog_title', 'Untitled')}`")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Target Audience", str(plan_dict.get("audience", "General")))
    mcol2.metric("Content Tone", str(plan_dict.get("tone", "Technical")))
    mcol3.metric("Format", str(plan_dict.get("blog_kind", "explainer")).upper())
    mcol4.metric("Plan Version", f"v{plan_version}")

    tasks = plan_dict.get("tasks", [])
    st.markdown("#### **Proposed Section Outline**")
    if tasks:
        for t in tasks:
            t_id = t.get("id", "")
            t_title = t.get("title", "")
            t_goal = t.get("goal", "")
            t_words = t.get("target_words", 0)
            t_bullets = t.get("bullets", [])
            t_research = "Yes" if t.get("requires_research") else "No"
            t_code = "Yes" if t.get("requires_code") else "No"
            t_tags = ", ".join(t.get("tags") or [])

            with st.expander(f"📌 **Section {t_id}: {t_title}** (~{t_words} words)", expanded=True):
                st.markdown(f"**Goal:** {t_goal}")
                st.markdown("**Key Subsections & Talking Points:**")
                for b in t_bullets:
                    st.markdown(f"- {b}")
                st.caption(f"Tags: `{t_tags}` | Needs Research: `{t_research}` | Code Snippet: `{t_code}`")

    if evidence_items:
        st.markdown("#### **Gathered Research Sources**")
        with st.expander("📚 View Gathered Research Sources", expanded=False):
            for e in evidence_items:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                st.markdown(f"- [{e.get('title')}]({e.get('url')})")

    st.divider()
    st.markdown("#### **Human Review Action**")

    if "show_change_request_form" not in st.session_state:
        st.session_state["show_change_request_form"] = False

    bcol1, bcol2, bcol3 = st.columns(3)

    action_selected = None
    feedback_selected = None

    with bcol1:
        if st.button("✅ Approve Plan", type="primary", use_container_width=True, key="hitl_approve_btn"):
            action_selected = "approve"
            st.session_state["show_change_request_form"] = False

    with bcol2:
        if st.button("✏️ Request Changes", type="secondary", use_container_width=True, key="hitl_request_changes_btn"):
            st.session_state["show_change_request_form"] = not st.session_state["show_change_request_form"]

    with bcol3:
        if st.button("🔄 Regenerate Plan", type="secondary", use_container_width=True, key="hitl_regenerate_btn"):
            action_selected = "regenerate"
            st.session_state["show_change_request_form"] = False

    if st.session_state.get("show_change_request_form"):
        st.markdown("---")
        st.markdown("##### ✏️ **What would you like to change in the plan?**")
        st.caption("Examples: *'Add a section explaining vector databases'*, *'Remove limitations section'*, *'Make article beginner friendly'*")
        feedback_input = st.text_area(
            "Change Instructions",
            placeholder="Type your feedback/modifications for the Planner agent...",
            key="hitl_feedback_text",
        )
        if st.button("📨 Submit Plan Changes", type="primary", key="hitl_submit_changes_btn"):
            if not feedback_input.strip():
                st.warning("Please provide modification instructions before submitting.")
            else:
                action_selected = "request_changes"
                feedback_selected = feedback_input.strip()
                st.session_state["show_change_request_form"] = False

    return action_selected, feedback_selected


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

