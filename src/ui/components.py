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


def render_supabase_auth_and_chat_sidebar():
    """
    Renders Supabase User Authentication controls and Chat Sessions Drawer in Sidebar.
    """
    from src.db import login_user, signup_user, create_session, list_user_sessions, GUEST_USER_ID, GUEST_USER_EMAIL

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = GUEST_USER_ID
        st.session_state["user_email"] = GUEST_USER_EMAIL
        st.session_state["is_logged_in"] = False

    user_id = st.session_state["user_id"]
    user_email = st.session_state.get("user_email", GUEST_USER_EMAIL)
    is_logged_in = st.session_state.get("is_logged_in", False) and user_id != GUEST_USER_ID

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 **User Account**")

    if not is_logged_in:
        st.sidebar.warning("🔒 **Login or Sign Up required to start chat.**")
        with st.sidebar.expander("🔑 Login / Sign Up Form", expanded=True):
            auth_email = st.text_input("Email", key="sidebar_auth_email")
            auth_password = st.text_input("Password", type="password", key="sidebar_auth_password")
            
            c_login, c_signup = st.columns(2)
            with c_login:
                if st.button("Login", key="sidebar_login_btn", use_container_width=True):
                    res = login_user(auth_email, auth_password)
                    if res.get("success"):
                        st.session_state["user_id"] = res["user_id"]
                        st.session_state["user_email"] = res["email"]
                        st.session_state["is_logged_in"] = True
                        st.session_state["active_session_id"] = None
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error(res.get("error", "Login failed"))
            with c_signup:
                if st.button("Sign Up", key="sidebar_signup_btn", use_container_width=True):
                    res = signup_user(auth_email, auth_password)
                    if res.get("success"):
                        st.session_state["user_id"] = res["user_id"]
                        st.session_state["user_email"] = res["email"]
                        st.session_state["is_logged_in"] = True
                        st.session_state["active_session_id"] = None
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.error(res.get("error", "Signup failed"))
    else:
        st.sidebar.caption(f"👤 Logged in as: `{user_email}`")
        if st.sidebar.button("🚪 Logout", key="sidebar_logout_btn", use_container_width=True):
            st.session_state["user_id"] = GUEST_USER_ID
            st.session_state["user_email"] = GUEST_USER_EMAIL
            st.session_state["is_logged_in"] = False
            st.session_state["active_session_id"] = None
            st.session_state["unified_messages"] = []
            st.rerun()

    # -----------------------------
    # Chat Sessions Drawer (Multi-Chat Support)
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 **Conversations**")

    if not is_logged_in:
        st.sidebar.caption("🔒 *Log in to view or start saved chat conversations.*")
        return

    if st.sidebar.button("➕ **New Chat**", key="btn_new_chat", use_container_width=True, type="primary"):
        new_sess = create_session(user_id, title="New Conversation")
        if new_sess:
            st.session_state["active_session_id"] = new_sess["id"]
            st.session_state["unified_messages"] = []
            st.rerun()

    # Fetch user sessions
    user_sessions = list_user_sessions(user_id)
    if not user_sessions:
        st.sidebar.caption("No past conversations.")
        if "active_session_id" not in st.session_state or not st.session_state["active_session_id"]:
            st.session_state["active_session_id"] = "default_session"
    else:
        if "active_session_id" not in st.session_state or not st.session_state["active_session_id"]:
            st.session_state["active_session_id"] = user_sessions[0]["id"]

        session_map = {s["id"]: f"💬 {s.get('title', 'Chat')} ({s.get('created_at', '')[:10]})" for s in user_sessions}
        curr_active = st.session_state.get("active_session_id")
        
        selected_sess_id = st.sidebar.radio(
            "Select Session",
            options=list(session_map.keys()),
            format_func=lambda x: session_map.get(x, x),
            index=list(session_map.keys()).index(curr_active) if curr_active in session_map else 0,
            key="sb_session_radio"
        )
        if selected_sess_id != curr_active:
            st.session_state["active_session_id"] = selected_sess_id
            st.session_state["unified_messages"] = []
            st.rerun()


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
    # These keys MUST match the actual LangGraph node names emitted while streaming
    # (see src/agent/graph.py): router -> research -> orchestrator -> worker -> reducer.
    steps = [
        ("router", "Route"),
        ("research", "Research"),
        ("orchestrator", "Plan"),
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
