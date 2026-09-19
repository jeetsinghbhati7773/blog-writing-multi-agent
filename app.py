from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

import streamlit as st

from src.agent.graph import app as graph_app
from src.ui.theme import apply_theme
import importlib

try:
    from src.ui.components import (
        render_studio_header,
        render_content_brief,
        render_workflow_progress,
        render_system_status_sidebar,
    )
except ImportError:
    import src.ui.components as comp_mod
    importlib.reload(comp_mod)
    from src.ui.components import (
        render_studio_header,
        render_content_brief,
        render_workflow_progress,
        render_system_status_sidebar,
    )

try:
    from src.ui.helpers import (
        try_stream,
        extract_latest_state,
        list_past_blogs,
        read_md_file,
        extract_title_from_md,
    )
except ImportError:
    import src.ui.helpers as helpers_mod
    importlib.reload(helpers_mod)
    from src.ui.helpers import (
        try_stream,
        extract_latest_state,
        list_past_blogs,
        read_md_file,
        extract_title_from_md,
    )

try:
    from src.ui.views import (
        render_article_workspace,
        render_library_workspace,
        render_knowledge_base_workspace,
    )
except ImportError:
    import src.ui.views as views_mod
    importlib.reload(views_mod)
    from src.ui.views import (
        render_article_workspace,
        render_library_workspace,
        render_knowledge_base_workspace,
    )

try:
    from src.ui.chat_views import (
        render_unified_chat_view,
    )
except ImportError:
    import src.ui.chat_views as chat_views_mod
    importlib.reload(chat_views_mod)
    from src.ui.chat_views import (
        render_unified_chat_view,
    )

# -----------------------------
# 1. Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Content Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply sleek dark design tokens and custom CSS rules
apply_theme()

# -----------------------------
# 2. Studio Shell Navigation (Callback Architecture)
# -----------------------------
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "📝 Article Generator"


def cb_nav_radio_change():
    st.session_state["app_mode"] = st.session_state["nav_radio_widget"]


def cb_create_new_article():
    st.session_state["last_out"] = None
    st.session_state["history_mode"] = False
    st.session_state["selected_past_article"] = None
    st.session_state["app_mode"] = "📝 Article Generator"


def cb_select_past_article(filename: str, md_text: str):
    st.session_state["selected_past_article"] = filename
    st.session_state["last_out"] = {
        "plan": None,
        "evidence": [],
        "image_specs": [],
        "final": md_text,
    }
    st.session_state["history_mode"] = True
    st.session_state["loaded_blog_name"] = filename
    st.session_state["app_mode"] = "📝 Article Generator"


def cb_new_conversation():
    st.session_state["unified_messages"] = []


def cb_clear_chat_history():
    st.session_state["unified_messages"] = []


with st.sidebar:
    st.markdown("## **🚀 AI Content Studio**")

    modes_list = [
        "📝 Article Generator",
        "💬 Document & AI Chat",
        "🗂 Knowledge Base",
    ]

    if st.session_state["app_mode"] not in modes_list:
        st.session_state["app_mode"] = "📝 Article Generator"
    curr_index = modes_list.index(st.session_state["app_mode"])

    st.radio(
        "Navigation",
        modes_list,
        index=curr_index,
        key="nav_radio_widget",
        on_change=cb_nav_radio_change,
    )

    app_mode = st.session_state["app_mode"]
    st.divider()
    
    # -----------------------------
    # Sidebar Section based on Sketches
    # -----------------------------
    if app_mode == "📝 Article Generator":
        # Action button (Sketch 1)
        st.button(
            "✨ Create a new Article",
            use_container_width=True,
            type="primary",
            key="sb_create_new_btn",
            on_click=cb_create_new_article,
        )

        st.markdown("#### **📜 Past / History**")
        past_files = list_past_blogs()
        if not past_files:
            st.caption("No saved articles found.")
        else:
            with st.container(height=260):
                for idx, p in enumerate(past_files):
                    num = len(past_files) - idx
                    try:
                        md_text = read_md_file(p)
                        title = extract_title_from_md(md_text, p.stem)
                    except Exception:
                        md_text = ""
                        title = p.stem
                    label = f"{num}. {title[:24]}…" if len(title) > 24 else f"{num}. {title}"
                    st.button(
                        label,
                        key=f"sb_hist_{p.name}",
                        use_container_width=True,
                        on_click=cb_select_past_article,
                        args=(p.name, md_text),
                    )

    elif app_mode == "💬 Document & AI Chat":
        # Action button (Sketch 2)
        st.button(
            "💬 Start a new conversation",
            use_container_width=True,
            type="primary",
            key="sb_new_chat_btn",
            on_click=cb_new_conversation,
        )

        st.markdown("#### **📜 History**")
        messages = st.session_state.get("unified_messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            st.caption("No active conversation turns.")
        else:
            with st.container(height=240):
                for idx, m in enumerate(user_msgs):
                    cnt = m["content"]
                    lbl = f"{idx + 1}. {cnt[:24]}…" if len(cnt) > 24 else f"{idx + 1}. {cnt}"
                    st.caption(lbl)

        st.button(
            "🗑️ Clear all past chat",
            use_container_width=True,
            key="sb_clear_chat_btn",
            on_click=cb_clear_chat_history,
        )

    from src.ui.components import render_supabase_auth_and_chat_sidebar
    render_supabase_auth_and_chat_sidebar()

    # System Status at the bottom of the sidebar as sketched in both wireframes
    render_system_status_sidebar()

# -----------------------------
# 3. Mode Router Execution (Enforce Global Auth Gate)
# -----------------------------
from src.db import GUEST_USER_ID

is_logged_in = st.session_state.get("is_logged_in", False) and st.session_state.get("user_id") != GUEST_USER_ID

if not is_logged_in:
    render_studio_header(
        title="Welcome to AI Content Studio",
        subtitle="Please Log In or Sign Up in the sidebar to access Article Generation, AI Chat, and Knowledge Base.",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.warning("🔒 **Authentication Required**: Access to AI Content Studio features is locked.")
        st.info(
            "🔑 **Getting Started**:\n\n"
            "1. Enter your email and password in the **User Account** form in the sidebar on the left.\n"
            "2. Click **Login** (or **Sign Up** to create a new account).\n"
            "3. Once authenticated, your personal workspace, articles, expenses, and chat history will be unlocked!"
        )
    st.stop()

if app_mode in ("💬 Document & AI Chat", "💬 AI Chat", "📚 Document Chat"):
    render_studio_header(
        title="Document Intelligence Chat",
        subtitle="Conversational RAG assistant grounded in your uploaded knowledge base.",
    )
    render_unified_chat_view()

elif app_mode == "🗂 Knowledge Base":
    render_studio_header(
        title="Knowledge Base Storage",
        subtitle="Manage vector database collections, documents, and chunk memory.",
    )
    render_knowledge_base_workspace()

else:
    # Mode 1: Article Generator / Viewer
    if "last_out" not in st.session_state:
        st.session_state["last_out"] = None
    if "history_mode" not in st.session_state:
        st.session_state["history_mode"] = False

    is_history = st.session_state.get("history_mode", False)
    out = st.session_state.get("last_out")
    loaded_name = st.session_state.get("loaded_blog_name", "")

    if is_history and out:
        # Dedicated Past Article Viewer Mode
        render_studio_header(
            title="Past Article Viewer",
            subtitle=f"Viewing saved technical article from archive: {loaded_name}",
        )
        render_article_workspace(out, is_history=True, blog_name=loaded_name)
    else:
        # New Article Generator Mode
        render_studio_header()

        logs: List[str] = []
        def log(msg: str):
            logs.append(msg)

        # Render Content Brief Editor
        run_btn, brief = render_content_brief()

        # Trigger Generation Workflow
        if run_btn:
            if not brief["topic_raw"]:
                st.warning("Please enter a valid article topic in the Content Brief above.")
                st.stop()

            st.session_state["history_mode"] = False

            inputs: Dict[str, Any] = {
                "topic": brief["full_prompt"],
                "mode": "",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "as_of": brief["as_of"].isoformat() if isinstance(brief["as_of"], date) else brief["as_of"],
                "recency_days": 7,
                "use_uploaded_docs": brief["use_uploaded_docs"],
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": "",
            }

            status = st.status("🚀 Running Autonomous Multi-Agent Pipeline...", expanded=True)
            progress_area = st.empty()

            current_state: Dict[str, Any] = {"_completed_nodes": []}
            last_node = None

            for kind, payload in try_stream(graph_app, inputs):
                if kind in ("updates", "values"):
                    node_name = None
                    if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                        node_name = next(iter(payload.keys()))
                    if node_name and node_name != last_node:
                        status.write(f"➡️ Active Agent Node: `{node_name}`")
                        if last_node and last_node not in current_state["_completed_nodes"]:
                            current_state["_completed_nodes"].append(last_node)
                        last_node = node_name

                    current_state = extract_latest_state(current_state, payload)

                    with progress_area.container():
                        render_workflow_progress(last_node or "", current_state)

                    log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

                elif kind == "final":
                    out = payload
                    st.session_state["last_out"] = out
                    status.update(label="✅ Article Generation Complete!", state="complete", expanded=False)
                    log("[final] Pipeline execution successfully finished.")

        # Render Newly Generated Results Workspace
        out = st.session_state.get("last_out")
        if out:
            st.divider()
            render_article_workspace(out, is_history=False)
