import json
import uuid
from datetime import date
from typing import Any, Dict, List

import streamlit as st
from langgraph.types import Command

from src.agent.graph import app as graph_app
from src.ui.theme import apply_theme
import importlib

try:
    from src.ui.components import (
        render_studio_header,
        render_content_brief,
        render_workflow_progress,
        render_system_status_sidebar,
        render_plan_approval_card,
    )
except ImportError:
    import src.ui.components as comp_mod
    importlib.reload(comp_mod)
    from src.ui.components import (
        render_studio_header,
        render_content_brief,
        render_workflow_progress,
        render_system_status_sidebar,
        render_plan_approval_card,
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
    )
except ImportError:
    import src.ui.views as views_mod
    importlib.reload(views_mod)
    from src.ui.views import (
        render_article_workspace,
        render_library_workspace,
    )

# -----------------------------
# 1. Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply sleek dark design tokens and custom CSS rules
apply_theme()

# -----------------------------
# 2. Session & Navigation Initialization
# -----------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
if "active_interrupt" not in st.session_state:
    st.session_state["active_interrupt"] = None
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "history_mode" not in st.session_state:
    st.session_state["history_mode"] = False
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "📝 Article Generator"


def cb_nav_radio_change():
    st.session_state["app_mode"] = st.session_state["nav_radio_widget"]


def cb_create_new_article():
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["active_interrupt"] = None
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
    st.session_state["active_interrupt"] = None
    st.session_state["loaded_blog_name"] = filename
    st.session_state["app_mode"] = "📝 Article Generator"


with st.sidebar:
    st.markdown("## **✍️ AI Blog Writing Agent**")

    modes_list = [
        "📝 Article Generator",
        "📜 Past Articles Archive",
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

    if app_mode == "📝 Article Generator":
        st.button(
            "✨ Create New Article",
            use_container_width=True,
            type="primary",
            key="sb_create_new_btn",
            on_click=cb_create_new_article,
        )

        st.markdown("#### **📜 Recent Articles**")
        past_files = list_past_blogs()
        if not past_files:
            st.caption("No saved articles found.")
        else:
            with st.container(height=280):
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

    # System Status at the bottom of the sidebar
    render_system_status_sidebar()

# -----------------------------
# 3. Main Workspace Router
# -----------------------------
if app_mode == "📜 Past Articles Archive":
    render_studio_header(
        title="Past Articles Archive",
        subtitle="Browse and read previously generated research articles",
    )
    render_library_workspace()

else:
    # Article Generator Mode
    is_history = st.session_state.get("history_mode", False)
    out = st.session_state.get("last_out")
    loaded_name = st.session_state.get("loaded_blog_name", "")

    if is_history and out:
        render_studio_header(
            title="Past Article Viewer",
            subtitle=f"Viewing saved technical article from archive: {loaded_name}",
        )
        render_article_workspace(out, is_history=True, blog_name=loaded_name)
    else:
        render_studio_header(
            title="AI Blog Writing Agent",
            subtitle="Autonomous Multi-Agent Content Generation with LangGraph & Human-in-the-Loop",
        )

        logs: List[str] = []
        def log(msg: str):
            logs.append(msg)

        # Helper to execute/resume graph
        def execute_graph(stream_input: Any):
            thread_id = st.session_state.get("thread_id") or str(uuid.uuid4())
            st.session_state["thread_id"] = thread_id
            config = {"configurable": {"thread_id": thread_id}}

            status = st.status("🚀 Running Autonomous Multi-Agent Pipeline...", expanded=True)
            progress_area = st.empty()

            current_state: Dict[str, Any] = {"_completed_nodes": []}
            last_node = None

            for kind, payload in try_stream(graph_app, stream_input, config=config):
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

            # Inspect checkpointer snapshot to handle HITL pause vs pipeline completion
            snapshot = graph_app.get_state(config)
            if snapshot.next and ("plan_approval" in snapshot.next or (snapshot.tasks and any(t.interrupts for t in snapshot.tasks))):
                st.session_state["active_interrupt"] = snapshot.tasks[0].interrupts[0].value if snapshot.tasks and snapshot.tasks[0].interrupts else snapshot.values
                st.session_state["last_out"] = None
                status.update(label="⏸️ Execution Paused: Human Plan Approval Required", state="running", expanded=False)
            else:
                st.session_state["active_interrupt"] = None
                st.session_state["last_out"] = snapshot.values
                status.update(label="✅ Article Generation Complete!", state="complete", expanded=False)

        # Render Content Brief Editor if not actively awaiting HITL approval or viewing generated article
        active_interrupt = st.session_state.get("active_interrupt")
        out = st.session_state.get("last_out")

        if not active_interrupt and not out:
            run_btn, brief = render_content_brief()

            if run_btn:
                if not brief["topic_raw"]:
                    st.warning("Please enter a valid article topic in the Content Brief above.")
                    st.stop()

                st.session_state["thread_id"] = str(uuid.uuid4())
                st.session_state["active_interrupt"] = None
                st.session_state["last_out"] = None
                st.session_state["history_mode"] = False

                inputs: Dict[str, Any] = {
                    "topic": brief["full_prompt"],
                    "audience": brief.get("audience"),
                    "tone": brief.get("tone"),
                    "mode": "",
                    "needs_research": False,
                    "queries": [],
                    "evidence": [],
                    "plan": None,
                    "plan_version": 1,
                    "approval_status": "pending",
                    "human_approval": None,
                    "human_feedback": None,
                    "as_of": brief["as_of"].isoformat() if isinstance(brief["as_of"], date) else brief["as_of"],
                    "recency_days": 7,
                    "sections": [],
                    "merged_md": "",
                    "md_with_placeholders": "",
                    "image_specs": [],
                    "final": "",
                    "errors": [],
                    "retry_count": 0,
                }
                execute_graph(inputs)
                st.rerun()

        # Render Human-in-the-Loop Plan Approval Interface
        active_interrupt = st.session_state.get("active_interrupt")
        if active_interrupt and not out:
            action, feedback = render_plan_approval_card(active_interrupt)
            if action:
                cmd = Command(resume={"action": action, "feedback": feedback})
                st.session_state["active_interrupt"] = None
                execute_graph(cmd)
                st.rerun()

        # Render Newly Generated Results Workspace
        out = st.session_state.get("last_out")
        if out:
            st.divider()
            render_article_workspace(out, is_history=False)

