from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.agent.graph import app as graph_app
from src.ui.helpers import (
    try_stream,
    extract_latest_state,
    list_past_blogs,
    read_md_file,
    extract_title_from_md,
)
from src.ui.views import (
    render_plan_tab,
    render_evidence_tab,
    render_preview_tab,
    render_images_tab,
    render_logs_tab,
)

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Blog Writer Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom Styling / CSS Token Injection
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stSidebarNav"] {
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# App Header
# -----------------------------
st.title("✍️ AI Blog Writer Agent")
st.caption("Autonomous multi-agent technical blog writer powered by LangGraph, Tavily Search, and Gemini Imagen.")

# -----------------------------
# Sidebar Setup
# -----------------------------
with st.sidebar:
    st.header("⚡ Agent Configuration")

    topic = st.text_area(
        "Blog Topic / Prompt",
        placeholder="e.g. Architecting High-Performance RAG Pipelines with Vector Search",
        height=120,
    )
    as_of = st.date_input("As-of Target Date", value=date.today())

    run_btn = st.button("🚀 Generate Technical Blog", type="primary", use_container_width=True)

    # API Status Indicators
    st.divider()
    st.subheader("🔑 API Key Status")
    openai_set = bool(os.getenv("OPENAI_API_KEY"))
    tavily_set = bool(os.getenv("TAVILY_API_KEY"))
    google_set = bool(os.getenv("GOOGLE_API_KEY"))

    st.markdown(f"- OpenAI API: {'✅ Configured' if openai_set else '⚠️ Missing'}")
    st.markdown(f"- Tavily Search: {'✅ Configured' if tavily_set else 'ℹ️ Disabled'}")
    st.markdown(f"- Gemini Images: {'✅ Configured' if google_set else 'ℹ️ Disabled'}")

    # History / Past Blogs Section
    st.divider()
    st.subheader("📂 Saved Blog History")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in output folder).")
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title[:35]}… ({p.name})" if len(title) > 35 else f"{title} ({p.name})"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox(
            "Select a blog to load",
            options=options,
        )
        if st.button("📖 Load Selected Blog", use_container_width=True):
            selected_file = file_by_label.get(selected_label)
            if selected_file:
                md_text = read_md_file(selected_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.toast(f"Loaded {selected_file.name}", icon="✅")

# -----------------------------
# Session State Initialization
# -----------------------------
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Main Tabs Setup
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Research Evidence", "📝 Markdown Preview", "🖼️ Diagrams & Images", "🧾 Logs"]
)

logs: List[str] = []

def log(msg: str):
    logs.append(msg)

# -----------------------------
# Execution Flow Trigger
# -----------------------------
if run_btn:
    if not topic.strip():
        st.warning("Please enter a valid blog topic in the sidebar.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Executing Multi-Agent Workflow…", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(graph_app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Executing Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                "evidence_count": len(current_state.get("evidence", []) or []),
                "tasks_planned": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "sections_completed": len(current_state.get("sections", []) or []),
                "images_planned": len(current_state.get("image_specs", []) or []),
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Blog Generation Complete!", state="complete", expanded=False)
            log("[final] Agent execution successfully finished.")

# -----------------------------
# Render Results in Tabs
# -----------------------------
out = st.session_state.get("last_out")
if out:
    with tab_plan:
        render_plan_tab(out)
    with tab_evidence:
        render_evidence_tab(out)
    with tab_preview:
        render_preview_tab(out)
    with tab_images:
        render_images_tab(out)
    with tab_logs:
        render_logs_tab(logs)
else:
    with tab_preview:
        st.info("👈 Enter a topic in the sidebar and click **Generate Technical Blog** to start.")
