from __future__ import annotations

import sys
import time
import types
from datetime import date

import pytest

from src.agent import nodes as agent_nodes
from src.agent import tools as agent_tools
from src.agent.nodes import (
    _insert_placeholders_into_md,
    route_next,
    router_node,
    worker_node,
    merge_content,
    fanout,
)
from src.agent.rate_limiter import RateLimiter
from src.agent.schemas import Plan, Task, RouterDecision


def test_insert_placeholders_into_md_with_section_title():
    sample_md = (
        "# Title\n\n"
        "## Introduction\n"
        "Intro text paragraph.\n\n"
        "## WebSocket Basics\n"
        "WebSocket explanation paragraph.\n\n"
        "## Summary\n"
        "Conclusion paragraph."
    )
    image_specs = [
        {
            "placeholder": "[[IMAGE_1]]",
            "filename": "ws_flow.png",
            "alt": "WebSocket diagram",
            "caption": "Handshake flow",
            "prompt": "Diagram prompt",
            "section_title": "WebSocket Basics",
        }
    ]

    result = _insert_placeholders_into_md(sample_md, image_specs)
    assert "[[IMAGE_1]]" in result
    ws_idx = result.find("## WebSocket Basics")
    img_idx = result.find("[[IMAGE_1]]")
    sum_idx = result.find("## Summary")
    assert ws_idx < img_idx < sum_idx


def test_insert_placeholders_into_md_fallback_even_distribution():
    sample_md = (
        "# Title\n\n"
        "## Section One\n"
        "One text.\n\n"
        "## Section Two\n"
        "Two text.\n\n"
        "## Section Three\n"
        "Three text."
    )
    image_specs = [
        {
            "placeholder": "[[IMAGE_1]]",
            "filename": "img1.png",
            "alt": "Alt 1",
            "caption": "Cap 1",
            "prompt": "P1",
            "section_title": "Unknown Section",
        }
    ]

    result = _insert_placeholders_into_md(sample_md, image_specs)
    assert "[[IMAGE_1]]" in result


def test_route_next_picks_research_when_needed():
    assert route_next({"needs_research": True}) == "research"
    assert route_next({"needs_research": False}) == "orchestrator"


@pytest.mark.parametrize(
    "mode,expected_days",
    [("open_book", 7), ("hybrid", 45), ("closed_book", 3650)],
)
def test_router_node_maps_mode_to_recency(monkeypatch, mode, expected_days):
    """router_node should translate the LLM's mode into the right recency window."""
    decision = RouterDecision(
        needs_research=(mode != "closed_book"),
        mode=mode,
        reason="test",
        queries=["q1", "q2"],
    )
    monkeypatch.setattr(
        agent_nodes, "structured_with_retry", lambda schema, messages: decision
    )

    out = router_node({"topic": "Transformers", "as_of": "2026-08-23"})
    assert out["mode"] == mode
    assert out["needs_research"] == (mode != "closed_book")
    assert out["recency_days"] == expected_days
    assert out["queries"] == ["q1", "q2"]


def _make_plan(num_tasks: int = 3) -> Plan:
    tasks = [
        Task(
            id=i,
            title=f"Section {i}",
            goal="Explain something.",
            bullets=["a", "b", "c"],
            target_words=200,
        )
        for i in range(1, num_tasks + 1)
    ]
    return Plan(blog_title="My Blog", audience="devs", tone="technical", tasks=tasks)


def test_fanout_emits_one_send_per_task():
    plan = _make_plan(3)
    state = {
        "topic": "t",
        "mode": "closed_book",
        "as_of": "2026-08-23",
        "recency_days": 3650,
        "plan": plan,
        "evidence": [],
    }
    sends = fanout(state)
    assert len(sends) == 3
    assert all(getattr(s, "node", None) == "worker" for s in sends)
    task_ids = sorted(s.arg["task"]["id"] for s in sends)
    assert task_ids == [1, 2, 3]


def test_worker_node_returns_single_ordered_section(monkeypatch):
    plan = _make_plan(2)
    task = plan.tasks[1]  # id == 2
    payload = {
        "task": task.model_dump(),
        "topic": "Transformers",
        "mode": "closed_book",
        "as_of": "2026-08-23",
        "recency_days": 3650,
        "plan": plan.model_dump(),
        "evidence": [],
    }

    monkeypatch.setattr(agent_nodes, "get_llm", lambda: object())
    monkeypatch.setattr(
        agent_nodes,
        "invoke_with_retry",
        lambda *a, **k: types.SimpleNamespace(content="## Section 2\n\nBody text."),
    )
    monkeypatch.setattr(agent_nodes.groq_rate_limiter, "acquire", lambda: 0.0)

    out = worker_node(payload)
    assert out["sections"] == [(2, "## Section 2\n\nBody text.")]


def test_merge_content_orders_sections_by_id():
    plan = _make_plan(2)
    state = {
        "plan": plan,
        "sections": [(2, "## Second section"), (1, "## First section")],
    }
    out = merge_content(state)
    merged = out["merged_md"]
    assert merged.startswith("# My Blog\n\n")
    assert merged.index("## First section") < merged.index("## Second section")


def test_rate_limiter_spaces_consecutive_acquires():
    rl = RateLimiter(min_interval=0.05)
    start = time.monotonic()
    rl.acquire()
    rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045


def test_rate_limiter_zero_interval_is_noop():
    rl = RateLimiter(min_interval=0)
    assert rl.acquire() == 0.0


def test_get_llm_raises_clear_error_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    fake_st = types.SimpleNamespace(
        secrets=types.SimpleNamespace(get=lambda *a, **k: None)
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    with pytest.raises(RuntimeError, match="No LLM API key"):
        agent_nodes.get_llm()


def test_iso_to_date_parses_and_tolerates_bad_input():
    assert agent_tools.iso_to_date("2026-08-23") == date(2026, 8, 23)
    assert agent_tools.iso_to_date(None) is None
    assert agent_tools.iso_to_date("not-a-date") is None


def test_safe_slug():
    assert agent_tools.safe_slug("Hello, World!") == "hello_world"
    assert agent_tools.safe_slug("   ") == "blog"


def test_tavily_search_returns_empty_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert agent_tools.tavily_search("anything") == []


# ------------------------------------------------------------
# HITL (Human-in-the-Loop) Workflow Tests
# ------------------------------------------------------------

from langgraph.types import Command
from src.agent.graph import app as test_graph_app
from src.agent.schemas import ApprovalStatus


def test_hitl_pause_after_orchestrator(monkeypatch):
    """Verify that graph execution pauses at plan_approval node and does not run Workers automatically."""
    mock_plan = _make_plan(2)
    monkeypatch.setattr(
        agent_nodes,
        "structured_with_retry",
        lambda schema, messages: (
            RouterDecision(needs_research=False, mode="closed_book", reason="test")
            if schema == RouterDecision
            else mock_plan
        ),
    )

    config = {"configurable": {"thread_id": "test_hitl_pause_1"}}
    inputs = {
        "topic": "Retrieval-Augmented Generation",
        "as_of": "2026-08-23",
        "recency_days": 7,
        "sections": [],
        "evidence": [],
    }

    # Stream execution
    chunks = list(test_graph_app.stream(inputs, config=config, stream_mode="values"))
    snapshot = test_graph_app.get_state(config)

    assert snapshot.next == ("plan_approval",)
    assert snapshot.values.get("plan") is not None
    assert snapshot.values.get("plan").blog_title == "My Blog"
    assert snapshot.values.get("sections") == []  # Workers have NOT executed yet!


def test_hitl_approve_resumes_to_workers(monkeypatch):
    """Verify approving plan transitions workflow from HITL to Workers and Reducer."""
    mock_plan = _make_plan(2)

    def mock_structured(schema, messages):
        if schema == RouterDecision:
            return RouterDecision(needs_research=False, mode="closed_book", reason="test")
        elif schema == Plan:
            return mock_plan
        elif schema == agent_nodes.GlobalImagePlan:
            return agent_nodes.GlobalImagePlan(images=[])
        return mock_plan

    monkeypatch.setattr(agent_nodes, "structured_with_retry", mock_structured)
    monkeypatch.setattr(agent_nodes, "get_llm", lambda: object())
    monkeypatch.setattr(
        agent_nodes,
        "invoke_with_retry",
        lambda *a, **k: types.SimpleNamespace(content="## Section\n\nContent body."),
    )
    monkeypatch.setattr(agent_nodes.groq_rate_limiter, "acquire", lambda: 0.0)

    config = {"configurable": {"thread_id": "test_hitl_approve_1"}}
    inputs = {
        "topic": "Retrieval-Augmented Generation",
        "as_of": "2026-08-23",
        "recency_days": 7,
        "sections": [],
        "evidence": [],
    }

    # 1. Run until HITL pause
    list(test_graph_app.stream(inputs, config=config))
    snapshot1 = test_graph_app.get_state(config)
    assert snapshot1.next == ("plan_approval",)

    # 2. Resume with Approve
    list(test_graph_app.stream(Command(resume={"action": "approve"}), config=config))
    snapshot2 = test_graph_app.get_state(config)

    assert snapshot2.next == ()  # Finished cleanly!
    assert snapshot2.values.get("approval_status") == ApprovalStatus.APPROVED
    assert len(snapshot2.values.get("sections")) == 2
    assert "## Section" in snapshot2.values.get("merged_md")


def test_hitl_request_changes_revises_plan_and_pauses_again(monkeypatch):
    """Verify requesting changes returns control to Planner, increments version, and pauses again."""
    plan_v1 = _make_plan(2)
    plan_v1.blog_title = "Title V1"

    plan_v2 = _make_plan(3)
    plan_v2.blog_title = "Title V2 (Revised)"

    call_count = {"plan": 0}

    def mock_structured(schema, messages):
        if schema == RouterDecision:
            return RouterDecision(needs_research=False, mode="closed_book", reason="test")
        elif schema == Plan:
            call_count["plan"] += 1
            return plan_v1 if call_count["plan"] == 1 else plan_v2
        return plan_v1

    monkeypatch.setattr(agent_nodes, "structured_with_retry", mock_structured)

    config = {"configurable": {"thread_id": "test_hitl_change_req_1"}}
    inputs = {
        "topic": "Retrieval-Augmented Generation",
        "as_of": "2026-08-23",
        "recency_days": 7,
        "sections": [],
        "evidence": [],
    }

    # 1. Initial run -> pauses at V1
    list(test_graph_app.stream(inputs, config=config))
    snap1 = test_graph_app.get_state(config)
    assert snap1.next == ("plan_approval",)
    assert snap1.values.get("plan").blog_title == "Title V1"
    assert snap1.values.get("plan_version") == 1

    # 2. Request Changes -> returns to Orchestrator -> pauses at V2
    list(test_graph_app.stream(Command(resume={"action": "request_changes", "feedback": "Add Vector DB section"}), config=config))
    snap2 = test_graph_app.get_state(config)

    assert snap2.next == ("plan_approval",)
    assert snap2.values.get("plan").blog_title == "Title V2 (Revised)"
    assert snap2.values.get("plan_version") == 2
    assert snap2.values.get("sections") == []  # Workers STILL have not executed!


def test_hitl_multiple_revisions_and_approved_plan_to_workers(monkeypatch):
    """Verify multiple plan revisions (v1 -> v2 -> v3) and ensure only v3 reaches workers."""
    plan_v1 = _make_plan(2)
    plan_v1.blog_title = "Plan V1"

    plan_v2 = _make_plan(2)
    plan_v2.blog_title = "Plan V2"

    plan_v3 = _make_plan(2)
    plan_v3.blog_title = "Plan V3 Approved"

    plans = [plan_v1, plan_v2, plan_v3]
    call_idx = {"idx": 0}

    def mock_structured(schema, messages):
        if schema == RouterDecision:
            return RouterDecision(needs_research=False, mode="closed_book", reason="test")
        elif schema == Plan:
            curr = plans[min(call_idx["idx"], 2)]
            call_idx["idx"] += 1
            return curr
        elif schema == agent_nodes.GlobalImagePlan:
            return agent_nodes.GlobalImagePlan(images=[])
        return plan_v1

    worker_received_plans = []

    def mock_worker(payload):
        worker_received_plans.append(payload["plan"]["blog_title"])
        return {"sections": [(payload["task"]["id"], "## Section\nText")]}

    monkeypatch.setattr(agent_nodes, "structured_with_retry", mock_structured)
    monkeypatch.setattr(agent_nodes, "worker_node", mock_worker)

    config = {"configurable": {"thread_id": "test_hitl_multi_rev_1"}}
    inputs = {
        "topic": "Retrieval-Augmented Generation",
        "as_of": "2026-08-23",
        "recency_days": 7,
        "sections": [],
        "evidence": [],
    }

    # Initial -> V1
    list(test_graph_app.stream(inputs, config=config))
    # Revision 1 -> V2
    list(test_graph_app.stream(Command(resume={"action": "request_changes", "feedback": "Feedback 1"}), config=config))
    # Revision 2 -> V3
    list(test_graph_app.stream(Command(resume={"action": "request_changes", "feedback": "Feedback 2"}), config=config))

    snap_v3 = test_graph_app.get_state(config)
    assert snap_v3.values.get("plan").blog_title == "Plan V3 Approved"
    assert snap_v3.values.get("plan_version") == 3

    # Approve V3
    list(test_graph_app.stream(Command(resume={"action": "approve"}), config=config))

    # Workers must receive ONLY Plan V3 Approved
    assert len(worker_received_plans) == 2
    assert all(t == "Plan V3 Approved" for t in worker_received_plans)

