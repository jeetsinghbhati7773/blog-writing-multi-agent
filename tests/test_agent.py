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
import src.mcp_client as mcp_client


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
    # Should be placed under WebSocket Basics
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


# ---------------------------------------------------------------------------
# Router / routing
# ---------------------------------------------------------------------------

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
    # Mock the structured LLM call so no network / API key is needed.
    monkeypatch.setattr(
        agent_nodes, "structured_with_retry", lambda schema, messages: decision
    )

    out = router_node({"topic": "Transformers", "as_of": "2026-08-23"})
    assert out["mode"] == mode
    assert out["needs_research"] == (mode != "closed_book")
    assert out["recency_days"] == expected_days
    assert out["queries"] == ["q1", "q2"]


# ---------------------------------------------------------------------------
# Orchestrator fanout + worker + reducer merge
# ---------------------------------------------------------------------------

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
    # Every Send targets the worker node...
    assert all(getattr(s, "node", None) == "worker" for s in sends)
    # ...and carries exactly one task each.
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

    # Mock the LLM + rate limiter so the test is fast, offline, and key-free.
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
        # Deliberately out of order to prove sorting by task id.
        "sections": [(2, "## Second section"), (1, "## First section")],
    }
    out = merge_content(state)
    merged = out["merged_md"]
    assert merged.startswith("# My Blog\n\n")
    assert merged.index("## First section") < merged.index("## Second section")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_spaces_consecutive_acquires():
    rl = RateLimiter(min_interval=0.05)
    start = time.monotonic()
    rl.acquire()  # first slot is immediate
    rl.acquire()  # second slot must wait ~min_interval
    elapsed = time.monotonic() - start
    assert elapsed >= 0.045  # allow a little scheduling slack


def test_rate_limiter_zero_interval_is_noop():
    rl = RateLimiter(min_interval=0)
    assert rl.acquire() == 0.0


# ---------------------------------------------------------------------------
# get_llm clean error
# ---------------------------------------------------------------------------

def test_get_llm_raises_clear_error_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Provide a fake streamlit whose secrets never yield a key, so the result is
    # deterministic regardless of any local .streamlit/secrets.toml.
    fake_st = types.SimpleNamespace(
        secrets=types.SimpleNamespace(get=lambda *a, **k: None)
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    with pytest.raises(RuntimeError, match="No LLM API key"):
        agent_nodes.get_llm()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MCP client / stock tool
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_get_stock_price_reports_rate_limit_note(monkeypatch):
    monkeypatch.setattr(
        mcp_client.requests,
        "get",
        lambda url, timeout=10: _FakeResp({"Note": "Thank you for using Alpha Vantage!"}),
    )
    result = mcp_client.get_stock_price.invoke({"symbol": "IBM"})
    assert "error" in result
    assert "raw" in result


def test_get_stock_price_uses_demo_key_when_unset(monkeypatch):
    captured = {}

    def fake_get(url, timeout=10):
        captured["url"] = url
        return _FakeResp({"Global Quote": {"05. price": "123.45"}})

    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(mcp_client.requests, "get", fake_get)

    result = mcp_client.get_stock_price.invoke({"symbol": "IBM"})
    assert "apikey=demo" in captured["url"]
    assert result.get("Global Quote", {}).get("05. price") == "123.45"
