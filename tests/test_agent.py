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
from src.agent.schemas import (
    Plan,
    Task,
    RouterDecision,
    EvidenceItem,
    SourceFilterScore,
    SourceFilterResult,
    ContradictionItem,
    ContradictionReport,
    ResearchResult,
    OutlineValidationResult,
    ClaimVerification,
    FactCheckReport,
    FactCheckItem,
    SEOPlan,
    SEOFAQItem,
    ApprovalStatus,
)


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
        elif schema == agent_nodes.CriticEvaluation:
            return agent_nodes.CriticEvaluation(score=8.5, passed=True, summary="Good draft", issues=[])
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

    # 1. Run until HITL #1 pause (Plan Approval)
    list(test_graph_app.stream(inputs, config=config))
    snapshot1 = test_graph_app.get_state(config)
    assert snapshot1.next == ("plan_approval",)

    # 2. Resume with Approve for HITL #1 -> pauses at HITL #2 (reducer)
    list(test_graph_app.stream(Command(resume={"action": "approve"}), config=config))
    snapshot2 = test_graph_app.get_state(config)
    assert snapshot2.next == ("reducer",)

    # 3. Resume HITL #2 (Final Content Review) -> completes execution
    list(test_graph_app.stream(Command(resume={"action": "approve"}), config=config))
    snapshot3 = test_graph_app.get_state(config)
    assert snapshot3.next == ()
    assert snapshot3.values.get("approval_status") == ApprovalStatus.APPROVED
    assert snapshot3.values.get("final_approval_status") == ApprovalStatus.APPROVED
    assert len(snapshot3.values.get("sections")) == 2
    assert "## Section" in snapshot3.values.get("merged_md")


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
        elif schema == agent_nodes.CriticEvaluation:
            return agent_nodes.CriticEvaluation(score=8.5, passed=True, summary="Good draft", issues=[])
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
        elif schema == agent_nodes.CriticEvaluation:
            return agent_nodes.CriticEvaluation(score=8.5, passed=True, summary="Good draft", issues=[])
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


# ------------------------------------------------------------
# Critic / Quality Agent Evaluator-Revision Loop Tests
# ------------------------------------------------------------

from src.agent.schemas import CriticEvaluation, CriticIssue


def test_critic_node_produces_evaluation(monkeypatch):
    """Verify critic_node evaluates draft markdown and produces structured CriticEvaluation."""
    eval_mock = CriticEvaluation(
        score=8.5,
        passed=True,
        summary="Draft is clear and well grounded.",
        issues=[
            CriticIssue(
                category="readability",
                severity="low",
                description="Minor wordiness in intro",
                suggestion="Tighten intro paragraph.",
            )
        ],
        revision_instructions=None,
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: eval_mock)

    state = {
        "merged_md": "# RAG\n\nDraft content...",
        "plan": _make_plan(2),
        "topic": "RAG",
        "critic_retry_count": 0,
    }

    out = agent_nodes.critic_node(state)
    assert out["critic_evaluation"].score == 8.5
    assert out["critic_evaluation"].passed is True
    assert len(out["critic_evaluation"].issues) == 1


def test_route_after_critic_branches_correctly():
    """Verify route_after_critic routes to decide_images when passed or max retries reached, else to revision."""
    passed_eval = CriticEvaluation(score=8.0, passed=True, summary="Good", issues=[])
    failed_eval = CriticEvaluation(score=6.0, passed=False, summary="Needs work", issues=[])

    # 1. Passed -> decide_images
    assert agent_nodes.route_after_critic({"critic_evaluation": passed_eval, "critic_retry_count": 0, "max_critic_retries": 2}) == "decide_images"

    # 2. Failed with retries remaining -> revision
    assert agent_nodes.route_after_critic({"critic_evaluation": failed_eval, "critic_retry_count": 0, "max_critic_retries": 2}) == "revision"

    # 3. Failed but hit max retries -> decide_images
    assert agent_nodes.route_after_critic({"critic_evaluation": failed_eval, "critic_retry_count": 2, "max_critic_retries": 2}) == "decide_images"


def test_revision_node_updates_markdown_and_increments_retry(monkeypatch):
    """Verify revision_node rewrites draft and increments critic_retry_count."""
    monkeypatch.setattr(agent_nodes, "get_llm", lambda: object())
    monkeypatch.setattr(
        agent_nodes,
        "invoke_with_retry",
        lambda *a, **k: types.SimpleNamespace(content="# Revised RAG Title\n\nImproved draft content."),
    )

    failed_eval = CriticEvaluation(
        score=6.0,
        passed=False,
        summary="Fix repetition",
        issues=[CriticIssue(category="repetition", severity="medium", description="Dup text", suggestion="Remove dup")],
        revision_instructions="Remove duplicated text.",
    )

    state = {
        "topic": "RAG",
        "merged_md": "# Original RAG\n\nOld text",
        "critic_evaluation": failed_eval,
        "critic_retry_count": 0,
    }

    out = agent_nodes.revision_node(state)
    assert "# Revised RAG Title" in out["merged_md"]
    assert out["critic_retry_count"] == 1


# ------------------------------------------------------------
# Research Enhancement Nodes Tests (Phase 1)
# ------------------------------------------------------------

from src.agent.schemas import (
    SourceFilterResult,
    SourceFilterScore,
    ContradictionReport,
    ContradictionItem,
    ResearchResult,
    EvidenceItem,
)


def test_source_filter_node_filters_evidence(monkeypatch):
    """Verify source_filter_node filters evidence to keep=True items."""
    filter_mock = SourceFilterResult(
        scores=[
            SourceFilterScore(url="https://good.org", relevance_score=0.9, authority_score=0.9, freshness_score=0.9, is_duplicate=False, keep=True, reason="Relevant"),
            SourceFilterScore(url="https://spam.org", relevance_score=0.2, authority_score=0.1, freshness_score=0.1, is_duplicate=True, keep=False, reason="Spam"),
        ]
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: filter_mock)

    ev1 = EvidenceItem(title="Good", url="https://good.org", snippet="High quality content")
    ev2 = EvidenceItem(title="Spam", url="https://spam.org", snippet="Low quality content")

    state = {"topic": "RAG", "evidence": [ev1, ev2]}
    out = agent_nodes.source_filter_node(state)

    assert len(out["evidence"]) == 1
    assert out["evidence"][0].url == "https://good.org"
    assert len(out["source_filter_scores"]) == 2


def test_contradiction_detector_node_identifies_conflicts(monkeypatch):
    """Verify contradiction_detector_node identifies conflicts across sources."""
    report_mock = ContradictionReport(
        contradictions=[
            ContradictionItem(
                topic_claim="Vector DB scaling",
                source_a_url="https://a.org",
                source_a_claim="Supports 10M vectors",
                source_b_url="https://b.org",
                source_b_claim="Max 1M vectors",
                resolution_guidance="Note version differences",
            )
        ]
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: report_mock)

    ev1 = EvidenceItem(title="Source A", url="https://a.org", snippet="Supports 10M vectors")
    ev2 = EvidenceItem(title="Source B", url="https://b.org", snippet="Max 1M vectors")

    state = {"topic": "RAG", "evidence": [ev1, ev2]}
    out = agent_nodes.contradiction_detector_node(state)

    assert len(out["contradictions"]) == 1
    assert out["contradictions"][0]["topic_claim"] == "Vector DB scaling"


def test_structured_research_node_synthesizes_research_result(monkeypatch):
    """Verify structured_research_node normalizes evidence into ResearchResult."""
    res_mock = ResearchResult(
        topic="RAG Architecture",
        key_facts=["RAG combines retrieval and generation"],
        technical_details=["Uses dense vector embeddings"],
        code_examples=["retriever.retrieve(query)"],
        sources=[],
        contradictions=[],
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: res_mock)

    ev1 = EvidenceItem(title="Source A", url="https://a.org", snippet="Content A")
    state = {"topic": "RAG Architecture", "evidence": [ev1], "contradictions": []}

    out = agent_nodes.structured_research_node(state)
    assert out["research_result"].topic == "RAG Architecture"
    assert "RAG combines retrieval and generation" in out["research_result"].key_facts
    assert out["research_result"].filtered_sources_count == 1


def test_outline_validator_node_evaluates_plan():
    """Verify outline_validator_node checks plan structure and warns on missing conclusions."""
    plan = Plan(
        blog_title="Test Post",
        audience="Devs",
        tone="Direct",
        blog_kind="explainer",
        tasks=[
            Task(id=1, title="Introduction", goal="Intro goal", bullets=["b1", "b2", "b3"], target_words=200),
            Task(id=2, title="Deep Dive", goal="Dive goal", bullets=["b1", "b2", "b3"], target_words=400),
        ]
    )
    state = {"plan": plan}
    out = agent_nodes.outline_validator_node(state)
    assert "outline_validation" in out
    val = out["outline_validation"]
    assert len(val.warnings) > 0
    categories = [w.category for w in val.warnings]
    assert "task_count" in categories or "conclusion_missing" in categories


def test_fact_checker_node_produces_fact_report(monkeypatch):
    """Verify fact_checker_node generates a FactCheckReport."""
    fc_mock = FactCheckReport(
        score=9.0,
        total_claims_checked=2,
        verified_claims_count=2,
        claims=[
            ClaimVerification(
                claim_id="C1",
                claim="LangGraph supports state graphs",
                classification="SUPPORTED",
                confidence=0.95,
                supporting_sources=["https://docs.langchain.com"],
                explanation="Supported by docs",
            ),
        ]
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: fc_mock)

    state = {
        "topic": "LangGraph",
        "merged_md": "# LangGraph\n\nLangGraph supports state graphs.",
        "evidence": [EvidenceItem(title="Docs", url="https://docs.langchain.com", snippet="State graphs supported")],
    }
    out = agent_nodes.fact_checker_node(state)
    assert out["fact_check_report"].score == 9.0
    assert len(out["fact_check_report"].claims) == 1
    assert out["fact_check_report"].claims[0].classification == "SUPPORTED"
    assert out["fact_check_report"].claims[0].verdict == "verified"


def test_seo_agent_node_produces_seo_plan(monkeypatch):
    """Verify seo_agent_node generates an SEOPlan with metadata and FAQs."""
    seo_mock = SEOPlan(
        meta_title="Mastering LangGraph in 2026",
        meta_description="A complete guide to building agent workflows with LangGraph.",
        primary_keyword="LangGraph",
        secondary_keywords=["Multi-agent", "Python", "State Graph"],
        suggested_slug="mastering-langgraph-2026",
        target_readability_level="Intermediate",
        faq_items=[SEOFAQItem(question="What is LangGraph?", answer="A multi-agent orchestrator framework.")],
        seo_score=9.5,
    )
    monkeypatch.setattr(agent_nodes, "structured_with_retry", lambda schema, msg: seo_mock)

    state = {
        "topic": "LangGraph",
        "keywords": ["LangGraph", "Multi-agent"],
        "merged_md": "# LangGraph Guide\n\nContent here.",
    }
    out = agent_nodes.seo_agent_node(state)
    assert out["seo_plan"].meta_title == "Mastering LangGraph in 2026"
    assert out["seo_plan"].seo_score == 9.5
    assert len(out["seo_plan"].faq_items) == 1




