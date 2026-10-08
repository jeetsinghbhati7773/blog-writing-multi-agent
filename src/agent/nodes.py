import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from langgraph.types import interrupt

from src.agent.schemas import (
    State,
    OverallState,
    SocialPosts,
    Task,
    Plan,
    EvidenceItem,
    RouterDecision,
    EvidencePack,
    GlobalImagePlan,
    ApprovalStatus,
    CriticIssue,
    CriticEvaluation,
    SourceFilterScore,
    SourceFilterResult,
    ContradictionItem,
    ContradictionReport,
    ResearchResult,
    OutlineValidationWarning,
    OutlineValidationResult,
    ClaimVerification,
    FactCheckItem,
    FactCheckReport,
    SEOFAQItem,
    SEOPlan,
    ArticleStyleProfile,
)
from src.agent.tools import (
    tavily_search,
    iso_to_date,
    gemini_generate_image_bytes,
    safe_slug,
)
from src.agent.rate_limiter import groq_rate_limiter
from src.paths import OUTPUTS_DIR, IMAGES_DIR, PROJECT_ROOT, ensure_dir

load_dotenv()

logger = logging.getLogger(__name__)

# Sync Streamlit Cloud secrets to os.environ if available
try:
    import streamlit as st
    for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "POLLINATIONS_API_KEY", "GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
        if k in st.secrets and k not in os.environ:
            os.environ[k] = str(st.secrets[k])
except Exception as e:
    logger.debug("Streamlit secrets sync skipped (not running on Streamlit Cloud?): %s", e)

def get_llm():
    """
    Returns an LLM instance (ChatGroq or ChatOpenAI fallback).
    Supports GROQ_MODEL environment variable override.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key:
        try:
            import streamlit as st
            groq_key = st.secrets.get("GROQ_API_KEY")
            openai_key = openai_key or st.secrets.get("OPENAI_API_KEY")
        except Exception as e:
            logger.debug("Could not read LLM keys from Streamlit secrets: %s", e)

    if groq_key:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_key,
            max_retries=5,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            max_retries=5,
        )
    else:
        # No key configured. Fail with a clear, actionable message instead of
        # returning a client with a dummy key (which used to fail later with a
        # cryptic auth error mid-generation). Safe to raise here: get_llm() is
        # only ever called lazily (via LazyLLM) or inside node/tool functions,
        # never at import time, so this does not crash module import/deployment.
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (or OPENAI_API_KEY) in your "
            ".env file or environment before generating content. See .env.example "
            "for the expected variables. You can get a free Groq key at "
            "https://console.groq.com/keys."
        )

class LazyLLM:
    """Lazy proxy wrapper to avoid instantiating LLM at module import time."""
    _instance = None

    def __getattr__(self, name: str):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return getattr(LazyLLM._instance, name)

    def invoke(self, *args, **kwargs):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return LazyLLM._instance.invoke(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return get_llm().with_structured_output(*args, **kwargs)


# Module-level lazy LLM instance for node functions
llm = LazyLLM()


def invoke_with_retry(llm_instance, messages, max_retries=5, initial_delay=5.0):
    """
    Executes LLM invocation with automatic exponential backoff for rate limits (Groq 429 / TPM limits).
    """
    for attempt in range(max_retries):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str or "quota" in err_str or "tpm" in err_str:
                if attempt == max_retries - 1:
                    raise
                wait_time = initial_delay * (2 ** attempt) + (attempt * 2.0)
                time.sleep(wait_time)
            else:
                raise


def structured_with_retry(schema_cls, messages, max_retries=5, initial_delay=5.0):
    """
    Executes LLM structured output with automatic exponential backoff for rate limits
    and fallback for JSON argument parsing / tool calling errors.
    """
    last_exception = None
    target_llm = get_llm()

    for attempt in range(max_retries):
        try:
            runnable = target_llm.with_structured_output(schema_cls)
            return runnable.invoke(messages)
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str or "quota" in err_str or "tpm" in err_str:
                if attempt == max_retries - 1:
                    break
                wait_time = initial_delay * (2 ** attempt) + (attempt * 2.0)
                time.sleep(wait_time)
            elif any(k in err_str for k in ["parse", "tool", "json", "400", "invalid_request_error", "failed_generation"]):
                # Tool calling format issue (e.g. Groq failing to parse JSON tool arguments)
                break
            else:
                raise

    # Fallback to json_mode if function/tool calling failed due to syntax/formatting errors
    try:
        runnable = target_llm.with_structured_output(schema_cls, method="json_mode")
        return runnable.invoke(messages)
    except Exception as fallback_err:
        if last_exception:
            raise last_exception from fallback_err
        raise fallback_err


# -----------------------------
# 1) Router Node
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decision = structured_with_retry(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "plan_version": state.get("plan_version", 1),
        "approval_status": state.get("approval_status", ApprovalStatus.PENDING),
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 2) Research Node
# -----------------------------
RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        raw.extend(tavily_search(q, max_results=5))

    doc_evidence: List[EvidenceItem] = []
    if state.get("use_uploaded_docs"):
        try:
            from src.rag.retriever import DocumentRetriever
            retriever = DocumentRetriever()
            doc_queries = queries if queries else [state.get("topic", "")]
            for dq in doc_queries[:3]:
                chunks = retriever.retrieve_relevant_chunks(dq, k=3)
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    fn = meta.get("filename", "Uploaded Document")
                    page = meta.get("page")
                    page_str = f" (Page {page})" if page else ""
                    doc_evidence.append(
                        EvidenceItem(
                            title=f"Uploaded Document: {fn}{page_str}",
                            url=f"doc://{fn}",
                            published_at=None,
                            snippet=chunk.get("content", "")[:350],
                            source=fn,
                        )
                    )
        except Exception as e:
            logger.warning("Uploaded-document retrieval failed during research: %s", e)

    if not raw:
        return {"evidence": doc_evidence}

    # Trim search results to prevent exceeding LLM token limits (keep under 12k TPM limit)
    trimmed_raw = [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": (item.get("content") or item.get("snippet") or "")[:350],
        }
        for item in raw[:15]
    ]

    pack = structured_with_retry(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{trimmed_raw}"
                )
            ),
        ],
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    for de in doc_evidence:
        if de.url not in dedup:
            dedup[de.url] = de

    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if e.url.startswith("doc://") or ((d := iso_to_date(e.published_at)) and d >= cutoff)]

    return {"evidence": evidence}


# -----------------------------
# 2.5) Source Quality Filter, Contradiction Detector & Structured Research Nodes
# -----------------------------
SOURCE_FILTER_SYSTEM = """You are a research quality filter agent.
Evaluate gathered evidence items for topic relevance, source authority, freshness, and duplicate information.

Rules:
- Assign scores (0.0 to 1.0) for relevance_score, authority_score, and freshness_score.
- Detect if the source is a duplicate of another snippet.
- Set keep=True ONLY for high-signal, relevant, non-duplicate sources.
- Return strictly SourceFilterResult schema.
"""

def source_filter_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    if not evidence:
        return {"evidence": [], "source_filter_scores": []}

    trimmed_evidence = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:12]]

    filter_res = structured_with_retry(
        SourceFilterResult,
        [
            SystemMessage(content=SOURCE_FILTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"As-of: {state.get('as_of')}\n"
                    f"Evidence Sources:\n{trimmed_evidence}"
                )
            ),
        ],
    )

    keep_urls = {s.url for s in filter_res.scores if s.keep}
    filtered_evidence = [e for e in evidence if (e.url if hasattr(e, "url") else e.get("url")) in keep_urls]
    if not filtered_evidence:
        filtered_evidence = evidence[:5]

    return {
        "evidence": filtered_evidence,
        "source_filter_scores": [s.model_dump() for s in filter_res.scores],
    }


CONTRADICTION_SYSTEM = """You are a research contradiction detector.
Analyze evidence snippets to identify conflicting technical claims, incompatible benchmarks, or contradictory claims between sources.

Rules:
- Identify genuine contradictions or conflicting claims across sources.
- If no contradictions exist, return contradictions=[].
- Return strictly ContradictionReport schema.
"""

def contradiction_detector_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    if not evidence or len(evidence) < 2:
        return {"contradictions": []}

    snippets = [f"URL: {e.url if hasattr(e, 'url') else e.get('url')} | Snippet: {e.snippet if hasattr(e, 'snippet') else e.get('snippet')}" for e in evidence[:8]]

    report = structured_with_retry(
        ContradictionReport,
        [
            SystemMessage(content=CONTRADICTION_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"Evidence Snippets:\n" + "\n".join(snippets)
                )
            ),
        ],
    )

    return {"contradictions": [c.model_dump() for c in report.contradictions]}


STRUCTURED_RESEARCH_SYSTEM = """You are a senior technical research synthesizer.
Synthesize evidence items and contradiction reports into a clean, normalized ResearchResult object.

Rules:
- Extract key_facts, technical_details, and code_examples.
- Include structured contradictions and sources.
- Normalize raw evidence into clear structured research state.
- Return strictly ResearchResult schema.
"""

def structured_research_node(state: State) -> dict:
    evidence = state.get("evidence", [])
    contradictions = state.get("contradictions", [])

    ev_dumps = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:8]]
    ct_dumps = contradictions[:5]

    research_res = structured_with_retry(
        ResearchResult,
        [
            SystemMessage(content=STRUCTURED_RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"Filtered Evidence Sources:\n{ev_dumps}\n\n"
                    f"Identified Contradictions:\n{ct_dumps}"
                )
            ),
        ],
    )
    research_res.sources = [EvidenceItem(**e) if isinstance(e, dict) else e for e in ev_dumps]
    research_res.filtered_sources_count = len(evidence)

    return {"research_result": research_res}




# -----------------------------
# 3) Orchestrator Node
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

STYLE_PROFILES = {
    "Technical Tutorial": "Focus on step-by-step implementation, code snippets, prerequisites, practical setup, and clear technical explanations.",
    "Beginner Friendly": "Use simple analogies, avoid jargon without explaining it, include an engaging overview, and step-by-step walkthroughs.",
    "Research Style": "Emphasize data, academic/industry literature context, citations, comparative analysis, and structured evidence synthesis.",
    "Developer Blog": "Write in a conversational yet expert developer voice, include code/architectural diagrams, practical tips, and real-world trade-offs.",
    "LinkedIn Post": "Keep paragraphs short and punchy, use bullet points, focus on high-impact takeaways, executive summary, and actionable key points.",
    "SEO Blog": "Optimize heading structures (H2/H3), focus on comprehensive coverage of target keywords, FAQ sections, and reader intent satisfaction."
}


def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    human_feedback = state.get("human_feedback")
    prev_plan = state.get("plan")
    approval_status = state.get("approval_status")
    plan_version = state.get("plan_version", 1)
    style_profile = state.get("style_profile")
    style_guidance = f"\nStyle Profile requested: {style_profile} - {STYLE_PROFILES.get(style_profile, '')}" if style_profile else ""

    res_result = state.get("research_result")
    res_text = ""
    if res_result:
        res_dict = res_result.model_dump() if hasattr(res_result, "model_dump") else res_result
        res_text = (
            f"\n\nNormalized Research Result State:\n"
            f"Key Facts: {res_dict.get('key_facts', [])}\n"
            f"Technical Details: {res_dict.get('technical_details', [])}\n"
            f"Code Examples: {res_dict.get('code_examples', [])}\n"
            f"Identified Contradictions: {res_dict.get('contradictions', [])}\n"
        )

    feedback_prompt = ""
    if approval_status == ApprovalStatus.CHANGES_REQUESTED or approval_status == "changes_requested":
        if human_feedback:
            prev_title = prev_plan.blog_title if hasattr(prev_plan, "blog_title") else (prev_plan.get("blog_title") if isinstance(prev_plan, dict) else "N/A")
            prev_tasks = prev_plan.tasks if hasattr(prev_plan, "tasks") else (prev_plan.get("tasks", []) if isinstance(prev_plan, dict) else [])
            feedback_prompt = (
                f"\n\nIMPORTANT HUMAN REVIEWER FEEDBACK (Plan Revision Version {plan_version}):\n"
                f"The human reviewer requested changes to the proposed outline:\n"
                f"\"{human_feedback}\"\n"
                f"Previous Plan Title: {prev_title}\n"
                f"Previous Outline Tasks:\n{prev_tasks}\n"
                f"Update and re-architect the plan to carefully address the human reviewer's feedback."
            )
    elif approval_status == ApprovalStatus.REGENERATE or approval_status == "regenerate":
        feedback_prompt = (
            f"\n\nIMPORTANT (Plan Regeneration Requested - Version {plan_version}):\n"
            f"The human reviewer requested a fresh alternative plan for this topic. "
            f"Generate an engaging alternative outline."
        )

    plan = structured_with_retry(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n"
                    f"{style_guidance}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:8]}"
                    f"{res_text}"
                    f"{feedback_prompt}"
                )
            ),
        ],
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {
        "plan": plan,
        "plan_version": plan_version,
        "approval_status": ApprovalStatus.PENDING,
    }


def outline_validator_node(state: State) -> dict:
    """
    Pre-planning validation node. Evaluates the generated Plan structure
    before presenting it to the user in HITL #1.
    """
    plan = state.get("plan")
    if not plan:
        return {"outline_validation": OutlineValidationResult(passed=True, warnings=[], suggestions=[])}

    plan_dump = plan.model_dump() if hasattr(plan, "model_dump") else plan
    tasks = plan_dump.get("tasks", [])

    warnings = []
    suggestions = []

    if len(tasks) < 3:
        warnings.append(
            OutlineValidationWarning(
                category="task_count",
                severity="warning",
                message=f"Outline has only {len(tasks)} tasks. Minimum recommended is 3-4 tasks for depth.",
            )
        )
    elif len(tasks) > 8:
        warnings.append(
            OutlineValidationWarning(
                category="task_count",
                severity="info",
                message=f"Outline has {len(tasks)} tasks. Consider consolidating sections to keep reader engagement high.",
            )
        )

    has_conclusion = any(
        kw in t.get("title", "").lower()
        for t in tasks
        for kw in ["conclusion", "summary", "wrap", "next step", "takeaway"]
    )
    if not has_conclusion:
        warnings.append(
            OutlineValidationWarning(
                category="conclusion_missing",
                severity="warning",
                message="No dedicated conclusion or summary section found in the outline.",
            )
        )

    word_targets = [t.get("target_words", 300) for t in tasks]
    total_words = sum(word_targets)
    if any(w < 100 for w in word_targets):
        warnings.append(
            OutlineValidationWarning(
                category="word_distribution",
                severity="info",
                message="Some tasks have target word counts under 100 words.",
            )
        )

    suggestions.append(f"Total target word count across {len(tasks)} sections is ~{total_words} words.")

    result = OutlineValidationResult(
        passed=len([w for w in warnings if w.severity == "error"]) == 0,
        warnings=warnings,
        suggestions=suggestions,
    )
    return {"outline_validation": result}


# -----------------------------
# 3.5) Human-in-the-Loop Plan Approval Node
# -----------------------------
def plan_approval_node(state: State) -> dict:
    """
    Human-in-the-Loop approval node.
    Pauses graph execution after Orchestrator generates the plan.
    Resumes when human submits approval/feedback via Streamlit UI.
    """
    plan = state.get("plan")
    plan_version = state.get("plan_version", 1)

    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan

    human_response = interrupt({
        "plan": plan_dict,
        "plan_version": plan_version,
        "topic": state.get("topic"),
        "approval_status": state.get("approval_status", ApprovalStatus.PENDING),
        "as_of": state.get("as_of"),
        "evidence": [e.model_dump() if hasattr(e, "model_dump") else e for e in state.get("evidence", [])],
    })

    if not isinstance(human_response, dict):
        action = str(human_response)
        feedback = ""
    else:
        action = human_response.get("action", "approve")
        feedback = human_response.get("feedback", "")

    if action == "approve":
        return {
            "approval_status": ApprovalStatus.APPROVED,
            "human_approval": human_response,
            "human_feedback": None,
        }
    elif action == "request_changes":
        return {
            "approval_status": ApprovalStatus.CHANGES_REQUESTED,
            "human_feedback": feedback,
            "human_approval": human_response,
            "plan_version": plan_version + 1,
        }
    elif action == "regenerate":
        return {
            "approval_status": ApprovalStatus.REGENERATE,
            "human_feedback": None,
            "human_approval": human_response,
            "plan_version": plan_version + 1,
        }
    else:
        return {
            "approval_status": ApprovalStatus.PENDING,
        }


def route_after_approval(state: State):
    """
    Conditional edge router following HITL plan approval.
    - APPROVED -> fanout to parallel worker nodes
    - CHANGES_REQUESTED or REGENERATE -> return to orchestrator node
    - PENDING -> remain in plan_approval node
    """
    status = state.get("approval_status")
    if status == ApprovalStatus.APPROVED or status == "approved":
        return fanout(state)
    elif status in (ApprovalStatus.CHANGES_REQUESTED, "changes_requested", ApprovalStatus.REGENERATE, "regenerate"):
        return "orchestrator"
    else:
        return "plan_approval"


# -----------------------------
# 4) Fanout & Worker Node
# -----------------------------

def fanout(state: State):
    from langgraph.types import Send
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url}"
        for e in evidence[:5]
    )

    # Smooth parallel worker bursts via a shared rate limiter instead of a fixed
    # per-task sleep. Workers run concurrently (LangGraph `Send` fanout), so this
    # spaces the START of each Groq call by GROQ_MIN_REQUEST_INTERVAL seconds to
    # avoid Groq TPM burst spikes, without the dead time of hard-coded staggering.
    groq_rate_limiter.acquire()

    response = invoke_with_retry(
        get_llm(),
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ],
    )
    section_md = response.content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# 5) Reducer & Critic Subgraph Nodes
# -----------------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


CRITIC_SYSTEM = """You are a senior technical blog auditor and critic.
Evaluate the provided draft article against the target outline and research evidence.

Evaluation Dimensions:
1. Grounding & Factual Consistency: Are claims supported by provided evidence URLs? No hallucinations.
2. Relevance & Plan Alignment: Does the text fulfill the goals of each planned section?
3. Completeness: Are key technical bullet points covered?
4. Readability & Tone: Is it technical yet accessible, engaging, and clear?
5. Repetition & Redundancy: Are there duplicate explanations across sections?
6. Structure & Formatting: Headers, code blocks, bullet points correctly formatted.
7. Adherence to User Instructions.

Rules:
- Assign a numerical score from 0.0 to 10.0.
- Set passed=True ONLY IF score >= 7.5 AND there are NO high-severity issues.
- Provide constructive, actionable revision instructions if revision is needed.
- Return strictly CriticEvaluation schema.
"""

def critic_node(state: State) -> dict:
    merged_md = state.get("merged_md", "")
    plan = state.get("plan")
    evidence = state.get("evidence", [])

    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan

    evaluation = structured_with_retry(
        CriticEvaluation,
        [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"Audience: {state.get('audience', 'Technical')}\n"
                    f"Tone: {state.get('tone', 'Practical')}\n"
                    f"Plan:\n{plan_dict}\n\n"
                    f"Evidence Sources:\n{[e.model_dump() if hasattr(e, 'model_dump') else e for e in evidence][:6]}\n\n"
                    f"Draft Article Text:\n{merged_md[:12000]}"
                )
            ),
        ],
    )

    return {
        "critic_evaluation": evaluation,
        "critic_retry_count": state.get("critic_retry_count", 0),
        "max_critic_retries": state.get("max_critic_retries", 2),
    }


REVISION_SYSTEM = """You are an expert technical editor and content polisher.
Rewrite and improve the draft article to address all identified Critic issues.

Constraints:
- Resolve all high and medium severity issues highlighted by the Critic.
- Eliminate repetitive content between sections.
- Keep all valid technical code snippets and citation markdown links [Source](URL).
- Maintain Markdown structure starting with '# <Blog Title>'.
- Output ONLY the revised Markdown text.
"""

def revision_node(state: State) -> dict:
    merged_md = state.get("merged_md", "")
    critic_eval = state.get("critic_evaluation")
    retry_count = state.get("critic_retry_count", 0)

    issues_text = ""
    instructions = ""
    if critic_eval:
        issues_text = "\n".join(f"- [{i.severity.upper()}] ({i.category}): {i.description} -> {i.suggestion}" for i in critic_eval.issues)
        instructions = critic_eval.revision_instructions or ""

    response = invoke_with_retry(
        get_llm(),
        [
            SystemMessage(content=REVISION_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"Critic Score: {critic_eval.score if critic_eval else 'N/A'}/10\n"
                    f"Critic Issues:\n{issues_text}\n\n"
                    f"Revision Instructions:\n{instructions}\n\n"
                    f"Current Draft Markdown:\n{merged_md}"
                )
            ),
        ],
    )

    revised_md = response.content.strip()
    return {
        "merged_md": revised_md,
        "critic_retry_count": retry_count + 1,
    }


def route_after_critic(state: State) -> str:
    """
    Conditional edge router following Critic evaluation.
    - If passed==True or critic_retry_count >= max_critic_retries: proceed to decide_images.
    - Else: proceed to revision node.
    """
    eval_obj = state.get("critic_evaluation")
    retry_count = state.get("critic_retry_count", 0)
    max_retries = state.get("max_critic_retries", 2)

    if not eval_obj:
        return "decide_images"

    if eval_obj.passed or retry_count >= max_retries:
        return "decide_images"
    else:
        return "revision"

from src.agent.tools import (
    tavily_search,
    iso_to_date,
    gemini_generate_image_bytes,
    safe_slug,
)
from src.agent.rate_limiter import groq_rate_limiter
from src.paths import OUTPUTS_DIR, IMAGES_DIR, PROJECT_ROOT, ensure_dir

load_dotenv()

logger = logging.getLogger(__name__)

# Sync Streamlit Cloud secrets to os.environ if available
try:
    import streamlit as st
    for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "POLLINATIONS_API_KEY", "GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]:
        if k in st.secrets and k not in os.environ:
            os.environ[k] = str(st.secrets[k])
except Exception as e:
    logger.debug("Streamlit secrets sync skipped (not running on Streamlit Cloud?): %s", e)

def get_llm():
    """
    Returns an LLM instance (ChatGroq or ChatOpenAI fallback).
    Supports GROQ_MODEL environment variable override.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key:
        try:
            import streamlit as st
            groq_key = st.secrets.get("GROQ_API_KEY")
            openai_key = openai_key or st.secrets.get("OPENAI_API_KEY")
        except Exception as e:
            logger.debug("Could not read LLM keys from Streamlit secrets: %s", e)

    if groq_key:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_key,
            max_retries=5,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            max_retries=5,
        )
    else:
        # No key configured. Fail with a clear, actionable message instead of
        # returning a client with a dummy key (which used to fail later with a
        # cryptic auth error mid-generation). Safe to raise here: get_llm() is
        # only ever called lazily (via LazyLLM) or inside node/tool functions,
        # never at import time, so this does not crash module import/deployment.
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (or OPENAI_API_KEY) in your "
            ".env file or environment before generating content. See .env.example "
            "for the expected variables. You can get a free Groq key at "
            "https://console.groq.com/keys."
        )

class LazyLLM:
    """Lazy proxy wrapper to avoid instantiating LLM at module import time."""
    _instance = None

    def __getattr__(self, name: str):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return getattr(LazyLLM._instance, name)

    def invoke(self, *args, **kwargs):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return LazyLLM._instance.invoke(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return get_llm().with_structured_output(*args, **kwargs)


# Module-level lazy LLM instance for node functions
llm = LazyLLM()


def invoke_with_retry(llm_instance, messages, max_retries=5, initial_delay=5.0):
    """
    Executes LLM invocation with automatic exponential backoff for rate limits (Groq 429 / TPM limits).
    """
    for attempt in range(max_retries):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str or "quota" in err_str or "tpm" in err_str:
                if attempt == max_retries - 1:
                    raise
                wait_time = initial_delay * (2 ** attempt) + (attempt * 2.0)
                time.sleep(wait_time)
            else:
                raise


def structured_with_retry(schema_cls, messages, max_retries=5, initial_delay=5.0):
    """
    Executes LLM structured output with automatic exponential backoff for rate limits
    and fallback for JSON argument parsing / tool calling errors.
    """
    last_exception = None
    target_llm = get_llm()

    for attempt in range(max_retries):
        try:
            runnable = target_llm.with_structured_output(schema_cls)
            return runnable.invoke(messages)
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "limit" in err_str or "quota" in err_str or "tpm" in err_str:
                if attempt == max_retries - 1:
                    break
                wait_time = initial_delay * (2 ** attempt) + (attempt * 2.0)
                time.sleep(wait_time)
            elif any(k in err_str for k in ["parse", "tool", "json", "400", "invalid_request_error", "failed_generation"]):
                # Tool calling format issue (e.g. Groq failing to parse JSON tool arguments)
                break
            else:
                raise

    # Fallback to json_mode if function/tool calling failed due to syntax/formatting errors
    try:
        runnable = target_llm.with_structured_output(schema_cls, method="json_mode")
        return runnable.invoke(messages)
    except Exception as fallback_err:
        if last_exception:
            raise last_exception from fallback_err
        raise fallback_err


# -----------------------------
# 1) Router Node
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decision = structured_with_retry(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "plan_version": state.get("plan_version", 1),
        "approval_status": state.get("approval_status", ApprovalStatus.PENDING),
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# -----------------------------
# 2) Research Node
# -----------------------------
RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        raw.extend(tavily_search(q, max_results=5))

    doc_evidence: List[EvidenceItem] = []
    if state.get("use_uploaded_docs"):
        try:
            from src.rag.retriever import DocumentRetriever
            retriever = DocumentRetriever()
            doc_queries = queries if queries else [state.get("topic", "")]
            for dq in doc_queries[:3]:
                chunks = retriever.retrieve_relevant_chunks(dq, k=3)
                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    fn = meta.get("filename", "Uploaded Document")
                    page = meta.get("page")
                    page_str = f" (Page {page})" if page else ""
                    doc_evidence.append(
                        EvidenceItem(
                            title=f"Uploaded Document: {fn}{page_str}",
                            url=f"doc://{fn}",
                            published_at=None,
                            snippet=chunk.get("content", "")[:350],
                            source=fn,
                        )
                    )
        except Exception as e:
            logger.warning("Uploaded-document retrieval failed during research: %s", e)

    if not raw:
        return {"evidence": doc_evidence}

    # Trim search results to prevent exceeding LLM token limits (keep under 12k TPM limit)
    trimmed_raw = [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": (item.get("content") or item.get("snippet") or "")[:350],
        }
        for item in raw[:15]
    ]

    pack = structured_with_retry(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{trimmed_raw}"
                )
            ),
        ],
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    for de in doc_evidence:
        if de.url not in dedup:
            dedup[de.url] = de

    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if e.url.startswith("doc://") or ((d := iso_to_date(e.published_at)) and d >= cutoff)]

    return {"evidence": evidence}



# -----------------------------
# 3) Orchestrator Node
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    human_feedback = state.get("human_feedback")
    prev_plan = state.get("plan")
    approval_status = state.get("approval_status")
    plan_version = state.get("plan_version", 1)

    feedback_prompt = ""
    if approval_status == ApprovalStatus.CHANGES_REQUESTED or approval_status == "changes_requested":
        if human_feedback:
            prev_title = prev_plan.blog_title if hasattr(prev_plan, "blog_title") else (prev_plan.get("blog_title") if isinstance(prev_plan, dict) else "N/A")
            prev_tasks = prev_plan.tasks if hasattr(prev_plan, "tasks") else (prev_plan.get("tasks", []) if isinstance(prev_plan, dict) else [])
            feedback_prompt = (
                f"\n\nIMPORTANT HUMAN REVIEWER FEEDBACK (Plan Revision Version {plan_version}):\n"
                f"The human reviewer requested changes to the proposed outline:\n"
                f"\"{human_feedback}\"\n"
                f"Previous Plan Title: {prev_title}\n"
                f"Previous Outline Tasks:\n{prev_tasks}\n"
                f"Update and re-architect the plan to carefully address the human reviewer's feedback."
            )
    elif approval_status == ApprovalStatus.REGENERATE or approval_status == "regenerate":
        feedback_prompt = (
            f"\n\nIMPORTANT (Plan Regeneration Requested - Version {plan_version}):\n"
            f"The human reviewer requested a fresh alternative plan for this topic. "
            f"Generate an engaging alternative outline."
        )

    plan = structured_with_retry(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:8]}"
                    f"{feedback_prompt}"
                )
            ),
        ],
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {
        "plan": plan,
        "plan_version": plan_version,
        "approval_status": ApprovalStatus.PENDING,
    }


# -----------------------------
# 3.5) Human-in-the-Loop Plan Approval Node
# -----------------------------
def plan_approval_node(state: State) -> dict:
    """
    Human-in-the-Loop approval node.
    Pauses graph execution after Orchestrator generates the plan.
    Resumes when human submits approval/feedback via Streamlit UI.
    """
    plan = state.get("plan")
    plan_version = state.get("plan_version", 1)

    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan

    human_response = interrupt({
        "plan": plan_dict,
        "plan_version": plan_version,
        "topic": state.get("topic"),
        "approval_status": state.get("approval_status", ApprovalStatus.PENDING),
        "as_of": state.get("as_of"),
        "evidence": [e.model_dump() if hasattr(e, "model_dump") else e for e in state.get("evidence", [])],
    })

    if not isinstance(human_response, dict):
        action = str(human_response)
        feedback = ""
    else:
        action = human_response.get("action", "approve")
        feedback = human_response.get("feedback", "")

    if action == "approve":
        return {
            "approval_status": ApprovalStatus.APPROVED,
            "human_approval": human_response,
            "human_feedback": None,
        }
    elif action == "request_changes":
        return {
            "approval_status": ApprovalStatus.CHANGES_REQUESTED,
            "human_feedback": feedback,
            "human_approval": human_response,
            "plan_version": plan_version + 1,
        }
    elif action == "regenerate":
        return {
            "approval_status": ApprovalStatus.REGENERATE,
            "human_feedback": None,
            "human_approval": human_response,
            "plan_version": plan_version + 1,
        }
    else:
        return {
            "approval_status": ApprovalStatus.PENDING,
        }


def route_after_approval(state: State):
    """
    Conditional edge router following HITL plan approval.
    - APPROVED -> fanout to parallel worker nodes
    - CHANGES_REQUESTED or REGENERATE -> return to orchestrator node
    - PENDING -> remain in plan_approval node
    """
    status = state.get("approval_status")
    if status == ApprovalStatus.APPROVED or status == "approved":
        return fanout(state)
    elif status in (ApprovalStatus.CHANGES_REQUESTED, "changes_requested", ApprovalStatus.REGENERATE, "regenerate"):
        return "orchestrator"
    else:
        return "plan_approval"


# -----------------------------
# 4) Fanout & Worker Node
# -----------------------------

def fanout(state: State):
    from langgraph.types import Send
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url}"
        for e in evidence[:5]
    )

    # Smooth parallel worker bursts via a shared rate limiter instead of a fixed
    # per-task sleep. Workers run concurrently (LangGraph `Send` fanout), so this
    # spaces the START of each Groq call by GROQ_MIN_REQUEST_INTERVAL seconds to
    # avoid Groq TPM burst spikes, without the dead time of hard-coded staggering.
    groq_rate_limiter.acquire()

    response = invoke_with_retry(
        get_llm(),
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ],
    )
    section_md = response.content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# 5) Reducer Subgraph Nodes
# -----------------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are a senior technical art director and visual planner for high-end technical publications.

Your goal is to create a diverse, high-impact Visual Plan for the article markdown.

Rules:
1. Max 2-3 images total across the entire article. Select ONLY sections where a visual materially enhances technical understanding.
2. Ensure Visual Diversity: Assign different visual_type values to each image (e.g. TECHNICAL_DIAGRAM for architecture, PROCESS_FLOW for workflow, COMPARISON_GRAPHIC for comparisons).
3. Do NOT repeat the same visual style, layout, or concept across sections.
4. For each image, construct a structured, detailed prompt covering:
   - Subject & Purpose
   - Composition & Spatial Layout (left-to-right, hierarchical boxes, flow)
   - Visual Style (clean technical editorial illustration)
   - Color Palette
   - Aspect Ratio (16:9 preferred for tech blogs)
   - Negative constraints (avoid generic AI brain imagery, circuit boards, excessive text, watermarks)
5. Use placeholders [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] sequentially.

Output must match GlobalImagePlan schema.
"""

def _insert_placeholders_into_md(merged_md: str, image_specs: List[dict]) -> str:
    if not image_specs:
        return merged_md

    if all(spec.get("placeholder", "") in merged_md for spec in image_specs if spec.get("placeholder")):
        return merged_md

    lines = merged_md.split("\n")
    headers_idx = [i for i, line in enumerate(lines) if line.startswith("## ")]

    if not headers_idx:
        md_out = merged_md
        for spec in image_specs:
            ph = spec.get("placeholder", "")
            if ph and ph not in md_out:
                md_out += f"\n\n{ph}\n\n"
        return md_out

    placed_placeholders = set()

    # Try placement by section_title matching
    for spec in image_specs:
        ph = spec.get("placeholder", "")
        if not ph or ph in merged_md:
            placed_placeholders.add(ph)
            continue

        sec_title = spec.get("section_title") or ""
        sec_title_clean = sec_title.lower().replace("#", "").strip()

        if sec_title_clean:
            for h_line_idx in headers_idx:
                header_text = lines[h_line_idx].lower().replace("#", "").strip()
                if sec_title_clean in header_text or header_text in sec_title_clean:
                    insert_at = h_line_idx + 1
                    while insert_at < len(lines) and lines[insert_at].strip() != "" and not lines[insert_at].startswith("#"):
                        insert_at += 1
                    lines.insert(insert_at, f"\n{ph}\n")
                    placed_placeholders.add(ph)
                    headers_idx = [i for i, l in enumerate(lines) if l.startswith("## ")]
                    break

    # Distribute any unplaced placeholders across available section headers
    unplaced_specs = [s for s in image_specs if s.get("placeholder") and s.get("placeholder") not in placed_placeholders and s.get("placeholder") not in "\n".join(lines)]
    if unplaced_specs:
        num_headers = len(headers_idx)
        for idx, spec in enumerate(unplaced_specs):
            ph = spec["placeholder"]
            target_h_pos = int((idx + 1) * num_headers / (len(unplaced_specs) + 1))
            target_h_pos = max(0, min(num_headers - 1, target_h_pos))
            h_line_idx = headers_idx[target_h_pos]

            insert_at = h_line_idx + 1
            while insert_at < len(lines) and lines[insert_at].strip() != "" and not lines[insert_at].startswith("#"):
                insert_at += 1
            lines.insert(insert_at, f"\n{ph}\n")
            headers_idx = [i for i, l in enumerate(lines) if l.startswith("## ")]

    return "\n".join(lines)


def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = structured_with_retry(
        GlobalImagePlan,
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Propose image specs (max 3) with target section titles for visual diagrams.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    image_specs = [img.model_dump() for img in image_plan.images]

    if image_plan.md_with_placeholders and all(s["placeholder"] in image_plan.md_with_placeholders for s in image_specs if s.get("placeholder")):
        md_out = image_plan.md_with_placeholders
    else:
        md_out = _insert_placeholders_into_md(merged_md, image_specs)

    return {
        "md_with_placeholders": md_out,
        "image_specs": image_specs,
    }


def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")
    topic = state.get("topic") or "blog"
    blog_title = plan.blog_title if plan else topic

    md = state.get("md_with_placeholders") or state.get("merged_md") or state.get("final", "")
    image_specs = state.get("image_specs", []) or []

    # Ensure outputs directory exists (anchored to the project root, not CWD)
    outputs_dir = ensure_dir(OUTPUTS_DIR)

    filename_str = f"{safe_slug(blog_title)}.md"
    out_file = outputs_dir / filename_str

    if not image_specs:
        if md:
            out_file.write_text(md, encoding="utf-8")
        return {"final": md}

    # Store images inside a topic-based subfolder under images/<topic_slug>/
    topic_slug = safe_slug(state.get("topic") or plan.blog_title)
    topic_images_dir = IMAGES_DIR / topic_slug
    topic_images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = topic_images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        # Store the link relative to the project root so the renderer (which
        # resolves relative image links against the project root) finds the file
        # regardless of the current working directory.
        try:
            img_rel_path = out_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            img_rel_path = out_path.resolve().as_posix()
        img_md = f"![{spec['alt']}]({img_rel_path})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    out_file.write_text(md, encoding="utf-8")
    return {"final": md}


# -----------------------------
# Fact & Citation Checker Node
# -----------------------------
FACT_CHECK_SYSTEM = """You are a rigorous factual verification agent.

Your task is to verify each factual claim in a draft markdown ONLY against the provided research evidence sources.

Instructions:
1. Extract key technical & numerical claims (up to 8 claims). Decompose multi-part compound claims into atomic subclaims.
2. For each claim, classify it into EXACTLY ONE of:
   - SUPPORTED: The provided evidence explicitly states or directly entails the claim.
   - PARTIALLY_SUPPORTED: The evidence supports some components of a multi-part claim, but conflicts with or lacks info for other components.
   - CONTRADICTED: The provided evidence explicitly states a fact that conflicts with the claim.
   - UNSUPPORTED: The provided evidence does not contain enough information to verify the claim.

CRITICAL RULES:
- Absence of evidence is NOT evidence of contradiction. If a source simply does not mention a claimed spec, classify as UNSUPPORTED, NOT CONTRADICTED.
- Do NOT use 'General Knowledge' as a source name. Use explicit Evidence URLs provided in evidence list.
- Do NOT infer missing specifications.

Output must match FactCheckReport schema.
"""

def fact_checker_node(state: State) -> dict:
    draft = state.get("merged_md", "")
    evidence = state.get("evidence", [])
    research_res = state.get("research_result")

    ev_dumps = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:10]]
    res_dump = research_res.model_dump() if hasattr(research_res, "model_dump") else research_res if research_res else {}

    report = structured_with_retry(
        FactCheckReport,
        [
            SystemMessage(content=FACT_CHECK_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic')}\n"
                    f"Evidence Sources:\n{ev_dumps}\n\n"
                    f"Research Summary:\n{res_dump}\n\n"
                    f"Draft Text:\n{draft[:4500]}"
                )
            ),
        ],
    )
    return {"fact_check_report": report}


# -----------------------------
# Fact Repair Specialist Node
# -----------------------------
FACT_REPAIR_SYSTEM = """You are a targeted Fact Repair Specialist.
Your job is to fix ONLY the specific sentences in the blog draft that contain CONTRADICTED or PARTIALLY_SUPPORTED claims identified by the Fact Checker.

Rules:
1. Do NOT rewrite unaffected sections or change the overall article layout/tone.
2. For CONTRADICTED claims, update numbers, specs, or names to match the exact values in the verified Evidence sources.
3. For UNSUPPORTED/PARTIALLY_SUPPORTED claims, rephrase or remove unverified numbers while keeping the paragraph structure intact.
4. Output the complete corrected markdown text starting with '# '.
"""

def fact_repair_node(state: State) -> dict:
    draft = state.get("merged_md", "")
    report = state.get("fact_check_report")
    evidence = state.get("evidence", [])

    if not report:
        return {"merged_md": draft}

    claims = report.claims if hasattr(report, "claims") else (report.get("claims", []) if isinstance(report, dict) else [])

    problem_claims = [
        c for c in claims
        if (c.classification if hasattr(c, "classification") else (c.get("classification") if isinstance(c, dict) else "")) in ("CONTRADICTED", "PARTIALLY_SUPPORTED")
    ]

    if not problem_claims:
        return {"merged_md": draft}

    ev_dumps = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:10]]
    p_dumps = [c.model_dump() if hasattr(c, "model_dump") else c for c in problem_claims]

    response = invoke_with_retry(
        get_llm(),
        [
            SystemMessage(content=FACT_REPAIR_SYSTEM),
            HumanMessage(
                content=(
                    f"Original Draft:\n{draft[:4500]}\n\n"
                    f"Factual Errors to Repair:\n{p_dumps}\n\n"
                    f"Verified Evidence Sources:\n{ev_dumps}"
                )
            ),
        ],
    )
    repaired_md = response.content.strip()
    return {"merged_md": repaired_md}


# -----------------------------
# SEO Optimization Agent Node
# -----------------------------
SEO_AGENT_SYSTEM = """You are an expert SEO and content strategist.
Analyze the article draft and generate an optimized SEO metadata package.

Required outputs:
1. meta_title: Catchy title (50-60 characters).
2. meta_description: Actionable meta description (120-155 characters).
3. primary_keyword: Main keyword targeted.
4. secondary_keywords: 3-5 relevant secondary keywords.
5. suggested_slug: Clean, URL-friendly slug.
6. target_readability_level: Target reader experience level.
7. faq_items: 2-4 FAQ items with concise answers.
8. seo_score: Overall SEO optimization score (0.0 to 10.0).

Output must match SEOPlan schema.
"""

def seo_agent_node(state: State) -> dict:
    draft = state.get("merged_md", "")
    topic = state.get("topic", "")
    keywords = state.get("keywords", [])

    seo = structured_with_retry(
        SEOPlan,
        [
            SystemMessage(content=SEO_AGENT_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"Target Keywords requested by user: {keywords}\n\n"
                    f"Draft Content:\n{draft[:4000]}"
                )
            ),
        ],
    )
    return {"seo_plan": seo}


# -----------------------------
# Human-in-the-Loop Final Content Approval Node (HITL #2)
# -----------------------------
def final_approval_node(state: State) -> dict:
    """
    HITL #2 Checkpoint: Pauses graph execution after draft, Critic, Fact Checker, and SEO Agent complete.
    Allows human user to review the finished article before final export/image placement.
    """
    final_status = state.get("final_approval_status", ApprovalStatus.PENDING)

    seo_dict = state.get("seo_plan").model_dump() if hasattr(state.get("seo_plan"), "model_dump") else state.get("seo_plan")
    fc_dict = state.get("fact_check_report").model_dump() if hasattr(state.get("fact_check_report"), "model_dump") else state.get("fact_check_report")
    critic_dict = state.get("critic_evaluation").model_dump() if hasattr(state.get("critic_evaluation"), "model_dump") else state.get("critic_evaluation")

    human_response = interrupt({
        "type": "final_content_approval",
        "topic": state.get("topic"),
        "final_approval_status": final_status,
        "merged_md": state.get("merged_md", ""),
        "seo_plan": seo_dict,
        "fact_check_report": fc_dict,
        "critic_evaluation": critic_dict,
    })

    if not isinstance(human_response, dict):
        action = str(human_response)
        feedback = ""
    else:
        action = human_response.get("action", "approve")
        feedback = human_response.get("feedback", "")

    if action == "approve":
        return {
            "final_approval_status": ApprovalStatus.APPROVED,
            "final_human_feedback": None,
        }
    elif action == "request_changes":
        return {
            "final_approval_status": ApprovalStatus.CHANGES_REQUESTED,
            "final_human_feedback": feedback,
        }
    else:
        return {
            "final_approval_status": ApprovalStatus.PENDING,
        }


def route_after_final_approval(state: State) -> str:
    """
    Conditional edge router following HITL #2 final content review.
    - APPROVED -> proceed to decide_images / export
    - CHANGES_REQUESTED -> route to revision node
    """
    status = state.get("final_approval_status")
    if status == ApprovalStatus.APPROVED or status == "approved":
        return "decide_images"
    elif status == ApprovalStatus.CHANGES_REQUESTED or status == "changes_requested":
        return "revision"
    return "final_approval"


# -----------------------------
# Social Media Syndication Node & Visual Mermaid Agent Node
# -----------------------------
SOCIAL_SYNDICATION_SYSTEM = """You are an expert content marketing and social media strategist.

Your task is to take a completed technical blog article and generate multi-channel marketing content.

Produce a JSON response matching the SocialPosts schema:
1. twitter_thread: A 5 to 7 tweet thread summarizing key insights. Each tweet must be <= 280 characters, with hooks, core takeaways, hashtags, and a concluding call-to-action.
2. linkedin_post: A structured, high-engagement LinkedIn post with bold headlines, bullet points, emojis, short paragraphs, and a call-to-action.
3. newsletter_summary: A 200-300 word email digest summary formatted for a technical newsletter audience.

Ensure all outputs strictly conform to the field constraints.
"""


def social_syndication_node(state: State) -> Dict[str, Any]:
    """
    Social Syndication Node: Generates multi-channel derivative marketing content 
    (Twitter/X thread, LinkedIn post, Email newsletter summary) from the final article.
    """
    article = state.get("final") or state.get("md_with_placeholders") or state.get("merged_md") or ""
    topic = state.get("topic", "")
    seo_plan = state.get("seo_plan")
    
    seo_context = ""
    if seo_plan:
        meta_title = getattr(seo_plan, "meta_title", "")
        keyword = getattr(seo_plan, "primary_keyword", "")
        seo_context = f"Meta Title: {meta_title}\nPrimary Keyword: {keyword}\n"

    try:
        social_posts = structured_with_retry(
            SocialPosts,
            [
                SystemMessage(content=SOCIAL_SYNDICATION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n"
                        f"{seo_context}\n"
                        f"Article Content:\n{article[:6000]}"
                    )
                ),
            ],
        )
    except Exception as e:
        logger.warning("Social syndication LLM generation failed, generating fallback social posts: %s", e)
        social_posts = SocialPosts(
            twitter_thread=[
                f"1/7 🚀 Deep dive into {topic}! Here is a thread breaking down the key concepts and architecture.",
                f"2/7 Key Takeaway: Understanding {topic} requires a structured approach to problem solving and design.",
                f"3/7 Implementation matters: Ensure robust error handling, scalability, and clean component isolation.",
                f"4/7 Best Practices: Monitor system metrics and optimize key bottlenecks continuously.",
                f"5/7 Performance & Security: Keep dependencies updated and enforce strict verification.",
                f"6/7 Read the full post for detailed code samples, benchmarks, and diagrams! #{safe_slug(topic)}",
                f"7/7 What are your thoughts on {topic}? Let us know in the comments below! 👇"
            ],
            linkedin_post=(
                f"💡 **Deep Dive into {topic}**\n\n"
                f"In our latest technical article, we break down everything you need to know about {topic}.\n\n"
                f"**Key Highlights:**\n"
                f"• Architecture & design principles\n"
                f"• Real-world implementation patterns\n"
                f"• Performance & scalability considerations\n\n"
                f"👉 Read the full article and let us know your thoughts! #{safe_slug(topic)} #TechBlog #SoftwareEngineering"
            ),
            newsletter_summary=(
                f"Welcome to this week's technical newsletter! Today, we're taking an in-depth look at {topic}. "
                f"Whether you are building complex distributed systems or optimizing individual services, understanding "
                f"these core principles will help you design better software. Check out the full post for code examples, "
                f"architecture diagrams, and implementation details."
            )
        )

    return {"social_posts": social_posts}


MERMAID_GENERATOR_SYSTEM = """You are an expert technical visualizer and system designer.

Your task is to analyze a technical blog article and synthesize 1 to 2 clean, valid Mermaid.js diagrams to be embedded into the markdown.

Rules:
1. Generate valid Mermaid.js diagram code blocks tagged strictly as ```mermaid ... ``` (e.g., flowchart TD, sequenceDiagram, or classDiagram).
2. Ensure syntactical validity: wrap node labels containing spaces, parentheses, or special characters in double quotes (e.g., A["User Request (HTTP)"]). Avoid unescaped special characters.
3. Choose 1-2 key sections (H2/H3) in the article where a Mermaid diagram visually clarifies the process, data flow, or system architecture.
4. Return the updated full markdown article text with the Mermaid diagram code block(s) seamlessly inserted right after the relevant H2/H3 header.
"""


def visual_agent_node(state: State) -> Dict[str, Any]:
    """
    Visual Agent Node: Generates visual images and synthesizes valid Mermaid.js diagrams 
    embedded directly into the article markdown.
    """
    # 1. First ensure images are generated and placed if image_specs exist
    image_res = generate_and_place_images(state)
    article = image_res.get("final") or state.get("md_with_placeholders") or state.get("merged_md") or ""

    # 2. Check if mermaid blocks already exist in the article
    if "```mermaid" in article:
        return {"final": article, "md_with_placeholders": article}

    topic = state.get("topic") or "System Architecture"
    
    try:
        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content=MERMAID_GENERATOR_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\n"
                        f"Please synthesize 1 to 2 clean, valid Mermaid.js diagrams and insert them into appropriate H2/H3 sections of the markdown below.\n\n"
                        f"Return ONLY the complete updated Markdown text:\n\n"
                        f"{article}"
                    )
                ),
            ],
        )
        updated_md = response.content.strip()
        if updated_md.startswith("```markdown"):
            updated_md = re.sub(r"^```markdown\s*", "", updated_md)
            updated_md = re.sub(r"\s*```$", "", updated_md)
        elif updated_md.startswith("```") and not updated_md.startswith("```mermaid"):
            updated_md = re.sub(r"^```\w*\s*", "", updated_md)
            updated_md = re.sub(r"\s*```$", "", updated_md)

        if "```mermaid" in updated_md:
            return {"final": updated_md, "md_with_placeholders": updated_md}
    except Exception as e:
        logger.warning("LLM Mermaid diagram generation failed, applying default Mermaid diagram: %s", e)

    # 3. Fallback: Generate a clean default Mermaid flowchart diagram and insert after the first H2 header
    topic_clean = topic.replace('"', "'")
    default_mermaid = (
        "\n\n```mermaid\n"
        "flowchart TD\n"
        f'    A["Client Input / Topic: {topic_clean}"] --> B["Processing Pipeline"]\n'
        '    B --> C["Core Logic & Operations"]\n'
        '    C --> D["Final System Output"]\n'
        "```\n\n"
    )

    lines = article.split("\n")
    inserted = False
    for i, line in enumerate(lines):
        if line.startswith("## ") and not inserted:
            lines.insert(i + 1, default_mermaid)
            inserted = True
            break

    if not inserted:
        lines.append(default_mermaid)

    final_md = "\n".join(lines)
    return {"final": final_md, "md_with_placeholders": final_md}


