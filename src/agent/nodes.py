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
    Task,
    Plan,
    EvidenceItem,
    RouterDecision,
    EvidencePack,
    GlobalImagePlan,
    ApprovalStatus,
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


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/architecture visual).
- For each image, specify section_title (the H2 section heading where the image belongs, e.g. "WebSocket Basics").
- Use placeholders [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] sequentially.
- If no images needed: return images=[].
Return strictly GlobalImagePlan.
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
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Ensure outputs directory exists (anchored to the project root, not CWD)
    outputs_dir = ensure_dir(OUTPUTS_DIR)

    filename_str = f"{safe_slug(plan.blog_title)}.md"
    out_file = outputs_dir / filename_str

    if not image_specs:
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
