from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.schemas import State
import src.agent.nodes as nodes

# -----------------------------
# 1) Build Reducer & Quality Subgraph
# -----------------------------
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", lambda s: nodes.merge_content(s))
reducer_graph.add_node("fact_checker", lambda s: nodes.fact_checker_node(s))
reducer_graph.add_node("fact_repair", lambda s: nodes.fact_repair_node(s))
reducer_graph.add_node("critic", lambda s: nodes.critic_node(s))
reducer_graph.add_node("revision", lambda s: nodes.revision_node(s))
reducer_graph.add_node("seo_agent", lambda s: nodes.seo_agent_node(s))
reducer_graph.add_node("final_approval", lambda s: nodes.final_approval_node(s))
reducer_graph.add_node("decide_images", lambda s: nodes.decide_images(s))
reducer_graph.add_node("generate_and_place_images", lambda s: nodes.generate_and_place_images(s))

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "fact_checker")
reducer_graph.add_edge("fact_checker", "fact_repair")
reducer_graph.add_edge("fact_repair", "critic")

reducer_graph.add_conditional_edges(
    "critic",
    lambda s: nodes.route_after_critic(s),
    {"decide_images": "seo_agent", "revision": "revision"},
)

reducer_graph.add_edge("revision", "critic")
reducer_graph.add_edge("seo_agent", "final_approval")

reducer_graph.add_conditional_edges(
    "final_approval",
    lambda s: nodes.route_after_final_approval(s),
    {"decide_images": "decide_images", "revision": "revision", "final_approval": "final_approval"},
)

reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


def reducer_node(state: State) -> dict:
    """
    Wrapper for Reducer Subgraph to avoid duplicating `sections` list
    via state reducer (operator.add) when subgraph outputs state back to main graph.
    """
    res = reducer_subgraph.invoke(state)
    return {
        "merged_md": res.get("merged_md", ""),
        "fact_check_report": res.get("fact_check_report"),
        "critic_evaluation": res.get("critic_evaluation"),
        "critic_retry_count": res.get("critic_retry_count", 0),
        "seo_plan": res.get("seo_plan"),
        "final_approval_status": res.get("final_approval_status"),
        "md_with_placeholders": res.get("md_with_placeholders", ""),
        "image_specs": res.get("image_specs", []),
        "final": res.get("final", ""),
    }


# -----------------------------
# 2) Build Main Workflow Graph with Checkpointer
# -----------------------------
checkpointer = MemorySaver()

g = StateGraph(State)
g.add_node("router", lambda s: nodes.router_node(s))
g.add_node("research", lambda s: nodes.research_node(s))
g.add_node("source_filter", lambda s: nodes.source_filter_node(s))
g.add_node("contradiction_detector", lambda s: nodes.contradiction_detector_node(s))
g.add_node("structured_research", lambda s: nodes.structured_research_node(s))
g.add_node("orchestrator", lambda s: nodes.orchestrator_node(s))
g.add_node("outline_validator", lambda s: nodes.outline_validator_node(s))
g.add_node("plan_approval", lambda s: nodes.plan_approval_node(s))
g.add_node("worker", lambda payload: nodes.worker_node(payload))
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", lambda s: nodes.route_next(s), {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "source_filter")
g.add_edge("source_filter", "contradiction_detector")
g.add_edge("contradiction_detector", "structured_research")
g.add_edge("structured_research", "orchestrator")
g.add_edge("orchestrator", "outline_validator")
g.add_edge("outline_validator", "plan_approval")

g.add_conditional_edges(
    "plan_approval",
    lambda s: nodes.route_after_approval(s),
    ["worker", "orchestrator", "plan_approval"],
)

g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile(checkpointer=checkpointer)
