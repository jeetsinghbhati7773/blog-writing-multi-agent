from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.schemas import State
import src.agent.nodes as nodes

# -----------------------------
# 1) Build Reducer Subgraph
# -----------------------------
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", lambda s: nodes.merge_content(s))
reducer_graph.add_node("decide_images", lambda s: nodes.decide_images(s))
reducer_graph.add_node("generate_and_place_images", lambda s: nodes.generate_and_place_images(s))

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
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
g.add_node("orchestrator", lambda s: nodes.orchestrator_node(s))
g.add_node("plan_approval", lambda s: nodes.plan_approval_node(s))
g.add_node("worker", lambda payload: nodes.worker_node(payload))
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", lambda s: nodes.route_next(s), {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_edge("orchestrator", "plan_approval")

g.add_conditional_edges(
    "plan_approval",
    lambda s: nodes.route_after_approval(s),
    ["worker", "orchestrator", "plan_approval"],
)

g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile(checkpointer=checkpointer)


