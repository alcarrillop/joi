from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agent.graph.edges import (
    select_workflow,
    should_summarize_conversation,
)
from agent.graph.memory_nodes import memory_extraction_node, memory_injection_node
from agent.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    image_node,
    learning_stats_update_node,
    progress_query_node,
    router_node,
    summarize_conversation_node,
)
from agent.graph.state import AICompanionState


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes
    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("learning_stats_update_node", learning_stats_update_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("image_node", image_node)
    graph_builder.add_node("audio_node", audio_node)
    graph_builder.add_node("progress_query_node", progress_query_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # Define the flow
    # First extract memories from user message
    graph_builder.add_edge(START, "memory_extraction_node")

    # Then update learning stats from user message
    graph_builder.add_edge("memory_extraction_node", "learning_stats_update_node")

    # Then determine response type
    graph_builder.add_edge("learning_stats_update_node", "router_node")

    # Then inject both context and memories
    graph_builder.add_edge("router_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")

    # Then proceed to appropriate response node
    graph_builder.add_conditional_edges("memory_injection_node", select_workflow)

    # Check for summarization after any response
    graph_builder.add_conditional_edges("conversation_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("image_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("audio_node", should_summarize_conversation)
    graph_builder.add_conditional_edges("progress_query_node", should_summarize_conversation)
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
graph = create_workflow_graph().compile()
