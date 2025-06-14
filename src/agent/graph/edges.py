from langgraph.graph import END
from typing_extensions import Literal

from agent.graph.state import AICompanionState
from agent.settings import get_settings


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]
    settings = get_settings()

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def select_workflow(
    state: AICompanionState,
) -> Literal["conversation_node", "image_node", "audio_node", "progress_query_node"]:
    workflow = state["workflow"]

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    elif workflow == "progress_query":
        return "progress_query_node"

    else:
        return "conversation_node"
