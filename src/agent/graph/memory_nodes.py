"""Memory-related graph nodes."""

import logging

from agent.graph.state import AICompanionState
from agent.modules.memory.long_term.memory_manager import get_memory_manager

workflow_logger = logging.getLogger("workflow")


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")

    workflow_logger.info(f"[MEMORY_EXTRACT] Processing memory extraction for user {user_id}")

    if not state["messages"]:
        workflow_logger.warning(f"[MEMORY_EXTRACT] No messages to process for user {user_id}")
        return {}

    memory_manager = get_memory_manager()

    if not user_id or not session_id:
        workflow_logger.warning("[MEMORY_EXTRACT] Missing user context for memory extraction")
        return {}

    last_message = state["messages"][-1]
    workflow_logger.debug(
        f"[MEMORY_EXTRACT] Analyzing message for user {user_id}: '{last_message.content[:100]}...'"
    )

    await memory_manager.extract_and_store_memories(last_message, user_id=user_id, session_id=session_id)

    workflow_logger.info(f"[MEMORY_EXTRACT] Completed memory extraction for user {user_id}")
    return {}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[MEMORY_INJECT] Injecting memories for user {user_id}")

    memory_manager = get_memory_manager()

    if not user_id:
        workflow_logger.warning("[MEMORY_INJECT] No user_id available, skipping memory injection")
        return {"memory_context": ""}

    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    workflow_logger.debug(
        f"[MEMORY_INJECT] Recent context for user {user_id}: '{recent_context[:100]}...'"
    )

    memories = memory_manager.get_relevant_memories(recent_context, user_id=user_id)
    memory_context = memory_manager.format_memories_for_prompt(memories)

    workflow_logger.info(f"[MEMORY_INJECT] Found {len(memories)} relevant memories for user {user_id}")
    workflow_logger.debug(
        f"[MEMORY_INJECT] Memory context for user {user_id}: '{memory_context[:200]}...'"
    )

    return {"memory_context": memory_context}
