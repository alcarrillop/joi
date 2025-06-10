"""
Memory-related graph nodes for the conversational AI agent.

This module handles memory extraction and injection operations to maintain
context and personalization across conversations.
"""

import logging

from agent.graph.state import AICompanionState
from agent.modules.memory.long_term.memory_manager import get_memory_manager

# Configure logger for memory operations
workflow_logger = logging.getLogger("workflow")


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message.

    Args:
        state: Current conversation state containing messages and user context

    Returns:
        Empty dict as this node only performs side effects (memory storage)
    """
    user_id = state.get("user_id")
    session_id = state.get("session_id")

    workflow_logger.info(f"[MEMORY_EXTRACT] Processing memory extraction for user {user_id}")

    # Validate required data
    if not state.get("messages"):
        workflow_logger.warning(f"[MEMORY_EXTRACT] No messages to process for user {user_id}")
        return {}

    if not user_id or not session_id:
        workflow_logger.warning("[MEMORY_EXTRACT] Missing user context for memory extraction")
        return {}

    try:
        memory_manager = get_memory_manager()
        last_message = state["messages"][-1]

        # Only extract from human messages
        if last_message.type != "human":
            workflow_logger.debug(f"[MEMORY_EXTRACT] Skipping non-human message for user {user_id}")
            return {}

        workflow_logger.debug(
            f"[MEMORY_EXTRACT] Analyzing message for user {user_id}: '{last_message.content[:100]}...'"
        )

        await memory_manager.extract_and_store_memories(message=last_message, user_id=user_id, session_id=session_id)

        workflow_logger.info(f"[MEMORY_EXTRACT] Successfully completed memory extraction for user {user_id}")

    except Exception as e:
        workflow_logger.error(f"[MEMORY_EXTRACT] Failed to extract memories for user {user_id}: {e}")
        # Continue execution even if memory extraction fails

    return {}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the conversation context.

    Args:
        state: Current conversation state containing messages and user context

    Returns:
        Dict containing memory_context for use in prompt generation
    """
    user_id = state.get("user_id")
    workflow_logger.info(f"[MEMORY_INJECT] Injecting memories for user {user_id}")

    # Early return if no user ID available
    if not user_id or user_id == "unknown":
        workflow_logger.warning("[MEMORY_INJECT] No valid user_id available, skipping memory injection")
        return {"memory_context": ""}

    try:
        memory_manager = get_memory_manager()

        # Build context from recent messages (only human messages for relevance)
        recent_messages = state.get("messages", [])[-3:]
        human_messages = [msg for msg in recent_messages if msg.type == "human"]

        if not human_messages:
            workflow_logger.debug(f"[MEMORY_INJECT] No recent human messages for user {user_id}")
            return {"memory_context": ""}

        recent_context = " ".join([msg.content for msg in human_messages])
        workflow_logger.debug(f"[MEMORY_INJECT] Recent context for user {user_id}: '{recent_context[:100]}...'")

        # Retrieve and format relevant memories
        memories = memory_manager.get_relevant_memories(recent_context, user_id=user_id)
        memory_context = memory_manager.format_memories_for_prompt(memories)

        workflow_logger.info(f"[MEMORY_INJECT] Found {len(memories)} relevant memories for user {user_id}")

        if memory_context:
            workflow_logger.debug(f"[MEMORY_INJECT] Memory context for user {user_id}: '{memory_context[:200]}...'")
        else:
            workflow_logger.debug(f"[MEMORY_INJECT] No memory context generated for user {user_id}")

        return {"memory_context": memory_context}

    except Exception as e:
        workflow_logger.error(f"[MEMORY_INJECT] Failed to inject memories for user {user_id}: {e}")
        # Return empty context on error to prevent conversation breakdown
        return {"memory_context": ""}
