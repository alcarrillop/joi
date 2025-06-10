"""
Graph nodes for the conversational AI agent.

This module contains the main node functions that handle different stages
of conversation processing, including routing, memory management, and response generation.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from agent.graph.state import AICompanionState
from agent.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from agent.graph.utils.helpers import (
    get_chat_model,
    get_text_to_speech_module,
    get_user_level_info,
)
from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

# Memory manager import removed as memory operations are handled in separate memory_nodes
from agent.modules.schedules.context_generation import ScheduleContextGenerator
from agent.settings import get_settings

# Configure logger for workflow monitoring
workflow_logger = logging.getLogger("workflow")
workflow_logger.setLevel(logging.INFO)

# Add console handler if not already present
if not workflow_logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    workflow_logger.addHandler(console_handler)


async def router_node(state: AICompanionState):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[ROUTER] Processing messages for user {user_id}")

    try:
        chain = get_router_chain()
        settings = get_settings()
        messages_to_analyze = state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]

        # Log the messages being analyzed
        workflow_logger.debug(f"[ROUTER] Analyzing {len(messages_to_analyze)} messages for user {user_id}")
        for i, msg in enumerate(messages_to_analyze):
            workflow_logger.debug(f"[ROUTER] Message {i}: {msg.type} - {msg.content[:100]}...")

        response = await chain.ainvoke({"messages": messages_to_analyze})

        workflow_logger.info(f"[ROUTER] Selected workflow: {response.response_type} for user {user_id}")
        return {"workflow": response.response_type}

    except Exception as e:
        # Handle API failures gracefully by defaulting to conversation
        workflow_logger.warning(f"[ROUTER] API error for user {user_id}, defaulting to conversation: {e}")
        return {"workflow": "conversation"}


def context_injection_node(state: AICompanionState):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[CONTEXT] Injecting context for user {user_id}")

    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
        workflow_logger.info(f"[CONTEXT] New activity detected for user {user_id}: {schedule_context}")
    else:
        apply_activity = False
        workflow_logger.debug(f"[CONTEXT] Same activity for user {user_id}: {schedule_context}")

    return {"apply_activity": apply_activity, "current_activity": schedule_context}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[CONVERSATION] Processing conversation for user {user_id}")

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    workflow_logger.debug(f"[CONVERSATION] Memory context length for user {user_id}: {len(memory_context)} chars")

    # Progress queries are now handled by progress_query_node
    # This node focuses on normal conversation

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    workflow_logger.info(f"[CONVERSATION] Generated response for user {user_id}: {len(response)} chars")
    return {"messages": AIMessage(content=response)}


async def image_node(state: AICompanionState, config: RunnableConfig):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[IMAGE] Processing image workflow for user {user_id}")

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    # ===== IMAGE GENERATION DISABLED =====
    # Image generation has been disabled - system can only describe images, not generate them
    # The code below is commented out but preserved for potential future use

    # text_to_image_module = get_text_to_image_module()
    # scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    # os.makedirs("generated_images", exist_ok=True)
    # img_path = f"generated_images/image_{str(uuid4())}.png"
    # await text_to_image_module.generate_image(scenario.image_prompt, img_path)
    # workflow_logger.info(f"[IMAGE] Generated image for user {user_id}: {img_path}")
    # scenario_message = HumanMessage(content=f"<image attached by Joi generated from prompt: {scenario.image_prompt}>")
    # updated_messages = state["messages"] + [scenario_message]

    # ===== FALLBACK TO CONVERSATION =====
    # Instead of generating images, respond conversationally about images
    workflow_logger.info(f"[IMAGE] Image generation disabled - falling back to conversation for user {user_id}")

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response)}


async def audio_node(state: AICompanionState, config: RunnableConfig):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[AUDIO] Processing audio workflow for user {user_id}")

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    workflow_logger.info(f"[AUDIO] Generated audio for user {user_id}: {len(output_audio)} bytes")

    return {"messages": AIMessage(content=response), "audio_buffer": output_audio}


async def summarize_conversation_node(state: AICompanionState):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[SUMMARY] Summarizing conversation for user {user_id}")

    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Joi and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        workflow_logger.info(f"[SUMMARY] Extending existing summary for user {user_id}")
    else:
        summary_message = (
            "Create a summary of the conversation above between Joi and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Joi and the user:"
        )
        workflow_logger.info(f"[SUMMARY] Creating new summary for user {user_id}")

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    settings = get_settings()
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]

    workflow_logger.info(f"[SUMMARY] Created summary for user {user_id}, removing {len(delete_messages)} old messages")

    return {"summary": response.content, "messages": delete_messages}


async def learning_stats_update_node(state: AICompanionState):
    """Update learning statistics based on user's message."""
    user_id = state.get("user_id")
    session_id = state.get("session_id")

    workflow_logger.info(f"[LEARNING_STATS] Updating learning stats for user {user_id}")

    if not state["messages"]:
        workflow_logger.warning(f"[LEARNING_STATS] No messages to process for user {user_id}")
        return {}

    if not user_id or not session_id:
        workflow_logger.warning("[LEARNING_STATS] Missing user context for learning stats update")
        return {}

    # Only process user messages (not agent responses)
    last_message = state["messages"][-1]
    if last_message.type != "human":
        workflow_logger.debug(f"[LEARNING_STATS] Skipping agent message for user {user_id}")
        return {}

    learning_manager = get_learning_stats_manager()

    # Update learning stats based on the user's message
    try:
        stats_update = await learning_manager.update_learning_stats(
            user_id=user_id, session_id=session_id, message_text=last_message.content
        )

        workflow_logger.info(
            f"[LEARNING_STATS] Updated stats for user {user_id}: {stats_update.get('vocabulary_analysis', {}).get('new_words_found', 0)} new words"
        )

        return {"learning_stats_update": stats_update}

    except Exception as e:
        workflow_logger.error(f"[LEARNING_STATS] Failed to update learning stats for user {user_id}: {e}")
        return {"learning_stats_error": str(e)}


async def progress_query_node(state: AICompanionState):
    """Handle user progress and learning statistics queries."""
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[PROGRESS_QUERY] Processing progress query for user {user_id}")

    if not user_id or user_id == "unknown":
        workflow_logger.warning("[PROGRESS_QUERY] Missing user ID for progress query")
        return {
            "messages": AIMessage(
                content="I need to know who you are to check your progress. Please share your information first."
            )
        }

    # Get the last human message to analyze query type
    last_human_message = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_message = msg
            break

    if not last_human_message:
        workflow_logger.warning(f"[PROGRESS_QUERY] No human message found for user {user_id}")
        return {
            "messages": AIMessage(
                content="I couldn't find your question. Could you ask again about your English progress?"
            )
        }

    message_lower = last_human_message.content.lower()

    # Don't trigger for image analysis messages
    if "[image analysis:" in message_lower:
        workflow_logger.debug(f"[PROGRESS_QUERY] Skipping image analysis message for user {user_id}")
        # Fallback to conversation
        return await conversation_node(state, None)

    try:
        # Get comprehensive progress information
        level_info = await get_user_level_info(user_id)
        workflow_logger.info(f"[PROGRESS_QUERY] Generated progress response for user {user_id}")
        return {"messages": AIMessage(content=level_info)}

    except Exception as e:
        workflow_logger.error(f"[PROGRESS_QUERY] Failed to get progress info for user {user_id}: {e}")
        # Fallback response
        error_response = (
            "I'm having trouble accessing your progress data right now. "
            "Please try asking again in a moment, or feel free to continue our conversation!"
        )
        return {"messages": AIMessage(content=error_response)}
