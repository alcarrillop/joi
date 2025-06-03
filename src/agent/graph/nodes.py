import os
import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from agent.graph.state import AICompanionState
from agent.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from agent.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from agent.modules.memory.long_term.memory_manager import get_memory_manager
from agent.modules.schedules.context_generation import ScheduleContextGenerator
from agent.settings import settings

# Configure logger for workflow monitoring
workflow_logger = logging.getLogger("workflow")
workflow_logger.setLevel(logging.INFO)

# Add console handler if not already present
if not workflow_logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    workflow_logger.addHandler(console_handler)


async def router_node(state: AICompanionState):
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[ROUTER] Processing messages for user {user_id}")
    
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    
    workflow_logger.info(f"[ROUTER] Selected workflow: {response.response_type} for user {user_id}")
    return {"workflow": response.response_type}


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
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)
    
    workflow_logger.info(f"[IMAGE] Generated image for user {user_id}: {img_path}")

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(content=f"<image attached by Joi generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


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

    return {"messages": response, "audio_buffer": output_audio}


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

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    
    workflow_logger.info(f"[SUMMARY] Created summary for user {user_id}, removing {len(delete_messages)} old messages")
    
    return {"summary": response.content, "messages": delete_messages}


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")
    
    workflow_logger.info(f"[MEMORY_EXTRACT] Processing memory extraction for user {user_id}")
    
    if not state["messages"]:
        workflow_logger.warning(f"[MEMORY_EXTRACT] No messages to process for user {user_id}")
        return {}

    memory_manager = get_memory_manager()
    
    # Get user_id and session_id from state
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    
    if not user_id or not session_id:
        # Skip memory extraction if we don't have user context
        workflow_logger.warning(f"[MEMORY_EXTRACT] Missing user context for memory extraction")
        return {}
    
    last_message = state["messages"][-1]
    workflow_logger.debug(f"[MEMORY_EXTRACT] Analyzing message for user {user_id}: '{last_message.content[:100]}...'")
    
    await memory_manager.extract_and_store_memories(
        last_message, 
        user_id=user_id, 
        session_id=session_id
    )
    
    workflow_logger.info(f"[MEMORY_EXTRACT] Completed memory extraction for user {user_id}")
    return {}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    user_id = state.get("user_id", "unknown")
    workflow_logger.info(f"[MEMORY_INJECT] Injecting memories for user {user_id}")
    
    memory_manager = get_memory_manager()

    # Get user_id from state
    user_id = state.get("user_id")
    
    if not user_id:
        # Skip memory injection if we don't have user context
        workflow_logger.warning(f"[MEMORY_INJECT] No user_id available, skipping memory injection")
        return {"memory_context": ""}

    # Get relevant memories based on recent conversation for this user
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    workflow_logger.debug(f"[MEMORY_INJECT] Recent context for user {user_id}: '{recent_context[:100]}...'")
    
    memories = memory_manager.get_relevant_memories(recent_context, user_id=user_id)

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)
    
    workflow_logger.info(f"[MEMORY_INJECT] Found {len(memories)} relevant memories for user {user_id}")
    workflow_logger.debug(f"[MEMORY_INJECT] Memory context for user {user_id}: '{memory_context[:200]}...'")

    return {"memory_context": memory_context}
