"""
Graph nodes for the conversational AI agent.

This module contains the main node functions that handle different stages
of conversation processing, including routing, memory management, and response generation.
"""

import logging
import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from agent.core.prompts import (
    CONVERSATION_PROMPT,
    LANGUAGE_INSTRUCTOR_PROMPT,
    ROUTER_PROMPT,
    SENTENCE_IMPROVER_PROMPT,
    SUMMARY_PROMPT,
)
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
from agent.modules.learning.learning_stats_manager import get_learning_stats_manager
from agent.modules.memory.long_term.memory_manager import get_memory_manager
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

    chain = get_router_chain()
    settings = get_settings()
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

    settings = get_settings()
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
        workflow_logger.warning("[MEMORY_EXTRACT] Missing user context for memory extraction")
        return {}

    last_message = state["messages"][-1]
    workflow_logger.debug(f"[MEMORY_EXTRACT] Analyzing message for user {user_id}: '{last_message.content[:100]}...'")

    await memory_manager.extract_and_store_memories(last_message, user_id=user_id, session_id=session_id)

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
        workflow_logger.warning("[MEMORY_INJECT] No user_id available, skipping memory injection")
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


async def learning_stats_update_node(state: AICompanionState):
    """Update learning statistics based on user's message."""
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")

    workflow_logger.info(f"[LEARNING_STATS] Updating learning stats for user {user_id}")

    if not state["messages"]:
        workflow_logger.warning(f"[LEARNING_STATS] No messages to process for user {user_id}")
        return {}

    # Get user_id and session_id from state
    user_id = state.get("user_id")
    session_id = state.get("session_id")

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


def router(state: AICompanionState) -> AICompanionState:
    """Route messages based on conversation context and recent interactions."""
    workflow_logger.info("Router node: Analyzing conversation flow")

    # Get conversation context based on recent messages
    settings = get_settings()
    messages_to_analyze = state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]

    # Build router prompt with recent messages
    messages_context = "\n".join([f"{msg.type}: {msg.content}" for msg in messages_to_analyze])

    # Create router LLM instance
    router_llm = ChatGroq(
        model=settings.SMALL_TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_retries=2,
    )

    # Analyze conversation flow
    router_prompt = ROUTER_PROMPT.format(messages=messages_context)
    chain = router_llm
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})

    # Extract decision from response
    response_text = response.content.lower()

    # Default routing logic based on conversation patterns
    if "practice" in response_text or "conversation" in response_text:
        next_node = "conversation"
    elif "learn" in response_text or "teach" in response_text or "explain" in response_text:
        next_node = "language_instructor"
    elif "improve" in response_text or "correct" in response_text or "better" in response_text:
        next_node = "sentence_improver"
    else:
        # Default to conversation for natural chat flow
        next_node = "conversation"

    workflow_logger.info(f"Router decision: {next_node}")

    # Update state with routing decision
    state["next"] = next_node
    return state


async def memory_retrieval(state: AICompanionState) -> AICompanionState:
    """Retrieve relevant memories based on the current conversation context."""
    workflow_logger.info("Memory retrieval node: Fetching relevant context")

    # Get the last message for context
    last_message = state["messages"][-1] if state["messages"] else None
    if not last_message or last_message.type != "human":
        workflow_logger.info("No human message found for memory retrieval")
        state["relevant_memories"] = []
        return state

    # Get user ID from metadata or set default
    user_id = state.get("user_id", "unknown")

    # Initialize memory manager
    memory_manager = get_memory_manager()

    # Search for relevant memories
    relevant_memories = memory_manager.get_relevant_memories(last_message.content, user_id)

    # Format memories for context
    memory_context = memory_manager.format_memories_for_prompt(relevant_memories)

    # Update state with retrieved memories
    state["relevant_memories"] = relevant_memories
    state["memory_context"] = memory_context

    workflow_logger.info(f"Retrieved {len(relevant_memories)} relevant memories for user {user_id}")

    return state


async def conversation(state: AICompanionState) -> AICompanionState:
    """Handle general conversation with memory-aware responses."""
    workflow_logger.info("Conversation node: Generating conversational response")

    # Get user context
    user_id = state.get("user_id", "unknown")
    memory_context = state.get("memory_context", "")

    # Create conversation LLM
    settings = get_settings()
    conversation_llm = ChatGroq(
        model=settings.TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
        max_retries=2,
    )

    # Build conversation prompt with memory context
    conversation_prompt = CONVERSATION_PROMPT.format(
        memory_context=memory_context, recent_conversation=state["messages"][-3:]
    )

    # Generate response
    chain = conversation_llm
    response = await chain.ainvoke([HumanMessage(content=conversation_prompt)])

    # Create AI response message
    ai_message = AIMessage(content=response.content)

    # Update conversation state
    state["messages"].append(ai_message)

    # Store memory asynchronously
    try:
        memory_manager = get_memory_manager()
        last_human_message = state["messages"][-2] if len(state["messages"]) >= 2 else None
        if last_human_message and last_human_message.type == "human":
            session_id = state.get("session_id", str(uuid4()))
            await memory_manager.extract_and_store_memories(last_human_message, user_id, session_id)
    except Exception as e:
        workflow_logger.warning(f"Failed to store memory: {e}")

    # Update learning statistics
    try:
        learning_manager = get_learning_stats_manager()
        last_human_message = state["messages"][-2] if len(state["messages"]) >= 2 else None
        if last_human_message and last_human_message.type == "human":
            session_id = state.get("session_id", str(uuid4()))
            await learning_manager.update_learning_stats(user_id, session_id, last_human_message.content)
    except Exception as e:
        workflow_logger.warning(f"Failed to update learning stats: {e}")

    workflow_logger.info("Conversation response generated successfully")
    return state


async def language_instructor(state: AICompanionState) -> AICompanionState:
    """Provide language instruction and explanations."""
    workflow_logger.info("Language instructor node: Providing educational content")

    # Get user context
    memory_context = state.get("memory_context", "")

    # Create instructor LLM
    settings = get_settings()
    instructor_llm = ChatGroq(
        model=settings.TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.3,
        max_retries=2,
    )

    # Build instruction prompt
    instruction_prompt = LANGUAGE_INSTRUCTOR_PROMPT.format(
        memory_context=memory_context, recent_conversation=state["messages"][-3:]
    )

    # Generate educational response
    chain = instructor_llm
    response = await chain.ainvoke([HumanMessage(content=instruction_prompt)])

    # Create AI response message
    ai_message = AIMessage(content=response.content)

    # Update state
    state["messages"].append(ai_message)

    workflow_logger.info("Language instruction provided successfully")
    return state


async def sentence_improver(state: AICompanionState) -> AICompanionState:
    """Improve user's sentences and provide corrections."""
    workflow_logger.info("Sentence improver node: Analyzing and improving user input")

    # Get the last human message for improvement
    last_human_message = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_message = msg
            break

    if not last_human_message:
        workflow_logger.warning("No human message found for sentence improvement")
        # Fallback to conversation
        return await conversation(state)

    # Create improver LLM
    settings = get_settings()
    improver_llm = ChatGroq(
        model=settings.SMALL_TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.2,
        max_retries=2,
    )

    # Build improvement prompt
    improvement_prompt = SENTENCE_IMPROVER_PROMPT.format(user_message=last_human_message.content)

    # Generate improved version
    chain = improver_llm
    response = await chain.ainvoke([HumanMessage(content=improvement_prompt)])

    # Create AI response message
    ai_message = AIMessage(content=response.content)

    # Update state
    state["messages"].append(ai_message)

    workflow_logger.info("Sentence improvement provided successfully")
    return state


async def summarize_conversation(state: AICompanionState) -> AICompanionState:
    """Summarize long conversations to manage context length."""
    workflow_logger.info("Summarize node: Condensing conversation history")

    # Check if we need to summarize
    settings = get_settings()
    total_messages = len(state["messages"])

    if total_messages < settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        workflow_logger.info("Conversation not long enough for summarization")
        return state

    # Create summarizer LLM
    summarizer_llm = ChatGroq(
        model=settings.SMALL_TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        max_retries=2,
    )

    # Get messages to summarize (exclude recent ones)
    messages_to_summarize = state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]
    recent_messages = state["messages"][-settings.TOTAL_MESSAGES_AFTER_SUMMARY :]

    # Build summary prompt
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages_to_summarize])

    summary_prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

    # Generate summary
    chain = summarizer_llm
    response = await chain.ainvoke([HumanMessage(content=summary_prompt)])

    # Create summary message
    summary_message = AIMessage(content=f"[Conversation Summary: {response.content}]")

    # Update state: replace old messages with summary + recent messages
    new_messages = [summary_message] + recent_messages
    state["messages"] = new_messages

    # Also remove the old messages from checkpointer if available
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    if delete_messages:
        state["messages"].extend(delete_messages)

    workflow_logger.info(f"Conversation summarized: {total_messages} → {len(new_messages)} messages")

    return state
