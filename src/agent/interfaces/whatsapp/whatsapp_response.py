import logging
import os
import time
from io import BytesIO
from time import perf_counter
from typing import Dict

import httpx
from fastapi import APIRouter, Request, Response
from langchain_core.messages import HumanMessage

from agent.core.database import (
    get_checkpointer,
    get_or_create_session,
    get_or_create_user,
    log_message,
)
from agent.graph import graph_builder
from agent.modules.image import ImageToText
from agent.modules.speech import SpeechToText, TextToSpeech

logger = logging.getLogger(__name__)

# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

# Router for WhatsApp response
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Message deduplication - simple in-memory cache
# In production, use Redis or database for persistence
_processed_messages: Dict[str, float] = {}
DEDUP_CACHE_TTL = 300  # 5 minutes


def _is_image_message(message: Dict) -> bool:
    """Return True if the incoming message contains image data."""
    if message.get("type") == "image":
        return True
    if message.get("type") == "document":
        mime_type = message.get("document", {}).get("mime_type", "")
        return mime_type.startswith("image/")
    return "image" in message


def _is_message_already_processed(message_id: str) -> bool:
    """Check if message was already processed and clean old entries."""
    current_time = time.time()

    # Clean old entries
    expired_keys = [k for k, v in _processed_messages.items() if current_time - v > DEDUP_CACHE_TTL]
    for key in expired_keys:
        del _processed_messages[key]

    # Check if already processed
    if message_id in _processed_messages:
        logger.info(f"Skipping duplicate message: {message_id}")
        return True

    # Mark as processed
    _processed_messages[message_id] = current_time
    return False


# Validate required WhatsApp configuration
def _validate_env() -> None:
    missing = [
        var for var in ["WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_VERIFY_TOKEN"] if not os.getenv(var)
    ]
    if missing:
        raise RuntimeError(f"Missing WhatsApp environment variables: {', '.join(missing)}")


_validate_env()

# Test environment detection
IS_TEST_ENV = os.getenv("TESTING") == "true" or not WHATSAPP_TOKEN


@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"])
async def whatsapp_handler(request: Request) -> Response:
    """Handle webhook events from WhatsApp Cloud API.

    **Example payload** (truncated):
    ```json
    {
      "entry": [{"changes": [{"value": {"messages": [{"from": "123", "text": {"body": "hi"}}]}}]}]
    }
    ```
    """
    start = perf_counter()

    if request.method == "GET":
        params = request.query_params
        if params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN"):
            return Response(content=params.get("hub.challenge"), status_code=200)
        return Response(content="Verification token mismatch", status_code=403)

    try:
        data = await request.json()
        change_value = data["entry"][0]["changes"][0]["value"]
        if "messages" in change_value:
            message = change_value["messages"][0]
            from_number = message["from"]
            message_id = message.get("id")

            # Skip if this message was already processed (deduplication)
            if message_id and _is_message_already_processed(message_id):
                return Response(content="Duplicate message ignored", status_code=200)

            # Extract user name from WhatsApp contacts if available
            user_name = None
            if "contacts" in change_value and len(change_value["contacts"]) > 0:
                contact = change_value["contacts"][0]
                if "profile" in contact and "name" in contact["profile"]:
                    user_name = contact["profile"]["name"]
                    logger.info(f"Extracted WhatsApp name: {user_name} for {from_number}")

            # Get or create user and session
            user_id = await get_or_create_user(from_number, user_name)
            session_id = await get_or_create_session(user_id)

            # Get user message and handle different message types
            content = ""
            user_message_type = "text"  # Default type

            # Debug: Log the incoming message structure
            logger.info(f"[DEBUG] Message structure for {from_number}: {message}")
            logger.info(f"[DEBUG] Message type field: {message.get('type')}")

            if message.get("type") == "audio":
                logger.info(f"[DEBUG] Processing audio message for {from_number}")
                content = await process_audio_message(message)
                user_message_type = "audio"
                logger.info(f"[DEBUG] Audio transcription result: {content}")
            elif _is_image_message(message):
                logger.info(f"[DEBUG] Processing image message for {from_number}")
                user_message_type = "image"
                # Get image caption if any
                content = message.get("image", {}).get("caption", "")
                image_id = message.get("image", {}).get("id") or message.get("document", {}).get("id")
                if image_id:
                    image_bytes = await download_media(image_id)
                    try:
                        logger.info(f"Analyzing image for user {from_number}, image size: {len(image_bytes)} bytes")
                        description = await image_to_text.analyze_image(
                            image_bytes,
                            "Please describe what you see in this image in the context of our conversation.",
                        )
                        content += f"\n[Image Analysis: {description}]"
                        logger.info(f"Image analysis successful for user {from_number}")
                    except Exception as e:
                        logger.error(f"Failed to analyze image for user {from_number}: {e}")
                        # Add a note about the failed analysis so the conversation can continue
                        content += "\n[Image received but analysis failed - I can see you sent an image but can't analyze it right now]"
                else:
                    logger.warning(f"Image message received but no image ID found for user {from_number}")
                    content += "\n[Image received but no image data found]"
            else:
                logger.info(f"[DEBUG] Processing text message for {from_number}")
                content = message.get("text", {}).get("body", "")
                user_message_type = "text"

            logger.info(f"[DEBUG] Final message type: {user_message_type}, content length: {len(content)}")

            # Log the incoming user message with type
            await log_message(session_id, "user", content, user_message_type)

            # Process message through the graph agent
            async with await get_checkpointer() as checkpointer:
                graph = graph_builder.compile(checkpointer=checkpointer)

                # Set initial state with user and session info
                initial_state = {
                    "messages": [HumanMessage(content=content)],
                    "user_id": user_id,
                    "session_id": session_id,
                }

                await graph.ainvoke(
                    initial_state,
                    {"configurable": {"thread_id": from_number}},  # Keep using phone number as thread_id for LangGraph
                )

                # Get the workflow type and response from the state
                output_state = await graph.aget_state(config={"configurable": {"thread_id": from_number}})

            workflow = output_state.values.get("workflow", "conversation")
            response_message = output_state.values["messages"][-1].content

            # Determine agent response type based on workflow
            agent_message_type = "text"  # Default type
            if workflow == "audio":
                agent_message_type = "audio"
            # Note: Image generation is disabled, so agent only sends text/audio

            # Log the agent response with type
            await log_message(session_id, "agent", response_message, agent_message_type)

            # Handle different response types based on workflow
            if workflow == "audio":
                audio_buffer = output_state.values["audio_buffer"]
                success = await send_response(from_number, response_message, "audio", audio_buffer)
            elif workflow == "image":
                # ===== IMAGE GENERATION DISABLED =====
                # Image generation has been disabled - respond with text instead
                # The code below is commented out but preserved for potential future use

                # image_path = output_state.values["image_path"]
                # with open(image_path, "rb") as f:
                #     image_data = f.read()
                # success = await send_response(from_number, response_message, "image", image_data)

                # Fallback to text response when image generation is disabled
                success = await send_response(from_number, response_message, "text")
            else:
                success = await send_response(from_number, response_message, "text")

            # In test environment, don't fail on WhatsApp API issues
            if not success and not IS_TEST_ENV:
                return Response(content="Failed to send message", status_code=500)

            duration = perf_counter() - start
            logger.info("Processed message from %s in %.2f seconds", from_number, duration)
            return Response(content="Message processed", status_code=200)

        elif "statuses" in change_value:
            duration = perf_counter() - start
            logger.info("Status update processed in %.2f seconds", duration)
            return Response(content="Status update received", status_code=200)

        else:
            duration = perf_counter() - start
            logger.warning("Unknown event type processed in %.2f seconds", duration)
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        duration = perf_counter() - start
        logger.error("Error processing message after %.2f seconds: %s", duration, e, exc_info=True)
        return Response(content="Internal server error", status_code=500)


async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    media_metadata_url = f"https://graph.facebook.com/v21.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

        media_response = await client.get(download_url, headers=headers)
        media_response.raise_for_status()
        return media_response.content


async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message."""
    audio_id = message["audio"]["id"]
    media_metadata_url = f"https://graph.facebook.com/v21.0/{audio_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

    # Download the audio file
    async with httpx.AsyncClient() as client:
        audio_response = await client.get(download_url, headers=headers)
        audio_response.raise_for_status()

    # Prepare for transcription
    audio_buffer = BytesIO(audio_response.content)
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()

    return await speech_to_text.transcribe(audio_data)


async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: bytes = None,
) -> bool:
    """Send response to user via WhatsApp API."""

    # Skip actual API calls in test environment
    if IS_TEST_ENV:
        logger.info(f"TEST MODE: Would send {message_type} message to {from_number}: {response_text[:100]}...")
        return True

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    if message_type in ["audio", "image"]:
        try:
            mime_type = "audio/mpeg" if message_type == "audio" else "image/png"
            media_buffer = BytesIO(media_content)
            media_id = await upload_media(media_buffer, mime_type)
            json_data = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": message_type,
                message_type: {"id": media_id},
            }

            # Add caption for images
            if message_type == "image":
                json_data["image"]["caption"] = response_text
        except Exception as e:
            logger.error(f"Media upload failed, falling back to text: {e}")
            message_type = "text"

    if message_type == "text":
        json_data = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "type": "text",
            "text": {"body": response_text},
        }

    logger.debug(headers)
    logger.debug(json_data)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Failed to send message to WhatsApp: %s", e)
            return False

    return True


async def upload_media(media_content: BytesIO, mime_type: str) -> str:
    """Upload media to WhatsApp servers."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("response.mp3", media_content, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v21.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        result = response.json()

    if "id" not in result:
        raise Exception("Failed to upload media")
    return result["id"]
