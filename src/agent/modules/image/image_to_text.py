import base64
import io
import logging
import os
from typing import Union

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from PIL import Image

from agent.core.exceptions import ImageToTextError
from agent.settings import get_settings


class ImageToText:
    """A class to handle image-to-text conversion using Groq's vision capabilities."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        """Initialize the ImageToText class and validate environment variables."""
        self._validate_env_vars()
        settings = get_settings()
        self.model_name = settings.ITT_MODEL_NAME
        self.groq_client = ChatGroq(
            model=self.model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
        )
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _detect_image_format(self, image_bytes: bytes) -> str:
        """Detect image format from bytes."""
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "JPEG"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "PNG"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
            return "WEBP"
        elif image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return "GIF"
        else:
            return "UNKNOWN"

    def _get_mime_type(self, format_name: str) -> str:
        """Get MIME type for image format."""
        mime_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
        return mime_map.get(format_name, "image/jpeg")  # Default to JPEG

    def _validate_and_standardize_image(self, image_bytes: bytes) -> tuple[bytes, str]:
        """Validate and standardize image data."""
        try:
            # Detect format
            format_detected = self._detect_image_format(image_bytes)
            self.logger.info(f"Detected image format: {format_detected}")

            if format_detected == "UNKNOWN":
                self.logger.warning("Unknown image format detected, attempting to process anyway")

            # Validate image data using PIL
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Verify the image is valid

            # Reopen for processing (verify() can only be called once)
            img = Image.open(io.BytesIO(image_bytes))
            self.logger.info(f"Image validation successful: {img.size}, mode: {img.mode}, format: {img.format}")

            # Convert to RGB if necessary for consistency
            if img.mode != "RGB":
                self.logger.info(f"Converting image from {img.mode} to RGB mode")
                img = img.convert("RGB")

            # Standardize to high-quality JPEG for Groq
            standardized_buffer = io.BytesIO()
            img.save(standardized_buffer, format="JPEG", quality=95)
            standardized_bytes = standardized_buffer.getvalue()

            self.logger.info(
                f"Image standardized: original {len(image_bytes)} bytes -> {len(standardized_bytes)} bytes"
            )

            return standardized_bytes, "image/jpeg"

        except Exception as e:
            self.logger.error(f"Image validation/standardization failed: {e}")
            raise ValueError(f"Invalid image data: {e}") from e

    async def analyze_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        """Analyze an image using Groq's vision capabilities.

        Args:
            image_data: Either a file path (str) or binary image data (bytes)
            prompt: Optional prompt to guide the image analysis

        Returns:
            str: Description or analysis of the image

        Raises:
            ValueError: If the image data is empty or invalid
            ImageToTextError: If the image analysis fails
        """
        try:
            # Handle file path
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
                self.logger.info(f"Loaded image from file: {image_data}")
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            self.logger.info(f"Processing image of size: {len(image_bytes)} bytes")

            # Validate and standardize the image
            try:
                standardized_bytes, mime_type = self._validate_and_standardize_image(image_bytes)
            except ValueError as e:
                self.logger.error(f"Image validation failed: {e}")
                raise ImageToTextError(f"Invalid image data: {e}") from e

            # Convert to base64
            base64_image = base64.b64encode(standardized_bytes).decode("utf-8")

            # Default prompt if none provided
            if not prompt:
                prompt = "Please describe what you see in this image in detail."

            self.logger.info(f"Using model: {self.model_name} for image analysis")
            self.logger.info(f"Analysis prompt: {prompt}")
            self.logger.info(f"Image data URL format: data:{mime_type};base64,[{len(base64_image)} chars]")

            # Create the message for LangChain ChatGroq
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            ]

            message = HumanMessage(content=message_content)

            # Make the API call using LangChain ChatGroq
            self.logger.info("Making API call to Groq vision model...")
            response = await self.groq_client.ainvoke([message])

            if not response or not response.content:
                self.logger.error("No response received from vision model")
                raise ImageToTextError("No response received from the vision model")

            description = response.content
            self.logger.info(f"Generated image description (length: {len(description)})")

            # Check if the model claims it can't see the image
            if any(
                phrase in description.lower()
                for phrase in [
                    "don't see an image",
                    "no image",
                    "can't see",
                    "unable to see",
                    "no visual content",
                    "image was not provided",
                ]
            ):
                self.logger.warning(f"Model claims no image detected: {description[:100]}...")
                # Still return the response but log the issue

            return description

        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}")
            raise ImageToTextError(f"Failed to analyze image: {str(e)}") from e
