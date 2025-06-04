import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from agent.modules.image.image_to_text import ImageToText
from agent.modules.image.text_to_image import TextToImage
from agent.modules.speech.text_to_speech import TextToSpeech
from agent.settings import get_settings


@lru_cache
def get_chat_model():
    """Get the chat model instance."""
    settings = get_settings()
    return ChatGroq(
        model=settings.TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
    )


@lru_cache
def get_text_to_speech_module():
    """Get the text-to-speech module instance."""
    return TextToSpeech()


@lru_cache
def get_text_to_image_module():
    """Get the text-to-image module instance."""
    return TextToImage()


@lru_cache
def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))
