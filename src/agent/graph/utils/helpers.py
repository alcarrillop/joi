import logging
import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from agent.modules.image.image_to_text import ImageToText
from agent.modules.image.text_to_image import TextToImage
from agent.modules.learning.learning_stats_manager import get_learning_stats_manager
from agent.modules.speech.text_to_speech import TextToSpeech
from agent.settings import get_settings

logger = logging.getLogger(__name__)


def get_chat_model(temperature=None):
    """Get the chat model instance."""
    settings = get_settings()
    temp = temperature if temperature is not None else 0.7
    return ChatGroq(
        model=settings.TEXT_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=temp,
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


async def get_user_level_info(user_id: str) -> str:
    """Get user's English level information with curriculum-based insights."""
    try:
        learning_manager = get_learning_stats_manager()

        # Get basic vocabulary statistics from learning manager
        vocab_count = await learning_manager.get_vocabulary_word_count(user_id)
        top_words = await learning_manager.get_user_top_words(user_id, limit=3)

        # Simple level estimation based on vocabulary count
        if vocab_count >= 400:
            level = "C1"
        elif vocab_count >= 250:
            level = "B2"
        elif vocab_count >= 150:
            level = "B1"
        elif vocab_count >= 75:
            level = "A2"
        else:
            level = "A1"

        level_description = {
            "A1": "Beginner - You're just starting your English journey!",
            "A2": "Elementary - You know basic English for daily situations.",
            "B1": "Intermediate - You can handle most everyday conversations.",
            "B2": "Upper Intermediate - You communicate quite well in English.",
            "C1": "Advanced - You're very fluent and sophisticated!",
            "C2": "Proficient - You're practically a native speaker!",
        }

        description = level_description.get(level, "Learning English")

        response = "📊 **Your English Learning Progress** 📊\n\n"
        response += f"🎯 **Current Level:** {level} ({description})\n\n"
        response += f"📚 **Vocabulary Learned:** {vocab_count} English words\n\n"

        # Add top words with frequencies
        if top_words:
            response += "🔥 **Your Most Used Words:**\n"
            for word_data in top_words:
                response += f"   • {word_data['word']} ({word_data['frequency']} times)\n"
            response += "\n"

        # Simple progress tracking
        level_thresholds = {
            "A1": 75,
            "A2": 150,
            "B1": 250,
            "B2": 400,
            "C1": 600,
            "C2": 800,
        }

        current_threshold = level_thresholds.get(level, 0)
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

        try:
            current_index = levels.index(level)
            next_level = levels[current_index + 1] if current_index < len(levels) - 1 else None
        except ValueError:
            next_level = None

        if next_level:
            next_threshold = level_thresholds.get(next_level, current_threshold)
            words_needed = max(0, next_threshold - vocab_count)

            if words_needed > 0:
                response += f"🚀 **Next Goal:** Reach {next_level} level\n"
                response += f"🎯 **Words Needed:** {words_needed} more vocabulary words\n\n"
            else:
                response += f"🎉 **Ready to advance to {next_level}!**\n\n"
        elif vocab_count >= 600:
            response += "🏆 **Congratulations!** You're at an advanced level!\n\n"

        # Add encouraging context based on level
        if vocab_count < 25:
            response += "Keep chatting with me! Every interaction teaches you new words. 🌱"
        elif vocab_count < 100:
            response += "Great progress! You're building a solid foundation. Keep going! 💪"
        elif vocab_count < 300:
            response += "Excellent work! Your vocabulary is growing steadily. 🔥"
        else:
            response += "Amazing progress! You're becoming quite fluent. 🌟"

        return response

    except Exception as e:
        logger.error(f"Error getting user level info: {e}")
        return "I'm having trouble accessing your learning progress right now, but I can see you're actively learning! Keep practicing and I'll help you improve! 🚀"
