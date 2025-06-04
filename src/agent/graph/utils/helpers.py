import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from agent.modules.curriculum.curriculum_manager import get_curriculum_manager
from agent.modules.image.image_to_text import ImageToText
from agent.modules.image.text_to_image import TextToImage
from agent.modules.learning.learning_stats_manager import get_learning_stats_manager
from agent.modules.speech.text_to_speech import TextToSpeech
from agent.settings import get_settings


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
    """Get user's English level information for conversation context."""
    try:
        curriculum_manager = get_curriculum_manager()
        learning_manager = get_learning_stats_manager()

        # Get level progress from curriculum manager
        level_progress = await curriculum_manager.estimate_level_progress(user_id)

        if "error" in level_progress:
            return "I don't have enough information about your English level yet. Keep chatting with me so I can assess your vocabulary!"

        # Get enhanced vocabulary statistics
        vocab_summary = await learning_manager.get_learning_summary(user_id)
        top_words = await learning_manager.get_user_top_words(user_id, limit=3)

        level = level_progress.get("estimated_level", "A1")
        vocab_count = vocab_summary["vocabulary"]["total_words_learned"]
        next_level = level_progress.get("next_level")
        words_needed = level_progress.get("words_needed_for_next_level", 0)
        progress_percent = level_progress.get("progress_to_next_level", 0)
        message_count = level_progress.get("total_messages", 0)

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
        response += f"📚 **Vocabulary Learned:** {vocab_count} English words\n"
        response += f"💬 **Conversations:** {message_count} messages exchanged\n\n"

        # Add top words with frequencies
        if top_words:
            response += "🔥 **Your Most Used Words:**\n"
            for word_data in top_words:
                response += f"   • {word_data['word']} ({word_data['frequency']} times)\n"
            response += "\n"

        if next_level and words_needed > 0:
            response += f"🚀 **Next Goal:** Reach {next_level} level\n"
            response += f"📈 **Progress:** {progress_percent:.1f}% to {next_level}\n"
            response += f"🎯 **Words Needed:** {words_needed} more vocabulary words\n\n"
        elif vocab_count >= 600:
            response += "🏆 **Congratulations!** You're at an advanced level!\n\n"

        # Add encouraging context based on level
        if vocab_count < 25:
            response += "Keep chatting with me! Every conversation teaches you new words. 🌱"
        elif vocab_count < 100:
            response += "Great progress! You're building a solid foundation. Keep going! 💪"
        elif vocab_count < 300:
            response += "Excellent work! Your vocabulary is growing steadily. 🔥"
        else:
            response += "Amazing progress! You're becoming quite fluent. 🌟"

        return response

    except Exception:
        return "I'm having trouble accessing your learning progress right now, but I can see you're actively learning! Keep practicing and I'll help you improve! 🚀"
