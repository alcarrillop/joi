import logging
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
        curriculum_manager = get_curriculum_manager()
        learning_manager = get_learning_stats_manager()

        # Get comprehensive level progress from enhanced curriculum manager
        level_progress = await curriculum_manager.estimate_level_progress(user_id)

        if "error" in level_progress:
            return "I don't have enough information about your English level yet. Keep chatting with me so I can assess your vocabulary!"

        # Get basic vocabulary statistics from learning manager
        top_words = await learning_manager.get_user_top_words(user_id, limit=3)

        level = level_progress.get("estimated_level", "A1")
        vocab_count = level_progress.get("vocabulary_learned", 0)
        next_level = level_progress.get("next_level")
        words_needed = level_progress.get("words_needed_for_next_level", 0)
        progress_percent = level_progress.get("progress_to_next_level", 0)
        message_count = level_progress.get("total_messages", 0)

        # Get curriculum insights
        curriculum_insights = level_progress.get("curriculum_insights", {})
        recommendations = level_progress.get("educational_recommendations", [])
        competencies = level_progress.get("level_competencies", [])

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
        response += f"💬 **Interactions:** {message_count} messages exchanged\n\n"

        # Add curriculum mastery information
        if curriculum_insights:
            mastery = curriculum_insights.get("curriculum_mastery_percentage", 0)
            response += f"🎓 **Curriculum Mastery:** {mastery:.0f}% of {level} vocabulary\n\n"

        # Add top words with frequencies
        if top_words:
            response += "🔥 **Your Most Used Words:**\n"
            for word_data in top_words:
                response += f"   • {word_data['word']} ({word_data['frequency']} times)\n"
            response += "\n"

        # Progress tracking
        if next_level and words_needed > 0:
            current_threshold = level_progress.get("current_level_threshold", 0)
            previous_threshold = level_progress.get("previous_level_threshold", 0)

            if current_threshold > previous_threshold:
                level_progress_percent = (
                    (vocab_count - previous_threshold) / (current_threshold - previous_threshold)
                ) * 100
                level_progress_percent = min(100, max(0, level_progress_percent))

                if vocab_count >= current_threshold:
                    response += f"🚀 **Next Goal:** Reach {next_level} level\n"
                    response += f"📈 **Progress:** Ready to advance! You've completed {level}\n"
                    response += f"🎯 **Words for {next_level}:** {words_needed} more vocabulary words\n\n"
                else:
                    words_to_complete_current = current_threshold - vocab_count
                    response += f"🚀 **Current Goal:** Complete {level} level\n"
                    response += f"📈 **Progress in {level}:** {level_progress_percent:.1f}% complete\n"
                    response += (
                        f"🎯 **Words to complete {level}:** {words_to_complete_current} more vocabulary words\n\n"
                    )
            else:
                response += f"🚀 **Next Goal:** Reach {next_level} level\n"
                response += f"📈 **Progress:** {progress_percent:.1f}% to {next_level}\n"
                response += f"🎯 **Words Needed:** {words_needed} more vocabulary words\n\n"
        elif vocab_count >= 600:
            response += "🏆 **Congratulations!** You're at an advanced level!\n\n"

        # Add educational recommendations from curriculum
        if recommendations:
            response += "💡 **Learning Recommendations:**\n"
            for rec in recommendations[:2]:  # Limit to top 2
                response += f"   • {rec}\n"
            response += "\n"

        # Add current level focus areas
        if competencies:
            response += f"📖 **Current {level} Focus Areas:**\n"
            for comp in competencies:
                response += f"   • {comp['name']} ({comp['skill_type']})\n"
            response += "\n"

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
