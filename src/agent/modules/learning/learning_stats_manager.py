"""
Learning Stats Manager - SIMPLIFIED VERSION
"""

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Set

from .db_operations import get_vocabulary_db_operations
from .vocabulary_data import get_basic_vocabulary, get_excluded_words, get_intermediate_vocabulary, get_stop_words

logger = logging.getLogger(__name__)


class SimpleLearningStatsManager:
    """Simple learning stats manager - basic functionality only."""

    def __init__(self):
        # Load vocabularies from external data
        self.basic_vocabulary: Set[str] = get_basic_vocabulary()
        self.intermediate_vocabulary: Set[str] = get_intermediate_vocabulary()
        self.excluded_words: Set[str] = get_excluded_words()
        self.stop_words: Set[str] = get_stop_words()

        # Database operations
        self.db_ops = get_vocabulary_db_operations()

    @lru_cache(maxsize=1000)
    def _normalize_word(self, word: str) -> str:
        """Simple word normalization."""
        if not word or len(word) < 2:
            return word.lower()

        word_lower = word.lower().strip()

        # Basic plural removal
        if word_lower.endswith("s") and len(word_lower) > 3:
            return word_lower[:-1]

        return word_lower

    def _is_valid_english_word(self, word: str) -> bool:
        """Check if word is valid for tracking."""
        if not word.isalpha():
            return False
        if len(word) < 3 or len(word) > 15:
            return False
        if word.lower() in self.stop_words:
            return False
        if word.lower() in self.excluded_words:
            return False
        return True

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return [word for word in words if self._is_valid_english_word(word)]

    async def update_learning_stats(self, user_id: str, session_id: str, message_text: str) -> Dict[str, Any]:
        """Update basic learning stats."""
        try:
            words = self._extract_words(message_text)
            processed_words = 0

            for word in words:
                normalized_word = self._normalize_word(word)
                if await self.db_ops.increment_word_frequency(user_id, normalized_word):
                    processed_words += 1

            return {
                "words_processed": processed_words,
                "total_words_in_message": len(words),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error updating learning stats: {e}")
            return {"error": str(e), "success": False}

    async def get_user_learned_vocabulary(self, user_id: str) -> List[str]:
        """Get user's learned vocabulary."""
        return await self.db_ops.get_user_learned_vocabulary(user_id)

    async def get_vocabulary_word_count(self, user_id: str) -> int:
        """Get vocabulary count."""
        return await self.db_ops.get_vocabulary_word_count(user_id)

    async def get_user_top_words(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top words."""
        return await self.db_ops.get_user_top_words(user_id, limit)

    async def get_learning_summary(self, user_id: str) -> Dict[str, Any]:
        """Get simple learning summary."""
        try:
            vocab_count = await self.get_vocabulary_word_count(user_id)
            top_words = await self.get_user_top_words(user_id, 5)

            return {
                "vocabulary": {
                    "total_words": vocab_count,
                    "top_words": top_words,
                },
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {"error": str(e), "success": False}


# Singleton pattern
_learning_manager = None


def get_learning_stats_manager() -> SimpleLearningStatsManager:
    """Get learning stats manager instance."""
    global _learning_manager
    if _learning_manager is None:
        _learning_manager = SimpleLearningStatsManager()
    return _learning_manager
