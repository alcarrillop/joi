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
        """Smart word normalization that preserves word meaning."""
        if not word or len(word) < 2:
            return word.lower()

        word_lower = word.lower().strip()

        # Don't normalize words that should stay as-is
        preserve_words = {
            "analysis",
            "basis",
            "emphasis",
            "hypothesis",
            "synthesis",
            "crisis",
            "oasis",
            "thesis",
            "diagnosis",
            "prognosis",
            "jesus",
            "virus",
            "campus",
            "bonus",
            "focus",
            "genius",
            "minus",
            "plus",
            "status",
            "census",
            "corpus",
            "opus",
        }

        if word_lower in preserve_words:
            return word_lower

        # Handle common irregular plurals
        irregular_plurals = {
            "children": "child",
            "people": "person",
            "men": "man",
            "women": "woman",
            "feet": "foot",
            "teeth": "tooth",
            "mice": "mouse",
            "geese": "goose",
        }

        if word_lower in irregular_plurals:
            return irregular_plurals[word_lower]

        # Smart plural removal - only for clear plurals
        if word_lower.endswith("ies") and len(word_lower) > 4:
            # stories -> story, companies -> company
            return word_lower[:-3] + "y"
        elif word_lower.endswith("es") and len(word_lower) > 4:
            # Only remove 'es' if it's clearly a plural (boxes, dishes, etc.)
            if word_lower.endswith(("shes", "ches", "xes", "zes", "sses")):
                return word_lower[:-2]
            # Don't touch words like "images", "games", "names"
            return word_lower
        elif word_lower.endswith("s") and len(word_lower) > 4:
            # Only remove 's' if it's not part of the root word
            if not word_lower.endswith(("ss", "us", "is", "as", "os")):
                return word_lower[:-1]

        return word_lower

    def _is_valid_english_word(self, word: str) -> bool:
        """Check if word is valid English vocabulary for tracking."""
        word_lower = word.lower()

        # Basic format checks
        if not word.isalpha():
            return False
        if len(word) < 3 or len(word) > 15:
            return False

        # Skip stop words and excluded words
        if word_lower in self.stop_words:
            return False
        if word_lower in self.excluded_words:
            return False

        # Skip common non-English patterns
        if self._is_likely_non_english(word_lower):
            return False

        # Must be a real English word (in our vocabularies or common patterns)
        if self._is_real_english_word(word_lower):
            return True

        return False

    def _is_likely_non_english(self, word: str) -> bool:
        """Check if word is likely not English."""
        # Spanish/other language patterns
        if word.endswith(("ión", "ción", "sión", "ía", "ío")):
            return True
        # Consecutive vowels common in other languages
        if any(vowel_seq in word for vowel_seq in ["aa", "ee", "ii", "oo", "uu", "ae", "ao"]):
            return True
        # Common non-English letter combinations
        if any(pattern in word for pattern in ["ñ", "ç", "ü", "ö", "ä", "ß"]):
            return True
        return False

    def _is_real_english_word(self, word: str) -> bool:
        """Check if word appears to be a real English word."""
        # Check if it's in our known vocabularies
        if word in self.basic_vocabulary or word in self.intermediate_vocabulary:
            return True

        # Common English word patterns/endings
        english_patterns = [
            # Verb endings
            "ing",
            "ed",
            "er",
            "est",
            "ly",
            "tion",
            "sion",
            "ness",
            "ment",
            "ful",
            "less",
            # Adjective endings
            "able",
            "ible",
            "al",
            "ial",
            "ous",
            "ive",
            "ic",
            "ical",
            # Noun endings
            "ity",
            "ism",
            "ist",
            "ure",
            "ance",
            "ence",
            "ship",
            "hood",
            "dom",
        ]

        # If word ends with common English pattern, likely English
        if any(word.endswith(pattern) for pattern in english_patterns):
            return True

        # Common English prefixes
        english_prefixes = ["un", "re", "pre", "dis", "mis", "over", "under", "out", "up"]
        if any(word.startswith(prefix) for prefix in english_prefixes):
            return True

        # Short words (3-4 letters) are usually valid if they passed other checks
        if len(word) <= 4:
            return True

        return False

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from user's natural text."""
        if not text:
            return []

        # Clean the text first - remove system messages and metadata
        cleaned_text = self._clean_user_text(text)

        # Extract alphabetic words only
        words = re.findall(r"\b[a-zA-Z]+\b", cleaned_text.lower())

        # Filter for valid English vocabulary words
        valid_words = []
        for word in words:
            if self._is_valid_english_word(word):
                valid_words.append(word)

        return valid_words

    def _clean_user_text(self, text: str) -> str:
        """Clean text to focus on natural language content."""
        # Remove system indicators
        text = re.sub(r"\[image analysis:.*?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[.*?\]", "", text)  # Remove any bracketed content

        # Remove URLs and email addresses
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove punctuation-heavy sequences (not natural language)
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

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
