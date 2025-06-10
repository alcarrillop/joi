"""
Learning Stats Manager - Refactored
===================================

Improved vocabulary learning tracking with better architecture:
- Cached normalization for performance
- Modular validation functions
- External vocabulary data
- Better error handling
- Separated database operations
"""

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from langchain_groq import ChatGroq
from pydantic import BaseModel

from agent.settings import get_settings

from .db_operations import get_vocabulary_db_operations
from .vocabulary_data import get_basic_vocabulary, get_excluded_words, get_intermediate_vocabulary, get_stop_words

logger = logging.getLogger(__name__)


class VocabularyAnalysis(BaseModel):
    """Analysis of vocabulary used in a message."""

    new_words: List[str]
    advanced_words: List[str]
    basic_words: List[str]


class LearningStatsManager:
    """Manages learning statistics for users - vocabulary only."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Load vocabularies from external data
        self.basic_vocabulary: Set[str] = get_basic_vocabulary()
        self.intermediate_vocabulary: Set[str] = get_intermediate_vocabulary()
        self.excluded_words: Set[str] = get_excluded_words()
        self.stop_words: Set[str] = get_stop_words()

        # Database operations
        self.db_ops = get_vocabulary_db_operations()

        # LLM will be loaded when needed
        self._llm: Optional[ChatGroq] = None

    @property
    def llm(self) -> ChatGroq:
        """Lazy load LLM to avoid initialization issues."""
        if self._llm is None:
            self._llm = self._load_llm()
        return self._llm

    def _load_llm(self) -> ChatGroq:
        """Load the LLM model."""
        settings = get_settings()
        return ChatGroq(
            model=settings.TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
        )

    @lru_cache(maxsize=5000)
    def _normalize_word(self, word: str) -> str:
        """Convert word to its base form (cached for performance)."""
        if not word or len(word) < 2:
            return word.lower()

        word_lower = word.lower().strip()

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

        # Pluralization rules
        if word_lower.endswith("ies") and len(word_lower) > 4:
            return word_lower[:-3] + "y"
        elif word_lower.endswith("es") and len(word_lower) > 3:
            if word_lower.endswith(("shes", "ches", "xes", "zes")):
                return word_lower[:-2]
            elif word_lower not in ("yes", "goes", "does"):
                return word_lower[:-1]
        elif word_lower.endswith("s") and len(word_lower) > 3:
            preserve_s = {
                "this",
                "yes",
                "his",
                "its",
                "was",
                "has",
                "business",
                "process",
                "success",
                "express",
                "address",
                "access",
                "progress",
                "stress",
                "class",
                "glass",
                "pass",
                "less",
                "boss",
                "loss",
                "cross",
            }
            if word_lower not in preserve_s:
                return word_lower[:-1]

        # Verb forms
        if word_lower.endswith("ing") and len(word_lower) > 5:
            base = word_lower[:-3]
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeioulrwxy":
                return base[:-1]
            return base
        elif word_lower.endswith("ed") and len(word_lower) > 4:
            base = word_lower[:-2]
            if base.endswith("i"):
                return base[:-1] + "y"
            return base

        return word_lower

    def normalize_batch(self, words: List[str]) -> List[str]:
        """Normalize multiple words efficiently."""
        return [self._normalize_word(word) for word in words]

    def _is_alpha(self, word: str) -> bool:
        """Check if word contains only alphabetic characters."""
        return word.isalpha()

    def _is_valid_length(self, word: str) -> bool:
        """Check if word has valid length."""
        return 3 <= len(word) <= 15

    def _is_stopword(self, word: str) -> bool:
        """Check if word is a stop word."""
        return word.lower() in self.stop_words

    def _is_excluded_word(self, word: str) -> bool:
        """Check if word is in excluded list."""
        return word.lower() in self.excluded_words

    def _is_proper_noun(self, word: str) -> bool:
        """Check if word is likely a proper noun."""
        word_lower = word.lower()
        return (
            word[0].isupper()
            and word_lower not in self.basic_vocabulary
            and word_lower not in self.intermediate_vocabulary
        )

    def _is_valid_english_word(self, word: str) -> bool:
        """Check if word is a valid English vocabulary word to track."""
        if not self._is_alpha(word):
            return False

        if not self._is_valid_length(word):
            return False

        if self._is_stopword(word):
            return False

        if self._is_excluded_word(word):
            return False

        if self._is_proper_noun(word):
            return False

        return True

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return [word for word in words if len(word) > 2 and not self._is_stopword(word)]

    async def analyze_vocabulary(self, text: str, user_level: str = "A1") -> VocabularyAnalysis:
        """Analyze vocabulary used in the text - English words only."""
        words = self._extract_words(text)

        new_words = []
        advanced_words = []
        basic_words = []

        for word in words:
            if not self._is_valid_english_word(word):
                continue

            word_lower = word.lower()

            if word_lower in self.basic_vocabulary:
                basic_words.append(word_lower)
            elif word_lower in self.intermediate_vocabulary:
                advanced_words.append(word_lower)
                new_words.append(word_lower)
            else:
                if len(word) >= 4:
                    new_words.append(word_lower)

        # Remove duplicates while preserving order
        new_words = list(dict.fromkeys(new_words))
        advanced_words = list(dict.fromkeys(advanced_words))
        basic_words = list(dict.fromkeys(basic_words))

        return VocabularyAnalysis(
            new_words=new_words[:10],
            advanced_words=advanced_words[:5],
            basic_words=basic_words[:15],
        )

    # Database operations - delegated to db_operations module
    async def get_user_learned_vocabulary(self, user_id: str) -> List[str]:
        """Get vocabulary already learned by user."""
        return await self.db_ops.get_user_learned_vocabulary(user_id)

    async def get_vocabulary_word_count(self, user_id: str) -> int:
        """Return the total number of words learned by the user."""
        return await self.db_ops.get_vocabulary_word_count(user_id)

    async def get_user_top_words(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's most frequently used words."""
        return await self.db_ops.get_user_top_words(user_id, limit)

    async def increment_word_frequency(self, user_id: str, word: str) -> bool:
        """Increment the frequency of a word for a user."""
        return await self.db_ops.increment_word_frequency(user_id, word)

    async def update_learning_stats(self, user_id: str, session_id: str, message_text: str) -> Dict[str, Any]:
        """Update learning statistics for a user based on their message."""
        self.logger.info(f"Updating learning stats for user {user_id}")

        try:
            # Analyze the message vocabulary
            vocab_analysis = await self.analyze_vocabulary(message_text)

            # Get existing vocabulary
            learned_vocab = await self.get_user_learned_vocabulary(user_id)

            # Normalize existing vocabulary to prevent duplicates
            normalized_existing = self.normalize_batch(learned_vocab)

            # Find truly new vocabulary
            new_vocab = []
            frequency_updates = []

            for word in vocab_analysis.new_words:
                normalized_word = self._normalize_word(word)

                if normalized_word in normalized_existing:
                    frequency_updates.append(normalized_word)
                else:
                    new_vocab.append(normalized_word)
                    normalized_existing.append(normalized_word)

            # Batch update frequencies
            all_words_to_update = new_vocab + frequency_updates
            success_count = await self.db_ops.batch_increment_words(user_id, all_words_to_update)

            stats = {
                "vocabulary_analysis": {
                    "new_words_found": len(new_vocab),
                    "new_words": new_vocab,
                    "frequency_updates": len(frequency_updates),
                    "advanced_words": vocab_analysis.advanced_words,
                    "total_learned": len(normalized_existing),
                    "update_success_rate": success_count / len(all_words_to_update) if all_words_to_update else 1.0,
                },
            }

            self.logger.info(
                f"Learning stats updated for user {user_id}: {len(new_vocab)} new words, "
                f"{len(frequency_updates)} frequency updates, {success_count}/{len(all_words_to_update)} successful"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Failed to update learning stats for user {user_id}: {e}")
            return {"error": str(e)}

    async def cleanup_user_vocabulary(self, user_id: str) -> Dict[str, Any]:
        """Clean up user's vocabulary by removing invalid words and duplicates."""
        try:
            # Get current vocabulary
            current_words = await self.db_ops.get_all_user_words(user_id)

            # Filter and normalize words
            valid_words = []
            removed_words = []

            for word, freq in current_words:
                if self._is_valid_english_word(word):
                    normalized = self._normalize_word(word)
                    valid_words.append((normalized, freq))
                else:
                    removed_words.append(word)

            # Remove duplicates (keeping highest frequency)
            word_freqs = {}
            for word, freq in valid_words:
                if word in word_freqs:
                    word_freqs[word] = max(word_freqs[word], freq)
                else:
                    word_freqs[word] = freq

            duplicates_removed = len(valid_words) - len(word_freqs)

            # Update database with cleaned vocabulary
            clean_result = await self.db_ops.cleanup_user_vocabulary(
                user_id, [(word, freq) for word, freq in word_freqs.items()]
            )

            if "error" in clean_result:
                return clean_result

            clean_result.update(
                {
                    "removed_words": removed_words,
                    "duplicates_removed": duplicates_removed,
                }
            )

            return clean_result

        except Exception as e:
            self.logger.error(f"Failed to cleanup vocabulary for user {user_id}: {e}")
            return {"error": str(e)}

    async def get_learning_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user's learning progress."""
        try:
            vocab_count = await self.get_vocabulary_word_count(user_id)
            top_words = await self.get_user_top_words(user_id, limit=5)
            recent_words = await self.db_ops.get_recent_words(user_id, limit=5)

            return {
                "vocabulary": {
                    "total_words_learned": vocab_count,
                    "top_words": top_words,
                    "recent_words": recent_words,
                },
                "statistics": {
                    "total_vocabulary": vocab_count,
                    "most_frequent_word": top_words[0]["word"] if top_words else None,
                    "highest_frequency": top_words[0]["frequency"] if top_words else 0,
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to get learning summary for user {user_id}: {e}")
            return {"vocabulary": {"total_words_learned": 0}, "error": str(e)}

    # Vocabulary insights moved to curriculum module for educational analysis


# Singleton instance
_learning_stats_manager = None


def get_learning_stats_manager() -> LearningStatsManager:
    """Get singleton instance of LearningStatsManager."""
    global _learning_stats_manager
    if _learning_stats_manager is None:
        _learning_stats_manager = LearningStatsManager()
    return _learning_stats_manager
