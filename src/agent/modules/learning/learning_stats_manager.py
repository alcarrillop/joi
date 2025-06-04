"""
Learning Stats Manager
=====================

Manages vocabulary learning tracking for English words only.
Updates learning_stats table with user vocabulary progress.
"""

import logging
import re
import uuid
from typing import Dict, List

import asyncpg
from langchain_groq import ChatGroq
from pydantic import BaseModel

from agent.core.database import get_database_url
from agent.settings import get_settings

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
        self.settings = get_settings()
        self.llm = ChatGroq(
            model=self.settings.TEXT_MODEL_NAME,
            api_key=self.settings.GROQ_API_KEY,
            temperature=0.3,
        )

        # Basic vocabulary for beginners (A1-A2 level)
        self.basic_vocabulary = {
            "hello",
            "goodbye",
            "yes",
            "no",
            "please",
            "thank",
            "you",
            "welcome",
            "sorry",
            "excuse",
            "help",
            "understand",
            "speak",
            "english",
            "learning",
            "practice",
            "study",
            "name",
            "age",
            "from",
            "live",
            "work",
            "home",
            "family",
            "friend",
            "time",
            "today",
            "tomorrow",
            "yesterday",
            "morning",
            "afternoon",
            "evening",
            "night",
            "week",
            "month",
            "year",
            "food",
            "water",
            "coffee",
            "tea",
            "book",
            "phone",
            "computer",
            "car",
            "house",
            "city",
            "country",
            "good",
            "bad",
            "nice",
            "beautiful",
            "big",
            "small",
            "hot",
            "cold",
            "happy",
            "sad",
            "tired",
            "hungry",
            "thirsty",
            "like",
            "love",
            "want",
            "need",
            "have",
            "get",
            "go",
            "come",
            "see",
            "hear",
            "talk",
            "listen",
            "read",
            "write",
            "eat",
            "drink",
            "walk",
            "run",
            "sleep",
            "wake",
            "open",
            "close",
            "buy",
            "sell",
            "give",
            "take",
            "make",
            "do",
            "say",
            "tell",
            "ask",
            "answer",
            "school",
            "teacher",
            "student",
        }

        # Advanced vocabulary for intermediate+ learners (B1+ level)
        self.intermediate_vocabulary = {
            "marketing",
            "business",
            "presentation",
            "meeting",
            "client",
            "project",
            "pronunciation",
            "accent",
            "fluent",
            "improve",
            "practice",
            "conference",
            "professional",
            "colleague",
            "manager",
            "responsibility",
            "deadline",
            "objective",
            "strategy",
            "networking",
            "communication",
            "opportunity",
            "development",
            "experience",
            "organization",
            "decision",
            "investment",
            "technology",
            "innovation",
            "collaboration",
            "achievement",
            "performance",
            "efficiency",
            "productivity",
            "creativity",
            "leadership",
            "motivation",
            "challenge",
            "solution",
            "analysis",
            "research",
            "investigation",
            "implementation",
            "evaluation",
            "recommendation",
        }

        # Common non-English words and proper nouns to exclude
        self.excluded_words = {
            "muy",  # Spanish
            "que",  # Spanish/French
            "el",
            "la",
            "los",
            "las",  # Spanish articles
            "de",
            "del",
            "en",
            "con",
            "por",
            "para",  # Spanish prepositions
            "whatsapp",
            "instagram",
            "facebook",
            "youtube",  # Platforms
            "zoom",
            "google",
            "microsoft",
            "apple",  # Brands
            "covid",
            "coronavirus",  # Current events
            "ok",
            "okay",
            "yeah",
            "yep",
            "nope",
            "hmm",
            "uhm",  # Interjections
        }

    def _normalize_word(self, word: str) -> str:
        """Convert word to its base form (singular, present tense)."""
        word_lower = word.lower().strip()

        # Simple pluralization rules - convert to singular
        if word_lower.endswith("ies") and len(word_lower) > 4:
            # stories -> story, cities -> city
            return word_lower[:-3] + "y"
        elif word_lower.endswith("es") and len(word_lower) > 3:
            # boxes -> box, wishes -> wish, but keep "yes"
            if word_lower.endswith(("shes", "ches", "xes", "zes")):
                return word_lower[:-2]
            elif word_lower not in ("yes", "goes", "does"):
                return word_lower[:-1]
        elif word_lower.endswith("s") and len(word_lower) > 3:
            # words -> word, cats -> cat, but keep "this", "yes", etc.
            if word_lower not in (
                "this",
                "yes",
                "his",
                "its",
                "was",
                "has",
                "bus",
                "gas",
                "glass",
                "class",
                "pass",
                "less",
                "business",
                "process",
                "success",
                "express",
                "address",
                "access",
                "congress",
                "progress",
                "stress",
            ):
                return word_lower[:-1]

        # Simple verb forms - convert to base form
        if word_lower.endswith("ing") and len(word_lower) > 5:
            # learning -> learn, wishing -> wish
            base = word_lower[:-3]
            # Handle double consonants (running -> run)
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
                return base[:-1]
            return base
        elif word_lower.endswith("ed") and len(word_lower) > 4:
            # learned -> learn, wished -> wish
            return word_lower[:-2]

        return word_lower

    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar forms of the same base word."""
        norm1 = self._normalize_word(word1)
        norm2 = self._normalize_word(word2)
        return norm1 == norm2

    def _deduplicate_vocabulary(self, vocab_list: List[str]) -> List[str]:
        """Remove duplicate words, keeping only the base/singular form."""
        seen_normalized = set()
        deduplicated = []

        for word in vocab_list:
            normalized = self._normalize_word(word)
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                deduplicated.append(normalized)  # Store the normalized form

        return deduplicated

    def _is_valid_english_word(self, word: str) -> bool:
        """Check if word is a valid English vocabulary word to track."""
        word_lower = word.lower()

        # Must be alphabetic only
        if not word.isalpha():
            return False

        # Must be reasonable length (3-15 characters)
        if len(word) < 3 or len(word) > 15:
            return False

        # Exclude proper nouns (capitalized words that aren't sentence starters)
        if (
            word[0].isupper()
            and word_lower not in self.basic_vocabulary
            and word_lower not in self.intermediate_vocabulary
        ):
            return False

        # Exclude non-English and excluded words
        if word_lower in self.excluded_words:
            return False

        # Exclude very common words that aren't useful for tracking
        stop_words = {
            "the",
            "and",
            "but",
            "or",
            "so",
            "if",
            "when",
            "where",
            "why",
            "how",
            "what",
            "who",
            "which",
            "this",
            "that",
            "these",
            "those",
            "here",
            "there",
            "now",
            "then",
            "very",
            "much",
            "many",
            "some",
            "any",
            "all",
            "every",
            "each",
            "more",
            "most",
            "less",
            "few",
            "little",
            "big",
            "small",
            "good",
            "bad",
            "well",
            "better",
            "best",
            "nice",
            "great",
            "said",
            "thank",
            "thanks",
        }

        if word_lower in stop_words:
            return False

        return True

    async def analyze_vocabulary(self, text: str, user_level: str = "A1") -> VocabularyAnalysis:
        """Analyze vocabulary used in the text - English words only."""
        words = self._extract_words(text)

        new_words = []
        advanced_words = []
        basic_words = []

        for word in words:
            # Only process valid English words
            if not self._is_valid_english_word(word):
                continue

            word_lower = word.lower()

            if word_lower in self.basic_vocabulary:
                basic_words.append(word_lower)
            elif word_lower in self.intermediate_vocabulary:
                advanced_words.append(word_lower)
                new_words.append(word_lower)  # Intermediate words are worth tracking
            else:
                # Check if it's a meaningful new word (not in our basic sets)
                if len(word) >= 4:  # Reasonable length for new vocabulary
                    new_words.append(word_lower)

        # Remove duplicates while preserving order
        new_words = list(dict.fromkeys(new_words))
        advanced_words = list(dict.fromkeys(advanced_words))
        basic_words = list(dict.fromkeys(basic_words))

        return VocabularyAnalysis(
            new_words=new_words[:10],  # Limit to top 10
            advanced_words=advanced_words[:5],
            basic_words=basic_words[:15],
        )

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        # Remove punctuation and split into words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        # Filter out very short words and common stop words
        stop_words = {
            "i",
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "as",
            "it",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
        }

        meaningful_words = [word for word in words if len(word) > 2 and word not in stop_words]

        return meaningful_words

    async def get_user_learned_vocabulary(self, user_id: str) -> List[str]:
        """Get vocabulary already learned by user."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            result = await conn.fetchrow(
                "SELECT vocab_learned FROM learning_stats WHERE user_id = $1", uuid.UUID(user_id)
            )

            if result and result["vocab_learned"]:
                return result["vocab_learned"]
            return []

        finally:
            await conn.close()

    async def get_vocabulary_word_count(self, user_id: str) -> int:
        """Return the total number of words learned by the user."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            result = await conn.fetchrow(
                "SELECT vocab_learned FROM learning_stats WHERE user_id = $1",
                uuid.UUID(user_id),
            )

            if result and result["vocab_learned"]:
                return len(result["vocab_learned"])
            return 0

        finally:
            await conn.close()

    async def update_learning_stats(self, user_id: str, session_id: str, message_text: str) -> Dict:
        """Update learning statistics for a user based on their message."""
        self.logger.info(f"Updating learning stats for user {user_id}")

        try:
            # Analyze the message vocabulary only
            vocab_analysis = await self.analyze_vocabulary(message_text)

            # Get existing vocabulary
            learned_vocab = await self.get_user_learned_vocabulary(user_id)

            # Normalize existing vocabulary to prevent duplicates
            normalized_existing = [self._normalize_word(word) for word in learned_vocab]

            # Find truly new vocabulary (English words only)
            new_vocab = []
            for word in vocab_analysis.new_words:
                normalized_word = self._normalize_word(word)
                # Check if this normalized word is already in the user's vocabulary
                if normalized_word not in normalized_existing:
                    new_vocab.append(normalized_word)
                    normalized_existing.append(normalized_word)  # Prevent duplicates within this session

            # Update database with new vocabulary only
            if new_vocab:
                await self._update_database_stats(user_id, new_vocab)

            stats = {
                "vocabulary_analysis": {
                    "new_words_found": len(new_vocab),
                    "new_words": new_vocab,
                    "advanced_words": vocab_analysis.advanced_words,
                    "total_learned": len(learned_vocab) + len(new_vocab),
                },
            }

            self.logger.info(f"Learning stats updated for user {user_id}: {len(new_vocab)} new words")
            return stats

        except Exception as e:
            self.logger.error(f"Failed to update learning stats for user {user_id}: {e}")
            return {"error": str(e)}

    async def _update_database_stats(self, user_id: str, new_vocab: List[str]):
        """Update the learning_stats table with new vocabulary only."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            # Get current vocabulary
            current_stats = await conn.fetchrow(
                "SELECT vocab_learned FROM learning_stats WHERE user_id = $1", uuid.UUID(user_id)
            )

            if current_stats:
                # Update existing stats
                current_vocab = current_stats["vocab_learned"] or []
                # Add new vocabulary
                updated_vocab = current_vocab + new_vocab

                # Update database
                await conn.execute(
                    """UPDATE learning_stats
                       SET vocab_learned = $1, last_updated = now()
                       WHERE user_id = $2""",
                    updated_vocab,
                    uuid.UUID(user_id),
                )

            else:
                # Create new stats entry
                await conn.execute(
                    """INSERT INTO learning_stats (user_id, vocab_learned, last_updated)
                       VALUES ($1, $2, now())""",
                    uuid.UUID(user_id),
                    new_vocab,
                )

        finally:
            await conn.close()

    async def cleanup_user_vocabulary(self, user_id: str) -> Dict:
        """Clean up user's vocabulary by removing invalid words and duplicates."""
        try:
            # Get current vocabulary
            current_vocab = await self.get_user_learned_vocabulary(user_id)

            # Filter out invalid words and normalize to singular forms
            valid_vocab = []
            removed_words = []

            for word in current_vocab:
                if self._is_valid_english_word(word):
                    normalized = self._normalize_word(word)
                    valid_vocab.append(normalized)
                else:
                    removed_words.append(word)

            # Remove duplicates (keeping only singular forms)
            deduplicated_vocab = self._deduplicate_vocabulary(valid_vocab)

            # Update database with cleaned vocabulary
            database_url = get_database_url()
            conn = await asyncpg.connect(database_url)

            try:
                await conn.execute(
                    """UPDATE learning_stats
                       SET vocab_learned = $1, last_updated = now()
                       WHERE user_id = $2""",
                    deduplicated_vocab,
                    uuid.UUID(user_id),
                )
            finally:
                await conn.close()

            duplicates_removed = len(valid_vocab) - len(deduplicated_vocab)

            self.logger.info(
                f"Cleaned vocabulary for user {user_id}: kept {len(deduplicated_vocab)}, removed {len(removed_words)} invalid, {duplicates_removed} duplicates"
            )

            return {
                "original_count": len(current_vocab),
                "cleaned_count": len(deduplicated_vocab),
                "removed_words": removed_words,
                "duplicates_removed": duplicates_removed,
                "final_vocabulary": deduplicated_vocab,
            }

        except Exception as e:
            self.logger.error(f"Failed to cleanup vocabulary for user {user_id}: {e}")
            return {"error": str(e)}

    async def get_learning_summary(self, user_id: str) -> Dict:
        """Get a summary of user's learning progress (vocabulary only)."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            stats = await conn.fetchrow(
                "SELECT vocab_learned, last_updated FROM learning_stats WHERE user_id = $1",
                uuid.UUID(user_id),
            )

            if stats:
                vocab_count = len(stats["vocab_learned"] or [])

                return {
                    "vocabulary": {
                        "total_words_learned": vocab_count,
                        "recent_words": (stats["vocab_learned"] or [])[-5:] if stats["vocab_learned"] else [],
                    },
                    "last_updated": stats["last_updated"],
                }

            return {"vocabulary": {"total_words_learned": 0}}

        finally:
            await conn.close()


# Singleton instance
_learning_stats_manager = None


def get_learning_stats_manager() -> LearningStatsManager:
    """Get singleton instance of LearningStatsManager."""
    global _learning_stats_manager
    if _learning_stats_manager is None:
        _learning_stats_manager = LearningStatsManager()
    return _learning_stats_manager
