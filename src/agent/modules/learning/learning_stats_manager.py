"""
Learning Stats Manager
=====================

Manages vocabulary learning tracking for English words only.
Updates user_word_stats table with user vocabulary progress using relational tracking.
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
        """Get vocabulary already learned by user from the new relational table."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            result = await conn.fetch(
                "SELECT word FROM user_word_stats WHERE user_id = $1 ORDER BY word", uuid.UUID(user_id)
            )

            return [row["word"] for row in result]

        finally:
            await conn.close()

    async def get_vocabulary_word_count(self, user_id: str) -> int:
        """Return the total number of words learned by the user from the new table."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM user_word_stats WHERE user_id = $1",
                uuid.UUID(user_id),
            )

            return result or 0

        finally:
            await conn.close()

    async def get_user_top_words(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's most frequently used words."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            result = await conn.fetch(
                """SELECT word, freq, last_used_at
                   FROM user_word_stats
                   WHERE user_id = $1
                   ORDER BY freq DESC, last_used_at DESC
                   LIMIT $2""",
                uuid.UUID(user_id),
                limit,
            )

            return [{"word": row["word"], "frequency": row["freq"], "last_used": row["last_used_at"]} for row in result]

        finally:
            await conn.close()

    async def increment_word_frequency(self, user_id: str, word: str) -> None:
        """Increment the frequency of a word for a user using the PostgreSQL function."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            await conn.execute("SELECT inc_word_freq($1, $2)", uuid.UUID(user_id), word)

        finally:
            await conn.close()

    async def update_learning_stats(self, user_id: str, session_id: str, message_text: str) -> Dict:
        """Update learning statistics for a user based on their message using the new relational table."""
        self.logger.info(f"Updating learning stats for user {user_id}")

        try:
            # Analyze the message vocabulary only
            vocab_analysis = await self.analyze_vocabulary(message_text)

            # Get existing vocabulary from the new table
            learned_vocab = await self.get_user_learned_vocabulary(user_id)

            # Normalize existing vocabulary to prevent duplicates
            normalized_existing = [self._normalize_word(word) for word in learned_vocab]

            # Find truly new vocabulary (English words only)
            new_vocab = []
            frequency_updates = []

            for word in vocab_analysis.new_words:
                normalized_word = self._normalize_word(word)

                if normalized_word in normalized_existing:
                    # Word already exists, just increment frequency
                    frequency_updates.append(normalized_word)
                else:
                    # New word, will be added with frequency 1
                    new_vocab.append(normalized_word)
                    normalized_existing.append(normalized_word)

            # Update frequencies for all normalized words (new and existing)
            all_words_to_update = new_vocab + frequency_updates

            for word in all_words_to_update:
                await self.increment_word_frequency(user_id, word)

            stats = {
                "vocabulary_analysis": {
                    "new_words_found": len(new_vocab),
                    "new_words": new_vocab,
                    "frequency_updates": len(frequency_updates),
                    "advanced_words": vocab_analysis.advanced_words,
                    "total_learned": len(normalized_existing),
                },
            }

            self.logger.info(
                f"Learning stats updated for user {user_id}: {len(new_vocab)} new words, {len(frequency_updates)} frequency updates"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Failed to update learning stats for user {user_id}: {e}")
            return {"error": str(e)}

    async def cleanup_user_vocabulary(self, user_id: str) -> Dict:
        """Clean up user's vocabulary by removing invalid words and duplicates from the new table."""
        try:
            database_url = get_database_url()
            conn = await asyncpg.connect(database_url)

            try:
                # Get current vocabulary from new table
                current_words = await conn.fetch(
                    "SELECT word, freq FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id)
                )

                # Filter and normalize words
                valid_words = []
                removed_words = []

                for row in current_words:
                    word = row["word"]
                    if self._is_valid_english_word(word):
                        normalized = self._normalize_word(word)
                        valid_words.append((normalized, row["freq"]))
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

                # Delete all existing entries for this user
                await conn.execute("DELETE FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id))

                # Insert cleaned vocabulary
                if word_freqs:
                    values = [(uuid.UUID(user_id), word, freq) for word, freq in word_freqs.items()]

                    await conn.executemany(
                        """INSERT INTO user_word_stats (user_id, word, freq, created_at, last_used_at)
                           VALUES ($1, $2, $3, NOW(), NOW())""",
                        values,
                    )

                self.logger.info(
                    f"Cleaned vocabulary for user {user_id}: kept {len(word_freqs)}, removed {len(removed_words)} invalid, {duplicates_removed} duplicates"
                )

                return {
                    "original_count": len(current_words),
                    "cleaned_count": len(word_freqs),
                    "removed_words": removed_words,
                    "duplicates_removed": duplicates_removed,
                    "final_vocabulary": list(word_freqs.keys()),
                }

            finally:
                await conn.close()

        except Exception as e:
            self.logger.error(f"Failed to cleanup vocabulary for user {user_id}: {e}")
            return {"error": str(e)}

    async def get_learning_summary(self, user_id: str) -> Dict:
        """Get a summary of user's learning progress from the new relational table."""
        try:
            # Get total word count
            vocab_count = await self.get_vocabulary_word_count(user_id)

            # Get top words with frequencies
            top_words = await self.get_user_top_words(user_id, limit=5)

            # Get recent words (last 5)
            database_url = get_database_url()
            conn = await asyncpg.connect(database_url)

            try:
                recent_words = await conn.fetch(
                    """SELECT word, freq, last_used_at
                       FROM user_word_stats
                       WHERE user_id = $1
                       ORDER BY last_used_at DESC
                       LIMIT 5""",
                    uuid.UUID(user_id),
                )

                recent_words_list = [
                    {"word": row["word"], "frequency": row["freq"], "last_used": row["last_used_at"]}
                    for row in recent_words
                ]
            finally:
                await conn.close()

            return {
                "vocabulary": {
                    "total_words_learned": vocab_count,
                    "top_words": top_words,
                    "recent_words": recent_words_list,
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


# Singleton instance
_learning_stats_manager = None


def get_learning_stats_manager() -> LearningStatsManager:
    """Get singleton instance of LearningStatsManager."""
    global _learning_stats_manager
    if _learning_stats_manager is None:
        _learning_stats_manager = LearningStatsManager()
    return _learning_stats_manager
