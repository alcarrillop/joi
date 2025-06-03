"""
Learning Stats Manager
=====================

Manages vocabulary learning and grammar error detection.
Updates learning_stats table with user progress.
"""

import json
import logging
import re
import uuid
from typing import Dict, List, Optional

import asyncpg
from agent.core.database import get_database_url
from agent.settings import settings
from langchain_groq import ChatGroq
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VocabularyAnalysis(BaseModel):
    """Analysis of vocabulary used in a message."""

    new_words: List[str]
    advanced_words: List[str]
    basic_words: List[str]


class GrammarAnalysis(BaseModel):
    """Analysis of grammar errors in a message."""

    has_errors: bool
    error_types: List[str]
    error_descriptions: List[str]
    corrected_text: Optional[str] = None


class LearningStatsManager:
    """Manages learning statistics for users."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=2,
        )

        # Common vocabulary by level
        self.basic_vocabulary = {
            "hello",
            "hi",
            "bye",
            "thank",
            "you",
            "please",
            "yes",
            "no",
            "good",
            "bad",
            "big",
            "small",
            "have",
            "do",
            "go",
            "come",
            "see",
            "know",
            "think",
            "want",
            "like",
            "love",
            "work",
            "home",
            "family",
            "friend",
            "time",
            "day",
            "night",
            "eat",
            "drink",
            "food",
            "water",
            "money",
            "house",
            "car",
            "book",
            "read",
            "write",
            "speak",
            "listen",
            "learn",
            "study",
            "school",
            "teacher",
            "student",
        }

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
        }

    async def analyze_vocabulary(self, text: str, user_level: str = "A1") -> VocabularyAnalysis:
        """Analyze vocabulary used in the text."""
        words = self._extract_words(text)

        new_words = []
        advanced_words = []
        basic_words = []

        for word in words:
            word_lower = word.lower()

            if word_lower in self.basic_vocabulary:
                basic_words.append(word)
            elif word_lower in self.intermediate_vocabulary:
                advanced_words.append(word)
                if len(word) > 3 and word.isalpha():  # Only meaningful new words
                    new_words.append(word)
            elif len(word) > 4 and word.isalpha():  # New technical/advanced words
                new_words.append(word)

        return VocabularyAnalysis(
            new_words=new_words[:10],  # Limit to top 10
            advanced_words=advanced_words[:5],
            basic_words=basic_words[:15],
        )

    async def analyze_grammar(self, text: str) -> GrammarAnalysis:
        """Analyze grammar errors in the text using LLM."""
        if len(text.strip()) < 5:
            return GrammarAnalysis(has_errors=False, error_types=[], error_descriptions=[])

        prompt = f"""
Analyze this English text for grammar errors:

Text: "{text}"

Return a JSON with:
1. has_errors: true/false
2. error_types: list of error types (e.g., ["verb_tense", "subject_verb_agreement", "article_usage"])
3. error_descriptions: list of brief descriptions of each error
4. corrected_text: corrected version if errors exist

Only identify clear, obvious grammar errors. Don't be overly strict with informal conversation.

JSON:"""

        try:
            response = await self.llm.ainvoke(prompt)

            # Try to extract JSON from response
            response_text = str(response.content) if hasattr(response, "content") else str(response)

            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)

                return GrammarAnalysis(
                    has_errors=result.get("has_errors", False),
                    error_types=result.get("error_types", []),
                    error_descriptions=result.get("error_descriptions", []),
                    corrected_text=result.get("corrected_text"),
                )

        except Exception as e:
            self.logger.warning(f"Grammar analysis failed: {e}")

        return GrammarAnalysis(has_errors=False, error_types=[], error_descriptions=[])

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

    async def update_learning_stats(self, user_id: str, session_id: str, message_text: str) -> Dict:
        """Update learning statistics for a user based on their message."""
        self.logger.info(f"Updating learning stats for user {user_id}")

        try:
            # Analyze the message
            vocab_analysis = await self.analyze_vocabulary(message_text)
            grammar_analysis = await self.analyze_grammar(message_text)

            # Get existing vocabulary
            learned_vocab = await self.get_user_learned_vocabulary(user_id)

            # Find truly new vocabulary
            new_vocab = []
            for word in vocab_analysis.new_words:
                if word.lower() not in [v.lower() for v in learned_vocab]:
                    new_vocab.append(word)

            # Update database
            await self._update_database_stats(user_id, new_vocab, grammar_analysis)

            stats = {
                "vocabulary_analysis": {
                    "new_words_found": len(new_vocab),
                    "new_words": new_vocab,
                    "advanced_words": vocab_analysis.advanced_words,
                    "total_learned": len(learned_vocab) + len(new_vocab),
                },
                "grammar_analysis": {
                    "has_errors": grammar_analysis.has_errors,
                    "error_count": len(grammar_analysis.error_types),
                    "error_types": grammar_analysis.error_types,
                },
            }

            self.logger.info(
                f"Learning stats updated for user {user_id}: {len(new_vocab)} new words, {len(grammar_analysis.error_types)} grammar issues"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Failed to update learning stats for user {user_id}: {e}")
            return {"error": str(e)}

    async def _update_database_stats(self, user_id: str, new_vocab: List[str], grammar_analysis: GrammarAnalysis):
        """Update the learning_stats table in database."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            # Get current stats
            current_stats = await conn.fetchrow(
                "SELECT vocab_learned, grammar_issues FROM learning_stats WHERE user_id = $1", uuid.UUID(user_id)
            )

            if current_stats:
                # Update existing stats
                current_vocab = current_stats["vocab_learned"] or []
                current_grammar_str = current_stats["grammar_issues"] or "{}"

                # Parse JSON string to dict
                try:
                    current_grammar = (
                        json.loads(current_grammar_str) if isinstance(current_grammar_str, str) else current_grammar_str
                    )
                except (json.JSONDecodeError, TypeError):
                    current_grammar = {}

                # Add new vocabulary
                updated_vocab = current_vocab + new_vocab

                # Update grammar issues
                updated_grammar = dict(current_grammar) if isinstance(current_grammar, dict) else {}
                if grammar_analysis.has_errors:
                    for error_type in grammar_analysis.error_types:
                        updated_grammar[error_type] = updated_grammar.get(error_type, 0) + 1

                # Update database
                await conn.execute(
                    """UPDATE learning_stats
                       SET vocab_learned = $1, grammar_issues = $2, last_updated = now()
                       WHERE user_id = $3""",
                    updated_vocab,
                    json.dumps(updated_grammar),
                    uuid.UUID(user_id),
                )

            else:
                # Create new stats entry
                grammar_dict = {}
                if grammar_analysis.has_errors:
                    for error_type in grammar_analysis.error_types:
                        grammar_dict[error_type] = 1

                await conn.execute(
                    """INSERT INTO learning_stats (user_id, vocab_learned, grammar_issues, last_updated)
                       VALUES ($1, $2, $3, now())""",
                    uuid.UUID(user_id),
                    new_vocab,
                    json.dumps(grammar_dict),
                )

        finally:
            await conn.close()

    async def get_learning_summary(self, user_id: str) -> Dict:
        """Get a summary of user's learning progress."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            stats = await conn.fetchrow(
                "SELECT vocab_learned, grammar_issues, last_updated FROM learning_stats WHERE user_id = $1",
                uuid.UUID(user_id),
            )

            if stats:
                vocab_count = len(stats["vocab_learned"] or [])
                grammar_issues_str = stats["grammar_issues"] or "{}"

                # Parse JSON string to dict
                try:
                    grammar_issues = (
                        json.loads(grammar_issues_str) if isinstance(grammar_issues_str, str) else grammar_issues_str
                    )
                except (json.JSONDecodeError, TypeError):
                    grammar_issues = {}

                total_grammar_errors = sum(grammar_issues.values()) if isinstance(grammar_issues, dict) else 0

                return {
                    "vocabulary": {
                        "total_words_learned": vocab_count,
                        "recent_words": (stats["vocab_learned"] or [])[-5:] if stats["vocab_learned"] else [],
                    },
                    "grammar": {
                        "total_errors_identified": total_grammar_errors,
                        "error_breakdown": grammar_issues,
                        "most_common_error": max(grammar_issues, key=grammar_issues.get)
                        if grammar_issues and isinstance(grammar_issues, dict)
                        else None,
                    },
                    "last_updated": stats["last_updated"],
                }

            return {"vocabulary": {"total_words_learned": 0}, "grammar": {"total_errors_identified": 0}}

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
