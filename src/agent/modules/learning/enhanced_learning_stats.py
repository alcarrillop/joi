"""
Enhanced Learning Stats Manager
=============================

Advanced vocabulary learning tracking with improved analysis capabilities.
Includes language detection, better normalization, and enhanced metrics.
"""

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_groq import ChatGroq
from pydantic import BaseModel

from agent.core.database import get_db_connection
from agent.settings import get_settings

logger = logging.getLogger(__name__)


class AdvancedVocabularyAnalysis(BaseModel):
    """Enhanced analysis of vocabulary used in a message."""

    new_words: List[str]
    advanced_words: List[str]
    basic_words: List[str]
    complexity_score: float
    language_confidence: float
    word_diversity: float
    avg_word_length: float
    reading_level: str


class VocabularyCategories:
    """Centralized vocabulary categorization with enhanced word lists."""

    def __init__(self):
        # Core A1-A2 vocabulary (essential basics)
        self.a1_vocabulary = {
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
            "eat",
            "drink",
            "food",
            "water",
            "coffee",
            "tea",
            "book",
            "phone",
            "house",
            "city",
        }

        # A2 vocabulary (elementary)
        self.a2_vocabulary = {
            "learning",
            "practice",
            "study",
            "computer",
            "car",
            "country",
            "tired",
            "hungry",
            "thirsty",
            "talk",
            "listen",
            "read",
            "write",
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
            "office",
            "hospital",
            "restaurant",
            "shopping",
            "money",
            "travel",
            "holiday",
            "weather",
            "season",
            "winter",
            "summer",
            "spring",
            "autumn",
            "birthday",
            "party",
        }

        # B1 vocabulary (intermediate)
        self.b1_vocabulary = {
            "business",
            "meeting",
            "client",
            "project",
            "presentation",
            "conference",
            "professional",
            "colleague",
            "manager",
            "responsibility",
            "deadline",
            "objective",
            "strategy",
            "communication",
            "opportunity",
            "development",
            "experience",
            "organization",
            "decision",
            "technology",
            "innovation",
            "achievement",
            "performance",
            "efficiency",
            "creativity",
            "leadership",
            "motivation",
            "challenge",
            "solution",
            "environment",
            "culture",
        }

        # B2+ vocabulary (upper intermediate and advanced)
        self.b2_plus_vocabulary = {
            "analysis",
            "research",
            "investigation",
            "implementation",
            "evaluation",
            "recommendation",
            "methodology",
            "collaboration",
            "productivity",
            "investment",
            "networking",
            "entrepreneurship",
            "sustainability",
            "globalization",
            "digitalization",
            "transformation",
            "optimization",
            "prioritization",
            "specialization",
            "diversification",
            "consolidation",
            "authentication",
            "authorization",
            "customization",
            "standardization",
        }

        # Academic and technical vocabulary
        self.academic_vocabulary = {
            "hypothesis",
            "methodology",
            "analysis",
            "synthesis",
            "critique",
            "paradigm",
            "phenomenon",
            "correlation",
            "implication",
            "interpretation",
            "perspective",
            "framework",
            "concept",
            "theory",
            "principle",
            "approach",
            "dimension",
            "component",
            "structure",
            "function",
        }

    @lru_cache(maxsize=1000)
    def get_word_level(self, word: str) -> str:
        """Get the CEFR level of a word."""
        word_lower = word.lower()

        if word_lower in self.a1_vocabulary:
            return "A1"
        elif word_lower in self.a2_vocabulary:
            return "A2"
        elif word_lower in self.b1_vocabulary:
            return "B1"
        elif word_lower in self.b2_plus_vocabulary:
            return "B2+"
        elif word_lower in self.academic_vocabulary:
            return "C1+"
        else:
            # Estimate based on word length and complexity
            if len(word) <= 4:
                return "A2"
            elif len(word) <= 6:
                return "B1"
            elif len(word) <= 8:
                return "B2"
            else:
                return "C1+"


class EnhancedLearningStatsManager:
    """Enhanced learning statistics manager with advanced analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.llm = ChatGroq(
            model=self.settings.TEXT_MODEL_NAME,
            api_key=self.settings.GROQ_API_KEY,
            temperature=0.3,
        )

        self.vocabulary_categories = VocabularyCategories()
        self._cache = {}

        # Enhanced stop words and exclusions
        self.stop_words = {
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
            "said",
            "thank",
            "thanks",
            "yeah",
            "okay",
            "well",
            "just",
            "really",
            "actually",
        }

        # Non-English patterns to exclude
        self.non_english_patterns = {
            # Spanish
            "muy",
            "que",
            "el",
            "la",
            "los",
            "las",
            "de",
            "del",
            "en",
            "con",
            "por",
            "para",
            "esta",
            "este",
            "son",
            "es",
            "está",
            "tienen",
            "tiene",
            # Common brands/platforms
            "whatsapp",
            "instagram",
            "facebook",
            "youtube",
            "zoom",
            "google",
            "microsoft",
            "apple",
            "amazon",
            "netflix",
            "spotify",
            "uber",
            # Internet slang
            "lol",
            "omg",
            "wtf",
            "btw",
            "fyi",
            "asap",
            "ok",
            "thx",
            "pls",
        }

    @lru_cache(maxsize=500)
    def _advanced_normalize_word(self, word: str) -> str:
        """Enhanced word normalization with better stemming."""
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

        # Enhanced verb forms
        if word_lower.endswith("ies") and len(word_lower) > 4:
            return word_lower[:-3] + "y"
        elif word_lower.endswith("ied") and len(word_lower) > 4:
            return word_lower[:-3] + "y"
        elif word_lower.endswith("ing") and len(word_lower) > 5:
            base = word_lower[:-3]
            # Handle double consonants
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeioulrwxy":
                return base[:-1]
            return base
        elif word_lower.endswith("ed") and len(word_lower) > 4:
            base = word_lower[:-2]
            # Handle -ied endings
            if base.endswith("i"):
                return base[:-1] + "y"
            return base
        elif word_lower.endswith("es") and len(word_lower) > 3:
            if word_lower.endswith(("shes", "ches", "xes", "zes", "sses")):
                return word_lower[:-2]
            elif word_lower not in ("yes", "goes", "does", "comes"):
                return word_lower[:-1]
        elif word_lower.endswith("s") and len(word_lower) > 3:
            # Preserve common words ending in 's'
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

        return word_lower

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Simple language detection based on common patterns."""
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z]+\b", text_lower)

        if not words:
            return "unknown", 0.0

        # Spanish indicators
        spanish_indicators = ["que", "es", "la", "el", "de", "en", "muy", "con", "por", "para"]
        spanish_count = sum(1 for word in words if word in spanish_indicators)

        # English indicators
        english_indicators = ["the", "and", "is", "are", "this", "that", "have", "with"]
        english_count = sum(1 for word in words if word in english_indicators)

        total_indicators = spanish_count + english_count
        if total_indicators == 0:
            return "unknown", 0.5

        if english_count > spanish_count:
            confidence = english_count / len(words)
            return "english", min(confidence * 2, 1.0)
        else:
            confidence = spanish_count / len(words)
            return "spanish", min(confidence * 2, 1.0)

    def _calculate_complexity_score(self, words: List[str]) -> float:
        """Calculate text complexity based on vocabulary sophistication."""
        if not words:
            return 0.0

        total_score = 0
        level_scores = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "B2+": 5, "C1+": 6}

        for word in words:
            level = self.vocabulary_categories.get_word_level(word)
            total_score += level_scores.get(level, 3)  # Default to B1 level

        return min(total_score / len(words), 6.0)

    def _calculate_word_diversity(self, words: List[str]) -> float:
        """Calculate lexical diversity (Type-Token Ratio)."""
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _estimate_reading_level(self, complexity_score: float, avg_word_length: float) -> str:
        """Estimate reading level based on complexity metrics."""
        if complexity_score <= 2.0 and avg_word_length <= 4.5:
            return "A1-A2 (Beginner)"
        elif complexity_score <= 3.0 and avg_word_length <= 5.5:
            return "B1 (Intermediate)"
        elif complexity_score <= 4.0 and avg_word_length <= 6.5:
            return "B2 (Upper Intermediate)"
        elif complexity_score <= 5.0:
            return "C1 (Advanced)"
        else:
            return "C2 (Proficient)"

    def _is_valid_english_word(self, word: str) -> bool:
        """Enhanced validation for English words."""
        if not word or len(word) < 2:
            return False

        word_lower = word.lower()

        # Must be alphabetic
        if not word.isalpha():
            return False

        # Reasonable length (2-20 characters)
        if len(word) < 2 or len(word) > 20:
            return False

        # Exclude non-English words
        if word_lower in self.non_english_patterns:
            return False

        # Exclude common stop words
        if word_lower in self.stop_words:
            return False

        # Exclude proper nouns (simple heuristic)
        if (
            word[0].isupper()
            and word_lower not in self.vocabulary_categories.a1_vocabulary
            and word_lower not in self.vocabulary_categories.a2_vocabulary
        ):
            return False

        return True

    async def analyze_vocabulary_enhanced(self, text: str) -> AdvancedVocabularyAnalysis:
        """Enhanced vocabulary analysis with advanced metrics."""
        # Language detection
        detected_lang, lang_confidence = self._detect_language(text)

        # Extract and filter words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        valid_words = [word for word in words if self._is_valid_english_word(word)]

        # Normalize words
        normalized_words = [self._advanced_normalize_word(word) for word in valid_words]
        normalized_words = [w for w in normalized_words if w and len(w) > 1]

        if not normalized_words:
            return AdvancedVocabularyAnalysis(
                new_words=[],
                advanced_words=[],
                basic_words=[],
                complexity_score=0.0,
                language_confidence=lang_confidence,
                word_diversity=0.0,
                avg_word_length=0.0,
                reading_level="Unknown",
            )

        # Categorize words by level
        new_words = []
        advanced_words = []
        basic_words = []

        for word in set(normalized_words):  # Remove duplicates for categorization
            level = self.vocabulary_categories.get_word_level(word)

            if level in ["A1", "A2"]:
                basic_words.append(word)
            elif level in ["B1"]:
                new_words.append(word)
            else:  # B2+, C1+
                advanced_words.append(word)
                new_words.append(word)

        # Calculate metrics
        complexity_score = self._calculate_complexity_score(normalized_words)
        word_diversity = self._calculate_word_diversity(normalized_words)
        avg_word_length = sum(len(word) for word in normalized_words) / len(normalized_words)
        reading_level = self._estimate_reading_level(complexity_score, avg_word_length)

        return AdvancedVocabularyAnalysis(
            new_words=new_words[:15],
            advanced_words=advanced_words[:10],
            basic_words=basic_words[:20],
            complexity_score=complexity_score,
            language_confidence=lang_confidence,
            word_diversity=word_diversity,
            avg_word_length=avg_word_length,
            reading_level=reading_level,
        )

    async def get_learning_trends(self, user_id: str, days: int = 7) -> Dict:
        """Get learning trends over time."""
        conn = await get_db_connection()

        try:
            # Get vocabulary growth over time
            since_date = datetime.now() - timedelta(days=days)

            daily_stats = await conn.fetch(
                """
                SELECT DATE(created_at) as day, COUNT(*) as words_learned
                FROM user_word_stats
                WHERE user_id = $1 AND created_at >= $2
                GROUP BY DATE(created_at)
                ORDER BY day
            """,
                uuid.UUID(user_id),
                since_date,
            )

            # Get most challenging words (low frequency, recent)
            challenging_words = await conn.fetch(
                """
                SELECT word, freq, last_used_at
                FROM user_word_stats
                WHERE user_id = $1 AND freq <= 3 AND last_used_at >= $2
                ORDER BY last_used_at DESC
                LIMIT 10
            """,
                uuid.UUID(user_id),
                since_date,
            )

            return {
                "daily_progress": [{"date": str(row[0]), "words": row[1]} for row in daily_stats],
                "challenging_words": [
                    {"word": row[0], "frequency": row[1], "last_used": row[2]} for row in challenging_words
                ],
                "trend_period": f"{days} days",
            }

        finally:
            await conn.close()

    async def get_vocabulary_insights(self, user_id: str) -> Dict:
        """Get detailed vocabulary insights and recommendations."""
        conn = await get_db_connection()

        try:
            # Get vocabulary distribution by level
            words = await conn.fetch("SELECT word, freq FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id))

            level_distribution = defaultdict(int)
            total_words = 0

            for word_row in words:
                word = word_row[0]
                level = self.vocabulary_categories.get_word_level(word)
                level_distribution[level] += 1
                total_words += 1

            # Calculate vocabulary gaps
            gaps = []
            if level_distribution.get("A1", 0) < 30:
                gaps.append("Need more basic A1 vocabulary")
            if level_distribution.get("B1", 0) < 20:
                gaps.append("Could benefit from more intermediate B1 words")

            return {
                "total_vocabulary": total_words,
                "level_distribution": dict(level_distribution),
                "vocabulary_gaps": gaps,
                "diversity_score": len(level_distribution) / 6.0,  # Max 6 levels
            }

        finally:
            await conn.close()


# Singleton instance
_enhanced_learning_manager = None


def get_enhanced_learning_manager() -> EnhancedLearningStatsManager:
    """Get singleton instance of EnhancedLearningStatsManager."""
    global _enhanced_learning_manager
    if _enhanced_learning_manager is None:
        _enhanced_learning_manager = EnhancedLearningStatsManager()
    return _enhanced_learning_manager
