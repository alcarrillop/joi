"""
Enhanced Curriculum Manager
===========================

Complete curriculum management using structured CEFR content.
Provides level estimation, progress tracking, and adaptive content selection.
Integrates with learning module for technical vocabulary data.
"""

import logging
import uuid
from typing import Dict, List, Optional

import asyncpg

from agent.core.database import get_database_url
from agent.modules.learning.db_operations import get_vocabulary_db_operations

from .curriculum_data import CurriculumData
from .models import CEFRLevel, Competency, LevelThresholds

logger = logging.getLogger(__name__)


class EnhancedCurriculumManager:
    """Enhanced curriculum manager with structured CEFR content and adaptive learning."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.curriculum_data = CurriculumData()
        self.level_thresholds = LevelThresholds()
        self.db_ops = get_vocabulary_db_operations()

        # Cache competencies for performance
        self._competencies_cache = None

    def get_level_progression(self) -> List[str]:
        """Get the standard CEFR level progression."""
        return ["A1", "A2", "B1", "B2", "C1", "C2"]

    def get_next_level(self, current_level: str) -> Optional[str]:
        """Get the next level in progression."""
        levels = self.get_level_progression()
        try:
            current_index = levels.index(current_level)
            if current_index < len(levels) - 1:
                return levels[current_index + 1]
        except ValueError:
            pass
        return None

    def get_all_competencies(self) -> List[Competency]:
        """Get all competencies across all levels (cached)."""
        if self._competencies_cache is None:
            self._competencies_cache = (
                self.curriculum_data.get_a1_competencies()
                + self.curriculum_data.get_a2_competencies()
                + self.curriculum_data.get_b1_competencies()
            )
        return self._competencies_cache

    def get_competencies_for_level(self, level: str) -> List[Competency]:
        """Get competencies for a specific CEFR level."""
        level_enum = CEFRLevel(level)
        if level_enum == CEFRLevel.A1:
            return self.curriculum_data.get_a1_competencies()
        elif level_enum == CEFRLevel.A2:
            return self.curriculum_data.get_a2_competencies()
        elif level_enum == CEFRLevel.B1:
            return self.curriculum_data.get_b1_competencies()
        else:
            return []  # Higher levels not implemented yet

    def get_vocabulary_for_level(self, level: str) -> List[str]:
        """Get target vocabulary for a specific level."""
        competencies = self.get_competencies_for_level(level)
        vocabulary = []
        for comp in competencies:
            vocabulary.extend(comp.key_vocabulary)
        return list(set(vocabulary))  # Remove duplicates

    async def get_learning_statistics(self, user_id: str) -> Dict:
        """Get basic learning statistics integrated with vocabulary data."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            # Get basic stats from existing tables
            user_stats = await conn.fetchrow(
                """
                SELECT u.created_at,
                       COUNT(DISTINCT s.id) as total_sessions,
                       COUNT(m.id) as total_messages
                FROM users u
                LEFT JOIN sessions s ON u.id = s.user_id
                LEFT JOIN messages m ON s.id = m.session_id AND m.sender = 'user'
                WHERE u.id = $1
                GROUP BY u.id, u.created_at
                """,
                uuid.UUID(user_id),
            )

            if not user_stats:
                return {"error": "User not found"}

            # Get vocabulary stats using db_operations
            vocab_count = await self.db_ops.get_vocabulary_word_count(user_id)

            return {
                "member_since": user_stats["created_at"],
                "total_sessions": user_stats["total_sessions"] or 0,
                "total_messages": user_stats["total_messages"] or 0,
                "vocabulary_learned": vocab_count,
                "has_learning_data": vocab_count > 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {e}")
            return {"error": str(e)}
        finally:
            await conn.close()

    async def estimate_level_progress(self, user_id: str) -> Dict:
        """Estimate learning progress with educational analysis and vocabulary insights."""
        try:
            stats = await self.get_learning_statistics(user_id)
            vocab_count = stats.get("vocabulary_learned", 0)
            message_count = stats.get("total_messages", 0)

            # CEFR level estimation using curriculum thresholds
            estimated_level = self._calculate_cefr_level(vocab_count, message_count)
            next_level = self.get_next_level(estimated_level)

            # Get curriculum-based insights
            vocabulary_insights = await self._analyze_vocabulary_against_curriculum(user_id, estimated_level)
            level_competencies = self.get_competencies_for_level(estimated_level)

            # Calculate progress metrics
            level_thresholds = {
                "A1": self.level_thresholds.A1,
                "A2": self.level_thresholds.A2,
                "B1": self.level_thresholds.B1,
                "B2": self.level_thresholds.B2,
                "C1": self.level_thresholds.C1,
                "C2": self.level_thresholds.C2,
            }

            current_threshold = level_thresholds.get(estimated_level, 0)
            previous_threshold = self._get_previous_threshold(estimated_level, level_thresholds)
            next_threshold = level_thresholds.get(next_level, current_threshold) if next_level else current_threshold

            progress_to_next = 0
            if next_level and next_threshold > current_threshold:
                progress_to_next = min(
                    100, ((vocab_count - previous_threshold) / (next_threshold - previous_threshold)) * 100
                )

            return {
                "estimated_level": estimated_level,
                "next_level": next_level,
                "vocabulary_learned": vocab_count,
                "total_messages": message_count,
                "progress_to_next_level": max(0, progress_to_next),
                "words_needed_for_next_level": max(0, next_threshold - vocab_count) if next_level else 0,
                "current_level_threshold": current_threshold,
                "previous_level_threshold": previous_threshold,
                "progress_indicators": {
                    "vocabulary_strength": min(100, (vocab_count / 600) * 100),
                    "conversation_practice": min(100, (message_count / 300) * 100),
                },
                "curriculum_insights": vocabulary_insights,
                "level_competencies": [
                    {
                        "name": comp.name,
                        "description": comp.description,
                        "skill_type": comp.skill_type.value,
                        "estimated_hours": comp.estimated_hours,
                    }
                    for comp in level_competencies[:3]  # Top 3 most relevant
                ],
                "educational_recommendations": self._generate_learning_recommendations(
                    estimated_level, vocab_count, vocabulary_insights
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to estimate level progress: {e}")
            return {"error": str(e)}

    def _calculate_cefr_level(self, vocab_count: int, message_count: int) -> str:
        """Calculate CEFR level based on vocabulary and interaction data."""
        if vocab_count >= 400 and message_count >= 250:
            return "C1"
        elif vocab_count >= 250 and message_count >= 150:
            return "B2"
        elif vocab_count >= 150 and message_count >= 80:
            return "B1"
        elif vocab_count >= 75 and message_count >= 30:
            return "A2"
        else:
            return "A1"

    def _get_previous_threshold(self, current_level: str, thresholds: Dict[str, int]) -> int:
        """Get the threshold for the previous level."""
        levels = self.get_level_progression()
        try:
            current_index = levels.index(current_level)
            previous_level = levels[current_index - 1] if current_index > 0 else None
            return thresholds.get(previous_level, 0) if previous_level else 0
        except ValueError:
            return 0

    async def _analyze_vocabulary_against_curriculum(self, user_id: str, level: str) -> Dict:
        """Analyze user's vocabulary against curriculum expectations."""
        try:
            user_words = await self.db_ops.get_user_learned_vocabulary(user_id)
            level_target_words = self.get_vocabulary_for_level(level)

            # Find overlap between user vocabulary and curriculum
            user_words_set = set(word.lower() for word in user_words)
            target_words_set = set(word.lower() for word in level_target_words)

            mastered_curriculum_words = user_words_set.intersection(target_words_set)
            missing_curriculum_words = target_words_set - user_words_set

            # Calculate curriculum mastery percentage
            mastery_percentage = (
                (len(mastered_curriculum_words) / len(target_words_set) * 100) if target_words_set else 0
            )

            return {
                "curriculum_mastery_percentage": round(mastery_percentage, 1),
                "mastered_curriculum_words": len(mastered_curriculum_words),
                "missing_curriculum_words": len(missing_curriculum_words),
                "total_curriculum_words": len(target_words_set),
                "key_missing_words": list(missing_curriculum_words)[:10],  # Top 10 missing
                "vocabulary_beyond_curriculum": len(user_words_set - target_words_set),
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze vocabulary against curriculum: {e}")
            return {"error": str(e)}

    def _generate_learning_recommendations(self, level: str, vocab_count: int, insights: Dict) -> List[str]:
        """Generate personalized learning recommendations."""
        recommendations = []

        mastery = insights.get("curriculum_mastery_percentage", 0)
        missing_words = insights.get("missing_curriculum_words", 0)

        if mastery < 50:
            recommendations.append(
                f"Focus on core {level} vocabulary - you've mastered {mastery:.0f}% of expected words"
            )
        elif mastery < 80:
            recommendations.append(
                f"Great progress! Complete your {level} vocabulary foundation ({missing_words} words remaining)"
            )
        else:
            recommendations.append(f"Excellent {level} vocabulary mastery! Ready to advance to more complex topics")

        if vocab_count < 50:
            recommendations.append("Build vocabulary through daily conversations about familiar topics")
        elif vocab_count < 200:
            recommendations.append("Expand vocabulary by discussing diverse topics and asking about new words")
        else:
            recommendations.append("Challenge yourself with advanced topics and specialized vocabulary")

        return recommendations


# Singleton instance
_curriculum_manager = None


def get_curriculum_manager() -> EnhancedCurriculumManager:
    """Get singleton instance of EnhancedCurriculumManager."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = EnhancedCurriculumManager()
    return _curriculum_manager
