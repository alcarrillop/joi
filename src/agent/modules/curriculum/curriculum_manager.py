"""
Simplified Curriculum Manager
============================

Lightweight curriculum management focused on vocabulary tracking only.
Provides level estimation and progress tracking based on English vocabulary learned.
"""

import logging
import uuid
from typing import Dict, List, Optional

import asyncpg

from agent.core.database import get_database_url

logger = logging.getLogger(__name__)


class SimplifiedCurriculumManager:
    """Simplified curriculum manager focused on vocabulary learning only."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

    async def get_learning_statistics(self, user_id: str) -> Dict:
        """Get basic learning statistics for a user."""
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

            # Get vocabulary stats from the new user_word_stats table
            vocab_count = await conn.fetchval(
                "SELECT COUNT(*) FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id)
            )

            return {
                "member_since": user_stats["created_at"],
                "total_sessions": user_stats["total_sessions"] or 0,
                "total_messages": user_stats["total_messages"] or 0,
                "vocabulary_learned": vocab_count or 0,
                "has_learning_data": vocab_count > 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {e}")
            return {"error": str(e)}
        finally:
            await conn.close()

    async def estimate_level_progress(self, user_id: str) -> Dict:
        """Estimate learning progress based on vocabulary learned."""
        try:
            stats = await self.get_learning_statistics(user_id)

            # Simple estimation based on vocabulary learned
            vocab_count = stats.get("vocabulary_learned", 0)
            message_count = stats.get("total_messages", 0)

            # CEFR level estimation based on vocabulary thresholds
            # Fixed logic: check requirements properly
            if vocab_count >= 400 and message_count >= 250:
                estimated_level = "C1"
            elif vocab_count >= 250 and message_count >= 150:
                estimated_level = "B2"
            elif vocab_count >= 150 and message_count >= 80:
                estimated_level = "B1"
            elif vocab_count >= 75 and message_count >= 30:
                estimated_level = "A2"
            else:
                estimated_level = "A1"

            next_level = self.get_next_level(estimated_level)

            # Calculate progress to next level - FIXED LOGIC
            level_thresholds = {"A1": 75, "A2": 150, "B1": 250, "B2": 400, "C1": 600, "C2": 800}

            # Get current and previous thresholds
            current_threshold = level_thresholds.get(estimated_level, 0)

            # For progress calculation, we need the previous level's threshold
            levels = self.get_level_progression()
            try:
                current_index = levels.index(estimated_level)
                previous_level = levels[current_index - 1] if current_index > 0 else None
            except ValueError:
                previous_level = None

            previous_threshold = level_thresholds.get(previous_level, 0) if previous_level else 0
            next_threshold = level_thresholds.get(next_level, current_threshold) if next_level else current_threshold

            progress_to_next = 0
            if next_level and next_threshold > current_threshold:
                # Progress from previous level to next level
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
                    "vocabulary_strength": min(100, (vocab_count / 600) * 100),  # Max at C1 level
                    "conversation_practice": min(100, (message_count / 300) * 100),
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to estimate level progress: {e}")
            return {"error": str(e)}


# Singleton instance
_curriculum_manager = None


def get_curriculum_manager() -> SimplifiedCurriculumManager:
    """Get singleton instance of SimplifiedCurriculumManager."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = SimplifiedCurriculumManager()
    return _curriculum_manager
