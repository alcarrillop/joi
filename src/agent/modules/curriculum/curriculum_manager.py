"""
Simplified Curriculum Manager
============================

Lightweight curriculum management focused on memory and basic learning tracking.
Removed complex competency and assessment systems to focus on essential functionality.
"""

import logging
from typing import Dict, List, Optional

import asyncpg

from agent.core.database import get_database_url

logger = logging.getLogger(__name__)


class SimplifiedCurriculumManager:
    """Simplified curriculum manager focused on memory and basic learning."""

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
                user_id,
            )

            if not user_stats:
                return {"error": "User not found"}

            # Get learning stats if available
            learning_stats = await conn.fetchrow(
                "SELECT vocab_learned, grammar_issues FROM learning_stats WHERE user_id = $1", user_id
            )

            vocab_count = (
                len(learning_stats["vocab_learned"]) if learning_stats and learning_stats["vocab_learned"] else 0
            )

            return {
                "member_since": user_stats["created_at"],
                "total_sessions": user_stats["total_sessions"] or 0,
                "total_messages": user_stats["total_messages"] or 0,
                "vocabulary_learned": vocab_count,
                "has_learning_data": learning_stats is not None,
            }

        except Exception as e:
            self.logger.error(f"Failed to get learning statistics: {e}")
            return {"error": str(e)}
        finally:
            await conn.close()

    async def estimate_level_progress(self, user_id: str) -> Dict:
        """Estimate learning progress based on vocabulary and messages."""
        try:
            stats = await self.get_learning_statistics(user_id)

            # Simple estimation based on vocabulary learned
            vocab_count = stats.get("vocabulary_learned", 0)
            message_count = stats.get("total_messages", 0)

            # Basic level estimation thresholds
            if vocab_count >= 500 and message_count >= 300:
                estimated_level = "C1"
            elif vocab_count >= 350 and message_count >= 200:
                estimated_level = "B2"
            elif vocab_count >= 200 and message_count >= 100:
                estimated_level = "B1"
            elif vocab_count >= 100 and message_count >= 50:
                estimated_level = "A2"
            else:
                estimated_level = "A1"

            next_level = self.get_next_level(estimated_level)

            return {
                "estimated_level": estimated_level,
                "next_level": next_level,
                "vocabulary_learned": vocab_count,
                "total_messages": message_count,
                "progress_indicators": {
                    "vocabulary_strength": min(100, (vocab_count / 500) * 100),
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
