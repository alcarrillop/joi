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

    async def get_user_level(self, user_id: str) -> str:
        """Get user's current level from the database."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            level = await conn.fetchval("SELECT current_level FROM users WHERE id = $1", user_id)
            return level or "A1"
        finally:
            await conn.close()

    async def update_user_level(self, user_id: str, new_level: str) -> bool:
        """Update user's level in the database."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            await conn.execute("UPDATE users SET current_level = $1 WHERE id = $2", new_level, user_id)
            self.logger.info(f"Updated user {user_id} level to {new_level}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update user level: {e}")
            return False
        finally:
            await conn.close()

    async def get_learning_statistics(self, user_id: str) -> Dict:
        """Get basic learning statistics for a user."""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        try:
            # Get basic stats from existing tables
            user_stats = await conn.fetchrow(
                """
                SELECT u.current_level, u.created_at,
                       COUNT(DISTINCT s.id) as total_sessions,
                       COUNT(m.id) as total_messages
                FROM users u
                LEFT JOIN sessions s ON u.id = s.user_id
                LEFT JOIN messages m ON s.id = m.session_id AND m.sender = 'user'
                WHERE u.id = $1
                GROUP BY u.id, u.current_level, u.created_at
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
                "current_level": user_stats["current_level"],
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

    async def should_level_up(self, user_id: str) -> Dict:
        """Simple logic to determine if user should level up."""
        try:
            stats = await self.get_learning_statistics(user_id)
            current_level = stats.get("current_level", "A1")

            # Simple criteria for leveling up
            vocab_threshold = {"A1": 50, "A2": 100, "B1": 200, "B2": 350, "C1": 500, "C2": 1000}

            message_threshold = {"A1": 20, "A2": 50, "B1": 100, "B2": 200, "C1": 300, "C2": 500}

            vocab_count = stats.get("vocabulary_learned", 0)
            message_count = stats.get("total_messages", 0)

            vocab_ready = vocab_count >= vocab_threshold.get(current_level, 1000)
            message_ready = message_count >= message_threshold.get(current_level, 1000)

            should_advance = vocab_ready and message_ready
            next_level = self.get_next_level(current_level)

            return {
                "should_advance": should_advance and next_level is not None,
                "current_level": current_level,
                "next_level": next_level,
                "criteria": {
                    "vocabulary_ready": vocab_ready,
                    "message_ready": message_ready,
                    "vocab_progress": f"{vocab_count}/{vocab_threshold.get(current_level, 1000)}",
                    "message_progress": f"{message_count}/{message_threshold.get(current_level, 1000)}",
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to check level up: {e}")
            return {"error": str(e)}


# Singleton instance
_curriculum_manager = None


def get_curriculum_manager() -> SimplifiedCurriculumManager:
    """Get singleton instance of SimplifiedCurriculumManager."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = SimplifiedCurriculumManager()
    return _curriculum_manager
