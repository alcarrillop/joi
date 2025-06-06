"""
Database Operations for Learning Stats
=====================================

Separated database operations for better maintainability and testing.
Handles all database interactions for vocabulary tracking.
"""

import logging
import uuid
from typing import Any, Dict, List, Tuple

import asyncpg

from agent.core.database import get_db_connection

logger = logging.getLogger(__name__)


class VocabularyDBOperations:
    """Database operations for vocabulary tracking."""

    async def get_user_learned_vocabulary(self, user_id: str) -> List[str]:
        """Get vocabulary already learned by user from the relational table."""
        conn = await get_db_connection()
        try:
            result = await conn.fetch(
                "SELECT word FROM user_word_stats WHERE user_id = $1 ORDER BY word",
                uuid.UUID(user_id),
            )
            return [row[0] for row in result]
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error getting vocabulary for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting vocabulary for user {user_id}: {e}")
            return []
        finally:
            await conn.close()

    async def get_vocabulary_word_count(self, user_id: str) -> int:
        """Return the total number of words learned by the user."""
        conn = await get_db_connection()
        try:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM user_word_stats WHERE user_id = $1",
                uuid.UUID(user_id),
            )
            return result if result else 0
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error getting word count for user {user_id}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error getting word count for user {user_id}: {e}")
            return 0
        finally:
            await conn.close()

    async def get_user_top_words(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's most frequently used words."""
        conn = await get_db_connection()
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
            return [{"word": row[0], "frequency": row[1], "last_used": row[2]} for row in result]
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error getting top words for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting top words for user {user_id}: {e}")
            return []
        finally:
            await conn.close()

    async def increment_word_frequency(self, user_id: str, word: str) -> bool:
        """Increment the frequency of a word for a user using the PostgreSQL function."""
        conn = await get_db_connection()
        try:
            await conn.execute("SELECT inc_word_freq($1, $2)", uuid.UUID(user_id), word)
            return True
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error incrementing word '{word}' for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error incrementing word '{word}' for user {user_id}: {e}")
            return False
        finally:
            await conn.close()

    async def get_recent_words(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recently used words."""
        conn = await get_db_connection()
        try:
            result = await conn.fetch(
                """SELECT word, freq, last_used_at
                       FROM user_word_stats
                       WHERE user_id = $1
                       ORDER BY last_used_at DESC
                       LIMIT $2""",
                uuid.UUID(user_id),
                limit,
            )
            return [{"word": row[0], "frequency": row[1], "last_used": row[2]} for row in result]
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error getting recent words for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting recent words for user {user_id}: {e}")
            return []
        finally:
            await conn.close()

    async def get_all_user_words(self, user_id: str) -> List[Tuple[str, int]]:
        """Get all words with their frequencies for analysis."""
        conn = await get_db_connection()
        try:
            result = await conn.fetch("SELECT word, freq FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id))
            return [(row[0], row[1]) for row in result]
        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error getting all words for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting all words for user {user_id}: {e}")
            return []
        finally:
            await conn.close()

    async def cleanup_user_vocabulary(self, user_id: str, valid_words: List[Tuple[str, int]]) -> Dict[str, Any]:
        """Clean up user's vocabulary by replacing with valid normalized words."""
        conn = await get_db_connection()
        try:
            async with conn.transaction():
                # Get current count
                original_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id)
                )

                # Delete all existing entries for this user
                await conn.execute("DELETE FROM user_word_stats WHERE user_id = $1", uuid.UUID(user_id))

                # Insert cleaned vocabulary
                if valid_words:
                    values = [(uuid.UUID(user_id), word, freq) for word, freq in valid_words]
                    await conn.executemany(
                        """INSERT INTO user_word_stats (user_id, word, freq, created_at, last_used_at)
                           VALUES ($1, $2, $3, NOW(), NOW())""",
                        values,
                    )

                logger.info(f"Cleaned vocabulary for user {user_id}: {original_count} → {len(valid_words)} words")

                return {
                    "original_count": original_count,
                    "cleaned_count": len(valid_words),
                    "final_vocabulary": [word for word, _ in valid_words],
                }

        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error during cleanup for user {user_id}: {e}")
            return {"error": f"Database error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error during cleanup for user {user_id}: {e}")
            return {"error": f"Unexpected error: {e}"}
        finally:
            await conn.close()

    async def batch_increment_words(self, user_id: str, words: List[str]) -> int:
        """Batch increment multiple words for efficiency."""
        success_count = 0
        conn = await get_db_connection()
        try:
            async with conn.transaction():
                for word in words:
                    try:
                        await conn.execute("SELECT inc_word_freq($1, $2)", uuid.UUID(user_id), word)
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to increment word '{word}' for user {user_id}: {e}")
                        continue

            logger.info(f"Batch increment: {success_count}/{len(words)} words updated for user {user_id}")
            return success_count

        except asyncpg.exceptions.PostgresError as e:
            logger.error(f"Database error during batch increment for user {user_id}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during batch increment for user {user_id}: {e}")
            return 0
        finally:
            await conn.close()


# Singleton instance
_vocab_db_ops = None


def get_vocabulary_db_operations() -> VocabularyDBOperations:
    """Get singleton instance of VocabularyDBOperations."""
    global _vocab_db_ops
    if _vocab_db_ops is None:
        _vocab_db_ops = VocabularyDBOperations()
    return _vocab_db_ops
