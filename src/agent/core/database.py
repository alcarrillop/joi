"""
Database configuration for PostgreSQL checkpointer with automatic setup
"""

import asyncio
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict

import asyncpg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Global flag to track if setup has been done
_setup_done = False

logger = logging.getLogger(__name__)


@lru_cache
def get_database_url() -> str:
    """Get the PostgreSQL database URL from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return database_url


async def get_db_connection():
    """Get a database connection using asyncpg."""
    database_url = get_database_url()
    return await asyncpg.connect(database_url, timeout=30)


async def execute_sql_script(script_path: str):
    """Execute a SQL script file against the database."""
    # Read the SQL script
    script_file = Path(script_path)
    if not script_file.exists():
        raise FileNotFoundError(f"SQL script not found: {script_path}")

    sql_content = script_file.read_text(encoding="utf-8")

    # Connect and execute
    conn = await get_db_connection()
    try:
        await conn.execute(sql_content)
        logger.info("SQL script executed successfully: %s", script_path)
    except Exception as e:
        logger.error("Error executing SQL script %s: %s", script_path, e)
        raise
    finally:
        await conn.close()


async def ensure_tables_exist():
    """Ensure PostgreSQL tables exist (run setup once)."""
    global _setup_done
    if not _setup_done:
        database_url = get_database_url()
        # Create a temporary checkpointer just for setup
        checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
        async with checkpointer as cp:
            await cp.setup()

        # Execute our custom schema
        await execute_sql_script("scripts/init_db.sql")
        _setup_done = True


async def get_or_create_user(phone_number: str, name: str = None) -> str:
    """Get existing user or create new user by phone number."""
    conn = await get_db_connection()

    try:
        # Try to find existing user
        user = await conn.fetchrow("SELECT id FROM users WHERE phone_number = $1", phone_number)

        if user:
            # Update name if provided and not already set
            if name:
                await conn.execute(
                    "UPDATE users SET name = $1 WHERE id = $2 AND (name IS NULL OR name = '')",
                    name,
                    user[0],
                )
            return str(user[0])

        # Create new user with name if provided
        user_result = await conn.fetchrow(
            "INSERT INTO users (phone_number, name) VALUES ($1, $2) RETURNING id",
            phone_number,
            name,
        )
        user_id = user_result[0]

        logger.info(
            "Created new user %s with ID %s%s",
            phone_number,
            user_id,
            f" and name: {name}" if name else "",
        )
        return str(user_id)

    finally:
        await conn.close()


async def update_user_name(user_id: str, name: str):
    """Update user name if not already set."""
    conn = await get_db_connection()

    try:
        await conn.execute(
            "UPDATE users SET name = $1 WHERE id = $2 AND (name IS NULL OR name = '')",
            name,
            uuid.UUID(user_id),
        )
        logger.info("Updated name for user %s: %s", user_id, name)

    finally:
        await conn.close()


async def get_or_create_session(user_id: str) -> str:
    """Get active session or create new session for user."""
    conn = await get_db_connection()

    try:
        # Try to find the most recent session for the user
        session = await conn.fetchrow(
            "SELECT id FROM sessions WHERE user_id = $1 ORDER BY started_at DESC LIMIT 1",
            uuid.UUID(user_id),
        )

        if session:
            return str(session[0])

        # Create new session
        session_result = await conn.fetchrow(
            "INSERT INTO sessions (user_id) VALUES ($1) RETURNING id",
            uuid.UUID(user_id),
        )
        session_id = session_result[0]

        logger.info("Created new session %s for user %s", session_id, user_id)
        return str(session_id)

    finally:
        await conn.close()


async def log_message(session_id: str, sender: str, message: str, message_type: str = "text"):
    """Log a message to the messages table with type information.

    Args:
        session_id: Session identifier
        sender: 'user' or 'agent'
        message: Message content
        message_type: Type of message ('text', 'audio', 'image')
                     Users can send: text, audio, image
                     Agents can send: text, audio
    """
    conn = await get_db_connection()

    try:
        await conn.execute(
            "INSERT INTO messages (session_id, sender, message, message_type) VALUES ($1, $2, $3, $4)",
            uuid.UUID(session_id),
            sender,
            message,
            message_type,
        )
        logger.debug("Logged %s %s message for session %s", sender, message_type, session_id)

    finally:
        await conn.close()


async def get_user_stats(user_id: str) -> Dict:
    """Get user learning statistics (vocabulary only) from the new user_word_stats table."""
    conn = await get_db_connection()

    try:
        vocab_words = await conn.fetch(
            "SELECT word FROM user_word_stats WHERE user_id = $1 ORDER BY word",
            uuid.UUID(user_id),
        )

        return {
            "vocab_learned": [row[0] for row in vocab_words],
        }

    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {"vocab_learned": []}
    finally:
        await conn.close()


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Get a PostgreSQL checkpointer instance with automatic table setup.
    This is the main function to use in the application.

    Returns:
        AsyncPostgresSaver: Ready-to-use checkpointer with tables created
    """
    # Ensure tables exist first
    await ensure_tables_exist()

    # Return a new checkpointer instance for use
    database_url = get_database_url()
    return AsyncPostgresSaver.from_conn_string(database_url)


async def test_checkpointer_connection():
    """Test that the checkpointer connection and setup works correctly."""
    try:
        await get_checkpointer()
        logger.info("PostgreSQL checkpointer connection and setup successful")
        return True
    except Exception as e:
        logger.error("PostgreSQL checkpointer connection failed: %s", e)
        return False


if __name__ == "__main__":
    # Test the connection and setup
    asyncio.run(test_checkpointer_connection())
