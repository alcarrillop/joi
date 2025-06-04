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


async def execute_sql_script(script_path: str):
    """Execute a SQL script file against the database."""
    database_url = get_database_url()

    # Read the SQL script
    script_file = Path(script_path)
    if not script_file.exists():
        raise FileNotFoundError(f"SQL script not found: {script_path}")

    sql_content = script_file.read_text(encoding="utf-8")

    # Connect and execute
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(sql_content)
        print(f"✅ SQL script executed successfully: {script_path}")
    except Exception as e:
        print(f"❌ Error executing SQL script {script_path}: {e}")
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
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        # Try to find existing user
        user = await conn.fetchrow("SELECT id FROM users WHERE phone_number = $1", phone_number)

        if user:
            # Update name if provided and not already set
            if name:
                await conn.execute(
                    "UPDATE users SET name = $1 WHERE id = $2 AND (name IS NULL OR name = '')", name, user["id"]
                )
            return str(user["id"])

        # Create new user with name if provided
        user_id = await conn.fetchval(
            "INSERT INTO users (phone_number, name) VALUES ($1, $2) RETURNING id", phone_number, name
        )

        print(f"✅ Created new user: {phone_number} with ID: {user_id}" + (f" and name: {name}" if name else ""))
        return str(user_id)

    finally:
        await conn.close()


async def update_user_name(user_id: str, name: str):
    """Update user name if not already set."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        await conn.execute(
            "UPDATE users SET name = $1 WHERE id = $2 AND (name IS NULL OR name = '')", name, uuid.UUID(user_id)
        )
        print(f"✅ Updated name for user {user_id}: {name}")

    finally:
        await conn.close()


async def get_or_create_session(user_id: str) -> str:
    """Get active session or create new session for user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        # Try to find the most recent session for the user
        session = await conn.fetchrow(
            "SELECT id FROM sessions WHERE user_id = $1 ORDER BY started_at DESC LIMIT 1", uuid.UUID(user_id)
        )

        if session:
            return str(session["id"])

        # Create new session
        session_id = await conn.fetchval("INSERT INTO sessions (user_id) VALUES ($1) RETURNING id", uuid.UUID(user_id))

        print(f"✅ Created new session: {session_id} for user: {user_id}")
        return str(session_id)

    finally:
        await conn.close()


async def log_message(session_id: str, sender: str, message: str):
    """Log a message to the messages table."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        await conn.execute(
            "INSERT INTO messages (session_id, sender, message) VALUES ($1, $2, $3)",
            uuid.UUID(session_id),
            sender,
            message,
        )
        print(f"✅ Logged {sender} message for session: {session_id}")

    finally:
        await conn.close()


async def get_user_stats(user_id: str) -> Dict:
    """Get user learning statistics (vocabulary only) from the new user_word_stats table."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        # Get vocabulary from the new user_word_stats table
        vocab_words = await conn.fetch(
            "SELECT word FROM user_word_stats WHERE user_id = $1 ORDER BY word",
            uuid.UUID(user_id),
        )

        return {
            "vocab_learned": [row["word"] for row in vocab_words],
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
        print("✅ PostgreSQL checkpointer connection and setup successful!")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL checkpointer connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection and setup
    asyncio.run(test_checkpointer_connection())
