"""
Database configuration for PostgreSQL checkpointer with automatic setup
"""
import os
import asyncio
import asyncpg
import uuid
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from agent.settings import settings

# Global flag to track if setup has been done
_setup_done = False

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
    
    sql_content = script_file.read_text(encoding='utf-8')
    
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


async def get_or_create_user(phone_number: str) -> str:
    """Get existing user or create new user by phone number."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        # Try to find existing user
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE phone_number = $1", 
            phone_number
        )
        
        if user:
            return str(user['id'])
        
        # Create new user
        user_id = await conn.fetchval(
            "INSERT INTO users (phone_number) VALUES ($1) RETURNING id",
            phone_number
        )
        
        # Create initial learning stats for new user
        await conn.execute(
            """INSERT INTO learning_stats (user_id, vocab_learned, grammar_issues) 
               VALUES ($1, $2, $3)""",
            user_id, [], json.dumps({})
        )
        
        print(f"✅ Created new user: {phone_number} with ID: {user_id}")
        return str(user_id)
        
    finally:
        await conn.close()


async def get_or_create_session(user_id: str) -> str:
    """Get active session or create new session for user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        # Try to find active session (ended_at is NULL)
        session = await conn.fetchrow(
            "SELECT id FROM sessions WHERE user_id = $1 AND ended_at IS NULL",
            uuid.UUID(user_id)
        )
        
        if session:
            return str(session['id'])
        
        # Create new session
        session_id = await conn.fetchval(
            "INSERT INTO sessions (user_id) VALUES ($1) RETURNING id",
            uuid.UUID(user_id)
        )
        
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
            uuid.UUID(session_id), sender, message
        )
        print(f"✅ Logged {sender} message for session: {session_id}")
        
    finally:
        await conn.close()


async def get_user_learning_stats(user_id: str) -> dict:
    """Get learning statistics for a user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        stats = await conn.fetchrow(
            """SELECT u.current_level, ls.vocab_learned, ls.grammar_issues 
               FROM users u 
               LEFT JOIN learning_stats ls ON u.id = ls.user_id 
               WHERE u.id = $1""",
            uuid.UUID(user_id)
        )
        
        if stats:
            return {
                'current_level': stats['current_level'] or 'A1',
                'vocab_learned': stats['vocab_learned'] or [],
                'grammar_issues': stats['grammar_issues'] or {}
            }
        
        return {'current_level': 'A1', 'vocab_learned': [], 'grammar_issues': {}}
        
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
        checkpointer = await get_checkpointer()
        print("✅ PostgreSQL checkpointer connection and setup successful!")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL checkpointer connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection and setup
    asyncio.run(test_checkpointer_connection()) 