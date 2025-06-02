"""
Database configuration for PostgreSQL checkpointer with automatic setup
"""
import os
import asyncio
from functools import lru_cache
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from joi.settings import settings

# Global flag to track if setup has been done
_setup_done = False

@lru_cache
def get_database_url() -> str:
    """Get the PostgreSQL database URL from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return database_url


async def ensure_tables_exist():
    """Ensure PostgreSQL tables exist (run setup once)."""
    global _setup_done
    if not _setup_done:
        database_url = get_database_url()
        # Create a temporary checkpointer just for setup
        checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
        async with checkpointer as cp:
            await cp.setup()
        _setup_done = True


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