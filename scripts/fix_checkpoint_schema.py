#!/usr/bin/env python3
"""
Fix LangGraph Checkpoint Schema
==============================

This script fixes the checkpoint table schema issues by:
1. Dropping problematic checkpoint tables
2. Recreating them with proper schema
3. Ensuring clean state for LangGraph checkpointer

This fixes the "task_path" column already exists error.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")
src_dir = os.path.normpath(src_dir)
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

# Import after path setup
import asyncpg  # noqa: E402

try:
    from agent.core.database import get_checkpointer, get_database_url
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"❌ Current working directory: {os.getcwd()}")
    print(f"❌ Script directory: {script_dir}")
    print(f"❌ Source directory: {src_dir}")
    print(f"❌ Source directory exists: {os.path.exists(src_dir)}")
    sys.exit(1)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log(message: str, color: str = Colors.BLUE, symbol: str = "ℹ"):
    """Log a message with color and timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}{symbol} [{timestamp}] {message}{Colors.END}")


def success(message: str):
    log(message, Colors.GREEN, "✅")


def warning(message: str):
    log(message, Colors.YELLOW, "⚠️")


def error(message: str):
    log(message, Colors.RED, "❌")


async def fix_checkpoint_schema():
    """Fix the checkpoint schema issues."""
    log("🔧 Fixing LangGraph checkpoint schema...", Colors.PURPLE)

    try:
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        # List of checkpoint tables that might have schema issues
        checkpoint_tables = ["checkpoint_migrations", "checkpoints", "checkpoint_blobs", "checkpoint_writes"]

        # Check which tables exist
        existing_tables = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = ANY($1::text[])
        """,
            checkpoint_tables,
        )

        existing_table_names = [table["table_name"] for table in existing_tables]
        log(f"Found checkpoint tables: {', '.join(existing_table_names)}")

        # Drop checkpoint tables in reverse dependency order
        drop_order = ["checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"]

        for table in drop_order:
            if table in existing_table_names:
                log(f"Dropping table: {table}")
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                success(f"Dropped table: {table}")

        # Verify tables are dropped
        remaining_tables = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = ANY($1::text[])
        """,
            checkpoint_tables,
        )

        if remaining_tables:
            warning(f"Some tables still exist: {[t['table_name'] for t in remaining_tables]}")
        else:
            success("All checkpoint tables successfully dropped")

        await conn.close()

        log("🧪 Testing checkpointer setup...", Colors.PURPLE)

        # Now test if we can create a fresh checkpointer
        async with await get_checkpointer():
            success("Checkpointer setup successful!")

        return True

    except Exception as e:
        error(f"Checkpoint schema fix failed: {e}")
        return False


async def main():
    """Main function."""
    log("🔧 Starting Checkpoint Schema Fix", Colors.CYAN, "🔧")
    log("=" * 60, Colors.CYAN)

    start_time = datetime.now()

    success_result = await fix_checkpoint_schema()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log("\n" + "=" * 60, Colors.CYAN)
    log("📊 SCHEMA FIX RESULTS", Colors.CYAN, "📊")
    log("=" * 60, Colors.CYAN)

    if success_result:
        log("  PASSED: Checkpoint Schema Fix", Colors.GREEN)
        log("🎉 Schema fixed! Ready for testing.", Colors.GREEN, "🎉")
    else:
        log("  FAILED: Checkpoint Schema Fix", Colors.RED)
        log("❌ Schema fix failed. Check logs above.", Colors.RED)

    log(f"\n⏱️ Total Duration: {duration:.1f} seconds", Colors.BLUE)

    return success_result


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
