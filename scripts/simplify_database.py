#!/usr/bin/env python3
"""
Simplify Database Schema
========================

This script removes unnecessary tables and keeps only the core components
needed for a good memory system:

ESSENTIAL TABLES (Keep):
- users: Basic user management
- sessions: Conversation sessions
- messages: Conversation history (core of memory system)
- learning_stats: Simple learning progress
- checkpoint_*: LangGraph workflow state

REMOVE:
- All complex assessment and curriculum tables
- Detailed progress tracking tables
- Competency and module tables
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
    from agent.core.database import get_database_url
except ImportError as e:
    print(f"❌ Import error: {e}")
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


# Tables to KEEP (essential for memory system)
ESSENTIAL_TABLES = {
    "users",
    "sessions",
    "messages",
    "learning_stats",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}

# Tables to REMOVE (complex features not needed for MVP)
TABLES_TO_REMOVE = {
    "assessment_configs",
    "assessments",
    "conversation_assessments",
    "detected_errors",
    "grammar_errors",
    "learning_goals",
    "learning_modules",
    "competencies",
    "level_detections",
    "practice_activities",
    "skill_progressions",
    "user_goals",
    "user_progress",
    "vocabulary_progress",
}


async def analyze_current_schema():
    """Analyze current database schema and data."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        # Get all tables
        all_tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)

        log("📊 Current Database Analysis", Colors.PURPLE)
        log("=" * 60, Colors.CYAN)

        current_tables = {table["table_name"] for table in all_tables}

        log(f"Total tables: {len(current_tables)}", Colors.BLUE)

        # Show what will be kept
        essential_found = current_tables & ESSENTIAL_TABLES
        log(f"\n✅ ESSENTIAL TABLES (Will Keep): {len(essential_found)}", Colors.GREEN)
        for table in sorted(essential_found):
            # Get row count
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            log(f"  • {table}: {count:,} rows", Colors.GREEN)

        # Show what will be removed
        to_remove_found = current_tables & TABLES_TO_REMOVE
        log(f"\n🗑️ TABLES TO REMOVE: {len(to_remove_found)}", Colors.RED)
        for table in sorted(to_remove_found):
            # Get row count
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            log(f"  • {table}: {count:,} rows", Colors.RED)

        # Show unknown tables
        unknown_tables = current_tables - ESSENTIAL_TABLES - TABLES_TO_REMOVE
        if unknown_tables:
            log(f"\n❓ UNKNOWN TABLES: {len(unknown_tables)}", Colors.YELLOW)
            for table in sorted(unknown_tables):
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                log(f"  • {table}: {count:,} rows", Colors.YELLOW)

        return to_remove_found, essential_found

    finally:
        await conn.close()


async def remove_unnecessary_tables(tables_to_remove):
    """Remove unnecessary tables from database."""
    if not tables_to_remove:
        success("No tables to remove!")
        return True

    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        log(f"🗑️ Removing {len(tables_to_remove)} unnecessary tables...", Colors.PURPLE)

        # Drop tables in reverse dependency order to avoid foreign key issues
        for table in sorted(tables_to_remove):
            try:
                log(f"Dropping table: {table}")
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                success(f"Removed: {table}")
            except Exception as e:
                error(f"Failed to drop {table}: {e}")

        return True

    except Exception as e:
        error(f"Failed to remove tables: {e}")
        return False
    finally:
        await conn.close()


async def verify_essential_data():
    """Verify essential data is still intact."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        log("🔍 Verifying essential data integrity...", Colors.PURPLE)

        # Check users
        user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        log(f"Users: {user_count:,}", Colors.BLUE)

        # Check sessions
        session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        log(f"Sessions: {session_count:,}", Colors.BLUE)

        # Check messages
        message_count = await conn.fetchval("SELECT COUNT(*) FROM messages")
        log(f"Messages: {message_count:,}", Colors.BLUE)

        # Check learning stats
        stats_count = await conn.fetchval("SELECT COUNT(*) FROM learning_stats")
        log(f"Learning Stats: {stats_count:,}", Colors.BLUE)

        # Test a basic query to ensure relationships work
        recent_messages = await conn.fetchval("""
            SELECT COUNT(*)
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            JOIN users u ON s.user_id = u.id
            WHERE m.timestamp >= NOW() - INTERVAL '7 days'
        """)
        log(f"Recent messages (last 7 days): {recent_messages:,}", Colors.BLUE)

        success("✅ All essential data verified!")
        return True

    except Exception as e:
        error(f"Data verification failed: {e}")
        return False
    finally:
        await conn.close()


async def main():
    """Main function."""
    log("🧹 Starting Database Simplification", Colors.CYAN, "🧹")
    log("=" * 60, Colors.CYAN)

    # 1. Analyze current schema
    tables_to_remove, essential_tables = await analyze_current_schema()

    if not tables_to_remove:
        success("Database is already simplified!")
        return True

    log("\n" + "=" * 60, Colors.CYAN)
    log("⚠️ WARNING: This will permanently delete data!", Colors.YELLOW, "⚠️")
    log("Essential tables for memory system will be preserved.", Colors.GREEN)
    log("=" * 60, Colors.CYAN)

    # Confirmation
    response = input("\n🤔 Proceed with database simplification? (yes/no): ").strip().lower()
    if response != "yes":
        log("❌ Operation cancelled by user.", Colors.YELLOW)
        return False

    # 2. Remove unnecessary tables
    log("\n" + "=" * 60, Colors.CYAN)
    removal_success = await remove_unnecessary_tables(tables_to_remove)

    if not removal_success:
        error("Failed to remove tables!")
        return False

    # 3. Verify essential data
    log("\n" + "=" * 60, Colors.CYAN)
    verification_success = await verify_essential_data()

    if not verification_success:
        error("Data verification failed!")
        return False

    # Summary
    log("\n" + "=" * 60, Colors.CYAN)
    log("📊 SIMPLIFICATION COMPLETE", Colors.CYAN, "📊")
    log("=" * 60, Colors.CYAN)

    success(f"✅ Removed {len(tables_to_remove)} unnecessary tables")
    success(f"✅ Kept {len(essential_tables)} essential tables")
    success("✅ Memory system is now streamlined for production!")

    log("\n🎯 Your system now focuses on what matters:", Colors.GREEN)
    log("  • User management (users)", Colors.GREEN)
    log("  • Conversation tracking (sessions, messages)", Colors.GREEN)
    log("  • Simple learning stats (learning_stats)", Colors.GREEN)
    log("  • Workflow state (checkpoint tables)", Colors.GREEN)

    log("\n🚀 Ready for real user testing!", Colors.GREEN, "🚀")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
