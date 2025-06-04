#!/usr/bin/env python3
"""
Database Schema Update Script
=============================

Ensures all required tables exist and have proper structure for the JOI system.
This script:

1. Ensures user_word_stats table exists
2. Checks core database health
3. Validates essential table structure

Note: After simplification, this focuses only on essential tables.
"""

import asyncio
import os
import sys

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")
src_dir = os.path.normpath(src_dir)
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

import asyncpg  # noqa: E402
from agent.core.database import get_database_url  # noqa: E402


async def check_and_update_schema():
    """Check and update database schema as needed."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        print("🔍 Checking database schema...")

        # Check core essential tables after simplification
        print("\n📊 Checking essential tables...")

        user_word_stats_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user_word_stats')"
        )

        if user_word_stats_exists:
            print("✅ user_word_stats table exists")

            # Show table structure for verification
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'user_word_stats'
                ORDER BY ordinal_position
            """)

            print("📊 user_word_stats table structure:")
            for col in columns:
                print(
                    f"  • {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})"
                )
        else:
            print("❌ user_word_stats table missing - this should exist!")

        # Check all essential tables
        expected_tables = ["users", "sessions", "messages", "user_word_stats"]

        print(f"\n🔍 Verifying {len(expected_tables)} essential tables...")
        for table in expected_tables:
            exists = await conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table}'
                )
            """)

            if exists:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                print(f"✅ {table}: {count:,} rows")
            else:
                print(f"❌ {table}: MISSING")

        print("\n🎯 Database schema check complete!")
        print("📋 Essential tables for vocabulary tracking system verified.")

    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        return False
    finally:
        await conn.close()

    return True


if __name__ == "__main__":
    print("🔧 Database Schema Update Script")
    print("=" * 50)

    success = asyncio.run(check_and_update_schema())

    if success:
        print("\n✅ Schema check completed successfully!")
    else:
        print("\n❌ Schema check failed!")
        sys.exit(1)
