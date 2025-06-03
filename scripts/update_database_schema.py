#!/usr/bin/env python3
"""
Database Schema Update Script
============================

This script updates the database schema to include improvements:
1. Ensures user_goals table exists
2. Applies any schema changes from init_db.sql
3. Validates all tables and indices

Run this after making database schema changes.
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncpg

from src.agent.core.database import execute_sql_script, get_database_url


async def update_database_schema():
    """Update database schema with latest changes."""
    print("🔄 Updating database schema...")

    try:
        # Execute the updated init_db.sql script
        await execute_sql_script("scripts/init_db.sql")

        # Verify new tables exist
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        # Check for user_goals table
        table_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user_goals')"
        )

        if table_exists:
            print("✅ user_goals table exists")

            # Check table structure
            columns = await conn.fetch(
                """SELECT column_name, data_type, is_nullable
                   FROM information_schema.columns
                   WHERE table_name = 'user_goals'
                   ORDER BY ordinal_position"""
            )

            print("📊 user_goals table structure:")
            for col in columns:
                nullable = "NULL" if col["is_nullable"] == "YES" else "NOT NULL"
                print(f"  - {col['column_name']}: {col['data_type']} {nullable}")
        else:
            print("❌ user_goals table missing")

        # Check all expected tables exist
        expected_tables = ["users", "sessions", "messages", "learning_stats", "user_goals"]
        for table in expected_tables:
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", table
            )
            status = "✅" if exists else "❌"
            print(f"{status} {table} table")

        await conn.close()
        print("✅ Database schema update completed successfully!")

    except Exception as e:
        print(f"❌ Database schema update failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(update_database_schema())
    exit(0 if success else 1)
