#!/usr/bin/env python3
"""
Smart Database Initialization
============================

This script intelligently initializes the database:
- Checks if tables exist
- Creates only missing tables
- Runs SQL migrations only when needed
- Safe for both development and production

This replaces the verification/fix scripts with a single smart init script.
"""

import asyncio

import asyncpg
from agent.core.database import get_database_url


async def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = await conn.fetchval(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)", table_name
    )
    return result


async def init_database():
    """Initialize database with smart table creation."""
    print("🚀 Smart Database Initialization")
    print("=" * 50)

    try:
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)

        # Define essential tables in order of dependency
        essential_tables = ["users", "sessions", "messages", "user_word_stats"]

        print("🔍 Checking existing tables...")

        missing_tables = []
        for table in essential_tables:
            exists = await check_table_exists(conn, table)
            if exists:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                print(f"✅ {table}: {count:,} rows")
            else:
                print(f"❌ {table}: MISSING")
                missing_tables.append(table)

        if not missing_tables:
            print("\n🎉 All tables exist! Database is ready.")
            await conn.close()
            return True

        print(f"\n🔧 Creating {len(missing_tables)} missing tables...")

        # Read and execute the SQL schema
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        sql_file = os.path.join(script_dir, "init_db.sql")

        if not os.path.exists(sql_file):
            print("❌ init_db.sql not found!")
            return False

        with open(sql_file, "r") as f:
            sql_content = f.read()

        # Execute the SQL (uses IF NOT EXISTS, so it's safe)
        await conn.execute(sql_content)

        print("✅ Database schema applied successfully!")

        # Verify all tables now exist
        print("\n🔍 Verifying table creation...")
        all_created = True
        for table in essential_tables:
            exists = await check_table_exists(conn, table)
            if exists:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                print(f"✅ {table}: {count:,} rows")
            else:
                print(f"❌ {table}: STILL MISSING")
                all_created = False

        if all_created:
            print("\n🎉 Database initialization complete!")
        else:
            print("\n❌ Some tables failed to create!")
            return False

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


async def main():
    """Main function."""
    success = await init_database()

    if success:
        print("\n✅ Ready for development/production!")
        print("📋 Next steps:")
        print("  • Development: langgraph dev")
        print("  • Tests: make test")
        print("  • Production: ./scripts/start.sh")
    else:
        print("\n❌ Initialization failed!")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
