#!/usr/bin/env python3
"""
Script to create missing tables in the test: learning_modules and competencies
"""

import asyncio
import os
import sys

import asyncpg

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core.database import get_database_url


async def create_missing_tables():
    """Create missing tables for the test"""
    database_url = get_database_url()
    print("🔗 Connecting to database...")

    conn = await asyncpg.connect(database_url)
    try:
        # Create learning_modules table
        print("📚 Creating learning_modules table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_modules (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                level VARCHAR(10) NOT NULL,
                order_index INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Create competencies table
        print("🎯 Creating competencies table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS competencies (
                id VARCHAR(255) PRIMARY KEY,
                module_id UUID REFERENCES learning_modules(id),
                name VARCHAR(255) NOT NULL,
                description TEXT,
                level VARCHAR(10) NOT NULL,
                skill_type VARCHAR(50) NOT NULL,
                competency_type VARCHAR(50) NOT NULL,
                prerequisites TEXT[],
                estimated_hours INTEGER DEFAULT 1,
                key_vocabulary TEXT[],
                grammar_points TEXT[],
                learning_objectives TEXT[],
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        # Insert sample data to make the system coherent
        print("📝 Inserting sample data...")

        # Insert sample module
        module_id = await conn.fetchval("""
            INSERT INTO learning_modules (name, description, level, order_index)
            VALUES ('A1 Foundation', 'Basic English fundamentals', 'A1', 1)
            RETURNING id
        """)

        # Insert sample competencies (based on real curriculum)
        await conn.execute(
            """
            INSERT INTO competencies (
                id, module_id, name, description, level, skill_type, competency_type,
                prerequisites, estimated_hours, key_vocabulary, grammar_points, learning_objectives
            ) VALUES
            ($1, $2, 'Introductions and Personal Information', 'Basic self-introduction skills', 'A1', 'speaking', 'communication',
             '{}', 2, '{"hello","name","from","live","work"}', '{"present simple","be verb"}', '{"Introduce yourself","State your name and origin"}'),
            ($3, $2, 'Essential Daily Vocabulary', 'Common words for daily life', 'A1', 'vocabulary', 'lexical',
             '{}', 3, '{"food","water","house","work","family"}', '{}', '{"Use basic daily vocabulary","Understand common words"}'),
            ($4, $2, 'Family and Relationships', 'Vocabulary for family members', 'A1', 'vocabulary', 'lexical',
             '{}', 3, '{"mother","father","sister","brother","family"}', '{}', '{"Name family members","Describe family relationships"}')
        """,
            "a1_introductions",
            module_id,
            "a1_daily_vocabulary",
            "a1_family_vocabulary",
        )

        print("✅ Tables created successfully!")

        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('learning_modules', 'competencies')
            ORDER BY table_name
        """)

        print("\n📋 Tables verified:")
        for table in tables:
            print(f"  ✓ {table['table_name']}")

        # Verify data
        module_count = await conn.fetchval("SELECT COUNT(*) FROM learning_modules")
        competency_count = await conn.fetchval("SELECT COUNT(*) FROM competencies")

        print("\n📊 Data inserted:")
        print(f"  📚 Modules: {module_count}")
        print(f"  🎯 Competencies: {competency_count}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(create_missing_tables())
