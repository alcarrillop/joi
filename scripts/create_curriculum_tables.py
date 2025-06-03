#!/usr/bin/env python3
"""
Script to create curriculum system tables
"""
import asyncio
import asyncpg
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core.database import get_database_url

async def create_curriculum_tables():
    """Create curriculum system tables"""
    database_url = get_database_url()
    print(f"🔗 Connecting to database...")
    
    conn = await asyncpg.connect(database_url)
    try:
        # Read SQL script
        script_path = os.path.join(os.path.dirname(__file__), 'curriculum_tables.sql')
        with open(script_path, 'r') as f:
            sql_script = f.read()
        
        # Execute script
        print("📊 Creating curriculum system tables...")
        await conn.execute(sql_script)
        print("✅ Curriculum tables created successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_progress', 'assessments', 'learning_goals', 
                              'practice_activities', 'vocabulary_progress', 'grammar_errors')
            ORDER BY table_name
        """)
        
        print("\n📋 Tables created:")
        for table in tables:
            print(f"  ✓ {table['table_name']}")
        
        # Verify views
        views = await conn.fetch("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_progress_summary', 'vocabulary_stats', 'grammar_error_analysis')
            ORDER BY table_name
        """)
        
        print("\n👁 Views created:")
        for view in views:
            print(f"  ✓ {view['table_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_curriculum_tables()) 