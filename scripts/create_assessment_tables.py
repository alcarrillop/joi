#!/usr/bin/env python3
"""
Script to create assessment system tables
"""
import asyncio
import asyncpg
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core.database import get_database_url

async def create_assessment_tables():
    """Create assessment system tables"""
    database_url = get_database_url()
    print(f"🔗 Connecting to database...")
    
    conn = await asyncpg.connect(database_url)
    try:
        # Read SQL script
        script_path = os.path.join(os.path.dirname(__file__), 'assessment_tables.sql')
        with open(script_path, 'r') as f:
            sql_script = f.read()
        
        # Execute script
        print("📊 Creating assessment system tables...")
        await conn.execute(sql_script)
        print("✅ Assessment tables created successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('conversation_assessments', 'level_detections', 'skill_progressions', 
                              'detected_errors', 'assessment_configs')
            ORDER BY table_name
        """)
        
        print("\n📋 Tables created:")
        for table in tables:
            print(f"  ✓ {table['table_name']}")
        
        # Verify views
        views = await conn.fetch("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name IN ('assessment_summary', 'skill_overview', 'error_patterns')
            ORDER BY table_name
        """)
        
        print("\n👁 Views created:")
        for view in views:
            print(f"  ✓ {view['table_name']}")
        
        # Verify triggers
        triggers = await conn.fetch("""
            SELECT trigger_name FROM information_schema.triggers 
            WHERE trigger_schema = 'public' 
            AND trigger_name IN ('trigger_update_level_detection_status', 'trigger_update_skill_progression_timestamp')
            ORDER BY trigger_name
        """)
        
        print("\n⚡ Triggers created:")
        for trigger in triggers:
            print(f"  ✓ {trigger['trigger_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_assessment_tables()) 