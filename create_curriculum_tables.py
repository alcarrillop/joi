#!/usr/bin/env python3
"""
Script para crear las tablas del sistema de currículo
"""
import asyncio
import asyncpg
from src.agent.core.database import get_database_url

async def create_curriculum_tables():
    """Crea las tablas del sistema de currículo"""
    database_url = get_database_url()
    print(f"🔗 Conectando a base de datos...")
    
    conn = await asyncpg.connect(database_url)
    try:
        # Leer el script SQL
        with open('scripts/curriculum_tables.sql', 'r') as f:
            sql_script = f.read()
        
        # Ejecutar el script
        print("📊 Creando tablas del sistema de currículo...")
        await conn.execute(sql_script)
        print("✅ Tablas de currículo creadas exitosamente!")
        
        # Verificar que las tablas fueron creadas
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_progress', 'assessments', 'learning_goals', 
                              'practice_activities', 'vocabulary_progress', 'grammar_errors')
            ORDER BY table_name
        """)
        
        print("\n📋 Tablas creadas:")
        for table in tables:
            print(f"  ✓ {table['table_name']}")
        
        # Verificar vistas
        views = await conn.fetch("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND table_name IN ('user_progress_summary', 'vocabulary_stats', 'grammar_error_analysis')
            ORDER BY table_name
        """)
        
        print("\n👁 Vistas creadas:")
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