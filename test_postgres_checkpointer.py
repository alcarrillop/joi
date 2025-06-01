#!/usr/bin/env python3
"""
Script para probar el checkpointer de PostgreSQL
"""

import os
import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Cargar variables de entorno
load_dotenv()

async def test_postgres_checkpointer():
    """Probar que el checkpointer de PostgreSQL funciona"""
    print("🔍 Probando PostgreSQL checkpointer...")
    
    try:
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            print("❌ DATABASE_URL no está configurada")
            return False
            
        print(f"   URL: {database_url[:50]}...")
        
        # Crear el checkpointer
        async with AsyncPostgresSaver.from_conn_string(database_url) as checkpointer:
            print("✅ PostgreSQL checkpointer creado exitosamente!")
            
            # Probar operación básica
            config = {"configurable": {"thread_id": "test_thread"}}
            
            # Verificar que no hay errores básicos
            try:
                # Esto debería funcionar sin errores
                checkpoint = await checkpointer.aget(config)
                print("✅ Operación de lectura funcionando!")
            except Exception as e:
                # Es normal que no haya checkpoint inicialmente
                if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                    print("✅ Checkpointer funcionando (sin datos previos)")
                else:
                    raise e
        
        return True
        
    except Exception as e:
        print(f"❌ Error con PostgreSQL checkpointer: {e}")
        return False

async def main():
    """Función principal"""
    print("🚀 Probando configuración PostgreSQL para LangGraph...\n")
    
    success = await test_postgres_checkpointer()
    
    if success:
        print("\n🎉 ¡PostgreSQL checkpointer configurado correctamente!")
        print("   Listo para usar con LangGraph.")
    else:
        print("\n🔧 Hay problemas con la configuración PostgreSQL.")

if __name__ == "__main__":
    asyncio.run(main()) 