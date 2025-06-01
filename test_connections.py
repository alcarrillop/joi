#!/usr/bin/env python3
"""
Script para probar las conexiones a Qdrant Cloud y Supabase
"""

import os
import asyncio
from datetime import datetime
from qdrant_client import QdrantClient
from supabase import create_client, Client
import asyncpg
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

def test_qdrant_connection():
    """Probar conexión a Qdrant Cloud"""
    print("🔍 Probando conexión a Qdrant Cloud...")
    
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            print("❌ QDRANT_URL o QDRANT_API_KEY no están configuradas")
            return False
            
        print(f"   URL: {qdrant_url}")
        print(f"   API Key: {qdrant_api_key[:20]}...")
        
        # Crear cliente
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Probar conexión listando colecciones
        collections = client.get_collections()
        print(f"✅ Conexión exitosa! Colecciones encontradas: {len(collections.collections)}")
        
        for collection in collections.collections:
            print(f"   - {collection.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a Qdrant: {e}")
        return False

def test_supabase_connection():
    """Probar conexión a Supabase"""
    print("\n🔍 Probando conexión a Supabase...")
    
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ SUPABASE_URL o SUPABASE_KEY no están configuradas")
            return False
            
        print(f"   URL: {supabase_url}")
        print(f"   Key: {supabase_key[:20]}...")
        
        # Crear cliente Supabase
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Hacer una consulta simple para probar la conexión
        result = supabase.table("test_table").select("*").limit(1).execute()
        print("✅ Conexión a Supabase exitosa!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a Supabase: {e}")
        # No es un error crítico si la tabla no existe
        if "table" in str(e).lower() and "does not exist" in str(e).lower():
            print("   (La tabla de prueba no existe, pero la conexión funciona)")
            return True
        return False

async def test_postgres_connection():
    """Probar conexión directa a PostgreSQL de Supabase"""
    print("\n🔍 Probando conexión directa a PostgreSQL...")
    
    try:
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            print("❌ DATABASE_URL no está configurada")
            return False
            
        print(f"   URL: {database_url[:50]}...")
        
        # Conectar usando asyncpg
        conn = await asyncpg.connect(database_url)
        
        # Ejecutar consulta simple
        version = await conn.fetchval("SELECT version()")
        print(f"✅ Conexión PostgreSQL exitosa!")
        print(f"   Versión: {version[:50]}...")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a PostgreSQL: {e}")
        return False

def test_embedding_model():
    """Probar que el modelo de embeddings funciona"""
    print("\n🔍 Probando modelo de embeddings...")
    
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Generar embedding de prueba
        test_text = "Esto es un texto de prueba para embeddings"
        embedding = model.encode(test_text)
        
        print(f"✅ Modelo de embeddings funcionando!")
        print(f"   Dimensiones: {len(embedding)}")
        print(f"   Tipo: {type(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error con modelo de embeddings: {e}")
        return False

async def main():
    """Función principal para ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas de conexión...\n")
    
    # Probar todas las conexiones
    qdrant_ok = test_qdrant_connection()
    supabase_ok = test_supabase_connection()
    postgres_ok = await test_postgres_connection()
    embedding_ok = test_embedding_model()
    
    print("\n" + "="*50)
    print("📊 RESUMEN DE PRUEBAS:")
    print("="*50)
    print(f"Qdrant Cloud:     {'✅' if qdrant_ok else '❌'}")
    print(f"Supabase:         {'✅' if supabase_ok else '❌'}")
    print(f"PostgreSQL:       {'✅' if postgres_ok else '❌'}")
    print(f"Embeddings:       {'✅' if embedding_ok else '❌'}")
    
    all_ok = qdrant_ok and (supabase_ok or postgres_ok) and embedding_ok
    print(f"\nEstado general:   {'✅ TODO FUNCIONANDO' if all_ok else '❌ HAY PROBLEMAS'}")
    
    if all_ok:
        print("\n🎉 ¡Todas las conexiones están funcionando correctamente!")
        print("   Ya puedes ejecutar el proyecto localmente.")
    else:
        print("\n🔧 Hay algunos problemas que necesitan resolverse.")

if __name__ == "__main__":
    asyncio.run(main()) 