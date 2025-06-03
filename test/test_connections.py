#!/usr/bin/env python3
"""
Connection tests for Qdrant Cloud and Supabase
"""
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

async def test_qdrant_connection():
    """Test connection to Qdrant Cloud"""
    print("🔍 Testing Qdrant Cloud connection...")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ QDRANT_URL or QDRANT_API_KEY not configured")
        return False
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Test connection by listing collections
        collections = client.get_collections()
        print(f"✅ Connection successful! Collections found: {len(collections.collections)}")
        
        # Show available collections
        if collections.collections:
            print("   Available collections:")
            for col in collections.collections:
                print(f"   - {col.name} ({col.points_count} points)")
        else:
            print("   No collections found (this is normal for a new instance)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return False

async def test_supabase_connection():
    """Test connection to Supabase"""
    print("\n🔍 Testing Supabase connection...")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ SUPABASE_URL or SUPABASE_KEY not configured")
        return False
    
    try:
        from supabase import create_client
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test connection with a simple query
        response = supabase.table("test_table").select("*").limit(1).execute()
        print("✅ Supabase connection successful!")
        
        return True
        
    except Exception as e:
        if "does not exist" in str(e):
            print("✅ Supabase connection successful!")
            print("   (Test table doesn't exist, but connection works)")
            return True
        else:
            print(f"❌ Error connecting to Supabase: {e}")
            return False

async def test_postgresql_connection():
    """Test direct connection to Supabase PostgreSQL"""
    print("\n🔍 Testing direct PostgreSQL connection...")
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not configured")
        return False
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Test with a simple query
        version = await conn.fetchval("SELECT version()")
        await conn.close()
        
        print(f"✅ PostgreSQL connection successful!")
        print(f"   Version: {version[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return False

async def test_qdrant_collections():
    """Test specific Qdrant collections for the project"""
    print("\n🔍 Testing project-specific Qdrant collections...")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ Qdrant not configured")
        return False
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Check if the memories collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "memories" in collection_names:
            print("✅ 'memories' collection exists")
            # Get collection info
            info = client.get_collection("memories")
            print(f"   Points: {info.points_count}")
            print(f"   Vector size: {info.config.params.vectors.size}")
        else:
            print("⚠️  'memories' collection not found")
            print("   This is normal if you haven't stored memories yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking collections: {e}")
        return False

async def main():
    """Main function to run all tests"""
    print("🚀 Starting connection tests...\n")
    
    results = {
        "qdrant": await test_qdrant_connection(),
        "supabase": await test_supabase_connection(), 
        "postgresql": await test_postgresql_connection(),
        "collections": await test_qdrant_collections()
    }
    
    print("\n" + "=" * 50)
    print("📋 Connection test summary:")
    
    for service, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {service.title()}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 All connections are working correctly!")
    else:
        print("\n⚠️  Some connections have issues. Check the configuration.")
        
        print("\n🔧 Configuration tips:")
        print("   - Verify environment variables in .env file")
        print("   - Check Qdrant Cloud credentials")
        print("   - Verify Supabase project settings")
        print("   - Ensure database is accessible")

if __name__ == "__main__":
    asyncio.run(main()) 