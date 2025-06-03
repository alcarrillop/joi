#!/usr/bin/env python3
"""
Debug tools for monitoring and analyzing the system via command line
"""
import asyncio
import asyncpg
import httpx
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from typing import List, Dict, Any

# API configuration
BASE_URL = "http://localhost:8000"

async def get_system_stats():
    """Show system statistics."""
    print("📊 System statistics:")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/debug/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"   👥 Users: {stats['total_users']}")
                print(f"   📞 Sessions: {stats['total_sessions']} (active: {stats['active_sessions']})")
                print(f"   💬 Messages: {stats['total_messages']}")
                print(f"   🧠 Memories: {stats['total_memories']}")
                print(f"   📊 Assessments: {stats['total_assessments']}")
                return stats
            else:
                print(f"   ❌ Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return None

async def list_users(limit: int = 10):
    """Show all users in the system."""
    print(f"👥 System users (limit {limit}):")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/debug/users?limit={limit}")
            if response.status_code == 200:
                users = response.json()
                for user in users:
                    phone = user.get('phone_number', 'Unknown')
                    name = user.get('name') or 'Unnamed'
                    created = user.get('created_at', '')[:10]
                    print(f"   📱 {phone} - {name} (created: {created})")
                return users
            else:
                print(f"   ❌ Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return []

async def show_user_memories(user_id: str, query: str = ""):
    """Show memories from a specific user."""
    print(f"🧠 User memories {user_id[:8]}{'...' if len(user_id) > 8 else ''}:")
    
    async with httpx.AsyncClient() as client:
        try:
            url = f"{BASE_URL}/debug/users/{user_id}/memories"
            if query:
                url += f"?query={query}"
            
            response = await client.get(url)
            if response.status_code == 200:
                memories = response.json()
                for memory in memories[:10]:  # Show first 10
                    content = memory.get('content', '')[:100]
                    timestamp = memory.get('timestamp', '')[:10]
                    print(f"   💭 {content}... ({timestamp})")
                return memories
            else:
                print(f"   ❌ Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return []

async def show_user_messages(user_id: str, limit: int = 10):
    print(f"💬 Last {limit} messages from user {user_id}:")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/debug/users/{user_id}/messages?limit={limit}")
            if response.status_code == 200:
                messages = response.json()
                for msg in messages:
                    content = msg.get('content', '')[:80]
                    timestamp = msg.get('timestamp', '')[:16]
                    msg_type = msg.get('message_type', 'text')
                    print(f"   💬 [{timestamp}] ({msg_type}) {content}...")
                return messages
            else:
                print(f"   ❌ Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return []

async def test_memory_search(query: str):
    """Test memory search."""
    print(f"🔍 Testing memory search for '{query}':")
    
    # First, get some users to test with
    users = await list_users(3)
    if not users:
        print("   ❌ No users found")
        return
    
    for user in users[:2]:  # Test with first 2 users
        user_id = user['id']
        memories = await show_user_memories(user_id, query)
        if memories:
            print(f"   ✅ Found {len(memories)} memories for user {user_id[:8]}")

async def show_recent_activity(limit: int = 10):
    print(f"⏰ Last {limit} interactions:")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/debug/recent-activity?limit={limit}")
            if response.status_code == 200:
                activities = response.json()
                for activity in activities:
                    content = activity.get('content', '')[:60]
                    timestamp = activity.get('timestamp', '')[:16]
                    user_phone = activity.get('user_phone', 'Unknown')
                    print(f"   📱 [{timestamp}] {user_phone}: {content}...")
                return activities
            else:
                print(f"   ❌ Error: {response.status_code}")
                return []
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return []

async def health_check():
    """Check system health."""
    print("🏥 System health check:")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/debug/health")
            if response.status_code == 200:
                health = response.json()
                print(f"   Database: {'✅' if health['database'] else '❌'}")
                print(f"   Qdrant: {'✅' if health['qdrant'] else '❌'}")
                print(f"   LangGraph: {'✅' if health['langgraph'] else '❌'}")
                print(f"   Overall: {'✅ Healthy' if health['status'] == 'healthy' else '❌ Issues'}")
                return health
            else:
                print(f"   ❌ Server error: {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return None

async def analyze_first_user():
    """Detailed analysis of the first user."""
    users = await list_users(1)
    if not users:
        print("❌ No users found")
        return
    
    user = users[0]
    user_id = user['id']
    
    print("🔍 Detailed analysis of first user:")
    print(f"   ID: {user_id}")
    print(f"   Phone: {user.get('phone_number', 'Unknown')}")
    print(f"   Name: {user.get('name') or 'Unnamed'}")
    
    # Get their messages
    await show_user_messages(user_id, 5)
    
    # Get their memories
    await show_user_memories(user_id)

async def main():
    """Main function with interactive menu."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "stats":
            await get_system_stats()
        elif command == "users":
            await list_users()
        elif command == "health":
            await health_check()
        elif command == "activity":
            await show_recent_activity()
        elif command == "analyze":
            await analyze_first_user()
        else:
            print(f"Unknown command: {command}")
        return
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        print("🔧 JOI DEBUG TOOLS")
        print("="*50)
        print("1. 🏥 Health check")
        print("2. 📊 Show statistics")
        print("3. 👥 List users")
        print("4. 💬 Show user messages")
        print("5. 🧠 Show user memories") 
        print("6. 🔍 Test memory search")
        print("7. ⏰ Recent activity")
        print("8. 🔍 Analyze first user")
        print("0. 🚪 Exit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            await health_check()
        elif choice == "2":
            await get_system_stats()
        elif choice == "3":
            await list_users()
        elif choice == "4":
            user_id = input("User ID: ").strip()
            limit = int(input("Number of messages (default 10): ") or "10")
            await show_user_messages(user_id, limit)
        elif choice == "5":
            user_id = input("User ID: ").strip()
            await show_user_memories(user_id)
        elif choice == "6":
            query = input("Search query: ").strip()
            await test_memory_search(query)
        elif choice == "7":
            limit = int(input("Number of activities (default 10): ") or "10")
            await show_recent_activity(limit)
        elif choice == "8":
            await analyze_first_user()
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    asyncio.run(main()) 