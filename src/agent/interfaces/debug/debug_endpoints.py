"""
Debug endpoints for development and testing
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.core.database import get_database_url
from agent.modules.learning.learning_stats_manager import get_learning_stats_manager
from agent.modules.memory.long_term.memory_manager import get_memory_manager

debug_router = APIRouter(prefix="/debug", tags=["debug"])


# Pydantic models for responses
class UserInfo(BaseModel):
    id: str
    phone_number: str
    name: Optional[str]
    created_at: datetime


class SessionInfo(BaseModel):
    id: str
    user_id: str
    started_at: datetime


class MessageInfo(BaseModel):
    id: str
    session_id: str
    sender: str
    message: str
    timestamp: datetime


class MemoryInfo(BaseModel):
    text: str
    score: Optional[float]
    metadata: Dict


class SystemStats(BaseModel):
    total_users: int
    total_sessions: int
    total_messages: int
    active_sessions: int
    total_memories: int
    total_vocabulary_entries: int


@debug_router.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system-wide statistics."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        # Basic statistics
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        active_sessions = total_sessions  # All sessions are considered active now
        total_vocabulary_entries = await conn.fetchval("SELECT COUNT(*) FROM user_word_stats")

        # Memory statistics
        memory_manager = get_memory_manager()
        vector_store = memory_manager.vector_store
        try:
            if vector_store._collection_exists():
                collection_info = vector_store.client.get_collection(vector_store.COLLECTION_NAME)
                total_memories = collection_info.points_count or 0
            else:
                total_memories = 0
        except Exception:
            total_memories = 0

        return SystemStats(
            total_users=total_users or 0,
            total_sessions=total_sessions or 0,
            total_messages=total_messages or 0,
            active_sessions=active_sessions or 0,
            total_memories=total_memories,
            total_vocabulary_entries=total_vocabulary_entries or 0,
        )
    finally:
        await conn.close()


@debug_router.get("/users", response_model=List[UserInfo])
async def get_all_users(limit: int = Query(50, ge=1, le=100)):
    """Get list of all users."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        users = await conn.fetch(
            "SELECT id, phone_number, name, created_at FROM users ORDER BY created_at DESC LIMIT $1",
            limit,
        )

        return [
            UserInfo(
                id=str(user["id"]),
                phone_number=user["phone_number"],
                name=user["name"],
                created_at=user["created_at"],
            )
            for user in users
        ]
    finally:
        await conn.close()


@debug_router.get("/users/{user_id}/sessions", response_model=List[SessionInfo])
async def get_user_sessions(user_id: str):
    """Get sessions for a specific user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        sessions = await conn.fetch(
            "SELECT id, user_id, started_at FROM sessions WHERE user_id = $1 ORDER BY started_at DESC",
            uuid.UUID(user_id),
        )

        return [
            SessionInfo(
                id=str(session["id"]),
                user_id=str(session["user_id"]),
                started_at=session["started_at"],
            )
            for session in sessions
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    finally:
        await conn.close()


@debug_router.get("/users/{user_id}/messages", response_model=List[MessageInfo])
async def get_user_messages(user_id: str, limit: int = Query(50, ge=1, le=200)):
    """Get messages from a specific user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        messages = await conn.fetch(
            """
            SELECT m.id, m.session_id, m.sender, m.message, m.timestamp
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.user_id = $1
            ORDER BY m.timestamp DESC
            LIMIT $2
        """,
            uuid.UUID(user_id),
            limit,
        )

        return [
            MessageInfo(
                id=str(msg["id"]),
                session_id=str(msg["session_id"]),
                sender=msg["sender"],
                message=msg["message"],
                timestamp=msg["timestamp"],
            )
            for msg in messages
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    finally:
        await conn.close()


@debug_router.get("/users/{user_id}/memories", response_model=List[MemoryInfo])
async def get_user_memories(user_id: str, query: str = Query("", description="Query to search for specific memories")):
    """Get memories from a specific user."""
    memory_manager = get_memory_manager()

    try:
        if query:
            # Specific search
            memories = memory_manager.vector_store.search_memories(query, user_id=user_id, k=20)
        else:
            # Get all user memories (generic search)
            memories = memory_manager.vector_store.search_memories("*", user_id=user_id, k=50)

        return [MemoryInfo(text=memory.text, score=memory.score, metadata=memory.metadata) for memory in memories]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memories: {str(e)}")


@debug_router.get("/sessions/{session_id}/messages", response_model=List[MessageInfo])
async def get_session_messages(session_id: str):
    """Get messages from a specific session."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        messages = await conn.fetch(
            "SELECT id, session_id, sender, message, timestamp FROM messages WHERE session_id = $1 ORDER BY timestamp ASC",
            uuid.UUID(session_id),
        )

        return [
            MessageInfo(
                id=str(msg["id"]),
                session_id=str(msg["session_id"]),
                sender=msg["sender"],
                message=msg["message"],
                timestamp=msg["timestamp"],
            )
            for msg in messages
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    finally:
        await conn.close()


@debug_router.get("/users/{user_id}/learning-stats")
async def get_user_learning_statistics(user_id: str):
    """Get comprehensive learning statistics for a user."""
    try:
        # Import here to avoid circular imports
        from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

        # Get enhanced curriculum-based analysis
        curriculum_manager = get_curriculum_manager()
        curriculum_progress = await curriculum_manager.estimate_level_progress(user_id)

        # Get basic technical stats from learning manager
        learning_stats_manager = get_learning_stats_manager()
        basic_stats = await learning_stats_manager.get_learning_summary(user_id)
        top_words = await learning_stats_manager.get_user_top_words(user_id, limit=10)

        # Get memory stats
        memory_manager = get_memory_manager()
        memory_stats = memory_manager.get_user_memory_stats(user_id)

        return {
            "user_id": user_id,
            "curriculum_analysis": {
                "estimated_level": curriculum_progress.get("estimated_level", "A1"),
                "vocabulary_learned": curriculum_progress.get("vocabulary_learned", 0),
                "curriculum_mastery": curriculum_progress.get("curriculum_insights", {}).get(
                    "curriculum_mastery_percentage", 0
                ),
                "educational_recommendations": curriculum_progress.get("educational_recommendations", []),
                "level_competencies": curriculum_progress.get("level_competencies", []),
                "progress_to_next_level": curriculum_progress.get("progress_to_next_level", 0),
                "next_level": curriculum_progress.get("next_level"),
            },
            "technical_stats": {
                "vocabulary_summary": basic_stats.get("vocabulary", {}),
                "top_words": top_words,
                "word_statistics": basic_stats.get("statistics", {}),
            },
            "memory_stats": memory_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving learning stats: {str(e)}")


@debug_router.post("/users/{user_id}/test-memory")
async def test_user_memory(user_id: str, query: str):
    """Test memory retrieval for a specific user."""
    memory_manager = get_memory_manager()

    try:
        memories = memory_manager.vector_store.search_memories(query, user_id=user_id, k=5)
        return {
            "query": query,
            "user_id": user_id,
            "memories_found": len(memories),
            "memories": [{"text": m.text, "score": m.score, "metadata": m.metadata} for m in memories],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing memory: {str(e)}")


@debug_router.delete("/users/{user_id}/memories")
async def delete_user_memories(user_id: str):
    """Delete all memories for a specific user."""
    memory_manager = get_memory_manager()

    try:
        deleted_count = memory_manager.vector_store.delete_user_memories(user_id)
        return {"user_id": user_id, "deleted_memories": deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memories: {str(e)}")


@debug_router.get("/recent-activity")
async def get_recent_activity(limit: int = Query(20, ge=1, le=100)):
    """Get recent system activity."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)

    try:
        recent_messages = await conn.fetch(
            """
            SELECT m.id, m.sender, m.message, m.timestamp, s.user_id, u.phone_number, u.name
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            JOIN users u ON s.user_id = u.id
            ORDER BY m.timestamp DESC
            LIMIT $1
        """,
            limit,
        )

        return [
            {
                "message_id": str(msg["id"]),
                "sender": msg["sender"],
                "message": msg["message"][:100] + "..." if len(msg["message"]) > 100 else msg["message"],
                "timestamp": msg["timestamp"],
                "user_id": str(msg["user_id"]),
                "user_phone": msg["phone_number"],
                "user_name": msg["name"],
            }
            for msg in recent_messages
        ]
    finally:
        await conn.close()


@debug_router.get("/health")
async def debug_health_check():
    """Check the health of various system components."""
    health = {}

    # Database health
    try:
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        await conn.close()
        health["database"] = {"status": "healthy", "user_count": user_count}
    except Exception as e:
        health["database"] = {"status": "unhealthy", "error": str(e)}

    # Memory system health
    try:
        memory_manager = get_memory_manager()
        vector_store = memory_manager.vector_store
        collection_exists = vector_store._collection_exists()
        if collection_exists:
            collection_info = vector_store.client.get_collection(vector_store.COLLECTION_NAME)
            memory_count = collection_info.points_count or 0
        else:
            memory_count = 0
        health["memory_system"] = {
            "status": "healthy" if collection_exists else "degraded",
            "collection_exists": collection_exists,
            "memory_count": memory_count,
        }
    except Exception as e:
        health["memory_system"] = {"status": "unhealthy", "error": str(e)}

    # Learning stats health
    try:
        get_learning_stats_manager()  # Just test if it can be imported
        health["learning_stats"] = {"status": "healthy"}
    except Exception as e:
        health["learning_stats"] = {"status": "unhealthy", "error": str(e)}

    return health


@debug_router.get("/dashboard", response_class=HTMLResponse)
async def debug_dashboard():
    """Simple HTML dashboard for debugging."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JOI Debug Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .stats { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .endpoint { background: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .endpoint h3 { margin-top: 0; color: #2c5aa0; }
            .endpoint code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🤖 JOI Debug Dashboard</h1>

        <div class="stats">
            <h2>📊 System Status</h2>
            <p><strong>Enhanced Curriculum Architecture:</strong> CEFR-based learning with adaptive content</p>
            <p><strong>Core Tables:</strong> users, sessions, messages, user_word_stats</p>
            <p><strong>Vector Store:</strong> Qdrant for semantic memory</p>
            <p><strong>Learning System:</strong> Curriculum + Technical tracking separation</p>
        </div>

        <h2>🔗 Available Endpoints</h2>

        <div class="endpoint">
            <h3>System Stats</h3>
            <p><code>GET /debug/stats</code> - Overall system statistics</p>
        </div>

        <div class="endpoint">
            <h3>Users</h3>
            <p><code>GET /debug/users</code> - List all users</p>
            <p><code>GET /debug/users/{user_id}/sessions</code> - User sessions</p>
            <p><code>GET /debug/users/{user_id}/messages</code> - User messages</p>
            <p><code>GET /debug/users/{user_id}/memories</code> - User memories</p>
            <p><code>GET /debug/users/{user_id}/learning-stats</code> - Comprehensive learning analysis (CEFR + technical)</p>
        </div>

        <div class="endpoint">
            <h3>Memory Testing</h3>
            <p><code>POST /debug/users/{user_id}/test-memory</code> - Test memory search</p>
            <p><code>DELETE /debug/users/{user_id}/memories</code> - Delete user memories</p>
        </div>

        <div class="endpoint">
            <h3>System Health</h3>
            <p><code>GET /debug/health</code> - Component health check</p>
            <p><code>GET /debug/recent-activity</code> - Recent system activity</p>
        </div>

        <div class="stats">
            <h2>🎯 Enhanced for Production</h2>
            <p>✅ <strong>Essential Memory System:</strong> Users, sessions, messages, vector store</p>
            <p>✅ <strong>Curriculum-Based Learning:</strong> CEFR levels, competencies, adaptive recommendations</p>
            <p>✅ <strong>Technical Tracking:</strong> Vocabulary normalization, frequency analysis, statistics</p>
            <p>✅ <strong>Educational Intelligence:</strong> Curriculum mastery analysis and personalized guidance</p>
            <p>🚀 <strong>Performance:</strong> ~9 seconds per message processing</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
