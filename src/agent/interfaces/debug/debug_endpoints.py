"""
Endpoints de debug para monitorear el funcionamiento interno del sistema
"""
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.core.database import get_database_url, get_user_learning_stats
from agent.modules.memory.long_term.memory_manager import get_memory_manager
from agent.modules.curriculum.curriculum_manager import get_curriculum_manager
from agent.settings import settings

debug_router = APIRouter(prefix="/debug", tags=["Debug"])

# Modelos para las respuestas
class UserInfo(BaseModel):
    id: str
    phone_number: str
    name: Optional[str]
    current_level: str
    created_at: datetime

class SessionInfo(BaseModel):
    id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    context: Optional[Dict]

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

class CompetencyInfo(BaseModel):
    id: str
    name: str
    description: str
    level: str
    skill_type: str
    estimated_hours: int
    key_vocabulary: List[str]
    prerequisites: List[str]

class SystemStats(BaseModel):
    total_users: int
    total_sessions: int
    total_messages: int
    active_sessions: int
    total_memories: int
    total_assessments: int
    total_competencies_completed: int

@debug_router.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Obtener estadísticas generales del sistema."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        # Estadísticas básicas
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        active_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL")
        
        # Estadísticas de currículo
        total_assessments = await conn.fetchval("SELECT COUNT(*) FROM assessments")
        total_competencies_completed = await conn.fetchval("""
            SELECT COUNT(*) FROM (
                SELECT user_id, unnest(completed_competencies) as competency 
                FROM user_progress
            ) as completed
        """)
        
        # Estadísticas de memoria
        memory_manager = get_memory_manager()
        vector_store = memory_manager.vector_store
        try:
            if vector_store._collection_exists():
                collection_info = vector_store.client.get_collection(vector_store.COLLECTION_NAME)
                total_memories = collection_info.points_count or 0
            else:
                total_memories = 0
        except:
            total_memories = 0
        
        return SystemStats(
            total_users=total_users or 0,
            total_sessions=total_sessions or 0, 
            total_messages=total_messages or 0,
            active_sessions=active_sessions or 0,
            total_memories=total_memories,
            total_assessments=total_assessments or 0,
            total_competencies_completed=total_competencies_completed or 0
        )
    finally:
        await conn.close()

@debug_router.get("/users", response_model=List[UserInfo])
async def get_all_users(limit: int = Query(50, ge=1, le=100)):
    """Obtener lista de todos los usuarios."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        users = await conn.fetch(
            "SELECT id, phone_number, name, current_level, created_at FROM users ORDER BY created_at DESC LIMIT $1",
            limit
        )
        
        return [
            UserInfo(
                id=str(user['id']),
                phone_number=user['phone_number'],
                name=user['name'],
                current_level=user['current_level'],
                created_at=user['created_at']
            ) for user in users
        ]
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/sessions", response_model=List[SessionInfo])
async def get_user_sessions(user_id: str):
    """Obtener sesiones de un usuario específico."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        sessions = await conn.fetch(
            "SELECT id, user_id, started_at, ended_at, context FROM sessions WHERE user_id = $1 ORDER BY started_at DESC",
            uuid.UUID(user_id)
        )
        
        return [
            SessionInfo(
                id=str(session['id']),
                user_id=str(session['user_id']),
                started_at=session['started_at'],
                ended_at=session['ended_at'],
                context=session['context']
            ) for session in sessions
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/messages", response_model=List[MessageInfo])
async def get_user_messages(user_id: str, limit: int = Query(50, ge=1, le=200)):
    """Obtener mensajes de un usuario específico."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        messages = await conn.fetch("""
            SELECT m.id, m.session_id, m.sender, m.message, m.timestamp 
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE s.user_id = $1 
            ORDER BY m.timestamp DESC 
            LIMIT $2
        """, uuid.UUID(user_id), limit)
        
        return [
            MessageInfo(
                id=str(msg['id']),
                session_id=str(msg['session_id']),
                sender=msg['sender'],
                message=msg['message'],
                timestamp=msg['timestamp']
            ) for msg in messages
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/memories", response_model=List[MemoryInfo])
async def get_user_memories(user_id: str, query: str = Query("", description="Query para buscar memorias específicas")):
    """Obtener memorias de un usuario específico."""
    memory_manager = get_memory_manager()
    
    try:
        if query:
            # Búsqueda específica
            memories = memory_manager.vector_store.search_memories(query, user_id=user_id, k=20)
        else:
            # Obtener todas las memorias del usuario (búsqueda genérica)
            memories = memory_manager.vector_store.search_memories("*", user_id=user_id, k=50)
        
        return [
            MemoryInfo(
                text=memory.text,
                score=memory.score,
                metadata=memory.metadata
            ) for memory in memories
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memories: {str(e)}")

@debug_router.get("/sessions/{session_id}/messages", response_model=List[MessageInfo])
async def get_session_messages(session_id: str):
    """Obtener mensajes de una sesión específica."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        messages = await conn.fetch(
            "SELECT id, session_id, sender, message, timestamp FROM messages WHERE session_id = $1 ORDER BY timestamp ASC",
            uuid.UUID(session_id)
        )
        
        return [
            MessageInfo(
                id=str(msg['id']),
                session_id=str(msg['session_id']),
                sender=msg['sender'],
                message=msg['message'],
                timestamp=msg['timestamp']
            ) for msg in messages
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/learning-stats")
async def get_user_learning_statistics(user_id: str):
    """Obtener estadísticas de aprendizaje de un usuario."""
    try:
        stats = await get_user_learning_stats(user_id)
        memory_manager = get_memory_manager()
        memory_stats = memory_manager.get_user_memory_stats(user_id)
        
        # Obtener estadísticas del currículo
        curriculum_manager = get_curriculum_manager()
        curriculum_stats = await curriculum_manager.get_learning_statistics(user_id)
        
        return {
            "user_id": user_id,
            "learning_stats": stats,
            "memory_stats": memory_stats,
            "curriculum_stats": curriculum_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving learning stats: {str(e)}")

@debug_router.get("/users/{user_id}/curriculum-progress")
async def get_user_curriculum_progress(user_id: str):
    """Obtener progreso del currículo de un usuario."""
    try:
        curriculum_manager = get_curriculum_manager()
        progress = await curriculum_manager.get_user_progress(user_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="User progress not found")
        
        # Obtener competencias disponibles
        available_competencies = await curriculum_manager.get_current_competencies(user_id)
        recommended = await curriculum_manager.get_next_recommended_competency(user_id)
        
        return {
            "user_id": user_id,
            "current_level": progress.current_level.value,
            "level_start_date": progress.level_start_date,
            "completed_competencies": list(progress.completed_competencies),
            "in_progress_competencies": list(progress.in_progress_competencies),
            "mastery_scores": progress.mastery_scores,
            "last_assessment_date": progress.last_assessment_date,
            "available_competencies": [
                {
                    "id": comp.id,
                    "name": comp.name,
                    "skill_type": comp.skill_type.value,
                    "estimated_hours": comp.estimated_hours,
                    "is_completed": comp.id in progress.completed_competencies,
                    "is_in_progress": comp.id in progress.in_progress_competencies
                } for comp in available_competencies
            ],
            "recommended_competency": {
                "id": recommended.id,
                "name": recommended.name,
                "description": recommended.description,
                "skill_type": recommended.skill_type.value,
                "estimated_hours": recommended.estimated_hours,
                "key_vocabulary": recommended.key_vocabulary[:10]  # Primeras 10 palabras
            } if recommended else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving curriculum progress: {str(e)}")

@debug_router.get("/users/{user_id}/assessments")
async def get_user_assessments(user_id: str, limit: int = Query(20, ge=1, le=100)):
    """Obtener evaluaciones de un usuario."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        assessments = await conn.fetch("""
            SELECT id, competency_id, skill_type, score, max_score, timestamp, 
                   feedback, areas_for_improvement, strengths
            FROM assessments 
            WHERE user_id = $1 
            ORDER BY timestamp DESC 
            LIMIT $2
        """, user_id, limit)
        
        return [
            {
                "id": str(assessment['id']),
                "competency_id": assessment['competency_id'],
                "skill_type": assessment['skill_type'],
                "score": assessment['score'],
                "max_score": assessment['max_score'],
                "score_percentage": assessment['score'] / assessment['max_score'] * 100,
                "timestamp": assessment['timestamp'],
                "feedback": assessment['feedback'],
                "areas_for_improvement": assessment['areas_for_improvement'],
                "strengths": assessment['strengths']
            } for assessment in assessments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")
    finally:
        await conn.close()

@debug_router.get("/curriculum/competencies")
async def get_all_competencies():
    """Obtener todas las competencias del currículo."""
    try:
        curriculum_manager = get_curriculum_manager()
        all_competencies = []
        
        for level, modules in curriculum_manager.learning_modules.items():
            for module in modules:
                for comp in module.competencies:
                    all_competencies.append({
                        "id": comp.id,
                        "name": comp.name,
                        "description": comp.description,
                        "level": comp.level.value,
                        "skill_type": comp.skill_type.value,
                        "competency_type": comp.competency_type.value,
                        "prerequisites": comp.prerequisites,
                        "estimated_hours": comp.estimated_hours,
                        "key_vocabulary": comp.key_vocabulary,
                        "grammar_points": comp.grammar_points,
                        "learning_objectives": comp.learning_objectives
                    })
        
        return all_competencies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving competencies: {str(e)}")

@debug_router.post("/users/{user_id}/test-memory")
async def test_user_memory(user_id: str, query: str):
    """Probar la búsqueda de memoria para un usuario con una query específica."""
    memory_manager = get_memory_manager()
    
    try:
        memories = memory_manager.get_relevant_memories(query, user_id)
        formatted = memory_manager.format_memories_for_prompt(memories)
        
        return {
            "user_id": user_id,
            "query": query,
            "memories_found": len(memories),
            "raw_memories": memories,
            "formatted_for_prompt": formatted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing memory: {str(e)}")

@debug_router.delete("/users/{user_id}/memories")
async def delete_user_memories(user_id: str):
    """Eliminar todas las memorias de un usuario (útil para testing)."""
    memory_manager = get_memory_manager()
    
    try:
        deleted_count = memory_manager.delete_user_memories(user_id)
        return {
            "user_id": user_id,
            "deleted_memories": deleted_count,
            "message": f"Deleted {deleted_count} memories for user {user_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting memories: {str(e)}")

@debug_router.get("/recent-activity")
async def get_recent_activity(limit: int = Query(20, ge=1, le=100)):
    """Obtener actividad reciente del sistema."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        recent_messages = await conn.fetch("""
            SELECT 
                m.timestamp,
                m.sender,
                m.message,
                u.phone_number,
                u.current_level,
                s.id as session_id
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            JOIN users u ON s.user_id = u.id
            ORDER BY m.timestamp DESC
            LIMIT $1
        """, limit)
        
        return [
            {
                "timestamp": msg['timestamp'],
                "sender": msg['sender'],
                "message": msg['message'][:100] + "..." if len(msg['message']) > 100 else msg['message'],
                "phone_number": msg['phone_number'],
                "user_level": msg['current_level'],
                "session_id": str(msg['session_id'])
            } for msg in recent_messages
        ]
    finally:
        await conn.close()

@debug_router.get("/health")
async def debug_health_check():
    """Verificar que todos los componentes estén funcionando."""
    health_status = {
        "timestamp": datetime.now(),
        "database": "unknown",
        "vector_store": "unknown", 
        "memory_manager": "unknown",
        "curriculum_manager": "unknown"
    }
    
    # Test database
    try:
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        await conn.fetchval("SELECT 1")
        await conn.close()
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
    
    # Test vector store
    try:
        memory_manager = get_memory_manager()
        vector_store = memory_manager.vector_store
        if vector_store._collection_exists():
            health_status["vector_store"] = "healthy"
        else:
            health_status["vector_store"] = "collection_missing"
    except Exception as e:
        health_status["vector_store"] = f"error: {str(e)}"
    
    # Test memory manager
    try:
        memory_manager = get_memory_manager()
        health_status["memory_manager"] = "healthy"
    except Exception as e:
        health_status["memory_manager"] = f"error: {str(e)}"
    
    # Test curriculum manager
    try:
        curriculum_manager = get_curriculum_manager()
        health_status["curriculum_manager"] = "healthy"
    except Exception as e:
        health_status["curriculum_manager"] = f"error: {str(e)}"
    
    return health_status

@debug_router.get("/dashboard", response_class=HTMLResponse)
async def debug_dashboard():
    """Servir la página de debug dashboard."""
    try:
        with open("debug_dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Debug Dashboard no encontrado</h1><p>Asegúrate de que debug_dashboard.html exista en el directorio raíz.</p>",
            status_code=404
        ) 