"""
Debug endpoints for monitoring internal system operations
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
    """Get general system statistics."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        # Basic statistics
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        active_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL")
        
        # Curriculum statistics
        total_assessments = await conn.fetchval("SELECT COUNT(*) FROM assessments")
        total_competencies_completed = await conn.fetchval("""
            SELECT COUNT(*) FROM (
                SELECT user_id, unnest(completed_competencies) as competency 
                FROM user_progress
            ) as completed
        """)
        
        # Memory statistics
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
    """Get list of all users."""
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
    """Get sessions for a specific user."""
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
    """Get messages from a specific user."""
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
    """Get messages from a specific session."""
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
    """Get learning statistics for a user."""
    try:
        stats = await get_user_learning_stats(user_id)
        memory_manager = get_memory_manager()
        memory_stats = memory_manager.get_user_memory_stats(user_id)
        
        # Get curriculum statistics
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
    """Get curriculum progress for a user."""
    try:
        curriculum_manager = get_curriculum_manager()
        progress = await curriculum_manager.get_user_progress(user_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="User progress not found")
        
        # Get available competencies
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
    """Get assessments for a user."""
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
    """Get all curriculum competencies."""
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
    """Test memory search for a user with a specific query."""
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
    """Delete all memories for a user (useful for testing)."""
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
    """Get recent system activity."""
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

@debug_router.get("/users/{user_id}/assessment-history")
async def get_user_assessment_history(user_id: str, limit: int = Query(20, ge=1, le=100)):
    """Get automatic assessment history for a user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        assessments = await conn.fetch("""
            SELECT ca.id, ca.timestamp, ca.user_message, ca.overall_level, 
                   ca.confidence_score, ca.skills_scores, ca.strengths, 
                   ca.areas_for_improvement, ca.vocab_level, ca.grammar_level,
                   ca.vocab_complexity, ca.grammar_accuracy, ca.fluency_score
            FROM conversation_assessments ca
            WHERE ca.user_id = $1 
            ORDER BY ca.timestamp DESC 
            LIMIT $2
        """, user_id, limit)
        
        return [
            {
                "id": str(assessment['id']),
                "timestamp": assessment['timestamp'],
                "user_message": assessment['user_message'][:100] + "..." if len(assessment['user_message']) > 100 else assessment['user_message'],
                "overall_level": assessment['overall_level'],
                "confidence_score": assessment['confidence_score'],
                "skills_scores": assessment['skills_scores'],
                "strengths": assessment['strengths'],
                "areas_for_improvement": assessment['areas_for_improvement'],
                "analysis": {
                    "vocab_level": assessment['vocab_level'],
                    "grammar_level": assessment['grammar_level'],
                    "vocab_complexity": assessment['vocab_complexity'],
                    "grammar_accuracy": assessment['grammar_accuracy'],
                    "fluency_score": assessment['fluency_score']
                }
            } for assessment in assessments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessment history: {str(e)}")
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/level-detection")
async def get_user_level_detection(user_id: str):
    """Get the most recent level detection for a user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        detection = await conn.fetchrow("""
            SELECT detected_level, confidence, evidence, recommendation, 
                   should_advance, should_review, assessment_count, detection_date
            FROM level_detections 
            WHERE user_id = $1 AND is_active = true
            ORDER BY detection_date DESC 
            LIMIT 1
        """, user_id)
        
        if not detection:
            # Realizar detección en tiempo real
            try:
                from src.agent.modules.assessment.assessment_manager import get_assessment_manager
                assessment_manager = get_assessment_manager()
                result = await assessment_manager.detect_user_level(user_id)
                
                return {
                    "user_id": user_id,
                    "detected_level": result.detected_level,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "recommendation": result.recommendation,
                    "should_advance": result.should_advance,
                    "should_review": result.should_review,
                    "assessment_count": len(result.assessment_history),
                    "detection_date": datetime.now(),
                    "real_time": True
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"No level detection found and unable to generate: {str(e)}")
        
        return {
            "user_id": user_id,
            "detected_level": detection['detected_level'],
            "confidence": detection['confidence'],
            "evidence": detection['evidence'],
            "recommendation": detection['recommendation'],
            "should_advance": detection['should_advance'],
            "should_review": detection['should_review'],
            "assessment_count": detection['assessment_count'],
            "detection_date": detection['detection_date'],
            "real_time": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving level detection: {str(e)}")
    finally:
        await conn.close()

@debug_router.get("/users/{user_id}/skill-progression")
async def get_user_skill_progression(user_id: str):
    """Get skill progression for a user."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        progressions = await conn.fetch("""
            SELECT skill, current_score, trend, level_estimate, next_milestone,
                   scores_history, last_updated
            FROM skill_progressions
            WHERE user_id = $1
            ORDER BY skill
        """, user_id)
        
        return [
            {
                "skill": progression['skill'],
                "current_score": progression['current_score'],
                "trend": progression['trend'],
                "level_estimate": progression['level_estimate'],
                "next_milestone": progression['next_milestone'],
                "scores_history": progression['scores_history'],
                "last_updated": progression['last_updated']
            } for progression in progressions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving skill progression: {str(e)}")
    finally:
        await conn.close()

@debug_router.get("/assessment/summary")
async def get_assessment_system_summary():
    """Get assessment system summary."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        # Estadísticas generales
        total_assessments = await conn.fetchval("SELECT COUNT(*) FROM conversation_assessments")
        active_detections = await conn.fetchval("SELECT COUNT(*) FROM level_detections WHERE is_active = true")
        users_with_progressions = await conn.fetchval("SELECT COUNT(DISTINCT user_id) FROM skill_progressions")
        total_errors_detected = await conn.fetchval("SELECT COUNT(*) FROM detected_errors")
        
        # Distribución de niveles detectados
        level_distribution = await conn.fetch("""
            SELECT detected_level, COUNT(*) as count
            FROM level_detections 
            WHERE is_active = true
            GROUP BY detected_level
            ORDER BY detected_level
        """)
        
        # Habilidades más evaluadas
        skill_stats = await conn.fetch("""
            SELECT skill, COUNT(*) as users, AVG(current_score) as avg_score
            FROM skill_progressions
            GROUP BY skill
            ORDER BY users DESC
        """)
        
        # Tendencias recientes
        recent_trends = await conn.fetch("""
            SELECT trend, COUNT(*) as count
            FROM skill_progressions
            GROUP BY trend
        """)
        
        return {
            "system_stats": {
                "total_assessments": total_assessments or 0,
                "active_level_detections": active_detections or 0,
                "users_with_progressions": users_with_progressions or 0,
                "total_errors_detected": total_errors_detected or 0
            },
            "level_distribution": [
                {"level": row['detected_level'], "count": row['count']} 
                for row in level_distribution
            ],
            "skill_statistics": [
                {
                    "skill": row['skill'], 
                    "users_count": row['users'],
                    "average_score": float(row['avg_score']) if row['avg_score'] else 0.0
                }
                for row in skill_stats
            ],
            "progression_trends": [
                {"trend": row['trend'], "count": row['count']}
                for row in recent_trends
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessment summary: {str(e)}")
    finally:
        await conn.close()

@debug_router.post("/users/{user_id}/assess-message")
async def assess_user_message_debug(user_id: str, message: str, session_id: Optional[str] = None):
    """Assess a specific user message for debugging."""
    try:
        from src.agent.modules.assessment.assessment_manager import get_assessment_manager
        import uuid
        
        assessment_manager = get_assessment_manager()
        
        # Usar session_id proporcionado o generar uno temporal
        test_session_id = session_id or str(uuid.uuid4())
        test_message_id = str(uuid.uuid4())
        
        # FORCE evaluation in debug mode by disabling filters
        original_frequency = assessment_manager.config.assessment_frequency
        original_min_words = assessment_manager.config.min_words_for_assessment
        
        # Debug configuration (always evaluate)
        assessment_manager.config.assessment_frequency = 1
        assessment_manager.config.min_words_for_assessment = 1
        assessment_manager.user_message_counts[user_id] = 0  # Reset counter
        
        # Perform evaluation
        assessment = await assessment_manager.assess_user_message(
            user_id=user_id,
            session_id=test_session_id,
            message_id=test_message_id,
            user_message=message,
            response_time=0.0
        )
        
        # Restore original configuration
        assessment_manager.config.assessment_frequency = original_frequency
        assessment_manager.config.min_words_for_assessment = original_min_words
        
        if not assessment:
            return {
                "message": "No assessment generated despite debug mode",
                "debug_info": {
                    "message_length": len(message.split()),
                    "debug_mode": True
                }
            }
        
        # Convertir a formato serializable
        return {
            "assessment_id": f"debug_{test_message_id}",
            "user_message": assessment.user_message,
            "overall_level": assessment.overall_level,
            "confidence_score": assessment.confidence_score,
            "skills_scores": {k.value: v for k, v in assessment.skills_scores.items()},
            "vocabulary_analysis": {
                "words_used": assessment.vocabulary_analysis.words_used,
                "total_words": assessment.vocabulary_analysis.total_words,
                "unique_words": assessment.vocabulary_analysis.unique_words,
                "advanced_words": assessment.vocabulary_analysis.advanced_words,
                "vocabulary_level": assessment.vocabulary_analysis.vocabulary_level,
                "complexity_score": assessment.vocabulary_analysis.complexity_score
            },
            "grammar_analysis": {
                "errors_detected": assessment.grammar_analysis.errors_detected,
                "error_count": assessment.grammar_analysis.error_count,
                "grammar_level": assessment.grammar_analysis.grammar_level,
                "accuracy_score": assessment.grammar_analysis.accuracy_score,
                "tenses_used": assessment.grammar_analysis.tenses_used
            },
            "fluency_analysis": {
                "message_length": assessment.fluency_analysis.message_length,
                "coherence_score": assessment.fluency_analysis.coherence_score,
                "clarity_score": assessment.fluency_analysis.clarity_score,
                "natural_flow_score": assessment.fluency_analysis.natural_flow_score
            },
            "feedback": {
                "strengths": assessment.strengths,
                "areas_for_improvement": assessment.areas_for_improvement
            },
            "competency_evidence": assessment.competency_evidence,
            "debug_mode": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assessing message: {str(e)}")

@debug_router.get("/assessment/error-patterns")
async def get_error_patterns(user_id: Optional[str] = None, days: int = Query(30, ge=1, le=365)):
    """Get detected error patterns."""
    database_url = get_database_url()
    conn = await asyncpg.connect(database_url)
    
    try:
        if user_id:
            # User-specific errors
            query = """
                SELECT error_type, COUNT(*) as error_count, 
                       array_agg(DISTINCT explanation) as explanations,
                       AVG(CASE WHEN severity = 'critical' THEN 4 
                                WHEN severity = 'major' THEN 3
                                WHEN severity = 'moderate' THEN 2
                                WHEN severity = 'minor' THEN 1
                                ELSE 0 END) as avg_severity_score,
                       MAX(detected_at) as last_occurrence
                FROM detected_errors
                WHERE user_id = $1 AND detected_at >= NOW() - INTERVAL '%s days'
                GROUP BY error_type
                ORDER BY error_count DESC
            """ % days
            params = [user_id]
        else:
            # Global patterns
            query = """
                SELECT error_type, COUNT(*) as error_count,
                       COUNT(DISTINCT user_id) as affected_users,
                       array_agg(DISTINCT explanation) as explanations,
                       AVG(CASE WHEN severity = 'critical' THEN 4 
                                WHEN severity = 'major' THEN 3
                                WHEN severity = 'moderate' THEN 2
                                WHEN severity = 'minor' THEN 1
                                ELSE 0 END) as avg_severity_score,
                       MAX(detected_at) as last_occurrence
                FROM detected_errors
                WHERE detected_at >= NOW() - INTERVAL '%s days'
                GROUP BY error_type
                ORDER BY error_count DESC
            """ % days
            params = []
        
        patterns = await conn.fetch(query, *params)
        
        return [
            {
                "error_type": pattern['error_type'],
                "error_count": pattern['error_count'],
                "affected_users": pattern.get('affected_users', 1),
                "explanations": pattern['explanations'],
                "avg_severity_score": float(pattern['avg_severity_score']) if pattern['avg_severity_score'] else 0.0,
                "last_occurrence": pattern['last_occurrence']
            } for pattern in patterns
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving error patterns: {str(e)}")
    finally:
        await conn.close() 