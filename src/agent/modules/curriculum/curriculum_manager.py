"""
Gestor principal del currículo y progresión de aprendizaje
"""
import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import asdict

import asyncpg

from agent.core.database import get_database_url
from .models import (
    UserProgress, CEFRLevel, Competency, AssessmentResult, 
    LevelTransitionCriteria, LearningGoal, SkillType
)
from .curriculum_data import CurriculumData

class CurriculumManager:
    """Gestiona el currículo, progresión y evaluaciones de usuarios"""
    
    def __init__(self):
        self.curriculum_data = CurriculumData()
        self.learning_modules = self.curriculum_data.get_learning_modules()
        self.transition_criteria = self.curriculum_data.get_level_transition_criteria()
    
    async def initialize_user_progress(self, user_id: str, initial_level: CEFRLevel = CEFRLevel.A1) -> UserProgress:
        """Inicializa el progreso de un nuevo usuario"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            progress = UserProgress(
                user_id=user_id,
                current_level=initial_level,
                level_start_date=datetime.now()
            )
            
            # Insertar en base de datos
            await conn.execute("""
                INSERT INTO user_progress (user_id, current_level, level_start_date, completed_competencies, 
                                         in_progress_competencies, mastery_scores, last_assessment_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (user_id) DO UPDATE SET
                    current_level = $2,
                    level_start_date = $3
            """, user_id, initial_level.value, progress.level_start_date, 
                list(progress.completed_competencies), list(progress.in_progress_competencies),
                json.dumps({}), None)
            
            return progress
            
        finally:
            await conn.close()
    
    async def get_user_progress(self, user_id: str) -> Optional[UserProgress]:
        """Obtiene el progreso actual de un usuario"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            row = await conn.fetchrow("""
                SELECT user_id, current_level, completed_competencies, in_progress_competencies,
                       mastery_scores, last_assessment_date, level_start_date
                FROM user_progress WHERE user_id = $1
            """, user_id)
            
            if not row:
                return None
            
            return UserProgress(
                user_id=row['user_id'],
                current_level=CEFRLevel(row['current_level']),
                completed_competencies=set(row['completed_competencies'] or []),
                in_progress_competencies=set(row['in_progress_competencies'] or []),
                mastery_scores=json.loads(row['mastery_scores']) if row['mastery_scores'] else {},
                last_assessment_date=row['last_assessment_date'],
                level_start_date=row['level_start_date']
            )
            
        finally:
            await conn.close()
    
    async def get_current_competencies(self, user_id: str) -> List[Competency]:
        """Obtiene las competencias disponibles para el nivel actual del usuario"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            # Usuario nuevo, inicializar en A1
            progress = await self.initialize_user_progress(user_id)
        
        current_level = progress.current_level
        modules = self.learning_modules.get(current_level, [])
        
        available_competencies = []
        for module in modules:
            for competency in module.competencies:
                # Verificar prerrequisitos
                if self._check_prerequisites(competency, progress.completed_competencies):
                    available_competencies.append(competency)
        
        return available_competencies
    
    async def get_next_recommended_competency(self, user_id: str) -> Optional[Competency]:
        """Recomienda la siguiente competencia que el usuario debería estudiar"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            progress = await self.initialize_user_progress(user_id)
        
        available_competencies = await self.get_current_competencies(user_id)
        
        # Filtrar competencias ya completadas o en progreso
        not_started = [
            comp for comp in available_competencies 
            if comp.id not in progress.completed_competencies 
            and comp.id not in progress.in_progress_competencies
        ]
        
        if not_started:
            # Priorizar por orden y tipo de habilidad
            not_started.sort(key=lambda x: (x.skill_type.value, x.estimated_hours))
            return not_started[0]
        
        # Si no hay competencias sin empezar, revisar las en progreso
        in_progress = [
            comp for comp in available_competencies 
            if comp.id in progress.in_progress_competencies
        ]
        
        if in_progress:
            # Priorizar la que tiene menor puntuación de dominio
            in_progress.sort(key=lambda x: progress.mastery_scores.get(x.id, 0))
            return in_progress[0]
        
        return None
    
    async def start_competency(self, user_id: str, competency_id: str):
        """Marca una competencia como en progreso"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Obtener progreso actual
            progress = await self.get_user_progress(user_id)
            if not progress:
                progress = await self.initialize_user_progress(user_id)
            
            # Añadir a competencias en progreso
            progress.in_progress_competencies.add(competency_id)
            
            # Actualizar en base de datos
            await conn.execute("""
                UPDATE user_progress 
                SET in_progress_competencies = $1
                WHERE user_id = $2
            """, list(progress.in_progress_competencies), user_id)
            
        finally:
            await conn.close()
    
    async def record_assessment(self, assessment: AssessmentResult):
        """Registra el resultado de una evaluación"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Insertar resultado de evaluación
            await conn.execute("""
                INSERT INTO assessments (id, user_id, competency_id, skill_type, score, max_score, 
                                       timestamp, feedback, areas_for_improvement, strengths)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, str(uuid.uuid4()), assessment.user_id, assessment.competency_id, 
                assessment.skill_type.value, assessment.score, assessment.max_score,
                assessment.timestamp, assessment.feedback, assessment.areas_for_improvement,
                assessment.strengths)
            
            # Actualizar progreso del usuario
            await self._update_user_progress_from_assessment(assessment)
            
        finally:
            await conn.close()
    
    async def _update_user_progress_from_assessment(self, assessment: AssessmentResult):
        """Actualiza el progreso del usuario basado en una evaluación"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            progress = await self.get_user_progress(assessment.user_id)
            if not progress:
                return
            
            # Calcular puntuación de dominio (0-1)
            mastery_score = assessment.score / assessment.max_score
            progress.mastery_scores[assessment.competency_id] = mastery_score
            progress.last_assessment_date = assessment.timestamp
            
            # Si la puntuación es suficientemente alta, marcar como completada
            if mastery_score >= 0.8:  # 80% para considerar completada
                progress.completed_competencies.add(assessment.competency_id)
                progress.in_progress_competencies.discard(assessment.competency_id)
            
            # Actualizar en base de datos
            await conn.execute("""
                UPDATE user_progress 
                SET completed_competencies = $1,
                    in_progress_competencies = $2,
                    mastery_scores = $3,
                    last_assessment_date = $4
                WHERE user_id = $5
            """, list(progress.completed_competencies), list(progress.in_progress_competencies),
                json.dumps(progress.mastery_scores), progress.last_assessment_date, assessment.user_id)
            
            # Verificar si el usuario puede avanzar de nivel
            await self._check_level_transition(assessment.user_id, progress)
            
        finally:
            await conn.close()
    
    async def _check_level_transition(self, user_id: str, progress: UserProgress):
        """Verifica si el usuario cumple los criterios para avanzar de nivel"""
        # Buscar criterios de transición para el nivel actual
        transition = None
        for criteria in self.transition_criteria:
            if criteria.from_level == progress.current_level:
                transition = criteria
                break
        
        if not transition:
            return  # No hay siguiente nivel definido
        
        # Verificar criterios
        if not self._meets_transition_criteria(progress, transition):
            return
        
        # El usuario cumple los criterios, avanzar de nivel
        await self._advance_user_level(user_id, transition.to_level)
    
    def _meets_transition_criteria(self, progress: UserProgress, criteria: LevelTransitionCriteria) -> bool:
        """Verifica si el usuario cumple todos los criterios de transición"""
        # 1. Número mínimo de competencias completadas
        if len(progress.completed_competencies) < criteria.min_competencies_completed:
            return False
        
        # 2. Puntuación promedio mínima
        if progress.mastery_scores:
            avg_score = sum(progress.mastery_scores.values()) / len(progress.mastery_scores)
            if avg_score < criteria.min_average_score:
                return False
        else:
            return False  # No hay puntuaciones registradas
        
        # 3. Competencias core requeridas
        for required_comp in criteria.required_core_competencies:
            if required_comp not in progress.completed_competencies:
                return False
        
        # 4. Tiempo mínimo en el nivel
        if progress.level_start_date:
            days_in_level = (datetime.now() - progress.level_start_date).days
            if days_in_level < criteria.min_time_in_level_days:
                return False
        
        return True
    
    async def _advance_user_level(self, user_id: str, new_level: CEFRLevel):
        """Avanza al usuario al siguiente nivel"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Actualizar nivel en user_progress
            await conn.execute("""
                UPDATE user_progress 
                SET current_level = $1, level_start_date = $2
                WHERE user_id = $3
            """, new_level.value, datetime.now(), user_id)
            
            # Actualizar nivel en users table
            await conn.execute("""
                UPDATE users 
                SET current_level = $1
                WHERE id = $2
            """, new_level.value, uuid.UUID(user_id))
            
            print(f"🎉 Usuario {user_id} ha avanzado a nivel {new_level.value}!")
            
        finally:
            await conn.close()
    
    def _check_prerequisites(self, competency: Competency, completed_competencies: Set[str]) -> bool:
        """Verifica si el usuario ha completado los prerrequisitos para una competencia"""
        for prereq in competency.prerequisites:
            if prereq not in completed_competencies:
                return False
        return True
    
    async def get_learning_statistics(self, user_id: str) -> Dict:
        """Obtiene estadísticas detalladas de aprendizaje"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            return {}
        
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Obtener evaluaciones recientes
            recent_assessments = await conn.fetch("""
                SELECT competency_id, skill_type, score, max_score, timestamp
                FROM assessments 
                WHERE user_id = $1 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, user_id)
            
            # Calcular estadísticas
            total_competencies = len(progress.completed_competencies) + len(progress.in_progress_competencies)
            completion_rate = len(progress.completed_competencies) / max(total_competencies, 1)
            
            avg_score = sum(progress.mastery_scores.values()) / len(progress.mastery_scores) if progress.mastery_scores else 0
            
            # Análisis por tipo de habilidad
            skill_scores = {}
            for comp_id, score in progress.mastery_scores.items():
                # Buscar la competencia para obtener su tipo de habilidad
                for level_modules in self.learning_modules.values():
                    for module in level_modules:
                        for comp in module.competencies:
                            if comp.id == comp_id:
                                skill_type = comp.skill_type.value
                                if skill_type not in skill_scores:
                                    skill_scores[skill_type] = []
                                skill_scores[skill_type].append(score)
                                break
            
            # Promedios por habilidad
            skill_averages = {
                skill: sum(scores) / len(scores)
                for skill, scores in skill_scores.items()
            }
            
            return {
                "user_id": user_id,
                "current_level": progress.current_level.value,
                "completed_competencies": len(progress.completed_competencies),
                "in_progress_competencies": len(progress.in_progress_competencies),
                "completion_rate": completion_rate,
                "average_score": avg_score,
                "skill_averages": skill_averages,
                "time_in_current_level": (datetime.now() - progress.level_start_date).days if progress.level_start_date else 0,
                "recent_assessments": [
                    {
                        "competency_id": a['competency_id'],
                        "skill_type": a['skill_type'], 
                        "score_percentage": a['score'] / a['max_score'] * 100,
                        "timestamp": a['timestamp']
                    } for a in recent_assessments
                ]
            }
            
        finally:
            await conn.close()
    
    async def create_learning_goal(self, goal: LearningGoal):
        """Crea una meta de aprendizaje personalizada"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            await conn.execute("""
                INSERT INTO learning_goals (id, user_id, title, description, target_competencies,
                                          target_completion_date, created_date, is_active, progress_percentage)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, goal.id, goal.user_id, goal.title, goal.description, goal.target_competencies,
                goal.target_completion_date, goal.created_date, goal.is_active, goal.progress_percentage)
            
        finally:
            await conn.close()
    
    async def get_user_learning_goals(self, user_id: str) -> List[LearningGoal]:
        """Obtiene las metas de aprendizaje activas de un usuario"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            rows = await conn.fetch("""
                SELECT id, user_id, title, description, target_competencies, target_completion_date,
                       created_date, is_active, progress_percentage
                FROM learning_goals 
                WHERE user_id = $1 AND is_active = true
                ORDER BY created_date DESC
            """, user_id)
            
            return [
                LearningGoal(
                    id=row['id'],
                    user_id=row['user_id'],
                    title=row['title'],
                    description=row['description'],
                    target_competencies=row['target_competencies'],
                    target_completion_date=row['target_completion_date'],
                    created_date=row['created_date'],
                    is_active=row['is_active'],
                    progress_percentage=row['progress_percentage']
                ) for row in rows
            ]
            
        finally:
            await conn.close()

# Instancia global del gestor de currículo
curriculum_manager = CurriculumManager()

def get_curriculum_manager() -> CurriculumManager:
    """Obtiene la instancia global del gestor de currículo"""
    return curriculum_manager 