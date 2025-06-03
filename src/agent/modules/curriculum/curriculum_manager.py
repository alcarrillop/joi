"""
Manages curriculum, progression and user assessments
"""
import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import asdict

import asyncpg

from src.agent.core.database import get_database_url
from .models import (
    UserProgress, CEFRLevel, Competency, AssessmentResult, 
    LevelTransitionCriteria, LearningGoal, SkillType
)
from .curriculum_data import CurriculumData

class CurriculumManager:
    """Manages curriculum, progression and user assessments"""
    
    def __init__(self):
        self.curriculum_data = CurriculumData()
        self.learning_modules = self.curriculum_data.get_learning_modules()
        self.transition_criteria = self.curriculum_data.get_level_transition_criteria()
        
        # Cache for user data
        self._user_progress_cache: Dict[str, UserProgress] = {}
        self._competencies_cache: Dict[str, List[Competency]] = {}
    
    async def initialize_user_progress(self, user_id: str, initial_level: CEFRLevel = CEFRLevel.A1) -> UserProgress:
        """Initialize progress for a new user"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            progress = UserProgress(
                user_id=user_id,
                current_level=initial_level,
                level_start_date=datetime.now()
            )
            
            # Insert into database
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
        """Get current progress of a user"""
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
        """Get available competencies for user's current level"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            progress = await self.initialize_user_progress(user_id)
        
        current_level = progress.current_level
        
        # Get all competencies for current level
        if current_level not in self.learning_modules:
            return []
        
        all_competencies = []
        for module in self.learning_modules[current_level]:
            all_competencies.extend(module.competencies)
        
        # Filter competencies already completed or in progress
        available_competencies = []
        for competency in all_competencies:
            # Check prerequisites are met
            if self._check_prerequisites(competency, progress):
                available_competencies.append(competency)
        
        # If no unstarted competencies, review in-progress ones
        unstarted = [c for c in available_competencies 
                    if c.id not in progress.completed_competencies 
                    and c.id not in progress.in_progress_competencies]
        
        if unstarted:
            return unstarted
        else:
            # Return in-progress competencies for review
            return [c for c in available_competencies 
                   if c.id in progress.in_progress_competencies]
    
    async def get_next_recommended_competency(self, user_id: str) -> Optional[Competency]:
        """Recommend the next competency the user should study"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            progress = await self.initialize_user_progress(user_id)
        
        available_competencies = await self.get_current_competencies(user_id)
        
        # Filter competencies not yet completed or in progress
        not_started = [
            comp for comp in available_competencies 
            if comp.id not in progress.completed_competencies 
            and comp.id not in progress.in_progress_competencies
        ]
        
        if not_started:
            # Prioritize by order and skill type
            not_started.sort(key=lambda x: (x.skill_type.value, x.estimated_hours))
            return not_started[0]
        
        # If no unstarted competencies, review in-progress ones
        in_progress = [
            comp for comp in available_competencies 
            if comp.id in progress.in_progress_competencies
        ]
        
        if in_progress:
            # Prioritize the one with lowest mastery score
            in_progress.sort(key=lambda x: progress.mastery_scores.get(x.id, 0))
            return in_progress[0]
        
        return None
    
    async def start_competency(self, user_id: str, competency_id: str):
        """Mark a competency as in progress"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Get current progress
            progress = await self.get_user_progress(user_id)
            if not progress:
                progress = await self.initialize_user_progress(user_id)
            
            # Add to in-progress competencies
            progress.in_progress_competencies.add(competency_id)
            
            # Update in database
            await conn.execute("""
                UPDATE user_progress 
                SET in_progress_competencies = $1
                WHERE user_id = $2
            """, list(progress.in_progress_competencies), user_id)
            
        finally:
            await conn.close()
    
    async def record_assessment(self, assessment: AssessmentResult):
        """Register an assessment result"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Insert assessment result
            await conn.execute("""
                INSERT INTO assessments (id, user_id, competency_id, skill_type, score, max_score, 
                                       timestamp, feedback, areas_for_improvement, strengths)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, str(uuid.uuid4()), assessment.user_id, assessment.competency_id, 
                assessment.skill_type.value, assessment.score, assessment.max_score,
                assessment.timestamp, assessment.feedback, assessment.areas_for_improvement,
                assessment.strengths)
            
            # Update user progress
            await self._update_user_progress(assessment)
            
        finally:
            await conn.close()
    
    async def _update_user_progress(self, assessment: AssessmentResult):
        """Update user progress based on an assessment"""
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            progress = await self.get_user_progress(assessment.user_id)
            if not progress:
                return
            
            # Calculate mastery score (0-1)
            mastery_score = assessment.score / assessment.max_score
            progress.mastery_scores[assessment.competency_id] = mastery_score
            progress.last_assessment_date = assessment.timestamp
            
            # If score is high enough, mark as completed
            if mastery_score >= 0.8:  # 80% to consider completed
                progress.completed_competencies.add(assessment.competency_id)
                progress.in_progress_competencies.discard(assessment.competency_id)
            
            # Update in database
            await conn.execute("""
                UPDATE user_progress 
                SET completed_competencies = $1,
                    in_progress_competencies = $2,
                    mastery_scores = $3,
                    last_assessment_date = $4
                WHERE user_id = $5
            """, list(progress.completed_competencies), list(progress.in_progress_competencies),
                json.dumps(progress.mastery_scores), progress.last_assessment_date, assessment.user_id)
            
            # Check if user can advance level
            await self._check_level_advancement(assessment.user_id)
            
        finally:
            await conn.close()
    
    async def _check_level_advancement(self, user_id: str) -> bool:
        """Check if user meets criteria to advance level"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            return False
            
        # Find transition criteria for current level
        transition = None
        for criteria in self.transition_criteria:
            if criteria.from_level == progress.current_level:
                transition = criteria
                break
        
        if not transition:
            return False  # No next level defined
        
        # Check criteria
        if not await self._meets_level_transition_criteria(user_id, transition.to_level):
            return False
        
        # User meets criteria, advance level
        await self._advance_user_level(user_id, progress.current_level)
        return True
    
    async def _meets_level_transition_criteria(self, user_id: str, current_level: CEFRLevel) -> bool:
        """Check if user meets all transition criteria"""
        # Get user progress
        progress = await self.get_user_progress(user_id)
        if not progress:
            return False
            
        # 1. Minimum number of competencies completed
        if len(progress.completed_competencies) < 3:  # Simplified criteria
            return False
        
        # 2. Minimum average score
        if progress.mastery_scores:
            avg_score = sum(progress.mastery_scores.values()) / len(progress.mastery_scores)
            if avg_score < 0.8:  # 80% threshold
                return False
        else:
            return False  # No scores recorded
        
        # 3. Time in level (optional check)
        if progress.level_start_date:
            days_in_level = (datetime.now() - progress.level_start_date).days
            if days_in_level < 7:  # At least 1 week
                return False
        
        return True
    
    async def _advance_user_level(self, user_id: str, current_level: CEFRLevel):
        """Advance user to next level"""
        next_level_map = {
            CEFRLevel.A1: CEFRLevel.A2,
            CEFRLevel.A2: CEFRLevel.B1,
            CEFRLevel.B1: CEFRLevel.B2,
            CEFRLevel.B2: CEFRLevel.C1,
            CEFRLevel.C1: CEFRLevel.C2,
        }
        
        new_level = next_level_map.get(current_level)
        if not new_level:
            return  # Already at highest level
        
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        try:
            await conn.execute("""
                UPDATE user_progress 
                SET current_level = $1, level_start_date = NOW()
                WHERE user_id = $2
            """, new_level.value, user_id)
            
            print(f"🎉 User {user_id} has advanced to level {new_level.value}!")
        finally:
            await conn.close()
    
    def _check_prerequisites(self, competency: Competency, user_progress: UserProgress) -> bool:
        """Check if user has completed prerequisites for a competency"""
        for prereq in competency.prerequisites:
            if prereq not in user_progress.completed_competencies:
                return False
        return True
    
    async def get_learning_statistics(self, user_id: str) -> Dict:
        """Get detailed learning statistics"""
        progress = await self.get_user_progress(user_id)
        if not progress:
            return {}
        
        database_url = get_database_url()
        conn = await asyncpg.connect(database_url)
        
        try:
            # Get recent assessments
            recent_assessments = await conn.fetch("""
                SELECT competency_id, skill_type, score, max_score, timestamp
                FROM assessments 
                WHERE user_id = $1 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, user_id)
            
            # Calculate statistics
            total_competencies = len(progress.completed_competencies) + len(progress.in_progress_competencies)
            completion_rate = len(progress.completed_competencies) / max(total_competencies, 1)
            
            avg_score = sum(progress.mastery_scores.values()) / len(progress.mastery_scores) if progress.mastery_scores else 0
            
            # Analysis by skill type
            skill_scores = {}
            for comp_id, score in progress.mastery_scores.items():
                # Search for competency to get its skill type
                for level_modules in self.learning_modules.values():
                    for module in level_modules:
                        for comp in module.competencies:
                            if comp.id == comp_id:
                                skill_type = comp.skill_type.value
                                if skill_type not in skill_scores:
                                    skill_scores[skill_type] = []
                                skill_scores[skill_type].append(score)
                                break
            
            # Averages by skill
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
        """Create a personalized learning goal"""
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
        """Get active learning goals for a user"""
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

    async def get_learning_goals(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active learning goals for a user"""
        # Get user's current competencies
        competencies = await self.get_current_competencies(user_id)
        
        goals = []
        for competency in competencies[:3]:  # Top 3 recommendations
            goals.append({
                "competency_id": competency.id,
                "name": competency.name,
                "description": competency.description,
                "estimated_hours": competency.estimated_hours,
                "skill_type": competency.skill_type.value,
                "priority": "high" if goals == [] else "medium"  # First is high priority
            })
        
        return goals

# Global curriculum manager instance
curriculum_manager = CurriculumManager()

def get_curriculum_manager() -> CurriculumManager:
    """Get the global curriculum manager instance"""
    return curriculum_manager 