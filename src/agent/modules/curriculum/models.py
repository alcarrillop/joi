"""
Data models for curriculum system and learning progression
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set


class CEFRLevel(Enum):
    """Niveles del Marco Común Europeo de Referencia para las Lenguas"""

    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient


class SkillType(Enum):
    """Types of language skills"""

    LISTENING = "listening"
    SPEAKING = "speaking"
    READING = "reading"
    WRITING = "writing"
    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    PRONUNCIATION = "pronunciation"


class CompetencyType(Enum):
    """Types of specific competencies"""

    # Vocabulary competencies
    BASIC_VOCABULARY = "basic_vocabulary"
    FAMILY_VOCABULARY = "family_vocabulary"
    FOOD_VOCABULARY = "food_vocabulary"
    WORK_VOCABULARY = "work_vocabulary"
    TRAVEL_VOCABULARY = "travel_vocabulary"

    # Grammar competencies
    PRESENT_SIMPLE = "present_simple"
    PRESENT_CONTINUOUS = "present_continuous"
    PAST_SIMPLE = "past_simple"
    FUTURE_SIMPLE = "future_simple"
    MODAL_VERBS = "modal_verbs"
    CONDITIONALS = "conditionals"

    # Communication competencies
    INTRODUCTIONS = "introductions"
    SMALL_TALK = "small_talk"
    ASKING_DIRECTIONS = "asking_directions"
    ORDERING_FOOD = "ordering_food"
    JOB_INTERVIEWS = "job_interviews"
    PRESENTATIONS = "presentations"


@dataclass
class Competency:
    """Una competencia específica del currículo"""

    id: str
    name: str
    description: str
    level: CEFRLevel
    skill_type: SkillType
    competency_type: CompetencyType
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    key_vocabulary: List[str] = field(default_factory=list)
    grammar_points: List[str] = field(default_factory=list)
    practice_activities: List[str] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    estimated_hours: int = 0


@dataclass
class LearningModule:
    """A learning module that groups competencies"""

    id: str
    name: str
    description: str
    level: CEFRLevel
    competencies: List[Competency]
    order: int
    is_core: bool = True  # True para módulos obligatorios, False para opcionales


@dataclass
class UserProgress:
    """User progress in the curriculum"""

    user_id: str
    current_level: CEFRLevel
    completed_competencies: Set[str] = field(default_factory=set)
    in_progress_competencies: Set[str] = field(default_factory=set)
    mastery_scores: Dict[str, float] = field(default_factory=dict)  # competency_id -> score (0-1)
    last_assessment_date: Optional[datetime] = None
    level_start_date: Optional[datetime] = None


@dataclass
class AssessmentResult:
    """Result of an assessment"""

    user_id: str
    competency_id: str
    skill_type: SkillType
    score: float  # 0-1
    max_score: float
    timestamp: datetime
    feedback: str
    areas_for_improvement: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


@dataclass
class LevelTransitionCriteria:
    """Criterios para avanzar de nivel"""

    from_level: CEFRLevel
    to_level: CEFRLevel
    min_competencies_completed: int
    min_average_score: float
    required_core_competencies: List[str]
    min_time_in_level_days: int = 7  # Mínimo tiempo en el nivel actual


@dataclass
class LearningGoal:
    """Meta de aprendizaje personalizada"""

    id: str
    user_id: str
    title: str
    description: str
    target_competencies: List[str]
    target_completion_date: datetime
    created_date: datetime
    is_active: bool = True
    progress_percentage: float = 0.0
