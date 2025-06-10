"""
Complete data models for structured curriculum and learning progression
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class CEFRLevel(Enum):
    """Common European Framework of Reference for Languages levels"""

    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient


class SkillType(Enum):
    """Core language skill types"""

    SPEAKING = "speaking"
    LISTENING = "listening"
    READING = "reading"
    WRITING = "writing"
    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    PRONUNCIATION = "pronunciation"


class CompetencyType(Enum):
    """Types of language competencies"""

    INTRODUCTIONS = "introductions"
    BASIC_VOCABULARY = "basic_vocabulary"
    PRESENT_SIMPLE = "present_simple"
    FAMILY_VOCABULARY = "family_vocabulary"
    DAILY_ROUTINES = "daily_routines"
    FOOD_VOCABULARY = "food_vocabulary"
    PAST_SIMPLE = "past_simple"
    FUTURE_PLANS = "future_plans"
    COMPARATIVES = "comparatives"
    PRESENT_PERFECT = "present_perfect"
    CONDITIONALS = "conditionals"
    ADVANCED_VOCABULARY = "advanced_vocabulary"


@dataclass
class VocabularyProgress:
    """User vocabulary learning progress"""

    user_id: str
    current_level: CEFRLevel
    total_words_learned: int = 0
    words_by_level: Dict[str, int] = field(default_factory=dict)  # level -> word count
    last_updated: Optional[datetime] = None
    level_start_date: Optional[datetime] = None


@dataclass
class LevelThresholds:
    """Vocabulary thresholds for CEFR levels"""

    A1: int = 75
    A2: int = 150
    B1: int = 250
    B2: int = 400
    C1: int = 600
    C2: int = 800


@dataclass
class Competency:
    """Individual learning competency with detailed structure"""

    id: str
    name: str
    description: str
    level: CEFRLevel
    skill_type: SkillType
    competency_type: CompetencyType
    learning_objectives: List[str] = field(default_factory=list)
    key_vocabulary: List[str] = field(default_factory=list)
    grammar_points: List[str] = field(default_factory=list)
    practice_activities: List[str] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_hours: int = 1


@dataclass
class LearningModule:
    """Collection of related competencies"""

    id: str
    name: str
    description: str
    level: CEFRLevel
    competencies: List[str] = field(default_factory=list)  # Competency IDs
    estimated_total_hours: int = 0


@dataclass
class LevelTransitionCriteria:
    """Criteria for advancing between CEFR levels"""

    from_level: CEFRLevel
    to_level: CEFRLevel
    min_vocabulary_words: int
    min_competencies_completed: int
    required_competency_types: List[CompetencyType] = field(default_factory=list)
    min_practice_hours: int = 0


@dataclass
class LearningGoal:
    """Simple learning goal for vocabulary"""

    id: str
    user_id: str
    title: str
    target_words: int
    target_level: CEFRLevel
    target_date: datetime
    created_date: datetime
    is_active: bool = True
    progress_percentage: float = 0.0
