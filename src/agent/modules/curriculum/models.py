"""
Simplified data models for vocabulary-focused learning progression
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional


class CEFRLevel(Enum):
    """Common European Framework of Reference for Languages levels"""

    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient


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
