"""
Simple Curriculum Manager - SIMPLIFIED VERSION
"""

import logging
from typing import Dict, List

from .curriculum_data import CurriculumData
from .models import CEFRLevel, Competency, LevelThresholds

logger = logging.getLogger(__name__)


class SimpleCurriculumManager:
    """Simplified curriculum manager - just basic functionality."""

    def __init__(self):
        self.curriculum_data = CurriculumData()
        self.level_thresholds = LevelThresholds()

    def get_competencies_for_level(self, level: str) -> List[Competency]:
        """Get competencies for a specific CEFR level."""
        try:
            level_enum = CEFRLevel(level)
            if level_enum == CEFRLevel.A1:
                return self.curriculum_data.get_a1_competencies()
            elif level_enum == CEFRLevel.A2:
                return self.curriculum_data.get_a2_competencies()
            elif level_enum == CEFRLevel.B1:
                return self.curriculum_data.get_b1_competencies()
            else:
                return []
        except Exception:
            return []

    def get_vocabulary_for_level(self, level: str) -> set:
        """Get target vocabulary for a specific level."""
        competencies = self.get_competencies_for_level(level)
        vocabulary = []
        for comp in competencies:
            vocabulary.extend(comp.key_vocabulary)
        return set(vocabulary)

    async def estimate_level_progress(self, user_id: str) -> Dict:
        """Simple level estimation - kept async for API compatibility."""
        return {
            "estimated_level": "A1",
            "vocabulary_learned": 0,
            "curriculum_insights": {},
            "educational_recommendations": [],
            "level_competencies": [],
        }

    def _analyze_vocabulary_against_curriculum(self, user_id: str, level: str) -> Dict:
        """Simple vocabulary analysis."""
        return {
            "curriculum_mastery_percentage": 0,
            "mastered_curriculum_words": [],
        }

    def _generate_learning_recommendations(self, level: str, vocab_count: int, insights: Dict) -> List[str]:
        """Simple recommendations."""
        return [
            "Practice basic vocabulary",
            "Focus on conversational skills",
        ]


# Singleton pattern
_curriculum_manager = None


def get_curriculum_manager() -> SimpleCurriculumManager:
    """Get curriculum manager instance."""
    global _curriculum_manager
    if _curriculum_manager is None:
        _curriculum_manager = SimpleCurriculumManager()
    return _curriculum_manager
