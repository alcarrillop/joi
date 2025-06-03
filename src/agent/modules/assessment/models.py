"""
Models for the automatic assessment system and level detection
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class AssessmentType(Enum):
    """Types of automatic assessment"""
    CONVERSATION = "conversation"
    VOCABULARY_USAGE = "vocabulary_usage"
    GRAMMAR_ACCURACY = "grammar_accuracy"
    FLUENCY = "fluency"
    COMPREHENSION = "comprehension"
    PRONUNCIATION = "pronunciation"

class ErrorSeverity(Enum):
    """Severity of detected errors"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

class LanguageSkill(Enum):
    """Language skills to assess"""
    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    FLUENCY = "fluency"
    ACCURACY = "accuracy"
    COMPLEXITY = "complexity"
    COHERENCE = "coherence"

@dataclass
class VocabularyAnalysis:
    """Analysis of vocabulary used by the user"""
    words_used: List[str]
    total_words: int
    unique_words: int
    advanced_words: List[str]
    basic_words: List[str]
    vocabulary_level: str  # A1, A2, B1, etc.
    complexity_score: float  # 0-1
    appropriateness_score: float  # 0-1

@dataclass
class GrammarAnalysis:
    """Grammatical analysis of user text"""
    errors_detected: List[Dict[str, str]]  # [{type, original, corrected, explanation}]
    error_count: int
    sentence_structures: List[str]
    tenses_used: List[str]
    grammar_level: str
    accuracy_score: float  # 0-1
    complexity_score: float  # 0-1

@dataclass
class FluencyAnalysis:
    """Fluency analysis in conversation"""
    response_time: float  # seconds
    message_length: int
    coherence_score: float  # 0-1
    clarity_score: float  # 0-1
    natural_flow_score: float  # 0-1

@dataclass
class ConversationAssessment:
    """Complete assessment of a conversation intervention"""
    user_id: str
    session_id: str
    message_id: str
    timestamp: datetime
    user_message: str
    
    # Detailed analysis
    vocabulary_analysis: VocabularyAnalysis
    grammar_analysis: GrammarAnalysis
    fluency_analysis: FluencyAnalysis
    
    # Overall results
    overall_level: str  # A1, A2, B1, B2, C1, C2
    confidence_score: float  # 0-1, confidence in the assessment
    skills_scores: Dict[LanguageSkill, float]
    
    # Generated feedback
    strengths: List[str]
    areas_for_improvement: List[str]
    competency_evidence: Dict[str, float]

@dataclass
class LevelDetectionResult:
    """Result of automatic level detection"""
    user_id: str
    detected_level: str
    confidence: float
    evidence: Dict[str, float]  # skill -> score
    assessment_history: List[ConversationAssessment]
    recommendation: str
    should_advance: bool
    should_review: bool

@dataclass
class SkillProgression:
    """Progression of a specific skill over time"""
    skill: LanguageSkill
    scores_history: List[Tuple[datetime, float]]
    current_score: float
    trend: str  # "improving", "stable", "declining"
    level_estimate: str
    next_milestone: Optional[str]

@dataclass
class AutoAssessmentConfig:
    """Configuration for automatic assessment"""
    min_words_for_assessment: int = 5
    assessment_frequency: int = 3  # evaluate every N messages
    confidence_threshold: float = 0.7
    level_change_threshold: float = 0.8
    grammar_weight: float = 0.3
    vocabulary_weight: float = 0.3
    fluency_weight: float = 0.2
    accuracy_weight: float = 0.2 