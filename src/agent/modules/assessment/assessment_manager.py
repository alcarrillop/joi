"""
Main manager for automatic assessment during conversations
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .analyzers import FluencyAnalyzer, GrammarAnalyzer, VocabularyAnalyzer
from .models import AutoAssessmentConfig, ConversationAssessment, LanguageSkill, LevelDetectionResult, SkillProgression


# Importaciones que se resuelven en tiempo de ejecución
def get_curriculum_manager():
    from src.agent.modules.curriculum.curriculum_manager import get_curriculum_manager as _get_curriculum_manager

    return _get_curriculum_manager()


def get_curriculum_models():
    from src.agent.modules.curriculum.models import AssessmentResult, CEFRLevel, SkillType

    return CEFRLevel, AssessmentResult, SkillType


class AssessmentManager:
    """Main automatic assessment manager"""

    def __init__(self, config: Optional[AutoAssessmentConfig] = None):
        self.config = config or AutoAssessmentConfig()
        self.vocabulary_analyzer = VocabularyAnalyzer()
        self.grammar_analyzer = GrammarAnalyzer()
        self.fluency_analyzer = FluencyAnalyzer()

        # Cache for recent assessments
        self.recent_assessments: Dict[str, List[ConversationAssessment]] = {}
        self.user_message_counts: Dict[str, int] = {}

    async def assess_user_message(
        self, user_id: str, session_id: str, message_id: str, user_message: str, response_time: float = 0.0
    ) -> Optional[ConversationAssessment]:
        """Assess a user message in real time"""

        # Check if the message meets minimum criteria
        if not self._should_assess_message(user_id, user_message):
            return None

        # Perform detailed analysis
        vocabulary_analysis = self.vocabulary_analyzer.analyze_vocabulary(user_message)
        grammar_analysis = self.grammar_analyzer.analyze_grammar(user_message)
        fluency_analysis = self.fluency_analyzer.analyze_fluency(user_message, response_time)

        # Determine overall level and scores
        overall_level = self._determine_overall_level(vocabulary_analysis, grammar_analysis, fluency_analysis)

        skills_scores = self._calculate_skills_scores(vocabulary_analysis, grammar_analysis, fluency_analysis)

        confidence_score = self._calculate_confidence(vocabulary_analysis, grammar_analysis, fluency_analysis)

        # Generate recommendations
        strengths, areas_for_improvement = self._generate_feedback(
            vocabulary_analysis, grammar_analysis, fluency_analysis
        )

        # Identify competency evidence
        competency_evidence = await self._identify_competency_evidence(
            user_id, user_message, vocabulary_analysis, grammar_analysis
        )

        # Create complete assessment
        assessment = ConversationAssessment(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.now(),
            user_message=user_message,
            vocabulary_analysis=vocabulary_analysis,
            grammar_analysis=grammar_analysis,
            fluency_analysis=fluency_analysis,
            overall_level=overall_level,
            confidence_score=confidence_score,
            skills_scores=skills_scores,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            competency_evidence=competency_evidence,
        )

        # Store assessment
        await self._store_assessment(assessment)

        # Update cache
        if user_id not in self.recent_assessments:
            self.recent_assessments[user_id] = []
        self.recent_assessments[user_id].append(assessment)

        # Keep only the last 10 assessments
        if len(self.recent_assessments[user_id]) > 10:
            self.recent_assessments[user_id] = self.recent_assessments[user_id][-10:]

        # Check if it's time to update competencies
        await self._update_curriculum_progress(assessment)

        return assessment

    async def detect_user_level(self, user_id: str, lookback_days: int = 7) -> LevelDetectionResult:
        """Detect the user's current level based on recent assessments"""

        # Get recent assessments
        assessments = await self._get_recent_assessments(user_id, lookback_days)

        if not assessments:
            return LevelDetectionResult(
                user_id=user_id,
                detected_level="A1",
                confidence=0.3,
                evidence={},
                assessment_history=[],
                recommendation="Need more conversation data for accurate assessment",
                should_advance=False,
                should_review=False,
            )

        # Analyze trends
        level_counts = {}
        skill_scores = {skill: [] for skill in LanguageSkill}

        for assessment in assessments:
            level = assessment.overall_level
            level_counts[level] = level_counts.get(level, 0) + 1

            for skill, score in assessment.skills_scores.items():
                skill_scores[skill].append(score)

        # Determine most frequent level
        detected_level = max(level_counts, key=level_counts.get)
        confidence = level_counts[detected_level] / len(assessments)

        # Calculate average evidence per skill
        evidence = {}
        for skill, scores in skill_scores.items():
            if scores:
                evidence[skill.value] = sum(scores) / len(scores)

        # Determine recommendations
        recommendation, should_advance, should_review = self._generate_level_recommendation(
            detected_level, confidence, evidence, assessments
        )

        return LevelDetectionResult(
            user_id=user_id,
            detected_level=detected_level,
            confidence=confidence,
            evidence=evidence,
            assessment_history=assessments,
            recommendation=recommendation,
            should_advance=should_advance,
            should_review=should_review,
        )

    async def get_skill_progression(self, user_id: str, skill: LanguageSkill) -> SkillProgression:
        """Obtiene la progresión de una habilidad específica"""

        # Obtener historial de evaluaciones
        assessments = await self._get_recent_assessments(user_id, days=30)

        # Extraer puntuaciones para la habilidad específica
        scores_history = []
        for assessment in assessments:
            if skill in assessment.skills_scores:
                scores_history.append((assessment.timestamp, assessment.skills_scores[skill]))

        if not scores_history:
            return SkillProgression(
                skill=skill,
                scores_history=[],
                current_score=0.0,
                trend="stable",
                level_estimate="A1",
                next_milestone=None,
            )

        # Calcular puntuación actual y tendencia
        current_score = scores_history[-1][1]
        trend = self._calculate_trend(scores_history)
        level_estimate = self._score_to_level(current_score)
        next_milestone = self._get_next_milestone(skill, level_estimate)

        return SkillProgression(
            skill=skill,
            scores_history=scores_history,
            current_score=current_score,
            trend=trend,
            level_estimate=level_estimate,
            next_milestone=next_milestone,
        )

    def _should_assess_message(self, user_id: str, message: str) -> bool:
        """Determine if a message should be assessed"""
        # Check minimum word count
        if len(message.split()) < self.config.min_words_for_assessment:
            return False

        # Check frequency
        self.user_message_counts[user_id] = self.user_message_counts.get(user_id, 0) + 1

        # Assess every N messages
        if self.user_message_counts[user_id] % self.config.assessment_frequency == 0:
            return True

        return False

    def _determine_overall_level(self, vocab_analysis, grammar_analysis, fluency_analysis) -> str:
        """Determina el nivel general basado en todos los análisis"""

        # Mapear niveles a números para cálculo
        level_map = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        reverse_map = {v: k for k, v in level_map.items()}

        # Obtener niveles numéricos
        vocab_level = level_map.get(vocab_analysis.vocabulary_level, 1)
        grammar_level = level_map.get(grammar_analysis.grammar_level, 1)

        # Fluency contributes less to overall level
        fluency_contribution = 1
        if fluency_analysis.natural_flow_score > 0.8:
            fluency_contribution = 1.5
        elif fluency_analysis.natural_flow_score > 0.6:
            fluency_contribution = 1.2

        # Weighted calculation
        weighted_level = (
            vocab_level * self.config.vocabulary_weight
            + grammar_level * self.config.grammar_weight
            + fluency_contribution * 0.3
        )

        # Round and convert back
        final_level = min(6, max(1, round(weighted_level)))
        return reverse_map[final_level]

    def _calculate_skills_scores(
        self, vocab_analysis, grammar_analysis, fluency_analysis
    ) -> Dict[LanguageSkill, float]:
        """Calcula puntuaciones por habilidad"""
        return {
            LanguageSkill.VOCABULARY: vocab_analysis.complexity_score,
            LanguageSkill.GRAMMAR: grammar_analysis.accuracy_score,
            LanguageSkill.FLUENCY: fluency_analysis.natural_flow_score,
            LanguageSkill.ACCURACY: grammar_analysis.accuracy_score,
            LanguageSkill.COMPLEXITY: (vocab_analysis.complexity_score + grammar_analysis.complexity_score) / 2,
            LanguageSkill.COHERENCE: fluency_analysis.coherence_score,
        }

    def _calculate_confidence(self, vocab_analysis, grammar_analysis, fluency_analysis) -> float:
        """Calcula la confianza en la evaluación"""

        # Factors that affect confidence
        word_count = vocab_analysis.total_words

        # Base confidence based on data amount
        if word_count >= 20:
            base_confidence = 0.9
        elif word_count >= 10:
            base_confidence = 0.7
        elif word_count >= 5:
            base_confidence = 0.5
        else:
            base_confidence = 0.3

        # Adjust for consistency in analysis
        consistency_penalty = 0
        if abs(vocab_analysis.complexity_score - grammar_analysis.complexity_score) > 0.4:
            consistency_penalty = 0.1

        return max(0.1, base_confidence - consistency_penalty)

    def _generate_feedback(self, vocab_analysis, grammar_analysis, fluency_analysis) -> Tuple[List[str], List[str]]:
        """Genera feedback de fortalezas y áreas de mejora"""

        strengths = []
        areas_for_improvement = []

        # Vocabulary analysis
        if vocab_analysis.complexity_score > 0.7:
            strengths.append("Good use of varied vocabulary")
        elif vocab_analysis.complexity_score < 0.4:
            areas_for_improvement.append("Try using more diverse vocabulary")

        if len(vocab_analysis.advanced_words) > 0:
            strengths.append(f"Uses advanced words: {', '.join(vocab_analysis.advanced_words[:3])}")

        # Grammar analysis
        if grammar_analysis.accuracy_score > 0.8:
            strengths.append("Excellent grammar accuracy")
        elif grammar_analysis.error_count > 0:
            error_types = set(error["type"] for error in grammar_analysis.errors_detected)
            areas_for_improvement.append(f"Focus on: {', '.join(error_types)}")

        if len(grammar_analysis.tenses_used) > 2:
            strengths.append("Uses variety of verb tenses")

        # Fluency analysis
        if fluency_analysis.coherence_score > 0.7:
            strengths.append("Clear and coherent communication")
        elif fluency_analysis.coherence_score < 0.5:
            areas_for_improvement.append("Work on connecting ideas more clearly")

        if fluency_analysis.natural_flow_score > 0.8:
            strengths.append("Natural conversational flow")

        return strengths, areas_for_improvement

    async def _identify_competency_evidence(
        self, user_id: str, message: str, vocab_analysis, grammar_analysis
    ) -> Dict[str, float]:
        """Identify evidence of specific curriculum competencies"""

        evidence = {}

        try:
            # Get user's current competencies
            curriculum_manager = get_curriculum_manager()
            current_competencies = await curriculum_manager.get_current_competencies(user_id)
            CEFRLevel, AssessmentResult, SkillType = get_curriculum_models()

            for competency in current_competencies:
                score = 0.0

                # Vocabulary evidence
                vocab_matches = sum(
                    1
                    for word in vocab_analysis.words_used
                    if word.lower() in [v.lower() for v in competency.key_vocabulary]
                )
                if competency.key_vocabulary:
                    vocab_score = vocab_matches / len(competency.key_vocabulary)
                    score += vocab_score * 0.4

                # Grammar evidence (simplified)
                if competency.skill_type == SkillType.GRAMMAR:
                    score += grammar_analysis.accuracy_score * 0.6
                elif competency.skill_type == SkillType.VOCABULARY:
                    score += vocab_analysis.complexity_score * 0.6
                elif competency.skill_type == SkillType.SPEAKING:
                    score += (vocab_analysis.appropriateness_score + grammar_analysis.accuracy_score) / 2 * 0.6

                if score > 0.3:  # Minimum threshold for evidence
                    evidence[competency.id] = min(1.0, score)
        except Exception as e:
            print(f"Error identifying competency evidence: {e}")

        return evidence

    async def _store_assessment(self, assessment: ConversationAssessment):
        """Store the assessment in the database"""

        try:
            import json

            import asyncpg

            from src.agent.core.database import get_database_url

            database_url = get_database_url()
            conn = await asyncpg.connect(database_url)

            try:
                await conn.execute(
                    """
                    INSERT INTO conversation_assessments (
                        id, user_id, session_id, message_id, timestamp, user_message,
                        overall_level, confidence_score, skills_scores, strengths,
                        areas_for_improvement, competency_evidence, vocab_level,
                        grammar_level, vocab_complexity, grammar_accuracy, fluency_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    str(uuid.uuid4()),
                    assessment.user_id,
                    assessment.session_id,
                    assessment.message_id,
                    assessment.timestamp,
                    assessment.user_message,
                    assessment.overall_level,
                    assessment.confidence_score,
                    json.dumps({k.value: v for k, v in assessment.skills_scores.items()}),
                    assessment.strengths,
                    assessment.areas_for_improvement,
                    json.dumps(assessment.competency_evidence),
                    assessment.vocabulary_analysis.vocabulary_level,
                    assessment.grammar_analysis.grammar_level,
                    assessment.vocabulary_analysis.complexity_score,
                    assessment.grammar_analysis.accuracy_score,
                    assessment.fluency_analysis.natural_flow_score,
                )
            finally:
                await conn.close()
        except Exception as e:
            print(f"Error storing assessment: {e}")

    async def _get_recent_assessments(self, user_id: str, days: int = 7) -> List[ConversationAssessment]:
        """Get recent assessments from the database"""

        # First check cache
        if user_id in self.recent_assessments:
            cached = self.recent_assessments[user_id]
            recent_cached = [a for a in cached if (datetime.now() - a.timestamp).days <= days]
            if recent_cached:
                return recent_cached

        # TODO: Implement complete DB search
        return []

    async def _update_curriculum_progress(self, assessment: ConversationAssessment):
        """Update curriculum progress based on assessment"""

        try:
            CEFRLevel, AssessmentResult, SkillType = get_curriculum_models()
            curriculum_manager = get_curriculum_manager()

            # If there's sufficient competency evidence, create assessments
            for competency_id, evidence_score in assessment.competency_evidence.items():
                if evidence_score > 0.7:  # High threshold for automatic assessment
                    # Create assessment for curriculum
                    curriculum_assessment = AssessmentResult(
                        user_id=assessment.user_id,
                        competency_id=competency_id,
                        skill_type=SkillType.SPEAKING,  # Default, could be more specific
                        score=evidence_score * 10,  # Convert to 0-10 scale
                        max_score=10.0,
                        timestamp=assessment.timestamp,
                        feedback="Automated assessment based on conversation analysis",
                        areas_for_improvement=assessment.areas_for_improvement,
                        strengths=assessment.strengths,
                    )

                    # Register in curriculum system
                    await curriculum_manager.record_assessment(curriculum_assessment)
        except Exception as e:
            print(f"Error updating curriculum progress: {e}")

    def _generate_level_recommendation(
        self, level: str, confidence: float, evidence: Dict, assessments: List
    ) -> Tuple[str, bool, bool]:
        """Genera recomendaciones de nivel"""

        should_advance = False
        should_review = False

        if confidence > self.config.confidence_threshold:
            avg_score = sum(evidence.values()) / len(evidence) if evidence else 0

            if avg_score > 0.85:
                should_advance = True
                recommendation = f"Strong performance at {level} level. Consider advancing to next level."
            elif avg_score < 0.6:
                should_review = True
                recommendation = f"May need more practice at {level} level before advancing."
            else:
                recommendation = f"Solid {level} level performance. Continue practicing current level."
        else:
            recommendation = f"Need more conversation data for confident {level} level assessment."

        return recommendation, should_advance, should_review

    def _calculate_trend(self, scores_history: List[Tuple[datetime, float]]) -> str:
        """Calcula la tendencia de una habilidad"""
        if len(scores_history) < 3:
            return "stable"

        # Simple trend analysis
        recent_scores = [score for _, score in scores_history[-5:]]
        early_avg = sum(recent_scores[: len(recent_scores) // 2]) / max(1, len(recent_scores) // 2)
        later_avg = sum(recent_scores[len(recent_scores) // 2 :]) / max(1, len(recent_scores) - len(recent_scores) // 2)

        if later_avg > early_avg + 0.1:
            return "improving"
        elif later_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"

    def _score_to_level(self, score: float) -> str:
        """Convierte puntuación a nivel CEFR"""
        if score >= 0.85:
            return "B2"
        elif score >= 0.7:
            return "B1"
        elif score >= 0.55:
            return "A2"
        else:
            return "A1"

    def _get_next_milestone(self, skill: LanguageSkill, current_level: str) -> Optional[str]:
        """Obtiene el próximo hito para una habilidad"""
        milestones = {
            LanguageSkill.VOCABULARY: {
                "A1": "Learn 500 basic words",
                "A2": "Master 1000 common words",
                "B1": "Use 2000+ words confidently",
                "B2": "Express complex ideas clearly",
            },
            LanguageSkill.GRAMMAR: {
                "A1": "Master present simple tense",
                "A2": "Use past and future tenses",
                "B1": "Handle complex sentence structures",
                "B2": "Use advanced grammar naturally",
            },
        }

        skill_milestones = milestones.get(skill, {})
        return skill_milestones.get(current_level)


# Global assessment manager instance
assessment_manager = AssessmentManager()


def get_assessment_manager() -> AssessmentManager:
    """Get the global assessment manager instance"""
    return assessment_manager
