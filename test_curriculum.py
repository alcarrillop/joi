#!/usr/bin/env python3
"""
Script de prueba para el sistema de currículo
"""
import asyncio
from datetime import datetime
from src.agent.modules.curriculum.curriculum_manager import get_curriculum_manager
from src.agent.modules.curriculum.models import AssessmentResult, SkillType, CEFRLevel

async def test_curriculum_system():
    """Prueba el sistema de currículo completo"""
    curriculum_manager = get_curriculum_manager()
    
    # Usar un usuario existente
    test_user_id = "ec467509-23d5-4ff1-8034-4676e3a85db8"
    
    print("🎓 Testing Curriculum System")
    print("=" * 50)
    
    # 1. Inicializar progreso del usuario
    print("\n1. Inicializando progreso del usuario...")
    progress = await curriculum_manager.initialize_user_progress(test_user_id, CEFRLevel.A1)
    print(f"✓ Usuario {test_user_id} inicializado en nivel {progress.current_level.value}")
    
    # 2. Obtener competencias disponibles
    print("\n2. Obteniendo competencias disponibles...")
    competencies = await curriculum_manager.get_current_competencies(test_user_id)
    print(f"✓ Competencias disponibles: {len(competencies)}")
    for comp in competencies[:3]:  # Mostrar las primeras 3
        print(f"   - {comp.name} ({comp.skill_type.value})")
    
    # 3. Obtener competencia recomendada
    print("\n3. Obteniendo competencia recomendada...")
    recommended = await curriculum_manager.get_next_recommended_competency(test_user_id)
    if recommended:
        print(f"✓ Recomendado: {recommended.name}")
        print(f"   - Vocabulario clave: {', '.join(recommended.key_vocabulary[:5])}...")
        print(f"   - Tiempo estimado: {recommended.estimated_hours} horas")
        
        # 4. Iniciar la competencia
        print(f"\n4. Iniciando competencia: {recommended.id}")
        await curriculum_manager.start_competency(test_user_id, recommended.id)
        print("✓ Competencia marcada como en progreso")
        
        # 5. Simular una evaluación
        print("\n5. Simulando evaluación...")
        assessment = AssessmentResult(
            user_id=test_user_id,
            competency_id=recommended.id,
            skill_type=recommended.skill_type,
            score=8.5,
            max_score=10.0,
            timestamp=datetime.now(),
            feedback="Excellent work on introductions! You're getting the hang of basic personal questions.",
            areas_for_improvement=["Pronunciation of 'th' sounds", "Intonation in questions"],
            strengths=["Clear articulation", "Good use of vocabulary", "Confident delivery"]
        )
        
        await curriculum_manager.record_assessment(assessment)
        print(f"✓ Evaluación registrada: {assessment.score}/{assessment.max_score} ({assessment.score/assessment.max_score*100:.1f}%)")
        print(f"   Feedback: {assessment.feedback}")
    
    # 6. Obtener estadísticas de aprendizaje
    print("\n6. Obteniendo estadísticas de aprendizaje...")
    stats = await curriculum_manager.get_learning_statistics(test_user_id)
    if stats:
        print(f"✓ Nivel actual: {stats['current_level']}")
        print(f"   Competencias completadas: {stats['completed_competencies']}")
        print(f"   Competencias en progreso: {stats['in_progress_competencies']}")
        print(f"   Tasa de completación: {stats['completion_rate']:.1%}")
        print(f"   Puntuación promedio: {stats['average_score']:.1%}")
        print(f"   Días en el nivel actual: {stats['time_in_current_level']}")
        
        if stats['skill_averages']:
            print("   Promedios por habilidad:")
            for skill, avg in stats['skill_averages'].items():
                print(f"     - {skill}: {avg:.1%}")
    
    # 7. Verificar progreso actualizado
    print("\n7. Verificando progreso actualizado...")
    updated_progress = await curriculum_manager.get_user_progress(test_user_id)
    if updated_progress:
        print(f"✓ Competencias completadas: {len(updated_progress.completed_competencies)}")
        print(f"   Competencias en progreso: {len(updated_progress.in_progress_competencies)}")
        print(f"   Puntuaciones de dominio: {len(updated_progress.mastery_scores)}")
        
        if updated_progress.mastery_scores:
            print("   Detalles de dominio:")
            for comp_id, score in updated_progress.mastery_scores.items():
                print(f"     - {comp_id}: {score:.1%}")
    
    print("\n" + "=" * 50)
    print("🎉 ¡Sistema de currículo funcionando correctamente!")

if __name__ == "__main__":
    asyncio.run(test_curriculum_system()) 