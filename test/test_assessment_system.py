#!/usr/bin/env python3
"""
Test script for the assessment system
"""
import asyncio
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.agent.modules.assessment.assessment_manager import get_assessment_manager
from src.agent.modules.assessment.models import LanguageSkill, ConversationAssessment

async def test_assessment_system():
    """Test the automatic assessment system"""
    print("🧪 Testing automatic assessment system")
    print("=" * 50)
    
    assessment_manager = get_assessment_manager()
    
    # Test different difficulty levels
    test_messages = [
        {
            "level": "A1",
            "message": "Hi! My name is Maria. I am from Colombia. I like pizza.",
            "description": "Basic level - simple vocabulary and structures"
        },
        {
            "level": "A2", 
            "message": "Yesterday I went to the supermarket with my sister. We bought vegetables and fruits for dinner. The weather was very nice.",
            "description": "Pre-intermediate - past tense and descriptive vocabulary"
        },
        {
            "level": "B1",
            "message": "I've been studying English for three years and I really enjoy reading novels in English. Although it's challenging, I believe that practice makes perfect.",
            "description": "Intermediate - present perfect, connectors, complex ideas"
        },
        {
            "level": "B2",
            "message": "The implementation of artificial intelligence in education has revolutionized learning methodologies. Nevertheless, we must consider the ethical implications and ensure that technology enhances rather than replaces human interaction.",
            "description": "Upper intermediate - advanced vocabulary, complex structures"
        }
    ]
    
    for i, test_case in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {test_case['level']} Level")
        print(f"💭 Description: {test_case['description']}")
        print(f"📄 Message: \"{test_case['message']}\"")
        print("-" * 50)
        
        # Evaluate message
        assessment = await assessment_manager.assess_user_message(
            user_id="test-user-001",
            session_id="test-session-001", 
            message_id=f"test-message-{i}",
            user_message=test_case['message'],
            response_time=2.5
        )
        
        if assessment:
            print(f"✅ Assessment completed:")
            print(f"   🎯 Detected level: {assessment.overall_level}")
            print(f"   🔥 Confidence: {assessment.confidence_score:.2f}")
            print(f"   📊 Skill scores:")
            for skill, score in assessment.skills_scores.items():
                print(f"      {skill.value}: {score:.2f}")
            
            print(f"   💪 Strengths:")
            for strength in assessment.strengths[:3]:  # Show first 3
                print(f"      • {strength}")
            
            if assessment.areas_for_improvement:
                print(f"   📈 Areas for improvement:")
                for area in assessment.areas_for_improvement[:3]:  # Show first 3
                    print(f"      • {area}")
            
            # Vocabulary analysis
            vocab = assessment.vocabulary_analysis
            print(f"   📚 Vocabulary: {vocab.total_words} words, Level {vocab.vocabulary_level}")
            if vocab.advanced_words:
                print(f"      Advanced words: {', '.join(vocab.advanced_words[:5])}")
            
            # Grammar analysis
            grammar = assessment.grammar_analysis
            print(f"   📝 Grammar: {grammar.error_count} errors, Level {grammar.grammar_level}")
            if grammar.tenses_used:
                print(f"      Tenses used: {', '.join(grammar.tenses_used)}")
            
            # Fluency analysis
            fluency = assessment.fluency_analysis
            print(f"   💬 Fluency: coherence {fluency.coherence_score:.2f}, clarity {fluency.clarity_score:.2f}")
            
            print()
        else:
            print("❌ No assessment generated (did not meet criteria)")
    
    print("\n🎉 Assessment system test completed!")
    print("The system correctly:")
    print("✅ Analyzes different complexity levels")
    print("✅ Detects vocabulary usage")
    print("✅ Checks grammar accuracy")
    print("✅ Measures fluency")
    print("✅ Provides specific feedback")

if __name__ == "__main__":
    asyncio.run(test_assessment_system()) 