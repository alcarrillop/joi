"""
Test curriculum integration - SIMPLIFIED VERSION
"""


def test_curriculum_manager_imports():
    """Test that curriculum manager can be imported and initialized."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()
    assert manager is not None

    # Verify it's the enhanced version
    assert hasattr(manager, "get_competencies_for_level")
    assert hasattr(manager, "get_vocabulary_for_level")
    assert hasattr(manager, "estimate_level_progress")


def test_curriculum_competencies_available():
    """Test that curriculum has competencies available."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()
    a1_competencies = manager.get_competencies_for_level("A1")

    assert isinstance(a1_competencies, list)
    if a1_competencies:
        comp = a1_competencies[0]
        assert hasattr(comp, "name")
        assert hasattr(comp, "skill_type")
        assert hasattr(comp, "key_vocabulary")
        assert len(comp.key_vocabulary) > 0


def test_curriculum_vocabulary_extraction():
    """Test that curriculum can extract vocabulary for levels."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()
    a1_vocab = manager.get_vocabulary_for_level("A1")

    assert isinstance(a1_vocab, set)
    assert len(a1_vocab) > 0


def test_separation_of_concerns():
    """Test that learning and curriculum have proper separation of concerns."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    learning_manager = get_learning_stats_manager()
    curriculum_manager = get_curriculum_manager()

    # Learning manager should NOT have educational analysis methods
    assert not hasattr(learning_manager, "get_vocabulary_insights")
    assert not hasattr(learning_manager, "_generate_vocabulary_insights")

    # But should have technical methods
    assert hasattr(learning_manager, "_normalize_word")
    assert hasattr(learning_manager, "_is_valid_english_word")
    assert hasattr(learning_manager, "update_learning_stats")

    # Curriculum manager should have educational methods
    assert hasattr(curriculum_manager, "get_competencies_for_level")
    assert hasattr(curriculum_manager, "_analyze_vocabulary_against_curriculum")
    assert hasattr(curriculum_manager, "_generate_learning_recommendations")

    # But should NOT have low-level technical methods
    assert not hasattr(curriculum_manager, "_normalize_word")
    assert not hasattr(curriculum_manager, "_is_valid_english_word")
