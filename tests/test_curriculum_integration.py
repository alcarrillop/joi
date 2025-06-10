"""
Test suite for curriculum and learning module integration after refactoring.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_curriculum_manager_imports():
    """Test that curriculum manager imports and loads correctly."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()
    assert manager is not None

    # Test singleton behavior
    manager2 = get_curriculum_manager()
    assert manager is manager2

    # Verify it's the enhanced version
    assert hasattr(manager, "get_competencies_for_level")
    assert hasattr(manager, "get_vocabulary_for_level")
    assert hasattr(manager, "estimate_level_progress")


@pytest.mark.asyncio
async def test_curriculum_competencies_available():
    """Test that curriculum competencies are properly loaded."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()

    # Test A1 competencies
    a1_competencies = manager.get_competencies_for_level("A1")
    assert len(a1_competencies) > 0

    # Test competency structure
    if a1_competencies:
        comp = a1_competencies[0]
        assert hasattr(comp, "name")
        assert hasattr(comp, "skill_type")
        assert hasattr(comp, "key_vocabulary")
        assert len(comp.key_vocabulary) > 0


@pytest.mark.asyncio
async def test_curriculum_vocabulary_extraction():
    """Test that vocabulary is correctly extracted from curriculum."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()

    # Test vocabulary extraction for A1
    a1_vocab = manager.get_vocabulary_for_level("A1")
    assert len(a1_vocab) > 0
    assert isinstance(a1_vocab, list)

    # Should be English words
    for word in a1_vocab[:5]:  # Check first 5
        assert isinstance(word, str)
        assert len(word) > 0


@pytest.mark.asyncio
async def test_estimate_level_progress_integration():
    """Test that estimate_level_progress uses both learning and curriculum data."""
    from agent.modules.curriculum.curriculum_manager import get_curriculum_manager

    manager = get_curriculum_manager()
    test_user_id = "test-user-123"

    # Mock database operations
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Mock basic user stats (from curriculum manager's get_learning_statistics)
        mock_conn.fetchrow.return_value = {"created_at": "2024-01-01", "total_sessions": 10, "total_messages": 50}

        # Mock vocabulary count (from db_operations via curriculum manager)
        mock_conn.fetchval.return_value = 100

        # Mock user vocabulary (for curriculum analysis)
        mock_conn.fetch.return_value = [("hello",), ("world",), ("learn",), ("study",), ("practice",)]

        # Test estimate_level_progress
        result = await manager.estimate_level_progress(test_user_id)

        # Should return comprehensive analysis
        assert isinstance(result, dict)
        assert "estimated_level" in result
        assert "curriculum_insights" in result
        assert "educational_recommendations" in result
        assert "level_competencies" in result

        # Curriculum insights should have analysis
        insights = result.get("curriculum_insights", {})
        assert "curriculum_mastery_percentage" in insights
        assert "mastered_curriculum_words" in insights


@pytest.mark.asyncio
async def test_debug_endpoint_compatibility():
    """Test that debug endpoint works with refactored modules."""
    from agent.interfaces.debug.debug_endpoints import get_user_learning_statistics

    test_user_id = "test-user-456"

    # Mock all database dependencies
    with (
        patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn,
        patch("agent.modules.memory.long_term.memory_manager.get_memory_manager") as mock_memory,
    ):
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Mock curriculum manager database calls
        mock_conn.fetchrow.return_value = {"created_at": "2024-01-01", "total_sessions": 15, "total_messages": 75}
        mock_conn.fetchval.return_value = 120  # vocab count
        mock_conn.fetch.return_value = [("hello",), ("learning",), ("progress",)]

        # Mock memory manager
        mock_memory_instance = AsyncMock()
        mock_memory_instance.get_user_memory_stats.return_value = {"total_memories": 25}
        mock_memory.return_value = mock_memory_instance

        # Test debug endpoint
        result = await get_user_learning_statistics(test_user_id)

        # Should return structured response
        assert isinstance(result, dict)
        assert "user_id" in result
        assert "curriculum_analysis" in result
        assert "technical_stats" in result
        assert "memory_stats" in result

        # Curriculum analysis should have new insights
        curriculum = result["curriculum_analysis"]
        assert "estimated_level" in curriculum
        assert "curriculum_mastery" in curriculum
        assert "educational_recommendations" in curriculum


@pytest.mark.asyncio
async def test_learning_manager_delegation():
    """Test that learning manager properly delegates to db_operations."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()
    test_user_id = "test-user-789"

    # Mock db_operations
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        mock_conn.fetchval.return_value = 85  # word count
        mock_conn.fetch.return_value = [{"word": "testing", "frequency": 5, "last_used": "2024-12-06"}]

        # Test methods that should delegate to db_ops
        count = await manager.get_vocabulary_word_count(test_user_id)
        assert count == 85

        top_words = await manager.get_user_top_words(test_user_id, limit=5)
        assert isinstance(top_words, list)

        summary = await manager.get_learning_summary(test_user_id)
        assert isinstance(summary, dict)
        assert "vocabulary" in summary


@pytest.mark.asyncio
async def test_separation_of_concerns():
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
