"""
Test suite for database operations with functional testing.
These tests verify that database operations exist and have proper async structure.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_db_operations_are_async():
    """Test that all database operations are properly async."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()

    # Check that all important methods are async
    async_methods = [
        "get_user_learned_vocabulary",
        "get_vocabulary_word_count",
        "get_user_top_words",
        "increment_word_frequency",
        "batch_increment_words",
        "get_recent_words",
        "get_all_user_words",
        "cleanup_user_vocabulary",
    ]

    for method_name in async_methods:
        method = getattr(db_ops, method_name)
        assert asyncio.iscoroutinefunction(method), f"{method_name} should be async"


@pytest.mark.asyncio
async def test_db_methods_handle_uuid_validation():
    """Test that database methods properly validate UUIDs."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()

    # Test with invalid UUID - should handle gracefully
    invalid_user_id = "not_a_uuid"

    # Mock database connection to avoid real DB connection
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # These should return default values on error, not crash
        count = await db_ops.get_vocabulary_word_count(invalid_user_id)
        assert count == 0  # Should return 0 on error

        vocab = await db_ops.get_user_learned_vocabulary(invalid_user_id)
        assert vocab == []  # Should return empty list on error

        top_words = await db_ops.get_user_top_words(invalid_user_id)
        assert top_words == []  # Should return empty list on error

        recent_words = await db_ops.get_recent_words(invalid_user_id)
        assert recent_words == []  # Should return empty list on error


@pytest.mark.asyncio
async def test_db_operations_use_proper_sql_structure():
    """Test that database operations use proper SQL and parameters."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()
    test_user_id = str(uuid.uuid4())

    # Mock only the database connection to capture SQL calls
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Test increment_word_frequency - should use PostgreSQL function
        await db_ops.increment_word_frequency(test_user_id, "testing")

        # Should call execute with inc_word_freq function
        mock_conn.execute.assert_called()
        sql_call = mock_conn.execute.call_args[0][0]
        assert "inc_word_freq" in sql_call

        # Test get_vocabulary_word_count - should use COUNT
        mock_conn.fetchval.return_value = 0
        await db_ops.get_vocabulary_word_count(test_user_id)

        sql_call = mock_conn.fetchval.call_args[0][0]
        assert "COUNT" in sql_call.upper()
        assert "user_word_stats" in sql_call


@pytest.mark.asyncio
async def test_batch_operations_exist():
    """Test that batch operations are implemented."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()
    test_user_id = str(uuid.uuid4())
    test_words = ["hello", "world", "testing"]

    # Test that batch_increment_words method exists and is callable
    assert hasattr(db_ops, "batch_increment_words")
    assert callable(db_ops.batch_increment_words)

    # Test with mock to avoid actual DB calls
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Should return integer (success count)
        result = await db_ops.batch_increment_words(test_user_id, test_words)
        assert isinstance(result, int)


@pytest.mark.asyncio
async def test_cleanup_operations_exist():
    """Test that cleanup operations are implemented."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()
    test_user_id = str(uuid.uuid4())
    test_words = [("hello", 5), ("world", 3)]

    # Test that cleanup method exists and is callable
    assert hasattr(db_ops, "cleanup_user_vocabulary")
    assert callable(db_ops.cleanup_user_vocabulary)

    # Test with mock
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.fetchval.return_value = 10  # original count

        # Should return dict with results
        result = await db_ops.cleanup_user_vocabulary(test_user_id, test_words)
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_learning_manager_uses_database():
    """Test that learning manager actually calls database operations."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()

    # Verify manager has database-dependent methods
    assert hasattr(manager, "update_learning_stats")
    assert hasattr(manager, "get_learning_summary")
    assert hasattr(manager, "get_user_top_words")

    # These methods should be async since they use database
    assert asyncio.iscoroutinefunction(manager.update_learning_stats)
    assert asyncio.iscoroutinefunction(manager.get_learning_summary)
    assert asyncio.iscoroutinefunction(manager.get_user_top_words)


@pytest.mark.asyncio
async def test_learning_manager_summary_structure():
    """Test that learning summary has proper structure."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()
    test_user_id = str(uuid.uuid4())

    # Mock database calls to test structure
    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Mock database responses
        mock_conn.fetchval.return_value = 25  # word count
        mock_conn.fetch.return_value = []  # empty results

        # Get summary
        summary = await manager.get_learning_summary(test_user_id)

        # Summary should be a dict with simplified structure
        assert isinstance(summary, dict)
        assert "vocabulary" in summary
        assert "success" in summary


@pytest.mark.asyncio
async def test_database_connection_cleanup():
    """Test that database connections are properly closed."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    db_ops = get_vocabulary_db_operations()
    test_user_id = str(uuid.uuid4())

    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Call a database operation
        await db_ops.get_vocabulary_word_count(test_user_id)

        # Connection should be closed in finally block
        mock_conn.close.assert_called()


@pytest.mark.asyncio
async def test_database_operations_singleton():
    """Test that database operations use singleton pattern."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    # Get two instances
    db_ops1 = get_vocabulary_db_operations()
    db_ops2 = get_vocabulary_db_operations()

    # Should be the same instance
    assert db_ops1 is db_ops2

    # Should be the correct type
    from agent.modules.learning.db_operations import VocabularyDBOperations

    assert isinstance(db_ops1, VocabularyDBOperations)


@pytest.mark.asyncio
async def test_end_to_end_with_real_structure():
    """Test end-to-end flow with real method signatures."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()
    test_user_id = str(uuid.uuid4())
    test_session_id = "session_123"
    test_message = "I am learning new vocabulary words today."

    with patch("agent.modules.learning.db_operations.get_db_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value = mock_conn

        # Mock all necessary database returns
        mock_conn.fetchval.return_value = 15  # word count
        mock_conn.fetch.return_value = [
            {"word": "learning", "frequency": 5, "last_used": "2024-12-06"},
            {"word": "vocabulary", "frequency": 3, "last_used": "2024-12-06"},
        ]

        # This should work without errors
        await manager.update_learning_stats(test_user_id, test_session_id, test_message)

        # Should have made database calls
        assert mock_get_conn.called

        # Get learning summary
        summary = await manager.get_learning_summary(test_user_id)

        # Should return something meaningful
        assert summary is not None
        assert (isinstance(summary, str) and len(summary) > 0) or isinstance(summary, dict)
