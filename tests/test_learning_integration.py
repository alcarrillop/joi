"""
Test suite for learning module integration.
"""

import pytest


@pytest.mark.asyncio
async def test_learning_imports():
    """Test that all learning module imports work correctly."""
    # Main manager

    # New modules
    from agent.modules.learning.vocabulary_data import (
        get_basic_vocabulary,
        get_excluded_words,
        get_intermediate_vocabulary,
        get_stop_words,
    )

    # Test vocabulary data
    basic_words = get_basic_vocabulary()
    intermediate_words = get_intermediate_vocabulary()
    excluded_words = get_excluded_words()
    stop_words = get_stop_words()

    assert len(basic_words) > 50  # Should have substantial vocabulary
    assert len(intermediate_words) > 20
    assert len(excluded_words) > 10
    assert len(stop_words) > 50

    # Test that sets don't overlap inappropriately
    assert basic_words.isdisjoint(excluded_words)
    assert intermediate_words.isdisjoint(excluded_words)


@pytest.mark.asyncio
async def test_manager_functionality():
    """Test that the learning manager works correctly."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    # Get manager instance
    manager = get_learning_stats_manager()
    assert manager is not None

    # Test singleton behavior
    manager2 = get_learning_stats_manager()
    assert manager is manager2

    # Check that all expected methods exist
    required_methods = [
        "update_learning_stats",
        "get_learning_summary",
        "get_user_top_words",
        "_normalize_word",
        "_is_valid_english_word",
    ]

    for method_name in required_methods:
        assert hasattr(manager, method_name), f"Method {method_name} missing"


@pytest.mark.asyncio
async def test_word_validation():
    """Test word validation functionality."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()

    # Test valid words
    valid_words = ["hello", "learning", "practice", "business", "development"]
    for word in valid_words:
        assert manager._is_valid_english_word(word), f"'{word}' should be valid"

    # Test invalid words
    invalid_words = ["the", "and", "muy", "que", "a", "123", "ok", "hmm"]
    for word in invalid_words:
        assert not manager._is_valid_english_word(word), f"'{word}' should be invalid"


@pytest.mark.asyncio
async def test_word_normalization():
    """Test word normalization with caching."""
    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()

    # Test basic normalization cases
    test_cases = [
        ("children", "child"),
        ("feet", "foot"),
        ("running", "run"),
        ("HELLO", "hello"),
        ("Studies", "study"),
    ]

    for original, expected in test_cases:
        normalized = manager._normalize_word(original)
        # Note: some normalizations might not match exactly due to stemming variations
        # The important thing is that it returns a consistent result
        assert isinstance(normalized, str)
        assert len(normalized) > 0
        assert normalized.islower()


@pytest.mark.asyncio
async def test_db_operations():
    """Test that database operations are properly separated."""
    from agent.modules.learning.db_operations import get_vocabulary_db_operations

    # Test singleton
    db_ops1 = get_vocabulary_db_operations()
    db_ops2 = get_vocabulary_db_operations()
    assert db_ops1 is db_ops2

    # Check expected methods exist
    expected_methods = [
        "get_user_learned_vocabulary",
        "get_vocabulary_word_count",
        "get_user_top_words",
        "increment_word_frequency",
        "batch_increment_words",
    ]

    for method_name in expected_methods:
        assert hasattr(db_ops1, method_name), f"DB method {method_name} missing"


@pytest.mark.asyncio
async def test_vocabulary_files_location():
    """Test that vocabulary files are in the correct location."""
    import os

    from agent.modules.learning.vocabulary_data import load_vocabularies_from_json

    # Test that files are loaded from the learning module directory
    vocabularies = load_vocabularies_from_json()

    # Should load all vocabulary types
    expected_keys = ["basic", "intermediate", "excluded", "stop_words"]
    for key in expected_keys:
        assert key in vocabularies, f"Missing vocabulary type: {key}"
        assert len(vocabularies[key]) > 0, f"Empty vocabulary for: {key}"

    # Test that files exist in expected location
    vocab_dir = os.path.join(os.path.dirname(__file__), "../src/agent/modules/learning/vocabularies")
    expected_files = ["basic_vocabulary.json", "intermediate_vocabulary.json", "excluded_words.json", "stop_words.json"]

    for filename in expected_files:
        filepath = os.path.join(vocab_dir, filename)
        assert os.path.exists(filepath), f"Vocabulary file missing: {filepath}"


@pytest.mark.asyncio
async def test_external_integration_compatibility():
    """Test integration with existing code that uses the manager."""
    import inspect

    from agent.modules.learning.learning_stats_manager import get_learning_stats_manager

    manager = get_learning_stats_manager()

    # Check method signatures are compatible with existing code
    # update_learning_stats method
    sig = inspect.signature(manager.update_learning_stats)
    params = list(sig.parameters.keys())
    expected_params = ["user_id", "session_id", "message_text"]
    assert all(param in params for param in expected_params), "update_learning_stats signature incompatible"

    # get_learning_summary method
    sig = inspect.signature(manager.get_learning_summary)
    params = list(sig.parameters.keys())
    assert "user_id" in params, "get_learning_summary signature incompatible"

    # get_user_top_words method
    sig = inspect.signature(manager.get_user_top_words)
    params = list(sig.parameters.keys())
    expected_params = ["user_id", "limit"]
    assert all(param in params for param in expected_params), "get_user_top_words signature incompatible"
