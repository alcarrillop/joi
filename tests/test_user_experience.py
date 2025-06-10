"""
Test suite for user experience elements like terminology and UI text.
"""

import pytest


@pytest.mark.asyncio
async def test_terminology_consistency():
    """Test that UI terminology is consistent and correct."""
    # Mock a response to check terminology
    # We can't easily test the full function without DB, but we can check the code
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    # Get the source code of the function
    source = inspect.getsource(get_user_level_info)

    # Check terminology consistency (simplified version doesn't show interactions count)
    assert "**Conversations:**" not in source, "Should not use 'Conversations' in stats display"

    # Check that motivational messages use "interaction" not "conversation"
    assert "Every interaction teaches" in source, "Should use 'interaction' in motivational text"
    assert "Every conversation teaches" not in source, "Should not use 'conversation' in motivational text"


@pytest.mark.asyncio
async def test_progress_display_elements():
    """Test that progress display has all required elements."""
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    source = inspect.getsource(get_user_level_info)

    # Check for expected UI elements in progress display (simplified version)
    expected_elements = [
        "**Your English Learning Progress**",
        "**Current Level:**",
        "**Vocabulary Learned:**",
        "**Your Most Used Words:**",
        "Keep chatting with me!",
    ]

    for element in expected_elements:
        assert element in source, f"Missing UI element: {element}"


@pytest.mark.asyncio
async def test_emoji_usage():
    """Test that emojis are used consistently in UI."""
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    source = inspect.getsource(get_user_level_info)

    # Check for consistent emoji usage (simplified version)
    expected_emojis = [
        "📊",  # Progress stats
        "🎯",  # Current level/goal
        "📚",  # Vocabulary
        "🔥",  # Most used words
        "🚀",  # Goals/advancement
    ]

    for emoji in expected_emojis:
        assert emoji in source, f"Missing expected emoji: {emoji}"


@pytest.mark.asyncio
async def test_level_descriptions():
    """Test that level descriptions are appropriate."""
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    source = inspect.getsource(get_user_level_info)

    # Check that all CEFR levels have descriptions
    expected_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

    for level in expected_levels:
        assert f'"{level}"' in source, f"Missing description for level: {level}"

    # Check for positive, encouraging language
    encouraging_terms = ["journey", "quite well", "very fluent", "practically a native"]

    # At least some encouraging terms should be present
    found_encouraging = any(term in source for term in encouraging_terms)
    assert found_encouraging, "Should have encouraging language in level descriptions"


@pytest.mark.asyncio
async def test_motivational_messages():
    """Test that motivational messages are varied and encouraging."""
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    source = inspect.getsource(get_user_level_info)

    # Check for varied motivational messages based on progress
    motivational_phrases = [
        "Keep chatting with me!",
        "Great progress!",
        "Excellent work!",
        "Amazing progress!",
        "building a solid foundation",
        "growing steadily",
        "becoming quite fluent",
    ]

    found_phrases = [phrase for phrase in motivational_phrases if phrase in source]
    assert len(found_phrases) >= 4, f"Should have varied motivational messages, found: {found_phrases}"


@pytest.mark.asyncio
async def test_progress_calculation_messaging():
    """Test that progress calculation messages are clear."""
    import inspect

    from agent.graph.utils.helpers import get_user_level_info

    source = inspect.getsource(get_user_level_info)

    # Check for clear progress messaging (simplified version)
    progress_terms = ["more vocabulary words", "Ready to advance"]

    for term in progress_terms:
        assert term in source, f"Missing progress term: {term}"

    # Check that we show clear goal messaging
    assert "**Next Goal:**" in source, "Should show next goal clearly"
