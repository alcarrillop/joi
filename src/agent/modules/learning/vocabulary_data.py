"""
Vocabulary Data
==============

External vocabulary sets for different CEFR levels.
This separates vocabulary data from code logic for better maintainability.
"""

import json
import os
from typing import Set

# Basic vocabulary for beginners (A1-A2 level)
BASIC_VOCABULARY = {
    "hello",
    "goodbye",
    "yes",
    "no",
    "please",
    "thank",
    "you",
    "welcome",
    "sorry",
    "excuse",
    "help",
    "understand",
    "speak",
    "english",
    "learning",
    "practice",
    "study",
    "name",
    "age",
    "from",
    "live",
    "work",
    "home",
    "family",
    "friend",
    "time",
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "afternoon",
    "evening",
    "night",
    "week",
    "month",
    "year",
    "food",
    "water",
    "coffee",
    "tea",
    "book",
    "phone",
    "computer",
    "car",
    "house",
    "city",
    "country",
    "good",
    "bad",
    "nice",
    "beautiful",
    "big",
    "small",
    "hot",
    "cold",
    "happy",
    "sad",
    "tired",
    "hungry",
    "thirsty",
    "like",
    "love",
    "want",
    "need",
    "have",
    "get",
    "go",
    "come",
    "see",
    "hear",
    "talk",
    "listen",
    "read",
    "write",
    "eat",
    "drink",
    "walk",
    "run",
    "sleep",
    "wake",
    "open",
    "close",
    "buy",
    "sell",
    "give",
    "take",
    "make",
    "do",
    "say",
    "tell",
    "ask",
    "answer",
    "school",
    "teacher",
    "student",
}

# Advanced vocabulary for intermediate+ learners (B1+ level)
INTERMEDIATE_VOCABULARY = {
    "marketing",
    "business",
    "presentation",
    "meeting",
    "client",
    "project",
    "pronunciation",
    "accent",
    "fluent",
    "improve",
    "practice",
    "conference",
    "professional",
    "colleague",
    "manager",
    "responsibility",
    "deadline",
    "objective",
    "strategy",
    "networking",
    "communication",
    "opportunity",
    "development",
    "experience",
    "organization",
    "decision",
    "investment",
    "technology",
    "innovation",
    "collaboration",
    "achievement",
    "performance",
    "efficiency",
    "productivity",
    "creativity",
    "leadership",
    "motivation",
    "challenge",
    "solution",
    "analysis",
    "research",
    "investigation",
    "implementation",
    "evaluation",
    "recommendation",
}

# Common non-English words and proper nouns to exclude
EXCLUDED_WORDS = {
    "muy",
    "que",
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "en",
    "con",
    "por",
    "para",
    "whatsapp",
    "instagram",
    "facebook",
    "youtube",
    "zoom",
    "google",
    "microsoft",
    "apple",
    "covid",
    "coronavirus",
    "ok",
    "okay",
    "yeah",
    "yep",
    "nope",
    "hmm",
    "uhm",
    # Technical/system words that shouldn't count as vocabulary
    "image",
    "analysi",  # Common misnormalization
    "photo",
    "attachment",
    "file",
    "upload",
    "download",
    "app",
    "bot",
    "system",
    "error",
    "message",
    "text",
    "emoji",
    "gif",
    "jpg",
    "png",
    "pdf",
    "url",
    "link",
    "email",
    "username",
    "password",
}

# Stop words (not useful for tracking)
STOP_WORDS = {
    "the",
    "and",
    "but",
    "or",
    "so",
    "if",
    "when",
    "where",
    "why",
    "how",
    "what",
    "who",
    "which",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
    "now",
    "then",
    "very",
    "much",
    "many",
    "some",
    "any",
    "all",
    "every",
    "each",
    "more",
    "most",
    "less",
    "few",
    "little",
    "big",
    "small",
    "good",
    "bad",
    "well",
    "better",
    "best",
    "nice",
    "great",
    "said",
    "thank",
    "thanks",
    "i",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "must",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "as",
    "it",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}


def get_basic_vocabulary() -> Set[str]:
    """Get basic vocabulary set."""
    return BASIC_VOCABULARY.copy()


def get_intermediate_vocabulary() -> Set[str]:
    """Get intermediate vocabulary set."""
    return INTERMEDIATE_VOCABULARY.copy()


def get_excluded_words() -> Set[str]:
    """Get excluded words set."""
    return EXCLUDED_WORDS.copy()


def get_stop_words() -> Set[str]:
    """Get stop words set."""
    return STOP_WORDS.copy()


def save_vocabularies_to_json(data_dir: str = None) -> None:
    """Save vocabularies to JSON files for external editing."""
    import os

    if data_dir is None:
        # Default to vocabularies directory within the learning module
        data_dir = os.path.join(os.path.dirname(__file__), "vocabularies")
    os.makedirs(data_dir, exist_ok=True)

    vocabularies = {
        "basic_vocabulary.json": list(BASIC_VOCABULARY),
        "intermediate_vocabulary.json": list(INTERMEDIATE_VOCABULARY),
        "excluded_words.json": list(EXCLUDED_WORDS),
        "stop_words.json": list(STOP_WORDS),
    }

    for filename, words in vocabularies.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sorted(words), f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(words)} words to {filepath}")


def load_vocabularies_from_json(data_dir: str = None) -> dict:
    """Load vocabularies from JSON files if they exist."""
    if data_dir is None:
        # Default to vocabularies directory within the learning module
        data_dir = os.path.join(os.path.dirname(__file__), "vocabularies")

    vocabularies = {}

    files = {
        "basic": "basic_vocabulary.json",
        "intermediate": "intermediate_vocabulary.json",
        "excluded": "excluded_words.json",
        "stop_words": "stop_words.json",
    }

    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    vocabularies[key] = set(json.load(f))
                print(f"✅ Loaded {len(vocabularies[key])} {key} words from {filepath}")
            except Exception as e:
                print(f"❌ Failed to load {filepath}: {e}")
                vocabularies[key] = globals()[key.upper() + "_VOCABULARY" if key != "stop_words" else "STOP_WORDS"]
        else:
            # Use default if file doesn't exist
            vocabularies[key] = globals()[key.upper() + "_VOCABULARY" if key != "stop_words" else "STOP_WORDS"]

    return vocabularies


if __name__ == "__main__":
    # Generate JSON files for external editing
    save_vocabularies_to_json()
