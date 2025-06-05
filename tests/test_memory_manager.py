import os
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Set required environment variables for testing
os.environ.setdefault("GROQ_API_KEY", "test_groq_key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test_elevenlabs_key")
os.environ.setdefault("ELEVENLABS_VOICE_ID", "test_voice_id")
os.environ.setdefault("TOGETHER_API_KEY", "test_together_key")
os.environ.setdefault("WHATSAPP_TOKEN", "test_whatsapp_token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "123456789")
os.environ.setdefault("WHATSAPP_VERIFY_TOKEN", "test_verify_token")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test_supabase_key")
os.environ.setdefault("QDRANT_URL", "memory")
os.environ.setdefault("QDRANT_API_KEY", "test_qdrant_key")
os.environ.setdefault("TESTING", "true")

try:
    from agent.modules.memory.long_term.memory_manager import MemoryAnalysis, MemoryManager
except Exception:  # pragma: no cover - skip if deps missing
    MemoryManager = None  # type: ignore
    MemoryAnalysis = None


class FakeVectorStore:
    def __init__(self):
        self.stored = []

    def find_similar_memory(self, text, user_id):
        return None

    def store_memory(self, text, metadata):
        self.stored.append((text, metadata))


@patch("agent.modules.memory.long_term.memory_manager.get_vector_store")
@pytest.mark.asyncio
async def test_extract_and_store_memories(mock_get_store):
    if MemoryManager is None:
        pytest.skip("Dependencies not available")
    store = FakeVectorStore()
    mock_get_store.return_value = store

    mgr = MemoryManager()
    mgr._analyze_memory = AsyncMock(return_value=MemoryAnalysis(is_important=True, formatted_memory="hello world"))

    @dataclass
    class Msg:
        content: str
        type: str = "human"

    msg = Msg("hi there")
    await mgr.extract_and_store_memories(msg, "u1", "s1")

    assert store.stored
