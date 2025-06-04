import pytest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import dataclass

try:
    from agent.modules.memory.long_term.memory_manager import MemoryManager, MemoryAnalysis
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

@patch('agent.modules.memory.long_term.memory_manager.get_vector_store')
@pytest.mark.asyncio
async def test_extract_and_store_memories(mock_get_store):
    if MemoryManager is None:
        pytest.skip("Dependencies not available")
    store = FakeVectorStore()
    mock_get_store.return_value = store

    mgr = MemoryManager()
    mgr._analyze_memory = AsyncMock(return_value=MemoryAnalysis(is_important=True, formatted_memory='hello world'))

    @dataclass
    class Msg:
        content: str
        type: str = "human"

    msg = Msg('hi there')
    await mgr.extract_and_store_memories(msg, 'u1', 's1')

    assert store.stored
