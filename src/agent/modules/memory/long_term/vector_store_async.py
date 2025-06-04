"""Async wrapper around Qdrant vector store operations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from agent.settings import get_settings


@dataclass
class Memory:
    text: str
    metadata: dict
    score: Optional[float] = None


class AsyncVectorStore:
    COLLECTION_NAME = "long_term_memory"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        self.model = SentenceTransformer(self.EMBEDDING_MODEL)

    async def _ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        if not any(c.name == self.COLLECTION_NAME for c in collections.collections):
            sample = self.model.encode("sample")
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
            )

    async def store_memory(self, text: str, metadata: dict) -> None:
        await self._ensure_collection()
        embedding = self.model.encode(text)
        point = PointStruct(id=metadata.get("id", hash(text)), vector=embedding.tolist(), payload={"text": text, **metadata})
        await self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])

    async def search_memories(self, query: str, user_id: str, k: int = 5) -> List[Memory]:
        await self._ensure_collection()
        query_emb = self.model.encode(query)
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        results = await self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_emb.tolist(),
            query_filter=user_filter,
            limit=k,
        )
        return [Memory(text=r.payload["text"], metadata=r.payload, score=r.score) for r in results]


@lru_cache
def get_async_vector_store() -> AsyncVectorStore:
    return AsyncVectorStore()
