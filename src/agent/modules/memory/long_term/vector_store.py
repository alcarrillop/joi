import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from agent.settings import get_settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None

    @property
    def user_id(self) -> Optional[str]:
        return self.metadata.get("user_id")

    @property
    def session_id(self) -> Optional[str]:
        return self.metadata.get("session_id")


class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._validate_env_vars()
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            settings = get_settings()
            self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            self._initialized = True

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists."""
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

        # Create index for user_id field to enable filtering
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME, field_name="user_id", field_schema=PayloadSchemaType.KEYWORD
        )

        # Create index for session_id field as well
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME, field_name="session_id", field_schema=PayloadSchemaType.KEYWORD
        )

    def find_similar_memory(self, text: str, user_id: str) -> Optional[Memory]:
        """Find if a similar memory already exists for a specific user.

        Args:
            text: The text to search for
            user_id: The user ID to filter by

        Returns:
            Optional Memory if a similar one is found for this user
        """
        results = self.search_memories(text, user_id=user_id, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, user_id, session_id, etc.)
        """
        if not self._collection_exists():
            self._create_collection()

        # Validate required metadata
        if "user_id" not in metadata:
            raise ValueError("user_id is required in metadata")
        if "session_id" not in metadata:
            raise ValueError("session_id is required in metadata")

        # Check if similar memory exists for this user
        similar_memory = self.find_similar_memory(text, metadata["user_id"])
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, user_id: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store for a specific user.

        Args:
            query: Text to search for
            user_id: The user ID to filter by
            k: Number of results to return

        Returns:
            List of Memory objects belonging to the specified user
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)

        # Create filter for user_id
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=user_filter,
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]

    def get_user_memory_count(self, user_id: str) -> int:
        """Get the total count of memories for a specific user.

        Args:
            user_id: The user ID to count memories for

        Returns:
            Total number of memories for the user
        """
        if not self._collection_exists():
            return 0

        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        collection_info = self.client.get_collection(self.COLLECTION_NAME)
        if collection_info.points_count == 0:
            return 0

        # Count points with the user filter
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=[0] * 384,  # Dummy vector for counting
            query_filter=user_filter,
            limit=10000,  # Large limit to get all matches
        )

        return len(results)

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a specific user.

        Args:
            user_id: The user ID whose memories should be deleted

        Returns:
            Number of memories deleted
        """
        if not self._collection_exists():
            return 0

        # Get all memories for the user first to count them
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

        memories_before = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=[0] * 384,  # Dummy vector
            query_filter=user_filter,
            limit=10000,  # Large limit to get all matches
        )

        memories_count = len(memories_before)

        if memories_count > 0:
            # Delete points with the user filter
            self.client.delete(collection_name=self.COLLECTION_NAME, points_selector=user_filter)

        return memories_count


@lru_cache
def get_vector_store() -> VectorStore:
    """Get singleton instance of VectorStore."""
    return VectorStore()
