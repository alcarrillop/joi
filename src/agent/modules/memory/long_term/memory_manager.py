import logging
import uuid
from datetime import datetime
from typing import List, Optional

from agent.core.prompts import MEMORY_ANALYSIS_PROMPT
from agent.modules.memory.long_term.vector_store import get_vector_store
from agent.settings import settings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")


class MemoryManager:
    """Manager class for handling long-term memory operations."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: BaseMessage, user_id: str, session_id: str) -> None:
        """Extract important information from a message and store in vector store.
        
        Args:
            message: The message to analyze
            user_id: The user ID who sent the message
            session_id: The session ID where the message was sent
        """
        if message.type != "human":
            return

        try:
            # Analyze the message for importance and formatting
            analysis = await self._analyze_memory(message.content)
            if analysis.is_important and analysis.formatted_memory:
                # Check if similar memory exists for this user
                similar = self.vector_store.find_similar_memory(analysis.formatted_memory, user_id)
                if similar:
                    # Skip storage if we already have a similar memory for this user
                    self.logger.info(f"Similar memory already exists for user {user_id}: '{analysis.formatted_memory}'")
                    return

                # Store new memory with user and session metadata
                self.logger.info(f"Storing new memory for user {user_id}: '{analysis.formatted_memory}'")
                self.vector_store.store_memory(
                    text=analysis.formatted_memory,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                )
        except Exception as e:
            self.logger.warning(f"Failed to extract and store memories for user {user_id}: {e}")

    def get_relevant_memories(self, context: str, user_id: str) -> List[str]:
        """Retrieve relevant memories based on the current context for a specific user.
        
        Args:
            context: The context to search for relevant memories
            user_id: The user ID to filter memories by
            
        Returns:
            List of relevant memory texts for the user
        """
        try:
            memories = self.vector_store.search_memories(context, user_id=user_id, k=settings.MEMORY_TOP_K)
            if memories:
                for memory in memories:
                    self.logger.debug(f"Memory for user {user_id}: '{memory.text}' (score: {memory.score:.2f})")
            return [memory.text for memory in memories]
        except Exception as e:
            self.logger.warning(f"Failed to retrieve memories for user {user_id}: {e}")
            return []

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)

    def get_user_memory_stats(self, user_id: str) -> dict:
        """Get memory statistics for a user.
        
        Args:
            user_id: The user ID to get stats for
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            memory_count = self.vector_store.get_user_memory_count(user_id)
            return {
                "total_memories": memory_count,
                "user_id": user_id,
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory stats for user {user_id}: {e}")
            return {"total_memories": 0, "user_id": user_id}

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user (GDPR compliance).
        
        Args:
            user_id: The user ID whose memories should be deleted
            
        Returns:
            Number of memories deleted
        """
        try:
            deleted_count = self.vector_store.delete_user_memories(user_id)
            self.logger.info(f"Deleted {deleted_count} memories for user {user_id}")
            return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to delete memories for user {user_id}: {e}")
            return 0


def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    return MemoryManager()
