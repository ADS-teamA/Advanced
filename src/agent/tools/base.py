"""
Base Tool Abstraction

Provides lightweight abstraction for all RAG tools with:
- Input validation via Pydantic
- Error handling
- Execution statistics
- Result formatting
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """
    Base input validation using Pydantic.

    All tool inputs inherit from this for automatic validation.
    """

    class Config:
        extra = "forbid"  # Reject unknown fields


@dataclass
class ToolResult:
    """
    Standardized tool execution result.

    All tools return this format for consistency.
    """

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    """
    Lightweight base class for all RAG tools.

    Provides:
    - Input validation (via Pydantic schemas)
    - Error handling (try/catch with graceful degradation)
    - Execution statistics (call count, avg time)
    - Result formatting (consistent ToolResult structure)

    Subclasses implement:
    - name: Tool identifier
    - description: What the tool does
    - tier: 1=basic, 2=advanced, 3=analysis
    - input_schema: Pydantic model for input validation
    - execute_impl(): Tool-specific logic
    """

    # Class attributes (override in subclasses)
    name: str = "base_tool"
    description: str = "Base tool (override in subclass)"
    tier: int = 1
    input_schema: type[ToolInput] = ToolInput

    # Metadata flags
    requires_kg: bool = False  # Requires knowledge graph
    requires_reranker: bool = False  # Requires reranker

    def __init__(
        self,
        vector_store,
        embedder,
        reranker=None,
        graph_retriever=None,
        knowledge_graph=None,
        context_assembler=None,
        config=None,
    ):
        """
        Initialize tool with pipeline components.

        Args:
            vector_store: HybridVectorStore instance
            embedder: EmbeddingGenerator instance
            reranker: CrossEncoderReranker (optional)
            graph_retriever: GraphEnhancedRetriever (optional)
            knowledge_graph: KnowledgeGraph (optional)
            context_assembler: ContextAssembler (optional)
            config: ToolConfig instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.graph_retriever = graph_retriever
        self.knowledge_graph = knowledge_graph
        self.context_assembler = context_assembler
        self.config = config

        # Statistics
        self.execution_count = 0
        self.total_time_ms = 0.0
        self.error_count = 0

    @abstractmethod
    def execute_impl(self, **kwargs) -> ToolResult:
        """
        Tool-specific execution logic.

        Args:
            **kwargs: Validated input parameters

        Returns:
            ToolResult with execution results
        """
        pass

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with validation and error handling.

        Flow:
        1. Validate inputs via Pydantic schema
        2. Execute tool logic with timing
        3. Track statistics
        4. Handle errors gracefully

        Args:
            **kwargs: Tool input parameters

        Returns:
            ToolResult
        """
        start_time = time.time()

        try:
            # Validate inputs
            validated_input = self.input_schema(**kwargs)
            validated_dict = validated_input.model_dump()

            # Execute tool logic
            result = self.execute_impl(**validated_dict)

            # Track statistics
            elapsed_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_time_ms += elapsed_ms
            result.execution_time_ms = elapsed_ms
            result.metadata["tool_name"] = self.name
            result.metadata["tier"] = self.tier

            logger.info(
                f"Tool '{self.name}' executed in {elapsed_ms:.0f}ms " f"(success={result.success})"
            )

            return result

        except Exception as e:
            # Handle errors gracefully
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(f"Tool '{self.name}' failed: {e}", exc_info=True)

            return ToolResult(
                success=False,
                data=None,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                },
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        avg_time = self.total_time_ms / self.execution_count if self.execution_count > 0 else 0
        success_rate = (
            (self.execution_count - self.error_count) / self.execution_count
            if self.execution_count > 0
            else 0
        )

        return {
            "name": self.name,
            "tier": self.tier,
            "execution_count": self.execution_count,
            "error_count": self.error_count,
            "success_rate": round(success_rate * 100, 1),
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(avg_time, 2),
        }

    def get_claude_sdk_definition(self) -> Dict[str, Any]:
        """
        Get Claude SDK tool definition.

        Returns:
            Tool definition dict for Claude SDK
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }
