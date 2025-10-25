"""
Base Tool Abstraction

Provides lightweight abstraction for all RAG tools with:
- Input validation via Pydantic
- Error handling
- Execution statistics
- Result formatting
"""

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


def estimate_tokens_from_result(result_data: Any) -> int:
    """
    Estimate token count from tool result data.

    Uses JSON serialization + character count / 4 heuristic.
    This is an approximation - actual tokenization depends on the model.

    Args:
        result_data: Tool result data (any JSON-serializable type)

    Returns:
        Estimated token count
    """
    try:
        # Serialize to JSON string
        json_str = json.dumps(result_data, ensure_ascii=False, default=str)

        # Estimate tokens: ~4 chars per token (approximation)
        # Actual ratio varies: 3-4 for English, 4-6 for code/JSON
        # Using ceil for conservative estimate (rounds up)
        estimated_tokens = math.ceil(len(json_str) / 4.0)

        return max(estimated_tokens, 1)  # Minimum 1 token
    except (TypeError, ValueError) as e:
        # Only catch serialization errors, not programming bugs
        logger.error(f"Failed to estimate tokens from result: {e}")
        return 0


class ToolInput(BaseModel):
    """
    Base input validation using Pydantic.

    All tool inputs inherit from this for automatic validation.
    """

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields


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
    estimated_tokens: int = 0  # Estimated token count of result data

    def __post_init__(self):
        """Validate ToolResult invariants."""
        # Validate execution time
        if self.execution_time_ms < 0:
            raise ValueError(f"Execution time cannot be negative: {self.execution_time_ms}")

        # Validate success/error relationship
        if self.success and self.error is not None:
            raise ValueError("Successful results cannot have errors")
        if not self.success and not self.error:
            raise ValueError("Failed results must have an error message")


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
    description: str = "Base tool (override in subclass)"  # Short description (API)
    detailed_help: str = ""  # Detailed help text (for get_tool_help)
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

            # Estimate token count from result data
            result.estimated_tokens = estimate_tokens_from_result(result.data)
            result.metadata["estimated_tokens"] = result.estimated_tokens

            logger.info(
                f"Tool '{self.name}' executed in {elapsed_ms:.0f}ms "
                f"(success={result.success}, ~{result.estimated_tokens} tokens)"
            )

            return result

        except ValidationError as e:
            # Pydantic validation errors - user input issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.warning(f"Tool '{self.name}' validation failed: {e}")

            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid input: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "validation",
                },
            )

        except (KeyError, AttributeError, IndexError, TypeError) as e:
            # Programming errors - these are bugs in tool implementation
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(
                f"Tool '{self.name}' implementation error: {e}",
                exc_info=True,
                extra={"kwargs": kwargs},
            )

            return ToolResult(
                success=False,
                data=None,
                error=f"Internal tool error - this is a bug. {type(e).__name__}: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "programming",
                },
            )

        except (OSError, RuntimeError, MemoryError) as e:
            # System errors - resource issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(f"Tool '{self.name}' system error: {e}", exc_info=True)

            return ToolResult(
                success=False,
                data=None,
                error=f"System error: {type(e).__name__}: {str(e)}. Try again or contact administrator.",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "system",
                },
            )

        except Exception as e:
            # Unexpected errors - catch-all for unknown issues
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1

            logger.error(
                f"Tool '{self.name}' unexpected error: {type(e).__name__}: {e}",
                exc_info=True,
                extra={"kwargs": kwargs},
            )

            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                metadata={
                    "tool_name": self.name,
                    "tier": self.tier,
                    "execution_time_ms": elapsed_ms,
                    "error_type": "unexpected",
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
