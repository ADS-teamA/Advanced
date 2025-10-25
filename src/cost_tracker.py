"""
Cost tracking for API usage (LLM and Embeddings).

Tracks token usage and calculates costs for:
- Anthropic Claude (Haiku, Sonnet, Opus)
- OpenAI (GPT-4o, GPT-5, o-series, embeddings)
- Voyage AI (embeddings)
- Local models (free)

Usage:
    from src.cost_tracker import CostTracker

    tracker = CostTracker()

    # Track LLM usage
    tracker.track_llm(
        provider="anthropic",
        model="claude-haiku-4-5",
        input_tokens=1000,
        output_tokens=500
    )

    # Track embedding usage
    tracker.track_embedding(
        provider="openai",
        model="text-embedding-3-large",
        tokens=10000
    )

    # Get total cost
    cost = tracker.get_total_cost()
    print(f"Total cost: ${cost:.4f}")
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ====================================================================
# PRICING DATA (2025)
# ====================================================================
# Source: https://docs.anthropic.com/pricing, https://openai.com/api/pricing/
# Updated: January 2025

PRICING = {
    # Anthropic Claude models (per 1M tokens)
    "anthropic": {
        # Haiku models
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "haiku": {"input": 1.00, "output": 5.00},
        "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
        "claude-haiku-3": {"input": 0.25, "output": 1.25},

        # Sonnet models
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "sonnet": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-sonnet-3-5": {"input": 3.00, "output": 15.00},

        # Opus models
        "claude-opus-4": {"input": 15.00, "output": 75.00},
        "claude-opus-4-1": {"input": 15.00, "output": 75.00},
        "opus": {"input": 15.00, "output": 75.00},
    },

    # OpenAI models (per 1M tokens)
    "openai": {
        # GPT-4o models
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},

        # GPT-5 models
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.50, "output": 2.00},  # Estimated
        "gpt-5-nano": {"input": 0.25, "output": 1.00},  # Estimated
        "gpt-5-pro": {"input": 5.00, "output": 20.00},  # Estimated
        "gpt-5-codex": {"input": 1.50, "output": 12.00},  # Estimated
        "gpt-5-chat": {"input": 1.25, "output": 10.00},  # Estimated

        # O-series reasoning models
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o3": {"input": 20.00, "output": 80.00},  # Estimated
        "o3-mini": {"input": 3.00, "output": 12.00},
        "o3-pro": {"input": 30.00, "output": 120.00},  # Estimated
        "o4-mini": {"input": 3.00, "output": 12.00},

        # Embeddings (per 1M tokens)
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    },

    # Voyage AI embeddings (per 1M tokens)
    "voyage": {
        "voyage-3-large": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-3": {"input": 0.06, "output": 0.0},
        "voyage-3-lite": {"input": 0.02, "output": 0.0},
        "voyage-law-2": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-finance-2": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-multilingual-2": {"input": 0.12, "output": 0.0},  # Estimated
        "kanon-2": {"input": 0.12, "output": 0.0},  # Estimated
    },

    # Local models (free)
    "huggingface": {
        "bge-m3": {"input": 0.0, "output": 0.0},
        "BAAI/bge-m3": {"input": 0.0, "output": 0.0},
        "bge-large": {"input": 0.0, "output": 0.0},
    },
}


@dataclass
class UsageEntry:
    """Single usage entry (LLM or embedding)."""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str  # "summary", "context", "embedding", "agent", etc.
    cache_creation_tokens: int = 0  # Tokens written to cache
    cache_read_tokens: int = 0  # Tokens read from cache


class CostTracker:
    """
    Track API costs across indexing pipeline and RAG agent.

    Features:
    - Track token usage for LLM and embeddings
    - Calculate costs based on current pricing
    - Support multiple providers (Anthropic, OpenAI, Voyage)
    - Session-based tracking (reset for each indexing/conversation)
    - Detailed breakdown by operation type
    - Immutable public interface (private fields with read-only properties)

    Usage:
        tracker = CostTracker()
        tracker.track_llm("anthropic", "haiku", 1000, 500, "summary")
        tracker.track_embedding("openai", "text-embedding-3-large", 10000, "indexing")

        print(tracker.get_summary())
        print(f"Total cost: ${tracker.total_cost:.4f}")
    """

    def __init__(self):
        """Initialize cost tracker with private fields."""
        # Private storage - prevents external mutation
        self._entries: List[UsageEntry] = []

        # Private accumulators
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0

        # Private breakdowns
        self._cost_by_provider: Dict[str, float] = {}
        self._cost_by_operation: Dict[str, float] = {}

    # Read-only properties for public access
    @property
    def entries(self) -> List[UsageEntry]:
        """Get copy of usage entries (read-only)."""
        return self._entries.copy()

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens."""
        return self._total_output_tokens

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return self._total_cost

    @property
    def cost_by_provider(self) -> Dict[str, float]:
        """Get cost breakdown by provider (read-only copy)."""
        return self._cost_by_provider.copy()

    @property
    def cost_by_operation(self) -> Dict[str, float]:
        """Get cost breakdown by operation (read-only copy)."""
        return self._cost_by_operation.copy()

    def track_llm(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "llm",
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0
    ) -> float:
        """
        Track LLM usage and calculate cost.

        Args:
            provider: "anthropic", "openai", "claude"
            model: Model name or alias
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Operation type ("summary", "context", "agent", etc.)
            cache_creation_tokens: Tokens written to cache (Anthropic only)
            cache_read_tokens: Tokens read from cache (Anthropic only)

        Returns:
            Cost in USD for this call
        """
        # Normalize provider name
        if provider == "claude":
            provider = "anthropic"

        # Get pricing
        cost = self._calculate_llm_cost(provider, model, input_tokens, output_tokens)

        # Store entry
        entry = UsageEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens
        )
        self._entries.append(entry)

        # Update accumulators
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost

        # Update breakdowns
        self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + cost
        self._cost_by_operation[operation] = self._cost_by_operation.get(operation, 0.0) + cost

        # Log with cache info if applicable
        if cache_creation_tokens > 0 or cache_read_tokens > 0:
            logger.debug(
                f"LLM usage tracked: {provider}/{model} - "
                f"{input_tokens} in, {output_tokens} out - ${cost:.6f} "
                f"(cache: {cache_read_tokens} read, {cache_creation_tokens} created)"
            )
        else:
            logger.debug(
                f"LLM usage tracked: {provider}/{model} - "
                f"{input_tokens} in, {output_tokens} out - ${cost:.6f}"
            )

        return cost

    def track_embedding(
        self,
        provider: str,
        model: str,
        tokens: int,
        operation: str = "embedding"
    ) -> float:
        """
        Track embedding usage and calculate cost.

        Args:
            provider: "openai", "voyage", "huggingface"
            model: Model name
            tokens: Number of tokens embedded
            operation: Operation type ("indexing", "query", etc.)

        Returns:
            Cost in USD for this call
        """
        # Get pricing (embeddings only have input cost)
        cost = self._calculate_embedding_cost(provider, model, tokens)

        # Store entry
        entry = UsageEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            cost=cost,
            operation=operation
        )
        self._entries.append(entry)

        # Update accumulators
        self._total_input_tokens += tokens
        self._total_cost += cost

        # Update breakdowns
        self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + cost
        self._cost_by_operation[operation] = self._cost_by_operation.get(operation, 0.0) + cost

        logger.debug(
            f"Embedding usage tracked: {provider}/{model} - "
            f"{tokens} tokens - ${cost:.6f}"
        )

        return cost

    def _calculate_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for LLM usage."""
        # Get pricing for this model
        pricing = PRICING.get(provider, {}).get(model)

        if not pricing:
            logger.warning(
                f"No pricing data for {provider}/{model}. "
                f"Cost calculation skipped. Add pricing to PRICING dict."
            )
            return 0.0

        # Calculate cost (prices are per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _calculate_embedding_cost(
        self,
        provider: str,
        model: str,
        tokens: int
    ) -> float:
        """Calculate cost for embedding usage."""
        # Get pricing for this model
        pricing = PRICING.get(provider, {}).get(model)

        if not pricing:
            logger.warning(
                f"No pricing data for {provider}/{model}. "
                f"Cost calculation skipped."
            )
            return 0.0

        # Calculate cost (prices are per 1M tokens)
        return (tokens / 1_000_000) * pricing["input"]

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        return self.total_cost

    def get_total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics (Anthropic prompt caching).

        Returns:
            Dictionary with cache_read_tokens and cache_creation_tokens
        """
        cache_read = 0
        cache_creation = 0

        for entry in self._entries:
            cache_read += entry.cache_read_tokens
            cache_creation += entry.cache_creation_tokens

        return {
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_creation
        }

    def get_session_cost_summary(self) -> str:
        """
        Get brief cost summary for current session (for CLI display).

        Returns:
            Single line cost summary string
        """
        total = self.get_total_cost()
        tokens = self.get_total_tokens()
        cache_stats = self.get_cache_stats()

        # Basic cost info
        summary = f"💰 Session cost: ${total:.4f} ({tokens:,} tokens)"

        # Add cache info if caching is being used
        if cache_stats["cache_read_tokens"] > 0:
            cache_read = cache_stats["cache_read_tokens"]
            summary += f" | 📦 Cache: {cache_read:,} tokens read (90% saved)"

        return summary

    def get_summary(self) -> str:
        """
        Get formatted cost summary.

        Returns:
            Multi-line string with cost breakdown
        """
        lines = []
        lines.append("=" * 60)
        lines.append("API COST SUMMARY")
        lines.append("=" * 60)

        # Total tokens and cost
        lines.append(f"Total tokens:  {self.get_total_tokens():,}")
        lines.append(f"  Input:       {self._total_input_tokens:,}")
        lines.append(f"  Output:      {self._total_output_tokens:,}")
        lines.append(f"Total cost:    ${self._total_cost:.4f}")
        lines.append("")

        # Cost by provider
        if self._cost_by_provider:
            lines.append("Cost by provider:")
            for provider, cost in sorted(self._cost_by_provider.items()):
                lines.append(f"  {provider:15s} ${cost:.4f}")
            lines.append("")

        # Cost by operation
        if self._cost_by_operation:
            lines.append("Cost by operation:")
            for operation, cost in sorted(self._cost_by_operation.items()):
                lines.append(f"  {operation:15s} ${cost:.4f}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self):
        """Reset tracker (for new indexing/conversation session)."""
        self._entries.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._cost_by_provider.clear()
        self._cost_by_operation.clear()

        logger.info("Cost tracker reset")


# Global instance for easy access
_global_tracker: Optional[CostTracker] = None


def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset global cost tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = CostTracker()

    # Simulate indexing pipeline
    print("Simulating indexing pipeline...")

    # PHASE 2: Summaries (Claude Haiku)
    tracker.track_llm("anthropic", "haiku", 5000, 750, "summary")
    tracker.track_llm("anthropic", "haiku", 3000, 500, "summary")

    # PHASE 3: Contextual retrieval (Claude Haiku)
    tracker.track_llm("anthropic", "haiku", 10000, 1500, "context")

    # PHASE 4: Embeddings (BGE-M3 local - free)
    tracker.track_embedding("huggingface", "bge-m3", 50000, "indexing")

    # Print summary
    print(tracker.get_summary())

    # Simulate RAG agent conversation
    print("\nSimulating RAG agent conversation...")
    tracker.reset()

    # Agent queries (Claude Sonnet)
    tracker.track_llm("anthropic", "sonnet", 2000, 500, "agent")
    tracker.track_llm("anthropic", "sonnet", 1500, 300, "agent")
    tracker.track_llm("anthropic", "sonnet", 3000, 800, "agent")

    # Query embeddings (text-embedding-3-large)
    tracker.track_embedding("openai", "text-embedding-3-large", 500, "query")
    tracker.track_embedding("openai", "text-embedding-3-large", 300, "query")

    # Print summary
    print(tracker.get_summary())
