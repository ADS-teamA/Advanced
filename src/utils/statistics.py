"""
Statistics utilities for MY_SUJBOT pipeline.

Provides standardized statistics tracking patterns across:
- reranker.py (RerankingStats)
- embedding_generator.py (cache stats)
- agent/agent_core.py (conversation stats)
- agent/tools/base.py (tool execution stats)

Features:
- Standardized OperationStats dataclass
- Helper functions for common calculations
- Consistent to_dict() format

Usage:
    from src.utils import OperationStats, compute_hit_rate

    stats = OperationStats(operation_name="embedding")
    stats.total_calls += 1
    stats.total_time_ms += 150.5

    print(stats.to_dict())
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationStats:
    """
    Standardized statistics for any operation.

    Tracks:
    - Call counts (total, success, errors)
    - Timing (total, average)
    - Success rate
    - Custom metadata

    Example:
        >>> stats = OperationStats(operation_name="reranking")
        >>> stats.total_calls = 10
        >>> stats.success_count = 9
        >>> stats.error_count = 1
        >>> stats.total_time_ms = 1500.0
        >>>
        >>> print(stats.to_dict())
        {
            "operation": "reranking",
            "total_calls": 10,
            "success_count": 9,
            "error_count": 1,
            "success_rate": 90.0,
            "total_time_ms": 1500.0,
            "avg_time_ms": 150.0
        }
    """

    operation_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate statistics invariants."""
        if self.total_calls < 0:
            raise ValueError(f"total_calls cannot be negative: {self.total_calls}")
        if self.total_time_ms < 0:
            raise ValueError(f"total_time_ms cannot be negative: {self.total_time_ms}")
        if self.success_count < 0:
            raise ValueError(f"success_count cannot be negative: {self.success_count}")
        if self.error_count < 0:
            raise ValueError(f"error_count cannot be negative: {self.error_count}")

        # Validate consistency
        if self.success_count + self.error_count > self.total_calls:
            raise ValueError(
                f"success_count ({self.success_count}) + error_count ({self.error_count}) "
                f"cannot exceed total_calls ({self.total_calls})"
            )

    @property
    def avg_time_ms(self) -> float:
        """Calculate average time per call."""
        return self.total_time_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        return self.success_count / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def success_rate_percent(self) -> float:
        """Calculate success rate as percentage (0-100)."""
        return self.success_rate * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0-1)."""
        return self.error_count / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def error_rate_percent(self) -> float:
        """Calculate error rate as percentage (0-100)."""
        return self.error_rate * 100

    def to_dict(self, include_percentages: bool = True) -> Dict[str, Any]:
        """
        Serialize to dictionary (standardized format).

        Args:
            include_percentages: Include success_rate as percentage

        Returns:
            Dictionary with standardized keys and rounded values

        Example:
            >>> stats.to_dict()
            {
                "operation": "embedding",
                "total_calls": 100,
                "success_count": 95,
                "error_count": 5,
                "success_rate": 95.0,
                "total_time_ms": 15000.0,
                "avg_time_ms": 150.0
            }
        """
        result = {
            "operation": self.operation_name,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
        }

        # Add success rate (as percentage or fraction)
        if include_percentages:
            result["success_rate"] = round(self.success_rate_percent, 1)  # e.g., 95.0%
        else:
            result["success_rate"] = round(self.success_rate, 3)  # e.g., 0.950

        # Merge custom metadata
        result.update(self.metadata)

        return result

    def record_call(self, success: bool, time_ms: float) -> None:
        """
        Record a single call (convenience method).

        Args:
            success: Whether call succeeded
            time_ms: Execution time in milliseconds

        Example:
            >>> stats = OperationStats("api_call")
            >>> stats.record_call(success=True, time_ms=150.5)
        """
        self.total_calls += 1
        self.total_time_ms += time_ms

        if success:
            self.success_count += 1
        else:
            self.error_count += 1


def compute_hit_rate(hits: int, misses: int) -> float:
    """
    Compute cache hit rate.

    Args:
        hits: Number of cache hits
        misses: Number of cache misses

    Returns:
        Hit rate as fraction (0-1)

    Example:
        >>> hit_rate = compute_hit_rate(hits=80, misses=20)
        >>> print(f"Hit rate: {hit_rate * 100:.1f}%")
        Hit rate: 80.0%
    """
    total = hits + misses
    return hits / total if total > 0 else 0.0


def compute_miss_rate(hits: int, misses: int) -> float:
    """
    Compute cache miss rate.

    Args:
        hits: Number of cache hits
        misses: Number of cache misses

    Returns:
        Miss rate as fraction (0-1)
    """
    return 1.0 - compute_hit_rate(hits, misses)


def compute_average(total: float, count: int) -> float:
    """
    Compute average (safe division).

    Args:
        total: Total value
        count: Number of items

    Returns:
        Average or 0.0 if count is 0

    Example:
        >>> avg = compute_average(total=1500.0, count=10)
        >>> print(f"Average: {avg}")
        Average: 150.0
    """
    return total / count if count > 0 else 0.0


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value as fraction (0-1)
        decimals: Number of decimal places

    Returns:
        Formatted percentage (e.g., "95.5%")

    Example:
        >>> format_percentage(0.955, decimals=1)
        '95.5%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_duration_ms(milliseconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        milliseconds: Duration in milliseconds

    Returns:
        Formatted duration (e.g., "150.5ms", "1.5s", "2.5m")

    Example:
        >>> format_duration_ms(150.5)
        '150.5ms'
        >>> format_duration_ms(1500)
        '1.5s'
        >>> format_duration_ms(150000)
        '2.5m'
    """
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds / 1000:.1f}s"
    else:
        return f"{milliseconds / 60000:.1f}m"


# Example usage
if __name__ == "__main__":
    print("=== Statistics Utilities Examples ===\n")

    # Example 1: Basic OperationStats
    print("1. Basic OperationStats...")
    stats = OperationStats(operation_name="reranking")
    stats.record_call(success=True, time_ms=150.5)
    stats.record_call(success=True, time_ms=125.0)
    stats.record_call(success=False, time_ms=200.0)

    print(f"   Stats: {stats.to_dict()}")

    # Example 2: Hit rate calculation
    print("\n2. Cache hit rate...")
    hit_rate = compute_hit_rate(hits=80, misses=20)
    print(f"   Hit rate: {format_percentage(hit_rate)}")

    # Example 3: Duration formatting
    print("\n3. Duration formatting...")
    durations = [150.5, 1500, 150000]
    for ms in durations:
        print(f"   {ms}ms → {format_duration_ms(ms)}")

    # Example 4: OperationStats with metadata
    print("\n4. OperationStats with custom metadata...")
    stats_with_meta = OperationStats(
        operation_name="embedding", metadata={"model": "bge-m3", "dimensions": 1024}
    )
    stats_with_meta.total_calls = 100
    stats_with_meta.success_count = 95
    stats_with_meta.error_count = 5
    stats_with_meta.total_time_ms = 15000.0

    print(f"   Stats: {stats_with_meta.to_dict()}")

    print("\n=== All examples completed ===")
