"""Tests for cost display in CLI."""

import pytest
from src.cost_tracker import CostTracker


class TestCostDisplayMethods:
    """Test new cost display methods in CostTracker."""

    def test_get_cache_stats_no_cache(self):
        """Test cache stats with no caching."""
        tracker = CostTracker()

        # Track without cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="agent"
        )

        stats = tracker.get_cache_stats()
        assert stats["cache_read_tokens"] == 0
        assert stats["cache_creation_tokens"] == 0

    def test_get_cache_stats_with_cache(self):
        """Test cache stats with caching."""
        tracker = CostTracker()

        # Track with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="agent",
            cache_creation_tokens=800,
            cache_read_tokens=200
        )

        # Track another with cache read
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=300,
            output_tokens=400,
            operation="agent",
            cache_read_tokens=5000
        )

        stats = tracker.get_cache_stats()
        assert stats["cache_read_tokens"] == 5200  # 200 + 5000
        assert stats["cache_creation_tokens"] == 800

    def test_get_session_cost_summary_basic(self):
        """Test session cost summary without caching (Variant C format)."""
        tracker = CostTracker()

        # Track some usage
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="agent"
        )

        summary = tracker.get_session_cost_summary()

        # Should contain per-message and session total (Variant C)
        assert "This message:" in summary
        assert "Session total:" in summary
        assert "$" in summary
        assert "tokens" in summary
        assert "1,500 tokens" in summary  # 1000 + 500

        # Should not mention cache if not used
        assert "Cache:" not in summary

    def test_get_session_cost_summary_with_cache(self):
        """Test session cost summary with caching (Variant C format)."""
        tracker = CostTracker()

        # Track with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="agent",
            cache_read_tokens=5000
        )

        summary = tracker.get_session_cost_summary()

        # Should contain per-message and session total (Variant C)
        assert "This message:" in summary
        assert "Session total:" in summary
        assert "$" in summary
        assert "Cache:" in summary
        assert "5,000 tokens read" in summary
        assert "90% saved" in summary
        # Total tokens should include cache: 1000 + 500 + 5000 = 6,500
        assert "6,500 tokens" in summary

    def test_get_session_cost_summary_multiple_calls(self):
        """Test session cost summary accumulates correctly (Variant C format)."""
        tracker = CostTracker()

        # Multiple calls
        for i in range(3):
            tracker.track_llm(
                provider="anthropic",
                model="claude-haiku-4-5",
                input_tokens=1000,
                output_tokens=500,
                operation="agent",
                cache_read_tokens=2000 if i > 0 else 0,  # Cache hit after first call
                cache_creation_tokens=500 if i == 0 else 0  # Cache created on first call
            )

        summary = tracker.get_session_cost_summary()

        # Total tokens now includes cache: 3*(1000+500) + 2*2000 = 4,500 + 4,000 = 8,500
        assert "8,500 tokens" in summary

        # Cache: 0 + 2000 + 2000 = 4,000 read
        assert "4,000 tokens read" in summary

    def test_session_cost_summary_format(self):
        """Test that summary has expected format."""
        tracker = CostTracker()

        tracker.track_llm(
            provider="anthropic",
            model="claude-sonnet-4-5",
            input_tokens=2000,
            output_tokens=1000,
            operation="agent",
            cache_read_tokens=10000
        )

        summary = tracker.get_session_cost_summary()

        # Check for emojis and formatting
        assert "💰" in summary
        assert "📦" in summary
        assert "|" in summary  # Separator
        assert "(" in summary and ")" in summary  # Token count in parens
