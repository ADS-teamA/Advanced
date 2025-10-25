"""
Test cost tracking accuracy with caching and tool results.

Verifies:
1. Token counting includes cache reads
2. Per-message vs session totals are accurate
3. Tool result tokens are counted correctly (via API usage, not estimates)
4. Cache statistics are accurate
"""

import pytest
from src.cost_tracker import CostTracker, PRICING


class TestCostTrackingAccuracy:
    """Test cost tracking accuracy."""

    def test_total_tokens_includes_cache_reads(self):
        """Verify get_total_tokens() includes cache reads."""
        tracker = CostTracker()

        # Track LLM usage with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="test",
            cache_read_tokens=2000  # Cache hit
        )

        # Total tokens should include cache reads
        assert tracker.get_total_tokens() == 3500  # 1000 + 500 + 2000

        # Billed tokens should discount cache (10% of cache)
        assert tracker.get_billed_tokens() == 1700  # 1000 + 500 + (2000 * 0.1)

    def test_billed_tokens_discounts_cache(self):
        """Verify billed tokens apply 90% cache discount."""
        tracker = CostTracker()

        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=500,
            output_tokens=200,
            cache_read_tokens=10000
        )

        # Billed equivalent: 500 + 200 + (10000 * 0.1) = 1700
        assert tracker.get_billed_tokens() == 1700

    def test_per_message_tracking(self):
        """Verify per-message cost tracking for CLI display."""
        tracker = CostTracker()

        # First message
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500
        )

        summary1 = tracker.get_session_cost_summary()

        # Should show first message = session total
        assert "This message:" in summary1
        assert "Session total:" in summary1

        # Second message
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=800,
            output_tokens=400
        )

        summary2 = tracker.get_session_cost_summary()

        # Should show different per-message vs session totals
        # Per-message should be ~1200 tokens (800 + 400)
        # Session should be ~2900 tokens (1500 + 1200)
        assert "This message:" in summary2
        assert "Session total:" in summary2

    def test_cache_statistics(self):
        """Verify cache statistics are accurate."""
        tracker = CostTracker()

        # Track multiple calls with caching
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=5000,  # First call creates cache
            cache_read_tokens=0
        )

        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=800,
            output_tokens=400,
            cache_creation_tokens=0,
            cache_read_tokens=5000  # Second call hits cache
        )

        cache_stats = tracker.get_cache_stats()
        assert cache_stats["cache_creation_tokens"] == 5000
        assert cache_stats["cache_read_tokens"] == 5000

    def test_cost_calculation_with_cache(self):
        """Verify cost calculation includes cache discount."""
        tracker = CostTracker()

        # Track with cache (Haiku: $1.00 input, $5.00 output per 1M)
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=10000,
            output_tokens=5000,
            cache_read_tokens=20000
        )

        # Cost calculation:
        # Input: 10000 * $1.00 / 1M = $0.01
        # Output: 5000 * $5.00 / 1M = $0.025
        # Cache: 20000 * $1.00 * 0.1 / 1M = $0.002
        # Total: $0.037

        expected_cost = (
            (10000 / 1_000_000) * 1.00 +  # Input
            (5000 / 1_000_000) * 5.00 +   # Output
            (20000 / 1_000_000) * 1.00 * 0.1  # Cache (10% of input price)
        )

        assert abs(tracker.get_total_cost() - expected_cost) < 0.0001

    def test_tool_results_not_double_counted(self):
        """
        Verify tool results are not double-counted.

        Tool estimates are informational only - actual tokens are counted
        when tool results are sent back to API in next message.
        """
        tracker = CostTracker()

        # Simulate first API call (user question)
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500
        )

        # Tool results are NOT tracked separately - they're just logged
        # (tool_estimated_tokens = 2000 for example, but NOT added to tracker)

        # Simulate second API call (includes tool results in input)
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=3000,  # This INCLUDES tool result tokens from API
            output_tokens=800
        )

        # Total should be: 1000 + 500 + 3000 + 800 = 5300
        # (tool results are already in the 3000 input tokens)
        assert tracker.get_total_tokens() == 5300

    def test_session_cost_summary_format(self):
        """Verify session cost summary has correct format (Variant C)."""
        tracker = CostTracker()

        # First message
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500
        )

        summary = tracker.get_session_cost_summary()

        # Should contain both per-message and session
        assert "This message:" in summary
        assert "Session total:" in summary
        assert "tokens)" in summary

        # Should contain $ amounts
        assert "$" in summary

    def test_cache_summary_display(self):
        """Verify cache info is displayed when caching is used."""
        tracker = CostTracker()

        # Track with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=5000
        )

        summary = tracker.get_session_cost_summary()

        # Should display cache info
        assert "📦 Cache:" in summary
        assert "5,000" in summary
        assert "90% saved" in summary

    def test_no_cache_no_display(self):
        """Verify cache info is not displayed when no caching."""
        tracker = CostTracker()

        # Track without cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500
        )

        summary = tracker.get_session_cost_summary()

        # Should NOT display cache info
        assert "📦 Cache:" not in summary

    def test_reset_clears_cache_tracking(self):
        """Verify reset clears cache tracking."""
        tracker = CostTracker()

        # Track with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=5000
        )

        # Reset
        tracker.reset()

        # Cache stats should be zero
        cache_stats = tracker.get_cache_stats()
        assert cache_stats["cache_creation_tokens"] == 0
        assert cache_stats["cache_read_tokens"] == 0
        assert tracker.get_total_tokens() == 0

    def test_multiple_providers_mixed(self):
        """Test tracking with multiple providers and caching."""
        tracker = CostTracker()

        # Anthropic with cache
        tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=2000
        )

        # OpenAI (no caching)
        tracker.track_embedding(
            provider="openai",
            model="text-embedding-3-large",
            tokens=5000
        )

        # Totals
        assert tracker.get_total_tokens() == 8500  # 1000 + 500 + 2000 + 5000
        assert tracker.total_input_tokens == 6000  # 1000 + 5000
        assert tracker.total_output_tokens == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
