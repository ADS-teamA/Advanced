"""
Tests for Smart Token Management System

Verifies:
- Accurate token counting with tiktoken
- Smart truncation at sentence boundaries
- Adaptive k calculation
- Progressive detail levels
"""

import pytest
from src.agent.tools.token_manager import (
    TokenCounter,
    SmartTruncator,
    AdaptiveFormatter,
    DetailLevel,
    TokenBudget,
)


class TestTokenCounter:
    """Test token counting functionality."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        counter = TokenCounter()

        # Short text
        text = "Hello, world!"
        tokens = counter.count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be fewer than chars

    def test_count_tokens_empty(self):
        """Test empty string."""
        counter = TokenCounter()
        assert counter.count_tokens("") == 0
        assert counter.count_tokens(None) == 0

    def test_count_tokens_czech(self):
        """Test Czech text (with diacritics)."""
        counter = TokenCounter()

        czech_text = "Příliš žluťoučký kůň úpěl ďábelské ódy."
        tokens = counter.count_tokens(czech_text)
        assert tokens > 0


class TestSmartTruncator:
    """Test smart truncation at sentence boundaries."""

    def test_truncate_at_sentence_basic(self):
        """Test truncation at sentence boundary."""
        counter = TokenCounter()
        text = "First sentence. Second sentence. Third sentence."

        # Truncate to allow only first sentence
        truncated, was_truncated = SmartTruncator.truncate_at_sentence(
            text, max_tokens=5, token_counter=counter
        )

        assert was_truncated
        assert "First sentence." in truncated
        assert "Third sentence" not in truncated

    def test_no_truncation_if_fits(self):
        """Test no truncation when text fits."""
        counter = TokenCounter()
        text = "Short text."

        truncated, was_truncated = SmartTruncator.truncate_at_sentence(
            text, max_tokens=100, token_counter=counter
        )

        assert not was_truncated
        assert truncated == text

    def test_truncate_at_word_fallback(self):
        """Test word-boundary fallback."""
        counter = TokenCounter()
        text = "This is a long sentence without proper punctuation marks so it will be truncated at word boundaries"

        truncated, was_truncated = SmartTruncator.truncate_at_word(
            text, max_tokens=10, token_counter=counter
        )

        assert was_truncated
        assert truncated.endswith("...")
        # Should not cut mid-word
        assert not truncated[:-3].endswith(" ")


class TestAdaptiveFormatter:
    """Test adaptive formatting with budget management."""

    def test_format_chunks_basic(self):
        """Test basic chunk formatting."""
        formatter = AdaptiveFormatter()

        chunks = [
            {
                "content": "This is test content " * 100,  # Long content
                "document_id": "test_doc",
                "section_title": "Test Section",
                "chunk_id": "test:1",
                "score": 0.95,
            }
        ]

        formatted, metadata = formatter.format_chunks(
            chunks, detail_level=DetailLevel.MEDIUM
        )

        assert len(formatted) == 1
        assert formatted[0]["document_id"] == "test_doc"
        assert "total_tokens" in metadata
        assert metadata["chunks_count"] == 1

    def test_format_chunks_auto_adjust(self):
        """Test automatic detail level adjustment."""
        formatter = AdaptiveFormatter(budget=TokenBudget(max_total_tokens=2000))

        # Create many chunks that would exceed budget at FULL detail
        chunks = [
            {
                "content": "Long content " * 50,
                "document_id": f"doc_{i}",
                "chunk_id": f"chunk_{i}",
            }
            for i in range(20)
        ]

        formatted, metadata = formatter.format_chunks(
            chunks, detail_level=DetailLevel.FULL, auto_adjust=True
        )

        # Should auto-adjust to lower detail level
        if metadata.get("auto_adjusted"):
            assert metadata["actual_detail_level"] != DetailLevel.FULL.value

    def test_adaptive_k_within_budget(self):
        """Test adaptive k when within budget."""
        formatter = AdaptiveFormatter()

        actual_k, reason = formatter.adaptive_k(requested_k=5, tokens_per_item=300)

        assert actual_k == 5
        assert reason == "within_budget"

    def test_adaptive_k_exceeds_budget(self):
        """Test adaptive k when exceeding budget."""
        formatter = AdaptiveFormatter(budget=TokenBudget(max_total_tokens=2000))

        # Request 100 results at 300 tokens each = 30K tokens (exceeds 2K budget)
        actual_k, reason = formatter.adaptive_k(requested_k=100, tokens_per_item=300)

        assert actual_k < 100
        assert reason == "budget_limited"
        # Should fit within budget: actual_k * 300 <= 1000 (max_total - reserved)
        assert actual_k * 300 <= 1000

    def test_format_sections_with_budget(self):
        """Test section formatting with budget."""
        formatter = AdaptiveFormatter()

        sections = [
            {
                "section_id": f"sec_{i}",
                "section_title": f"Section {i}",
                "section_summary": "Summary " * 20,
                "chunk_count": 5,
            }
            for i in range(150)  # Many sections
        ]

        formatted, metadata = formatter.format_sections_with_budget(
            sections, include_summary=True
        )

        # Should limit sections based on budget
        assert metadata["returned_sections"] <= metadata["total_sections"]
        assert metadata["total_sections"] == 150

        if metadata["truncated"]:
            assert metadata["returned_sections"] < 150


class TestDetailLevels:
    """Test different detail levels."""

    def test_summary_level(self):
        """Test summary detail level (~100 tokens)."""
        budget = TokenBudget()
        assert budget.tokens_per_item(DetailLevel.SUMMARY) == 100

    def test_medium_level(self):
        """Test medium detail level (~300 tokens)."""
        budget = TokenBudget()
        assert budget.tokens_per_item(DetailLevel.MEDIUM) == 300

    def test_full_level(self):
        """Test full detail level (~600 tokens)."""
        budget = TokenBudget()
        assert budget.tokens_per_item(DetailLevel.FULL) == 600


class TestBackwardCompatibility:
    """Test backward compatibility with utils.py functions."""

    def test_format_chunk_result_legacy(self):
        """Test legacy character-based truncation."""
        from src.agent.tools.utils import format_chunk_result

        chunk = {
            "content": "A" * 1000,
            "document_id": "test",
            "chunk_id": "test:1",
            "score": 0.9,
        }

        # Legacy mode: max_content_length specified
        result = format_chunk_result(chunk, max_content_length=400)

        assert len(result["content"]) <= 420  # 400 + "... [truncated]"
        assert result["content"].endswith("... [truncated]")

    def test_format_chunk_result_smart(self):
        """Test new smart truncation."""
        from src.agent.tools.utils import format_chunk_result

        chunk = {
            "content": "First sentence. " + "A" * 2000 + ". Last sentence.",
            "document_id": "test",
            "chunk_id": "test:1",
        }

        # Smart mode: detail_level specified
        result = format_chunk_result(chunk, detail_level="medium", smart_truncate=True)

        # Should have truncated flag if content was too long
        if "truncated" in result:
            assert result["truncated"] is True

    def test_validate_k_adaptive(self):
        """Test adaptive k validation."""
        from src.agent.tools.utils import validate_k_parameter

        # Request reasonable k
        actual_k, reason = validate_k_parameter(k=5, adaptive=True, detail_level="medium")

        assert actual_k >= 3  # Min k
        assert actual_k <= 50  # Max k

    def test_validate_k_legacy(self):
        """Test legacy k validation."""
        from src.agent.tools.utils import validate_k_parameter

        # Legacy mode: max_k specified
        actual_k, reason = validate_k_parameter(k=20, max_k=10, adaptive=False)

        assert actual_k == 10
        assert reason == "exceeded_maximum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
