"""
Comprehensive unit tests for contextual_retrieval.py

Tests cover:
- Initialization with different providers
- Context generation
- Batch processing
- Error handling and fallbacks
- Rate limiting
- API key sanitization
- XML escaping
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextual_retrieval import ContextualRetrieval, ChunkContext
from config import ContextGenerationConfig


class TestContextualRetrievalInit:
    """Test initialization with different providers."""

    def test_init_anthropic_with_api_key(self):
        """Test initialization with Anthropic provider and API key."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic") as mock_anthropic:
                retrieval = ContextualRetrieval(config=config)
                assert retrieval.provider_type == "anthropic"
                mock_anthropic.assert_called_once()

    def test_init_openai_with_api_key(self):
        """Test initialization with OpenAI provider and API key."""
        config = ContextGenerationConfig(provider="openai", model="gpt-4o-mini")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("contextual_retrieval.OpenAI") as mock_openai:
                retrieval = ContextualRetrieval(config=config)
                assert retrieval.provider_type == "openai"
                mock_openai.assert_called_once()

    def test_init_anthropic_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key required"):
                ContextualRetrieval(config=config)

    def test_init_openai_missing_api_key(self):
        """Test that missing OpenAI API key raises ValueError."""
        config = ContextGenerationConfig(provider="openai", model="gpt-4o-mini")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                ContextualRetrieval(config=config)

    def test_init_unsupported_provider(self):
        """Test that unsupported provider raises ValueError."""
        config = ContextGenerationConfig(provider="invalid", model="test")

        with pytest.raises(ValueError, match="Unsupported provider"):
            ContextualRetrieval(config=config)


class TestContextGeneration:
    """Test context generation functionality."""

    @pytest.fixture
    def mock_anthropic_retrieval(self):
        """Create ContextualRetrieval with mocked Anthropic client."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic") as mock_anthropic:
                retrieval = ContextualRetrieval(config=config)

                # Mock the response
                mock_response = Mock()
                mock_response.content = [Mock(text="This chunk discusses safety parameters.")]
                retrieval.client.messages.create = Mock(return_value=mock_response)

                yield retrieval

    def test_generate_context_success(self, mock_anthropic_retrieval):
        """Test successful context generation."""
        chunk = "The primary cooling circuit operates at 15.7 MPa."

        context = mock_anthropic_retrieval.generate_context(
            chunk=chunk,
            document_summary="Nuclear reactor safety specification",
            section_title="Pressure Limits"
        )

        assert context == "This chunk discusses safety parameters."
        assert mock_anthropic_retrieval.client.messages.create.called

    def test_generate_context_with_surrounding_chunks(self, mock_anthropic_retrieval):
        """Test context generation with surrounding chunks."""
        chunk = "Middle chunk"
        preceding = "Previous chunk"
        following = "Next chunk"

        context = mock_anthropic_retrieval.generate_context(
            chunk=chunk,
            preceding_chunk=preceding,
            following_chunk=following
        )

        assert context == "This chunk discusses safety parameters."

        # Verify prompt includes surrounding chunks
        call_args = mock_anthropic_retrieval.client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "preceding_chunk" in prompt
        assert "following_chunk" in prompt


class TestXMLEscaping:
    """Test XML tag escaping for prompt injection prevention."""

    @pytest.fixture
    def mock_retrieval(self):
        """Create retrieval instance for testing."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic"):
                retrieval = ContextualRetrieval(config=config)
                yield retrieval

    def test_escape_xml_tags(self, mock_retrieval):
        """Test that XML tags are properly escaped."""
        text_with_tags = "This </chunk> is malicious </document>"
        escaped = mock_retrieval._escape_xml_tags(text_with_tags)

        assert "&lt;/chunk&gt;" in escaped
        assert "&lt;/document&gt;" in escaped
        assert "</chunk>" not in escaped

    def test_no_escape_for_safe_text(self, mock_retrieval):
        """Test that safe text is not escaped."""
        safe_text = "This is normal text with <emphasis> tags"
        escaped = mock_retrieval._escape_xml_tags(safe_text)

        # Should not escape since it doesn't contain problematic tags
        assert escaped == safe_text


class TestAPIKeySanitization:
    """Test API key sanitization in error messages."""

    @pytest.fixture
    def mock_retrieval(self):
        """Create retrieval instance for testing."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic"):
                retrieval = ContextualRetrieval(config=config)
                yield retrieval

    def test_sanitize_anthropic_key(self, mock_retrieval):
        """Test sanitization of Anthropic API keys."""
        error_msg = "Error with key sk-ant-1234567890abcdefghijklmnopqrstuvwxyz"
        sanitized = mock_retrieval._sanitize_error(error_msg)

        assert "sk-ant-***" in sanitized
        assert "1234567890abcdefghijklmnopqrstuvwxyz" not in sanitized

    def test_sanitize_openai_key(self, mock_retrieval):
        """Test sanitization of OpenAI API keys."""
        error_msg = "Error with key sk-1234567890abcdefghijklmnopqrstuvwxyz"
        sanitized = mock_retrieval._sanitize_error(error_msg)

        assert "sk-***" in sanitized
        assert "1234567890abcdefghijklmnopqrstuvwxyz" not in sanitized

    def test_sanitize_bearer_token(self, mock_retrieval):
        """Test sanitization of Bearer tokens."""
        error_msg = "Authorization: Bearer abc123def456ghi789jkl012"
        sanitized = mock_retrieval._sanitize_error(error_msg)

        assert "Bearer ***" in sanitized
        assert "abc123def456ghi789jkl012" not in sanitized


class TestRateLimiting:
    """Test rate limiting and retry logic."""

    @pytest.fixture
    def mock_anthropic_retrieval(self):
        """Create retrieval with mocked Anthropic client."""
        config = ContextGenerationConfig(provider="anthropic", model="haiku")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic") as mock_anthropic:
                retrieval = ContextualRetrieval(config=config)
                yield retrieval

    def test_rate_limit_retry_success(self, mock_anthropic_retrieval):
        """Test that rate limit errors trigger retry and eventually succeed."""
        # First call raises rate limit, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text="Success after retry")]

        rate_limit_error = Exception("Rate limit exceeded (429)")
        mock_anthropic_retrieval.client.messages.create = Mock(
            side_effect=[rate_limit_error, mock_response]
        )

        with patch("time.sleep"):  # Don't actually sleep in tests
            prompt = "Test prompt"
            result = mock_anthropic_retrieval._generate_with_anthropic(prompt)

        assert result == "Success after retry"
        assert mock_anthropic_retrieval.client.messages.create.call_count == 2

    def test_rate_limit_max_retries_exceeded(self, mock_anthropic_retrieval):
        """Test that max retries are respected."""
        rate_limit_error = Exception("Rate limit exceeded (429)")
        mock_anthropic_retrieval.client.messages.create = Mock(
            side_effect=rate_limit_error
        )

        with patch("time.sleep"):  # Don't actually sleep in tests
            with pytest.raises(Exception, match="Rate limit exceeded"):
                prompt = "Test prompt"
                mock_anthropic_retrieval._generate_with_anthropic(prompt)

        # Should try 3 times (initial + 2 retries)
        assert mock_anthropic_retrieval.client.messages.create.call_count == 3


class TestBatchProcessing:
    """Test batch context generation."""

    @pytest.fixture
    def mock_anthropic_retrieval(self):
        """Create retrieval with mocked Anthropic client."""
        config = ContextGenerationConfig(
            provider="anthropic",
            model="haiku",
            batch_size=2,
            max_workers=2
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic") as mock_anthropic:
                retrieval = ContextualRetrieval(config=config)

                # Mock successful responses
                mock_response = Mock()
                mock_response.content = [Mock(text="Generated context")]
                retrieval.client.messages.create = Mock(return_value=mock_response)

                yield retrieval

    def test_batch_processing_preserves_order(self, mock_anthropic_retrieval):
        """Test that batch processing preserves chunk order."""
        chunks = [
            ("Chunk 1", {"document_summary": "Doc summary"}),
            ("Chunk 2", {"document_summary": "Doc summary"}),
            ("Chunk 3", {"document_summary": "Doc summary"}),
            ("Chunk 4", {"document_summary": "Doc summary"}),
        ]

        results = mock_anthropic_retrieval.generate_contexts_batch(chunks)

        # Verify all chunks processed
        assert len(results) == 4

        # Verify all successful
        assert all(r.success for r in results)

        # Verify order preserved
        for i, (chunk_text, _) in enumerate(chunks):
            assert results[i].chunk_text == chunk_text

    def test_batch_processing_handles_failures(self, mock_anthropic_retrieval):
        """Test that batch processing handles individual failures gracefully."""
        # Make some calls fail
        def side_effect(*args, **kwargs):
            if mock_anthropic_retrieval.client.messages.create.call_count % 2 == 0:
                raise Exception("API error")
            mock_response = Mock()
            mock_response.content = [Mock(text="Success")]
            return mock_response

        mock_anthropic_retrieval.client.messages.create = Mock(side_effect=side_effect)

        chunks = [
            ("Chunk 1", {"document_summary": "Doc"}),
            ("Chunk 2", {"document_summary": "Doc"}),
            ("Chunk 3", {"document_summary": "Doc"}),
        ]

        results = mock_anthropic_retrieval.generate_contexts_batch(chunks)

        # Should have results for all chunks
        assert len(results) == 3

        # Some should succeed, some fail
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        assert len(successes) > 0
        assert len(failures) > 0


class TestPromptConstruction:
    """Test prompt construction with various inputs."""

    @pytest.fixture
    def mock_retrieval(self):
        """Create retrieval instance for testing."""
        config = ContextGenerationConfig(
            provider="anthropic",
            model="haiku",
            include_surrounding_chunks=True
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic"):
                retrieval = ContextualRetrieval(config=config)
                yield retrieval

    def test_prompt_with_all_metadata(self, mock_retrieval):
        """Test prompt construction with full metadata."""
        prompt = mock_retrieval._build_context_prompt(
            chunk="Test chunk",
            document_summary="Doc summary",
            section_title="Section Title",
            section_path="Ch1 > Sec1.1",
            preceding_chunk="Previous",
            following_chunk="Next"
        )

        assert "Doc summary" in prompt
        assert "Section hierarchy: Ch1 > Sec1.1" in prompt
        assert "Test chunk" in prompt
        assert "preceding_chunk" in prompt
        assert "following_chunk" in prompt

    def test_prompt_minimal_metadata(self, mock_retrieval):
        """Test prompt construction with minimal metadata."""
        prompt = mock_retrieval._build_context_prompt(
            chunk="Test chunk",
            document_summary=None,
            section_title=None,
            section_path=None
        )

        assert "Test chunk" in prompt
        assert "No additional context available" in prompt

    def test_prompt_without_surrounding_chunks(self):
        """Test prompt when surrounding chunks are disabled."""
        config = ContextGenerationConfig(
            provider="anthropic",
            model="haiku",
            include_surrounding_chunks=False
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("contextual_retrieval.Anthropic"):
                retrieval = ContextualRetrieval(config=config)

                prompt = retrieval._build_context_prompt(
                    chunk="Test chunk",
                    preceding_chunk="Should not appear",
                    following_chunk="Should not appear"
                )

                assert "preceding_chunk" not in prompt
                assert "following_chunk" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
