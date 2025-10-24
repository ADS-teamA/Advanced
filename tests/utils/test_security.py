"""
Unit tests for utils.security module.

CRITICAL: These tests verify API key sanitization to prevent leaks.
"""

import pytest
from src.utils.security import sanitize_error, mask_api_key


class TestSanitizeError:
    """Test sanitize_error function."""

    def test_anthropic_key_masked(self):
        """Test that Anthropic API keys are masked."""
        error = "Error: sk-ant-api03-1234567890abcdef-1234567890"
        result = sanitize_error(error)
        assert "sk-ant-***" in result
        assert "1234567890" not in result

    def test_openai_key_masked(self):
        """Test that OpenAI API keys are masked."""
        error = "Error: sk-1234567890abcdefghijklmnopqrst"
        result = sanitize_error(error)
        assert "sk-***" in result
        assert "1234567890abcdef" not in result

    def test_voyage_key_masked(self):
        """Test that Voyage AI API keys are masked."""
        error = "Error: pa-1234567890abcdefghijklmnopqrst"
        result = sanitize_error(error)
        assert "pa-***" in result
        assert "1234567890abcdef" not in result

    def test_generic_api_key_masked(self):
        """Test that generic api_key= patterns are masked."""
        error = "Error: api_key=secret123"
        result = sanitize_error(error)
        assert "api_key=***" in result
        assert "secret123" not in result

    def test_bearer_token_masked(self):
        """Test that Bearer tokens are masked."""
        error = "Authorization: Bearer secret_token_123"
        result = sanitize_error(error)
        assert "Bearer ***" in result
        assert "secret_token_123" not in result

    def test_multiple_keys_masked(self):
        """Test that multiple API keys in same error are all masked."""
        error = "Error: sk-ant-api03-key1 and sk-key2"
        result = sanitize_error(error)
        assert "sk-ant-***" in result
        assert "sk-***" in result
        assert "key1" not in result
        assert "key2" not in result

    def test_exception_sanitized(self):
        """Test that Exception objects are sanitized."""
        error = Exception("Failed with key sk-1234567890")
        result = sanitize_error(error)
        assert "sk-***" in result
        assert "1234567890" not in result

    def test_no_api_key_unchanged(self):
        """Test that errors without API keys are unchanged."""
        error = "File not found: document.pdf"
        result = sanitize_error(error)
        assert result == error

    def test_empty_string(self):
        """Test empty string handling."""
        result = sanitize_error("")
        assert result == ""

    def test_none_raises_error(self):
        """Test that None raises TypeError."""
        with pytest.raises((TypeError, AttributeError)):
            sanitize_error(None)


class TestMaskApiKey:
    """Test mask_api_key function."""

    def test_anthropic_key_detection(self):
        """Test Anthropic key pattern detection."""
        key = "sk-ant-api03-1234567890abcdef"
        result = mask_api_key(key)
        assert result == "sk-ant-***"

    def test_openai_key_detection(self):
        """Test OpenAI key pattern detection."""
        key = "sk-1234567890abcdefghijklmnopqrst"
        result = mask_api_key(key)
        assert result == "sk-***"

    def test_voyage_key_detection(self):
        """Test Voyage AI key pattern detection."""
        key = "pa-1234567890abcdefghijklmnopqrst"
        result = mask_api_key(key)
        assert "pa-***" in result

    def test_non_api_key_unchanged(self):
        """Test non-API key strings are unchanged."""
        text = "regular text without keys"
        result = mask_api_key(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string handling."""
        result = mask_api_key("")
        assert result == ""


# Integration tests
class TestSecurityIntegration:
    """Integration tests for security module."""

    def test_real_world_error_message(self):
        """Test sanitization of realistic error message."""
        error_msg = """
        RateLimitError: Rate limit exceeded for sk-ant-api03-secret123.
        Please check your API key (sk-ant-api03-secret123) and try again.
        Authorization header: Bearer sk-ant-api03-secret123
        """
        result = sanitize_error(error_msg)

        # Verify all keys are masked
        assert "sk-ant-***" in result
        assert "secret123" not in result

        # Verify message structure is preserved
        assert "RateLimitError" in result
        assert "Rate limit exceeded" in result
        assert "Please check your API key" in result

    def test_openai_rate_limit_error(self):
        """Test OpenAI rate limit error sanitization."""
        error = "429 Rate Limit Error: sk-1234567890abcdefghij with api_key=sk-1234567890abcdefghij"
        result = sanitize_error(error)

        assert "sk-***" in result
        assert "1234567890abcdefghij" not in result
        assert "429 Rate Limit Error" in result

    def test_mixed_keys_in_traceback(self):
        """Test sanitization of error with multiple key types."""
        error = """
        Traceback (most recent call last):
          File "main.py", line 10, in <module>
            client = Anthropic(api_key="sk-ant-api03-key1")
          File "openai.py", line 5, in init
            self.key = "sk-key2"
        ValueError: Invalid key
        """
        result = sanitize_error(error)

        # Both keys should be masked
        assert "sk-ant-***" in result or "sk-***" in result
        assert "key1" not in result
        assert "key2" not in result

        # Traceback structure preserved
        assert "Traceback" in result
        assert "ValueError" in result
