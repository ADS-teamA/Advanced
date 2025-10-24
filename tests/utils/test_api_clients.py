"""
Unit tests for utils.api_clients module.

Tests API client factory with mocking to avoid real API calls.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock


# Test class for APIClientFactory
class TestAPIClientFactory:
    """Test APIClientFactory class."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"})
    @patch("src.utils.api_clients.anthropic")
    def test_create_anthropic_with_env_key(self, mock_anthropic):
        """Test creating Anthropic client with environment variable."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Create client
        client = APIClientFactory.create_anthropic()

        # Verify
        assert client == mock_client
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant-test123")

    @patch("src.utils.api_clients.anthropic")
    def test_create_anthropic_with_parameter_key(self, mock_anthropic):
        """Test creating Anthropic client with parameter."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Create client with explicit key
        client = APIClientFactory.create_anthropic(api_key="sk-ant-explicit123")

        # Verify
        assert client == mock_client
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant-explicit123")

    @patch.dict(os.environ, {}, clear=True)
    def test_create_anthropic_missing_key(self):
        """Test creating Anthropic client without API key raises ValueError."""
        from src.utils.api_clients import APIClientFactory

        with pytest.raises(ValueError, match="Anthropic API key required"):
            APIClientFactory.create_anthropic()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "invalid-key"})
    @patch("src.utils.api_clients.anthropic")
    def test_create_anthropic_invalid_format_warning(self, mock_anthropic, caplog):
        """Test warning for invalid Anthropic API key format."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Create client with invalid format
        with caplog.at_level("WARNING"):
            APIClientFactory.create_anthropic()

        # Verify warning was logged
        assert "unexpected format" in caplog.text.lower()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    @patch("src.utils.api_clients.openai")
    def test_create_openai_with_env_key(self, mock_openai):
        """Test creating OpenAI client with environment variable."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Create client
        client = APIClientFactory.create_openai()

        # Verify
        assert client == mock_client
        mock_openai.OpenAI.assert_called_once_with(api_key="sk-test123")

    @patch("src.utils.api_clients.openai")
    def test_create_openai_with_parameter_key(self, mock_openai):
        """Test creating OpenAI client with parameter."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client

        # Create client with explicit key
        client = APIClientFactory.create_openai(api_key="sk-explicit123")

        # Verify
        assert client == mock_client
        mock_openai.OpenAI.assert_called_once_with(api_key="sk-explicit123")

    @patch.dict(os.environ, {}, clear=True)
    def test_create_openai_missing_key(self):
        """Test creating OpenAI client without API key raises ValueError."""
        from src.utils.api_clients import APIClientFactory

        with pytest.raises(ValueError, match="OpenAI API key required"):
            APIClientFactory.create_openai()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    @patch("src.utils.api_clients.openai")
    def test_create_openai_connection_validation_success(self, mock_openai):
        """Test OpenAI client connection validation success."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client and models.list()
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_openai.OpenAI.return_value = mock_client

        # Create client with validation
        client = APIClientFactory.create_openai(validate_connection=True)

        # Verify
        assert client == mock_client
        mock_client.models.list.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    @patch("src.utils.api_clients.openai")
    def test_create_openai_connection_validation_failure(self, mock_openai):
        """Test OpenAI client connection validation failure."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client to fail validation
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Invalid API key")
        mock_openai.OpenAI.return_value = mock_client

        # Verify validation failure raises RuntimeError
        with pytest.raises(RuntimeError, match="OpenAI API key validation failed"):
            APIClientFactory.create_openai(validate_connection=True)

    @patch.dict(os.environ, {"VOYAGE_API_KEY": "pa-test123"})
    @patch("src.utils.api_clients.voyageai")
    def test_create_voyage_with_env_key(self, mock_voyageai):
        """Test creating Voyage AI client with environment variable."""
        from src.utils.api_clients import APIClientFactory

        # Mock the client
        mock_client = Mock()
        mock_voyageai.Client.return_value = mock_client

        # Create client
        client = APIClientFactory.create_voyage()

        # Verify
        assert client == mock_client
        mock_voyageai.Client.assert_called_once_with(api_key="pa-test123")

    @patch.dict(os.environ, {}, clear=True)
    def test_create_voyage_missing_key(self):
        """Test creating Voyage AI client without API key raises ValueError."""
        from src.utils.api_clients import APIClientFactory

        with pytest.raises(ValueError, match="Voyage AI API key required"):
            APIClientFactory.create_voyage()

    def test_import_error_anthropic(self):
        """Test ImportError when anthropic package not installed."""
        from src.utils.api_clients import APIClientFactory

        # Mock import to fail
        with patch("src.utils.api_clients.anthropic", side_effect=ImportError):
            with pytest.raises(ImportError, match="anthropic package required"):
                # Force re-import by calling the method
                import src.utils.api_clients
                import importlib
                importlib.reload(src.utils.api_clients)
                src.utils.api_clients.APIClientFactory.create_anthropic(api_key="test")

    def test_import_error_openai(self):
        """Test ImportError when openai package not installed."""
        from src.utils.api_clients import APIClientFactory

        # Mock import to fail
        with patch("src.utils.api_clients.openai", side_effect=ImportError):
            with pytest.raises(ImportError, match="openai package required"):
                # Force re-import by calling the method
                import src.utils.api_clients
                import importlib
                importlib.reload(src.utils.api_clients)
                src.utils.api_clients.APIClientFactory.create_openai(api_key="test")


# Integration tests with real imports (no API calls)
class TestAPIClientFactoryIntegration:
    """Integration tests for APIClientFactory."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"})
    def test_mask_api_key_in_logs(self, caplog):
        """Test that API keys are masked in log messages."""
        from src.utils.api_clients import APIClientFactory

        # Mock anthropic to avoid actual client creation
        with patch("src.utils.api_clients.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = Mock()

            with caplog.at_level("INFO"):
                APIClientFactory.create_anthropic()

            # Verify key is masked in logs
            assert "sk-ant-***" in caplog.text
            assert "test123" not in caplog.text

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-secret123"})
    def test_sanitize_error_in_exceptions(self):
        """Test that errors are sanitized before raising."""
        from src.utils.api_clients import APIClientFactory

        # Mock anthropic to raise error with API key
        with patch("src.utils.api_clients.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.side_effect = Exception("Error: sk-ant-secret123 is invalid")

            with pytest.raises(RuntimeError) as exc_info:
                APIClientFactory.create_anthropic()

            # Verify API key is masked in error
            error_msg = str(exc_info.value)
            assert "sk-ant-***" in error_msg
            assert "secret123" not in error_msg
