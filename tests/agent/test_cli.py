"""
Tests for CLI initialization and error handling.

Tests comprehensive CLI behavior:
- Initialization errors (vector store missing, corrupt, etc.)
- Graceful degradation (reranker fails, KG missing)
- Platform-specific model selection
- Error message quality
- Component initialization sequence
- Session statistics
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import platform

from src.agent.cli import AgentCLI
from src.agent.config import AgentConfig, CLIConfig, ToolConfig


@pytest.fixture
def tmp_vector_store(tmp_path):
    """Create a temporary vector store directory with required files."""
    store_path = tmp_path / "vector_store"
    store_path.mkdir()

    # Create minimal required files (FAISS index files)
    (store_path / "layer1.index").touch()
    (store_path / "layer2.index").touch()
    (store_path / "layer3.index").touch()

    # Create metadata files
    (store_path / "metadata.json").write_text('{"documents": [], "chunks": []}')

    return store_path


@pytest.fixture
def valid_config(tmp_vector_store):
    """Create a valid AgentConfig for testing."""
    return AgentConfig(
        anthropic_api_key="sk-ant-test123456789",
        vector_store_path=tmp_vector_store,
        model="claude-sonnet-4-5-20250929",
        debug_mode=False,
        enable_knowledge_graph=False,
        cli_config=CLIConfig(enable_streaming=True),
    )


@pytest.fixture
def minimal_config():
    """Create a minimal config without optional features."""
    return AgentConfig(
        anthropic_api_key="sk-ant-test123456789",
        vector_store_path=Path("/fake/path"),  # Will be mocked in tests
        model="claude-sonnet-4-5-20250929",
        enable_knowledge_graph=False,
        tool_config=ToolConfig(enable_reranking=False),
    )


class TestCLIInitialization:
    """Test AgentCLI initialization."""

    def test_cli_creation(self, valid_config):
        """Test creating AgentCLI instance."""
        cli = AgentCLI(valid_config)

        assert cli.config == valid_config
        assert cli.agent is None  # Not initialized yet

    def test_cli_with_custom_config(self, tmp_vector_store):
        """Test CLI with custom configuration."""
        config = AgentConfig(
            anthropic_api_key="sk-ant-custom123",
            vector_store_path=tmp_vector_store,
            model="claude-haiku-4-5",
            temperature=0.7,
            max_tokens=2048,
        )

        cli = AgentCLI(config)

        assert cli.config.model == "claude-haiku-4-5"
        assert cli.config.temperature == 0.7
        assert cli.config.max_tokens == 2048


class TestCLIInitializationErrors:
    """Test CLI initialization error handling."""

    def test_cli_initialization_with_missing_vector_store(self, minimal_config):
        """Test that missing vector store raises actionable error."""
        minimal_config.vector_store_path = Path("/nonexistent/path")

        cli = AgentCLI(minimal_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)
        assert "Vector store not found" in error_msg
        assert "/nonexistent/path" in error_msg
        assert "python run_pipeline.py" in error_msg  # Actionable fix

    def test_cli_initialization_with_corrupt_vector_store(self, tmp_path, minimal_config):
        """Test that corrupt vector store raises actionable error."""
        # Create directory with missing required files
        corrupt_store = tmp_path / "corrupt_store"
        corrupt_store.mkdir()
        # Only create one file instead of all three required
        (corrupt_store / "layer1.index").touch()

        minimal_config.vector_store_path = corrupt_store

        cli = AgentCLI(minimal_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)
        assert "Vector store loading failed" in error_msg or "not found" in error_msg.lower()

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    def test_cli_initialization_with_embedder_api_key_error(
        self, mock_embedder_class, mock_vector_store_class, valid_config
    ):
        """Test embedder API key error provides actionable message."""
        # Mock vector store to succeed
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        # Mock embedder to raise API key error
        mock_embedder_class.side_effect = Exception("API key not found")

        cli = AgentCLI(valid_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)
        assert "Embedder initialization failed" in error_msg
        assert "api key" in error_msg.lower()
        # Should provide actionable fix
        assert "openai_api_key" in error_msg.lower() or "embedding_model" in error_msg.lower()

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    def test_cli_initialization_with_generic_embedder_error(
        self, mock_embedder_class, mock_vector_store_class, valid_config
    ):
        """Test generic embedder error provides helpful message."""
        # Mock vector store to succeed
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        # Mock embedder to raise generic error
        mock_embedder_class.side_effect = Exception("Model not found")

        cli = AgentCLI(valid_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)
        assert "Embedder initialization failed" in error_msg
        assert "Model not found" in error_msg
        assert "model is supported" in error_msg.lower()


class TestCLIGracefulDegradation:
    """Test graceful degradation when optional components fail."""

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    @patch("src.agent.cli.CrossEncoderReranker")
    @patch("src.agent.cli.get_registry")
    @patch("src.agent.cli.AgentCore")
    def test_cli_degradation_when_reranker_fails(
        self,
        mock_agent_core,
        mock_registry,
        mock_reranker_class,
        mock_embedder_class,
        mock_vector_store_class,
        valid_config,
    ):
        """Test CLI continues when reranker fails to load."""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        # Reranker fails
        mock_reranker_class.side_effect = Exception("Reranker model not found")

        mock_registry_instance = MagicMock()
        mock_registry_instance.__len__.return_value = 26
        mock_registry.return_value = mock_registry_instance

        mock_agent = MagicMock()
        mock_agent_core.return_value = mock_agent

        # Enable reranking in config
        valid_config.tool_config.enable_reranking = True
        valid_config.tool_config.lazy_load_reranker = False

        cli = AgentCLI(valid_config)

        # Should not raise - degraded mode
        cli.initialize_agent()

        # Reranking should be disabled after failure
        assert valid_config.tool_config.enable_reranking is False

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    @patch("src.agent.cli.get_registry")
    @patch("src.agent.cli.AgentCore")
    @patch("src.graph.models.KnowledgeGraph.load_json")
    def test_cli_degradation_when_kg_missing(
        self,
        mock_kg_load,
        mock_agent_core,
        mock_registry,
        mock_embedder_class,
        mock_vector_store_class,
        tmp_vector_store,
    ):
        """Test CLI continues when knowledge graph is missing."""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        # KG file not found
        mock_kg_load.side_effect = FileNotFoundError("KG not found")

        mock_registry_instance = MagicMock()
        mock_registry_instance.__len__.return_value = 26
        mock_registry.return_value = mock_registry_instance

        mock_agent = MagicMock()
        mock_agent_core.return_value = mock_agent

        config = AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=tmp_vector_store,
            enable_knowledge_graph=True,
            knowledge_graph_path=Path("/nonexistent/kg.json"),
        )

        cli = AgentCLI(config)

        # Should not raise - degraded mode
        cli.initialize_agent()

        # KG should be disabled after failure
        assert config.enable_knowledge_graph is False

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    @patch("src.agent.cli.CrossEncoderReranker")
    @patch("src.agent.cli.get_registry")
    @patch("src.agent.cli.AgentCore")
    @patch("src.graph.models.KnowledgeGraph.load_json")
    def test_cli_initialization_success_all_components(
        self,
        mock_kg_load,
        mock_agent_core,
        mock_registry,
        mock_reranker_class,
        mock_embedder_class,
        mock_vector_store_class,
        tmp_vector_store,
        capsys,
    ):
        """Test successful initialization with all components."""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        mock_kg = MagicMock()
        mock_kg.entities = [Mock()] * 10
        mock_kg.relationships = [Mock()] * 5
        mock_kg_load.return_value = mock_kg

        mock_registry_instance = MagicMock()
        mock_registry_instance.__len__.return_value = 26
        mock_registry.return_value = mock_registry_instance

        mock_agent = MagicMock()
        mock_agent_core.return_value = mock_agent

        # Create KG file
        kg_path = tmp_vector_store / "kg.json"
        kg_path.write_text('{"entities": [], "relationships": []}')

        config = AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=tmp_vector_store,
            enable_knowledge_graph=True,
            knowledge_graph_path=kg_path,
            tool_config=ToolConfig(enable_reranking=True, lazy_load_reranker=False),
        )

        cli = AgentCLI(config)
        cli.initialize_agent()

        # Verify all components initialized
        assert cli.agent is not None
        mock_agent.initialize_with_documents.assert_called_once()

        # Check console output
        captured = capsys.readouterr()
        assert "Agent ready!" in captured.out
        # Should NOT show degraded mode
        assert "DEGRADED MODE" not in captured.out


class TestPlatformDetection:
    """Test platform-specific model selection."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("platform.system")
    def test_cli_platform_detection_apple_silicon(self, mock_platform_system):
        """Test Apple Silicon detection uses bge-m3."""
        mock_platform_system.return_value = "Darwin"

        # Mock torch with MPS support
        with patch("torch.backends.mps.is_available", return_value=True):
            from src.agent.config import _detect_optimal_embedding_model

            model = _detect_optimal_embedding_model()

        assert model == "bge-m3"

    @patch.dict("os.environ", {}, clear=True)
    @patch("platform.system")
    def test_cli_platform_detection_linux_gpu(self, mock_platform_system):
        """Test Linux with GPU uses bge-m3."""
        mock_platform_system.return_value = "Linux"

        # Mock torch with CUDA support
        with patch("torch.cuda.is_available", return_value=True):
            from src.agent.config import _detect_optimal_embedding_model

            model = _detect_optimal_embedding_model()

        assert model == "bge-m3"

    @patch("platform.system")
    @patch.dict("os.environ", {}, clear=True)
    def test_cli_platform_detection_windows(self, mock_platform_system):
        """Test Windows uses cloud embeddings."""
        mock_platform_system.return_value = "Windows"

        # Mock torch to not have GPU
        with patch("torch.cuda.is_available", return_value=False):
            from src.agent.config import _detect_optimal_embedding_model

            model = _detect_optimal_embedding_model()

            assert model == "text-embedding-3-large"

    @patch.dict("os.environ", {"EMBEDDING_MODEL": "custom-model"})
    def test_cli_platform_detection_env_override(self):
        """Test environment variable overrides platform detection."""
        from src.agent.config import _detect_optimal_embedding_model

        model = _detect_optimal_embedding_model()

        assert model == "custom-model"


class TestErrorMessageQuality:
    """Test that error messages are actionable and clear."""

    def test_cli_error_messages_are_actionable(self, minimal_config):
        """Test all error messages provide clear next steps."""
        minimal_config.vector_store_path = Path("/missing/path")

        cli = AgentCLI(minimal_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)

        # Check error message quality
        assert error_msg.startswith("❌")  # Visual indicator
        assert "Vector store not found" in error_msg  # Clear problem
        assert "/missing/path" in error_msg  # Specific location
        assert "python run_pipeline.py" in error_msg  # Actionable fix
        assert "data/" in error_msg  # Example usage

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    def test_cli_embedder_error_suggests_alternatives(
        self, mock_embedder_class, mock_vector_store_class, valid_config
    ):
        """Test embedder errors suggest alternative models."""
        # Mock vector store to succeed
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        # Mock embedder to raise API key error
        mock_embedder_class.side_effect = Exception("API key required")

        cli = AgentCLI(valid_config)

        with pytest.raises(RuntimeError) as exc_info:
            cli.initialize_agent()

        error_msg = str(exc_info.value)

        # Should suggest local model as alternative
        assert "bge-m3" in error_msg
        assert "export EMBEDDING_MODEL" in error_msg


class TestSessionStatistics:
    """Test session statistics tracking."""

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    @patch("src.agent.cli.get_registry")
    @patch("src.agent.cli.AgentCore")
    def test_cli_session_stats_accuracy(
        self,
        mock_agent_core,
        mock_registry,
        mock_embedder_class,
        mock_vector_store_class,
        valid_config,
        capsys,
    ):
        """Test that session statistics are accurate."""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        # Mock registry with stats
        mock_registry_instance = MagicMock()
        mock_registry_instance.__len__.return_value = 26
        mock_registry_instance.get_stats.return_value = {
            "total_tools": 26,
            "total_calls": 42,
            "total_errors": 2,
            "success_rate": 95.2,
            "total_time_ms": 1500.0,
            "avg_time_ms": 35.7,
            "tools": [
                {"name": "simple_search", "execution_count": 20, "avg_time_ms": 50.0},
                {"name": "get_document_list", "execution_count": 10, "avg_time_ms": 10.0},
            ],
        }
        mock_registry.return_value = mock_registry_instance

        # Mock agent with conversation stats
        mock_agent = MagicMock()
        mock_agent.get_conversation_stats.return_value = {
            "message_count": 10,
            "tool_calls": 15,
            "tools_used": ["simple_search", "get_document_list"],
        }
        mock_agent_core.return_value = mock_agent

        cli = AgentCLI(valid_config)
        cli.initialize_agent()

        # Show stats
        cli._show_stats()

        captured = capsys.readouterr()

        # Verify stats output
        assert "Total tools: 26" in captured.out
        assert "Total calls: 42" in captured.out
        assert "Total errors: 2" in captured.out
        assert "Success rate: 95.2%" in captured.out
        assert "Messages: 10" in captured.out
        assert "Tool calls: 15" in captured.out
        assert "simple_search" in captured.out


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    def test_cli_handles_invalid_config(self):
        """Test CLI handles invalid configuration gracefully."""
        # Create config with invalid temperature
        config = AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=Path("/fake/path"),
            temperature=1.5,  # Invalid: > 1.0
        )

        # Should fail during validation
        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "temperature" in error_msg.lower()

    def test_cli_config_validation(self, tmp_vector_store):
        """Test config validation catches errors."""
        config = AgentConfig(
            anthropic_api_key="invalid-key",  # Wrong format
            vector_store_path=tmp_vector_store,
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "invalid format" in error_msg.lower()
        assert "sk-ant-" in error_msg

    @patch("src.agent.cli.HybridVectorStore")
    @patch("src.agent.cli.EmbeddingGenerator")
    @patch("src.agent.cli.get_registry")
    @patch("src.agent.cli.AgentCore")
    def test_cli_shows_config_correctly(
        self,
        mock_agent_core,
        mock_registry,
        mock_embedder_class,
        mock_vector_store_class,
        valid_config,
        capsys,
    ):
        """Test that _show_config displays all settings."""
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_vector_store_class.load.return_value = mock_vector_store

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_registry_instance = MagicMock()
        mock_registry_instance.__len__.return_value = 26
        mock_registry.return_value = mock_registry_instance

        mock_agent = MagicMock()
        mock_agent_core.return_value = mock_agent

        cli = AgentCLI(valid_config)
        cli.initialize_agent()

        # Show config
        cli._show_config()

        captured = capsys.readouterr()

        # Verify config output
        assert "Current Configuration" in captured.out
        assert "claude-sonnet-4-5-20250929" in captured.out
        assert "Temperature: 0.3" in captured.out
        assert "Streaming: True" in captured.out


class TestCLIStartupValidation:
    """Test CLI startup validation."""

    @patch("src.agent.validation.AgentValidator")
    def test_cli_startup_validation_fails_stops_initialization(
        self, mock_validator_class, valid_config
    ):
        """Test that failed validation prevents initialization."""
        # Mock validator to fail
        mock_validator = MagicMock()
        mock_validator.validate_all.return_value = False
        mock_validator.print_summary.return_value = None
        mock_validator_class.return_value = mock_validator

        cli = AgentCLI(valid_config)

        # Validation should fail
        result = cli.startup_validation()

        assert result is False
        mock_validator.validate_all.assert_called_once()
        mock_validator.print_summary.assert_called_once()

    @patch("src.agent.validation.AgentValidator")
    def test_cli_startup_validation_success_continues(self, mock_validator_class, valid_config):
        """Test that successful validation allows initialization."""
        # Mock validator to pass
        mock_validator = MagicMock()
        mock_validator.validate_all.return_value = True
        mock_validator.print_summary.return_value = None
        mock_validator_class.return_value = mock_validator

        cli = AgentCLI(valid_config)

        # Validation should pass
        result = cli.startup_validation()

        assert result is True
        mock_validator.validate_all.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
