"""
Tests for Agent Validation system.

Tests:
- API key exposure protection
- Validation results
- Component checking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.agent.validation import AgentValidator, ValidationResult


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock()
    config.anthropic_api_key = "sk-ant-test123456789"
    config.vector_store_path = Path("/fake/path")
    config.enable_knowledge_graph = False
    config.knowledge_graph_path = None
    config.model = "claude-sonnet-4-5-20250929"
    return config


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            name="Test Check", passed=True, message="Test passed", details={"foo": "bar"}
        )

        assert result.name == "Test Check"
        assert result.passed is True
        assert result.message == "Test passed"
        assert result.details == {"foo": "bar"}

    def test_validation_result_repr_pass(self):
        """Test string representation of passing result."""
        result = ValidationResult(name="Test", passed=True, message="OK")

        assert "✅ PASS" in repr(result)
        assert "Test" in repr(result)
        assert "OK" in repr(result)

    def test_validation_result_repr_fail(self):
        """Test string representation of failing result."""
        result = ValidationResult(name="Test", passed=False, message="FAIL")

        assert "❌ FAIL" in repr(result)
        assert "Test" in repr(result)
        assert "FAIL" in repr(result)


class TestAPIKeyExposure:
    """Test that API keys are not exposed in validation."""

    def test_anthropic_key_prefix_not_exposed(self, mock_config):
        """Test that Anthropic API key prefix is redacted."""
        validator = AgentValidator(mock_config, debug=False)
        validator._check_api_keys()

        # Find the Anthropic key result
        anthropic_result = next((r for r in validator.results if "ANTHROPIC" in r.name), None)

        assert anthropic_result is not None
        assert anthropic_result.passed is True

        # Check that key is redacted
        key_prefix = anthropic_result.details.get("key_prefix")
        assert key_prefix == "sk-ant-***"
        # Ensure actual key is not in details
        assert "test123456789" not in str(anthropic_result.details)

    def test_invalid_anthropic_key_not_exposed(self, mock_config):
        """Test that invalid Anthropic API key is not exposed."""
        mock_config.anthropic_api_key = "invalid-key-12345"

        validator = AgentValidator(mock_config, debug=False)
        validator._check_api_keys()

        anthropic_result = next((r for r in validator.results if "ANTHROPIC" in r.name), None)

        assert anthropic_result is not None
        assert anthropic_result.passed is False

        # Check that invalid key is redacted
        key_prefix = anthropic_result.details.get("key_prefix")
        assert key_prefix == "***"
        assert "invalid-key-12345" not in str(anthropic_result.details)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123456789"})
    def test_openai_key_prefix_not_exposed(self, mock_config):
        """Test that OpenAI API key prefix is redacted."""
        validator = AgentValidator(mock_config, debug=False)
        validator._check_api_keys()

        openai_result = next((r for r in validator.results if "OPENAI" in r.name), None)

        assert openai_result is not None

        # Check that key is redacted
        key_prefix = openai_result.details.get("key_prefix")
        if openai_result.passed:
            assert key_prefix == "sk-***"
            assert "test123456789" not in str(openai_result.details)


class TestPythonVersionCheck:
    """Test Python version checking."""

    def test_python_version_check_passes(self, mock_config):
        """Test that current Python version passes (should be 3.10+)."""
        validator = AgentValidator(mock_config, debug=False)
        validator._check_python_version()

        assert len(validator.results) == 1
        result = validator.results[0]

        # Should pass since we're running on 3.10+
        assert result.passed is True
        assert "Python" in result.name


class TestDependencyChecks:
    """Test dependency checking."""

    def test_required_dependencies_checked(self, mock_config):
        """Test that required dependencies are checked."""
        validator = AgentValidator(mock_config, debug=False)
        validator._check_dependencies()

        # Should check for anthropic, pydantic, faiss, numpy
        dependency_results = [r for r in validator.results if "Dependency:" in r.name]

        assert len(dependency_results) >= 4

        # Check that we're looking for key dependencies
        dep_names = [r.name for r in dependency_results]
        assert any("anthropic" in name.lower() for name in dep_names)
        assert any("pydantic" in name.lower() for name in dep_names)

    def test_optional_dependencies_not_critical(self, mock_config):
        """Test that optional dependencies don't fail validation."""
        validator = AgentValidator(mock_config, debug=False)
        validator._check_dependencies()

        # Optional dependencies should not have [CRITICAL] tag
        optional_results = [r for r in validator.results if "Optional:" in r.name]

        for result in optional_results:
            assert "[CRITICAL]" not in result.name


class TestVectorStoreValidation:
    """Test vector store validation."""

    def test_missing_vector_store_fails(self, mock_config):
        """Test that missing vector store fails validation."""
        mock_config.vector_store_path = Path("/nonexistent/path")

        validator = AgentValidator(mock_config, debug=False)
        validator._check_vector_store()

        # Should have a critical failure
        vs_result = next((r for r in validator.results if "Vector Store" in r.name), None)

        assert vs_result is not None
        assert vs_result.passed is False
        assert "[CRITICAL]" in vs_result.name


class TestKnowledgeGraphValidation:
    """Test knowledge graph validation."""

    def test_kg_disabled_passes(self, mock_config):
        """Test that disabled KG passes validation."""
        mock_config.enable_knowledge_graph = False

        validator = AgentValidator(mock_config, debug=False)
        validator._check_knowledge_graph()

        kg_result = next((r for r in validator.results if "Knowledge Graph" in r.name), None)

        assert kg_result is not None
        assert kg_result.passed is True

    def test_kg_enabled_without_path_fails(self, mock_config):
        """Test that enabled KG without path fails."""
        mock_config.enable_knowledge_graph = True
        mock_config.knowledge_graph_path = None

        validator = AgentValidator(mock_config, debug=False)
        validator._check_knowledge_graph()

        kg_result = next((r for r in validator.results if "Knowledge Graph" in r.name), None)

        assert kg_result is not None
        assert kg_result.passed is False
        assert "[CRITICAL]" in kg_result.name


class TestValidationSummary:
    """Test overall validation logic."""

    def test_validate_all_runs_all_checks(self, mock_config):
        """Test that validate_all runs all check methods."""
        # Use a real path that might exist
        mock_config.vector_store_path = Path(".")

        validator = AgentValidator(mock_config, debug=False)

        # Mock the vector store check to avoid actual loading
        with patch.object(validator, "_check_vector_store"):
            result = validator.validate_all()

        # Should have results from multiple checks
        assert len(validator.results) > 5

        # Should have checks for:
        # - Python version
        # - Dependencies
        # - API keys
        # - etc.

    def test_critical_failures_cause_validation_failure(self, mock_config):
        """Test that critical failures cause validation to fail."""
        # Set an invalid config that will cause critical failure
        mock_config.anthropic_api_key = None  # Missing required key

        validator = AgentValidator(mock_config, debug=False)

        # Just run API key check for simplicity
        validator._check_api_keys()

        # Manually check for critical failures
        critical_failures = [
            r for r in validator.results if not r.passed and "[CRITICAL]" in r.name
        ]

        assert len(critical_failures) > 0
