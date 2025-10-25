"""Tests for prompt caching functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.agent.config import AgentConfig
from src.agent.agent_core import AgentCore


class TestPromptCachingConfig:
    """Test prompt caching configuration."""

    def test_default_caching_enabled(self):
        """Test that prompt caching is enabled by default."""
        config = AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=Path("tests/fixtures/vector_store")
        )
        assert config.enable_prompt_caching is True

    def test_caching_can_be_disabled(self):
        """Test that prompt caching can be disabled."""
        config = AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=Path("tests/fixtures/vector_store"),
            enable_prompt_caching=False
        )
        assert config.enable_prompt_caching is False


class TestPromptCachingHelpers:
    """Test prompt caching helper methods."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with vector store."""
        return AgentConfig(
            anthropic_api_key="sk-ant-test123",
            vector_store_path=Path("tests/fixtures/vector_store"),
            enable_prompt_caching=True
        )

    @pytest.fixture
    def mock_agent(self, mock_config, monkeypatch):
        """Create agent with mocked components."""
        # Mock registry
        mock_registry = MagicMock()
        mock_registry.__len__ = Mock(return_value=10)
        mock_registry.get_claude_sdk_tools = Mock(return_value=[
            {"name": "tool1", "description": "Test tool 1"},
            {"name": "tool2", "description": "Test tool 2"},
        ])

        monkeypatch.setattr("src.agent.agent_core.get_registry", lambda: mock_registry)

        # Mock config validation to bypass vector store check
        monkeypatch.setattr("src.agent.config.AgentConfig.validate", lambda self: None)

        agent = AgentCore(mock_config)
        return agent

    def test_prepare_system_prompt_with_cache(self, mock_agent):
        """Test system prompt preparation with cache control."""
        system_prompt = mock_agent._prepare_system_prompt_with_cache()

        assert isinstance(system_prompt, list)
        assert len(system_prompt) == 1
        assert system_prompt[0]["type"] == "text"
        assert "text" in system_prompt[0]
        assert "cache_control" in system_prompt[0]
        assert system_prompt[0]["cache_control"]["type"] == "ephemeral"

    def test_prepare_system_prompt_without_cache(self, mock_agent):
        """Test system prompt without caching."""
        mock_agent.config.enable_prompt_caching = False
        system_prompt = mock_agent._prepare_system_prompt_with_cache()

        assert isinstance(system_prompt, list)
        assert len(system_prompt) == 1
        assert system_prompt[0]["type"] == "text"
        assert "cache_control" not in system_prompt[0]

    def test_prepare_tools_with_cache(self, mock_agent):
        """Test tools preparation with cache control on last tool."""
        tools = [
            {"name": "tool1", "description": "Test 1"},
            {"name": "tool2", "description": "Test 2"},
            {"name": "tool3", "description": "Test 3"},
        ]

        cached_tools = mock_agent._prepare_tools_with_cache(tools)

        # Check that cache_control is only on last tool
        assert "cache_control" not in cached_tools[0]
        assert "cache_control" not in cached_tools[1]
        assert "cache_control" in cached_tools[2]
        assert cached_tools[2]["cache_control"]["type"] == "ephemeral"

        # Original tools should not be modified
        assert "cache_control" not in tools[2]

    def test_prepare_tools_without_cache(self, mock_agent):
        """Test tools preparation without caching."""
        mock_agent.config.enable_prompt_caching = False
        tools = [
            {"name": "tool1", "description": "Test 1"},
            {"name": "tool2", "description": "Test 2"},
        ]

        cached_tools = mock_agent._prepare_tools_with_cache(tools)

        # No cache control should be added
        assert "cache_control" not in cached_tools[0]
        assert "cache_control" not in cached_tools[1]

    def test_add_cache_control_to_messages(self, mock_agent):
        """Test adding cache control to initialization messages."""
        messages = [
            {"role": "user", "content": "Document list..."},
            {"role": "assistant", "content": "I understand. I have access..."},
            {"role": "user", "content": "What is in document X?"},
        ]

        cached_messages = mock_agent._add_cache_control_to_messages(messages)

        # Check that 2nd message (assistant acknowledgment) has cache control
        assert isinstance(cached_messages[1]["content"], list)
        assert cached_messages[1]["content"][0]["type"] == "text"
        assert "cache_control" in cached_messages[1]["content"][0]

        # Original messages should not be modified
        assert isinstance(messages[1]["content"], str)

    def test_add_cache_control_without_caching(self, mock_agent):
        """Test messages without cache control."""
        mock_agent.config.enable_prompt_caching = False
        messages = [
            {"role": "user", "content": "Document list..."},
            {"role": "assistant", "content": "I understand..."},
        ]

        cached_messages = mock_agent._add_cache_control_to_messages(messages)

        # Messages should be unchanged
        assert cached_messages == messages


class TestCostTrackingWithCache:
    """Test cost tracking with cache statistics."""

    def test_cost_tracker_accepts_cache_params(self):
        """Test that cost tracker accepts cache parameters."""
        from src.cost_tracker import CostTracker

        tracker = CostTracker()

        # Track with cache stats
        cost = tracker.track_llm(
            provider="anthropic",
            model="claude-sonnet-4-5",
            input_tokens=1000,
            output_tokens=500,
            operation="agent",
            cache_creation_tokens=800,
            cache_read_tokens=200
        )

        assert cost > 0
        assert len(tracker.entries) == 1

        entry = tracker.entries[0]
        assert entry.cache_creation_tokens == 800
        assert entry.cache_read_tokens == 200

    def test_cost_tracker_works_without_cache_params(self):
        """Test backward compatibility without cache params."""
        from src.cost_tracker import CostTracker

        tracker = CostTracker()

        # Track without cache stats (backward compatibility)
        cost = tracker.track_llm(
            provider="anthropic",
            model="claude-haiku-4-5",
            input_tokens=500,
            output_tokens=300,
            operation="agent"
        )

        assert cost > 0
        assert len(tracker.entries) == 1

        entry = tracker.entries[0]
        assert entry.cache_creation_tokens == 0
        assert entry.cache_read_tokens == 0
