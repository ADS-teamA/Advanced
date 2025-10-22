"""
Tests for Tool Registry.

Tests:
- Tool registration
- Unavailable tool tracking
- Tool execution
- Statistics
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.agent.tools.registry import ToolRegistry, register_tool
from src.agent.tools.base import BaseTool, ToolResult, ToolInput
from pydantic import Field


# Mock tool classes for testing
class MockToolInput(ToolInput):
    """Mock input for testing."""

    query: str = Field(..., description="Test query")


class MockBasicTool(BaseTool):
    """Mock basic tool for testing."""

    name = "mock_basic"
    description = "Mock basic tool"
    tier = 1
    input_schema = MockToolInput
    requires_kg = False
    requires_reranker = False

    def execute_impl(self, query: str) -> ToolResult:
        return ToolResult(
            success=True, data={"result": f"Processed: {query}"}, citations=[]
        )


class MockKGTool(BaseTool):
    """Mock tool that requires knowledge graph."""

    name = "mock_kg_tool"
    description = "Mock KG tool"
    tier = 2
    input_schema = MockToolInput
    requires_kg = True
    requires_reranker = False

    def execute_impl(self, query: str) -> ToolResult:
        return ToolResult(
            success=True, data={"result": f"KG processed: {query}"}, citations=[]
        )


class MockRerankerTool(BaseTool):
    """Mock tool that requires reranker."""

    name = "mock_reranker_tool"
    description = "Mock reranker tool"
    tier = 1
    input_schema = MockToolInput
    requires_kg = False
    requires_reranker = True

    def execute_impl(self, query: str) -> ToolResult:
        return ToolResult(
            success=True, data={"result": f"Reranked: {query}"}, citations=[]
        )


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ToolRegistry()


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for tool initialization."""
    return {
        "vector_store": Mock(),
        "embedder": Mock(),
        "reranker": Mock(),
        "graph_retriever": None,
        "knowledge_graph": None,
        "context_assembler": Mock(),
        "config": Mock(),
    }


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool_class(self, registry):
        """Test registering a tool class."""
        registry.register_tool_class(MockBasicTool)

        assert "mock_basic" in registry._tool_classes
        assert registry._tool_classes["mock_basic"] == MockBasicTool

    def test_register_multiple_tools(self, registry):
        """Test registering multiple tool classes."""
        registry.register_tool_class(MockBasicTool)
        registry.register_tool_class(MockKGTool)

        assert len(registry._tool_classes) == 2
        assert "mock_basic" in registry._tool_classes
        assert "mock_kg_tool" in registry._tool_classes


class TestToolInitialization:
    """Test tool initialization with dependencies."""

    def test_initialize_basic_tool(self, registry, mock_dependencies):
        """Test initializing a tool without special requirements."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        assert "mock_basic" in registry._tools
        assert len(registry._tools) == 1
        assert len(registry._unavailable_tools) == 0

    def test_kg_tool_without_kg_marked_unavailable(self, registry, mock_dependencies):
        """Test that KG-requiring tools are marked unavailable without KG."""
        registry.register_tool_class(MockKGTool)

        # Initialize without knowledge graph
        mock_dependencies["knowledge_graph"] = None
        registry.initialize_tools(**mock_dependencies)

        # Tool should not be initialized
        assert "mock_kg_tool" not in registry._tools

        # Tool should be marked unavailable
        assert "mock_kg_tool" in registry._unavailable_tools
        assert "knowledge graph" in registry._unavailable_tools["mock_kg_tool"].lower()

    def test_kg_tool_with_kg_initialized(self, registry, mock_dependencies):
        """Test that KG-requiring tools are initialized when KG is available."""
        registry.register_tool_class(MockKGTool)

        # Initialize WITH knowledge graph
        mock_dependencies["knowledge_graph"] = Mock()
        registry.initialize_tools(**mock_dependencies)

        # Tool should be initialized
        assert "mock_kg_tool" in registry._tools
        assert "mock_kg_tool" not in registry._unavailable_tools

    def test_reranker_tool_without_reranker_still_initialized(
        self, registry, mock_dependencies
    ):
        """Test that reranker tools are initialized even without reranker (just logs warning)."""
        registry.register_tool_class(MockRerankerTool)

        # Initialize without reranker
        mock_dependencies["reranker"] = None
        registry.initialize_tools(**mock_dependencies)

        # Tool should still be initialized (reranker is optional)
        assert "mock_reranker_tool" in registry._tools


class TestUnavailableToolTracking:
    """Test unavailable tool tracking feature."""

    def test_get_unavailable_tools_empty_initially(self, registry):
        """Test that unavailable tools dict is empty initially."""
        unavailable = registry.get_unavailable_tools()
        assert unavailable == {}

    def test_get_unavailable_tools_returns_copy(self, registry, mock_dependencies):
        """Test that get_unavailable_tools returns a copy (not reference)."""
        registry.register_tool_class(MockKGTool)
        mock_dependencies["knowledge_graph"] = None
        registry.initialize_tools(**mock_dependencies)

        unavailable = registry.get_unavailable_tools()
        unavailable["new_tool"] = "test"

        # Original should not be modified
        assert "new_tool" not in registry._unavailable_tools

    def test_unavailable_tools_in_stats(self, registry, mock_dependencies):
        """Test that unavailable tools appear in statistics."""
        registry.register_tool_class(MockBasicTool)
        registry.register_tool_class(MockKGTool)

        # Initialize without KG
        mock_dependencies["knowledge_graph"] = None
        registry.initialize_tools(**mock_dependencies)

        stats = registry.get_stats()

        assert stats["total_registered"] == 2
        assert stats["total_tools"] == 1  # Only MockBasicTool initialized
        assert stats["unavailable_tools"] == 1
        assert "mock_kg_tool" in stats["unavailable"]


class TestToolExecution:
    """Test tool execution through registry."""

    def test_execute_existing_tool(self, registry, mock_dependencies):
        """Test executing a registered and initialized tool."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        result = registry.execute_tool("mock_basic", query="test query")

        assert result.success is True
        assert "Processed: test query" in result.data["result"]

    def test_execute_nonexistent_tool_returns_error(self, registry, mock_dependencies):
        """Test that executing nonexistent tool returns error result."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        result = registry.execute_tool("nonexistent_tool", query="test")

        assert result.success is False
        assert "not found" in result.error.lower()
        assert "available_tools" in result.metadata

    def test_execute_unavailable_tool_returns_error(self, registry, mock_dependencies):
        """Test that executing unavailable tool returns error."""
        registry.register_tool_class(MockKGTool)

        # Initialize without KG (makes tool unavailable)
        mock_dependencies["knowledge_graph"] = None
        registry.initialize_tools(**mock_dependencies)

        result = registry.execute_tool("mock_kg_tool", query="test")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestToolRetrieval:
    """Test tool retrieval methods."""

    def test_get_tool_returns_instance(self, registry, mock_dependencies):
        """Test that get_tool returns tool instance."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        tool = registry.get_tool("mock_basic")

        assert tool is not None
        assert isinstance(tool, MockBasicTool)

    def test_get_tool_returns_none_for_nonexistent(self, registry, mock_dependencies):
        """Test that get_tool returns None for nonexistent tool."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        tool = registry.get_tool("nonexistent")

        assert tool is None

    def test_get_all_tools(self, registry, mock_dependencies):
        """Test getting all initialized tools."""
        registry.register_tool_class(MockBasicTool)
        registry.register_tool_class(MockRerankerTool)
        registry.initialize_tools(**mock_dependencies)

        all_tools = registry.get_all_tools()

        assert len(all_tools) == 2

    def test_get_tools_by_tier(self, registry, mock_dependencies):
        """Test getting tools by tier."""
        registry.register_tool_class(MockBasicTool)  # Tier 1
        registry.register_tool_class(MockKGTool)  # Tier 2
        registry.register_tool_class(MockRerankerTool)  # Tier 1

        # Initialize with KG so all tools are available
        mock_dependencies["knowledge_graph"] = Mock()
        registry.initialize_tools(**mock_dependencies)

        tier1_tools = registry.get_tools_by_tier(1)
        tier2_tools = registry.get_tools_by_tier(2)

        assert len(tier1_tools) == 2
        assert len(tier2_tools) == 1


class TestRegistryStatistics:
    """Test registry statistics."""

    def test_stats_with_no_tools(self, registry):
        """Test statistics with no tools."""
        stats = registry.get_stats()

        assert stats["total_tools"] == 0
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 100.0

    def test_stats_after_initialization(self, registry, mock_dependencies):
        """Test statistics after tool initialization."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        stats = registry.get_stats()

        assert stats["total_tools"] == 1
        assert stats["total_registered"] == 1
        assert stats["unavailable_tools"] == 0


class TestRegistryMagicMethods:
    """Test registry magic methods."""

    def test_len(self, registry, mock_dependencies):
        """Test __len__ returns number of initialized tools."""
        registry.register_tool_class(MockBasicTool)
        registry.register_tool_class(MockKGTool)

        # Only basic tool initialized (no KG)
        mock_dependencies["knowledge_graph"] = None
        registry.initialize_tools(**mock_dependencies)

        assert len(registry) == 1

    def test_contains(self, registry, mock_dependencies):
        """Test __contains__ checks for initialized tools."""
        registry.register_tool_class(MockBasicTool)
        registry.initialize_tools(**mock_dependencies)

        assert "mock_basic" in registry
        assert "nonexistent" not in registry
