"""
Tests for Tier 1 Basic Tools.

TODO: This test file is outdated and needs to be rewritten.
Many tools referenced here have been removed or consolidated into SearchTool.

Current tier1_basic.py tools (6 total):
- GetToolHelpTool
- SearchTool (unified search with query expansion)
- GetDocumentListTool
- ListAvailableToolsTool
- GetDocumentInfoTool
- ExactMatchSearchTool

Tests all basic retrieval tools with:
- Valid input scenarios
- Error cases (empty results, missing dependencies)
- Input validation (Pydantic schemas)
- ToolResult structure verification
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

# Skip entire file - these tests are for tools that no longer exist
# They were removed/consolidated during the refactoring to SearchTool
pytest.skip("Tests outdated - tools have been refactored", allow_module_level=True)

from src.agent.tools.tier1_basic import (
    SearchTool,
    GetDocumentListTool,
    ListAvailableToolsTool,
)
from src.agent.tools.base import ToolResult


# ===== Fixtures =====


@pytest.fixture
def mock_embedder():
    """Create mock EmbeddingGenerator."""
    embedder = Mock()
    embedder.embed_texts = Mock(return_value=np.random.rand(1, 3072))
    embedder.dimensions = 3072
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock HybridVectorStore."""
    store = Mock()

    # Mock hierarchical_search
    store.hierarchical_search = Mock(
        return_value={
            "layer1": [],
            "layer2": [],
            "layer3": [
                {
                    "chunk_id": "chunk_001",
                    "document_id": "GRI_306",
                    "section_id": "sec_3.2",
                    "section_title": "Disclosure 306-3",
                    "content": "Organizations shall report total weight of hazardous waste.",
                    "raw_content": "Organizations shall report total weight of hazardous waste.",
                    "rrf_score": 0.85,
                    "page_number": 15,
                },
                {
                    "chunk_id": "chunk_002",
                    "document_id": "GRI_306",
                    "section_id": "sec_3.4",
                    "section_title": "Disclosure 306-4",
                    "content": "Organizations shall report waste diverted from disposal.",
                    "raw_content": "Organizations shall report waste diverted from disposal.",
                    "rrf_score": 0.72,
                    "page_number": 17,
                },
            ],
        }
    )

    # Mock metadata layers
    store.metadata_layer1 = [
        {
            "document_id": "GRI_306",
            "content": "Standard for waste management reporting and disclosure requirements.",
        },
        {
            "document_id": "GRI_305",
            "content": "Standard for emissions reporting and climate change disclosure.",
        },
    ]

    store.metadata_layer2 = [
        {
            "document_id": "GRI_306",
            "section_id": "sec_3.2",
            "section_title": "Disclosure 306-3",
            "section_path": "/GRI_306/Section_3/Disclosure_306-3",
            "content": "Requirements for reporting hazardous waste.",
            "page_number": 15,
        },
        {
            "document_id": "GRI_306",
            "section_id": "sec_3.4",
            "section_title": "Disclosure 306-4",
            "section_path": "/GRI_306/Section_3/Disclosure_306-4",
            "content": "Requirements for reporting waste diverted from disposal.",
            "page_number": 17,
        },
    ]

    store.metadata_layer3 = [
        {
            "chunk_id": "chunk_001",
            "document_id": "GRI_306",
            "section_id": "sec_3.2",
            "section_title": "Disclosure 306-3",
            "content": "Organizations shall report total weight of hazardous waste.",
            "raw_content": "Organizations shall report total weight of hazardous waste.",
            "page_number": 15,
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "GRI_306",
            "section_id": "sec_3.2",
            "section_title": "Disclosure 306-3",
            "content": "This includes waste generated and waste diverted.",
            "raw_content": "This includes waste generated and waste diverted.",
            "page_number": 15,
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "GRI_306",
            "section_id": "sec_3.2",
            "section_title": "Disclosure 306-3",
            "content": "Waste should be categorized by composition and disposal method.",
            "raw_content": "Waste should be categorized by composition and disposal method.",
            "page_number": 16,
        },
    ]

    # Mock BM25 store
    bm25_store = Mock()
    bm25_store.search_layer3 = Mock(
        return_value=[
            {
                "chunk_id": "chunk_001",
                "document_id": "GRI_306",
                "section_title": "Disclosure 306-3",
                "content": "Organizations shall report total weight of hazardous waste.",
                "bm25_score": 12.5,
            },
        ]
    )
    store.bm25_store = bm25_store

    return store


@pytest.fixture
def mock_reranker():
    """Create mock CrossEncoderReranker."""
    reranker = Mock()
    reranker.rerank = Mock(side_effect=lambda query, candidates, top_k: candidates[:top_k])
    return reranker


@pytest.fixture
def mock_knowledge_graph():
    """Create mock KnowledgeGraph."""
    kg = Mock()
    kg.entities = []
    kg.relationships = []
    return kg


@pytest.fixture
def mock_config():
    """Create mock ToolConfig."""
    config = Mock()
    config.context_window = 2
    return config


# ===== Tool 1: SearchTool =====


class TestSearchTool:
    """Test SearchTool (unified hybrid search with optional query expansion)."""

    @pytest.fixture
    def mock_config(self):
        """Create mock AgentConfig."""
        config = Mock()
        config.tool_config = Mock()
        config.tool_config.query_expansion_provider = "openai"
        config.tool_config.query_expansion_model = "gpt-5-nano"
        config.anthropic_api_key = None
        config.openai_api_key = "sk-test-key"
        return config

    def test_search_without_expansion_with_reranker(self, mock_vector_store, mock_embedder, mock_reranker, mock_config):
        """Test search without expansion (num_expands=1) with reranker."""
        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
            config=mock_config,
        )

        result = tool.execute(query="waste disposal requirements", k=2, num_expands=1)

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 2
        assert result.data[0]["document_id"] == "GRI_306"
        assert "score" in result.data[0]
        assert len(result.citations) == 2
        assert result.metadata["method"] == "hybrid+rerank"
        assert result.metadata["k"] == 2
        assert result.metadata["num_expands"] == 1
        assert result.metadata["expansion_metadata"]["expansion_method"] == "none"

    def test_search_with_expansion(self, mock_vector_store, mock_embedder, mock_reranker, mock_config):
        """Test search with query expansion (num_expands=3)."""
        from unittest.mock import patch

        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            reranker=mock_reranker,
            config=mock_config,
        )

        # Mock QueryExpander
        with patch.object(tool, '_get_query_expander') as mock_get_expander:
            mock_expander = Mock()
            mock_expansion_result = Mock()
            mock_expansion_result.expanded_queries = [
                "waste disposal requirements",
                "requirements for waste management",
                "disposal of waste regulations"
            ]
            mock_expansion_result.expansion_method = "llm"
            mock_expansion_result.model_used = "gpt-5-nano"
            mock_expander.expand.return_value = mock_expansion_result
            mock_get_expander.return_value = mock_expander

            result = tool.execute(query="waste disposal requirements", k=2, num_expands=3)

            assert result.success is True
            assert result.metadata["num_expands"] == 3
            assert result.metadata["expansion_metadata"]["expansion_method"] == "llm"
            assert result.metadata["expansion_metadata"]["model_used"] == "gpt-5-nano"
            assert result.metadata["expansion_metadata"]["queries_generated"] == 3
            assert result.metadata["fusion_method"] == "rrf"

    def test_search_expansion_fallback_on_error(self, mock_vector_store, mock_embedder, mock_config):
        """Test search falls back to original query when expansion fails."""
        from unittest.mock import patch

        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            reranker=None,
            config=mock_config,
        )

        # Mock QueryExpander to raise error
        with patch.object(tool, '_get_query_expander') as mock_get_expander:
            mock_expander = Mock()
            mock_expander.expand.side_effect = Exception("Expansion failed")
            mock_get_expander.return_value = mock_expander

            result = tool.execute(query="test query", k=5, num_expands=3)

            # Should still succeed with fallback
            assert result.success is True
            assert result.metadata["expansion_metadata"]["expansion_method"] == "failed"

    def test_search_without_reranker(self, mock_vector_store, mock_embedder, mock_config):
        """Test search without reranker."""
        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            reranker=None,
            config=mock_config,
        )

        result = tool.execute(query="waste disposal", k=6, num_expands=1)

        assert result.success is True
        assert result.metadata["method"] == "hybrid"
        assert len(result.data) <= 6

    def test_search_empty_results(self, mock_embedder, mock_config):
        """Test search with no results."""
        empty_store = Mock()
        empty_store.hierarchical_search = Mock(
            return_value={"layer1": [], "layer2": [], "layer3": []}
        )

        tool = SearchTool(
            vector_store=empty_store,
            embedder=mock_embedder,
            reranker=None,
            config=mock_config,
        )

        result = tool.execute(query="nonexistent query", k=6, num_expands=1)

        assert result.success is True
        assert result.data == []
        assert result.citations == []


# ===== Tool 2: EntitySearchTool =====


class TestEntitySearchTool:
    """Test EntitySearchTool (find chunks mentioning entities)."""

    def test_entity_search_found(self, mock_vector_store, mock_embedder):
        """Test entity search with matching results."""
        tool = EntitySearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(entity_value="hazardous waste", k=6)

        assert result.success is True
        assert len(result.data) > 0
        # Check that returned chunks actually contain the entity (case-insensitive)
        for chunk in result.data:
            assert "hazardous waste" in chunk["content"].lower()
        assert result.metadata["entity"] == "hazardous waste"
        assert result.metadata["matches_found"] > 0
        assert result.metadata["no_results"] is False

    def test_entity_search_not_found(self, mock_embedder):
        """Test entity search with no matching results."""
        empty_store = Mock()
        empty_store.hierarchical_search = Mock(
            return_value={
                "layer3": [
                    {
                        "chunk_id": "chunk_999",
                        "content": "This chunk does not contain the entity.",
                        "document_id": "TEST",
                        "section_title": "Test Section",
                    }
                ]
            }
        )

        tool = EntitySearchTool(
            vector_store=empty_store,
            embedder=mock_embedder,
        )

        result = tool.execute(entity_value="nonexistent entity", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["no_results"] is True
        assert result.metadata["matches_found"] == 0

    def test_entity_search_case_insensitive(self, mock_vector_store, mock_embedder):
        """Test that entity search is case-insensitive."""
        tool = EntitySearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        # Search with different case
        result = tool.execute(entity_value="HAZARDOUS WASTE", k=6)

        assert result.success is True
        assert len(result.data) > 0


# ===== Tool 3: DocumentSearchTool =====


class TestDocumentSearchTool:
    """Test DocumentSearchTool (search within specific document)."""

    def test_document_search_found(self, mock_vector_store, mock_embedder):
        """Test document search with results."""
        tool = DocumentSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="waste", document_id="GRI_306", k=6)

        assert result.success is True
        assert len(result.data) > 0
        # All results should be from the specified document
        for chunk in result.data:
            assert chunk["document_id"] == "GRI_306"
        assert result.metadata["document_id"] == "GRI_306"
        assert result.metadata["results_count"] > 0

    def test_document_search_wrong_document(self, mock_vector_store, mock_embedder):
        """Test document search with wrong document ID."""
        tool = DocumentSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="waste", document_id="WRONG_DOC", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["no_results"] is True

    def test_document_search_empty_results(self, mock_embedder):
        """Test document search with no results."""
        empty_store = Mock()
        empty_store.hierarchical_search = Mock(return_value={"layer3": []})

        tool = DocumentSearchTool(
            vector_store=empty_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="test", document_id="TEST_DOC", k=6)

        assert result.success is True
        assert result.data == []


# ===== Tool 4: SectionSearchTool =====


class TestSectionSearchTool:
    """Test SectionSearchTool (search within sections)."""

    def test_section_search_found(self, mock_vector_store, mock_embedder):
        """Test section search with matching results."""
        tool = SectionSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="waste", section_title="Disclosure 306-3", k=6)

        assert result.success is True
        assert len(result.data) > 0
        # All results should match section title (case-insensitive partial match)
        for chunk in result.data:
            assert "disclosure 306-3" in chunk["section_title"].lower()
        assert result.metadata["section_title"] == "Disclosure 306-3"

    def test_section_search_partial_match(self, mock_vector_store, mock_embedder):
        """Test section search with partial title match."""
        tool = SectionSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="waste", section_title="306-3", k=6)

        assert result.success is True
        assert len(result.data) > 0

    def test_section_search_not_found(self, mock_embedder):
        """Test section search with no matching sections."""
        store = Mock()
        store.hierarchical_search = Mock(
            return_value={
                "layer3": [
                    {
                        "chunk_id": "chunk_001",
                        "section_title": "Different Section",
                        "content": "Content",
                        "document_id": "TEST",
                    }
                ]
            }
        )

        tool = SectionSearchTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="test", section_title="Nonexistent Section", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["no_results"] is True


# ===== Tool 5: KeywordSearchTool =====


class TestKeywordSearchTool:
    """Test KeywordSearchTool (pure BM25 search)."""

    def test_keyword_search_with_bm25_store(self, mock_vector_store, mock_embedder):
        """Test keyword search using BM25 store."""
        tool = KeywordSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(keywords="hazardous waste", k=6)

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["method"] == "bm25"
        assert result.metadata["keywords"] == "hazardous waste"

    def test_keyword_search_without_bm25_store(self, mock_embedder):
        """Test keyword search fallback without BM25 store."""
        store = Mock()
        # No bm25_store attribute - should use fallback
        # Need spec to make hasattr() return False
        del store.bm25_store
        store.hierarchical_search = Mock(
            return_value={
                "layer3": [
                    {
                        "chunk_id": "chunk_001",
                        "content": "Test content with keywords.",
                        "document_id": "TEST",
                        "section_title": "Section",
                    }
                ]
            }
        )

        tool = KeywordSearchTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(keywords="test keywords", k=6)

        assert result.success is True
        assert len(result.data) > 0

    def test_keyword_search_empty_results(self, mock_embedder):
        """Test keyword search with no results."""
        store = Mock()
        store.bm25_store = Mock()
        store.bm25_store.search_layer3 = Mock(return_value=[])

        tool = KeywordSearchTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(keywords="nonexistent keywords", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["no_results"] is True


# ===== Tool 6: GetDocumentListTool =====


class TestGetDocumentListTool:
    """Test GetDocumentListTool (list all documents)."""

    def test_get_document_list_success(self, mock_vector_store, mock_embedder):
        """Test getting document list."""
        tool = GetDocumentListTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert "documents" in result.data
        assert "count" in result.data
        assert len(result.data["documents"]) == 2  # GRI_306 and GRI_305

        # Check document structure
        doc = result.data["documents"][0]
        assert "id" in doc
        assert "summary" in doc

        assert result.metadata["total_documents"] == 2

    def test_get_document_list_with_faiss_store(self, mock_embedder):
        """Test getting document list from faiss_store attribute."""
        store = Mock()
        # Ensure metadata_layer1 doesn't exist on store itself
        del store.metadata_layer1

        faiss_store = Mock()
        faiss_store.metadata_layer1 = [
            {"document_id": "DOC_1", "content": "Summary 1"},
            {"document_id": "DOC_2", "content": "Summary 2"},
        ]
        store.faiss_store = faiss_store

        tool = GetDocumentListTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert len(result.data["documents"]) == 2

    def test_get_document_list_empty(self, mock_embedder):
        """Test getting document list when empty."""
        store = Mock()
        store.metadata_layer1 = []

        tool = GetDocumentListTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert result.data["count"] == 0


# ===== Tool 7: GetDocumentSummaryTool =====


class TestGetDocumentSummaryTool:
    """Test GetDocumentSummaryTool (get document summary)."""

    def test_get_document_summary_found(self, mock_vector_store, mock_embedder):
        """Test getting document summary."""
        tool = GetDocumentSummaryTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="GRI_306")

        assert result.success is True
        assert result.data["document_id"] == "GRI_306"
        assert "summary" in result.data
        assert len(result.data["summary"]) > 0
        assert result.metadata["summary_length"] > 0

    def test_get_document_summary_not_found(self, mock_vector_store, mock_embedder):
        """Test getting summary for nonexistent document."""
        tool = GetDocumentSummaryTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="NONEXISTENT")

        assert result.success is True
        assert result.data is None
        assert result.metadata["found"] is False

    def test_get_document_summary_with_faiss_store(self, mock_embedder):
        """Test getting summary from faiss_store attribute."""
        store = Mock()
        # Ensure metadata_layer1 doesn't exist on store itself
        del store.metadata_layer1

        faiss_store = Mock()
        faiss_store.metadata_layer1 = [
            {"document_id": "TEST", "content": "Test summary"},
        ]
        store.faiss_store = faiss_store

        tool = GetDocumentSummaryTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="TEST")

        assert result.success is True
        assert result.data["summary"] == "Test summary"


# ===== Tool 8: GetDocumentSectionsTool =====


class TestGetDocumentSectionsTool:
    """Test GetDocumentSectionsTool (list document sections)."""

    def test_get_document_sections_found(self, mock_vector_store, mock_embedder):
        """Test getting document sections."""
        tool = GetDocumentSectionsTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="GRI_306")

        assert result.success is True
        assert "sections" in result.data
        assert "count" in result.data
        assert len(result.data["sections"]) == 2

        # Check section structure
        section = result.data["sections"][0]
        assert "section_id" in section
        assert "section_title" in section
        assert "section_path" in section

        assert result.metadata["section_count"] == 2

    def test_get_document_sections_not_found(self, mock_vector_store, mock_embedder):
        """Test getting sections for nonexistent document."""
        tool = GetDocumentSectionsTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="NONEXISTENT")

        assert result.success is True
        assert result.data is None
        assert result.metadata["found"] is False

    def test_get_document_sections_sorted(self, mock_embedder):
        """Test that sections are sorted by section_id."""
        store = Mock()
        store.metadata_layer2 = [
            {"document_id": "TEST", "section_id": "sec_3", "section_title": "Section 3"},
            {"document_id": "TEST", "section_id": "sec_1", "section_title": "Section 1"},
            {"document_id": "TEST", "section_id": "sec_2", "section_title": "Section 2"},
        ]

        tool = GetDocumentSectionsTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="TEST")

        assert result.success is True
        # Check sorted order
        assert result.data["sections"][0]["section_id"] == "sec_1"
        assert result.data["sections"][1]["section_id"] == "sec_2"
        assert result.data["sections"][2]["section_id"] == "sec_3"


# ===== Tool 9: GetSectionDetailsTool =====


class TestGetSectionDetailsTool:
    """Test GetSectionDetailsTool (get section details)."""

    def test_get_section_details_found(self, mock_vector_store, mock_embedder):
        """Test getting section details."""
        tool = GetSectionDetailsTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="GRI_306", section_id="sec_3.2")

        assert result.success is True
        assert result.data["document_id"] == "GRI_306"
        assert result.data["section_id"] == "sec_3.2"
        assert "section_title" in result.data
        assert "summary" in result.data
        assert "chunk_count" in result.data
        assert result.data["chunk_count"] == 3  # 3 chunks in sec_3.2

    def test_get_section_details_not_found(self, mock_vector_store, mock_embedder):
        """Test getting details for nonexistent section."""
        tool = GetSectionDetailsTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="GRI_306", section_id="NONEXISTENT")

        assert result.success is True
        assert result.data is None
        assert result.metadata["found"] is False

    def test_get_section_details_with_chunk_count(self, mock_embedder):
        """Test chunk count calculation."""
        store = Mock()
        store.metadata_layer2 = [
            {
                "document_id": "TEST",
                "section_id": "sec_1",
                "section_title": "Section 1",
                "content": "Summary",
            }
        ]
        store.metadata_layer3 = [
            {"document_id": "TEST", "section_id": "sec_1"},
            {"document_id": "TEST", "section_id": "sec_1"},
            {"document_id": "TEST", "section_id": "sec_2"},  # Different section
        ]

        tool = GetSectionDetailsTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="TEST", section_id="sec_1")

        assert result.success is True
        assert result.data["chunk_count"] == 2


# ===== Tool 10: GetDocumentMetadataTool =====


class TestGetDocumentMetadataTool:
    """Test GetDocumentMetadataTool (comprehensive metadata)."""

    def test_get_document_metadata_found(self, mock_vector_store, mock_embedder):
        """Test getting document metadata."""
        tool = GetDocumentMetadataTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="GRI_306")

        assert result.success is True
        assert result.data["document_id"] == "GRI_306"
        assert "summary" in result.data
        assert "section_count" in result.data
        assert "chunk_count" in result.data
        assert "estimated_chars" in result.data
        assert "estimated_words" in result.data
        assert "sections" in result.data

        assert result.data["section_count"] == 2
        assert result.data["chunk_count"] == 3

    def test_get_document_metadata_not_found(self, mock_embedder):
        """Test getting metadata for nonexistent document."""
        store = Mock()
        store.metadata_layer1 = []
        store.metadata_layer2 = []
        store.metadata_layer3 = []

        tool = GetDocumentMetadataTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="NONEXISTENT")

        assert result.success is True
        assert result.data is None
        assert result.metadata["found"] is False

    def test_get_document_metadata_estimates(self, mock_embedder):
        """Test character and word count estimates."""
        store = Mock()
        store.metadata_layer1 = [{"document_id": "TEST", "content": "Summary"}]
        store.metadata_layer2 = [{"document_id": "TEST", "section_title": "Section 1"}]
        store.metadata_layer3 = [
            {"document_id": "TEST", "content": "A" * 100},  # 100 chars
            {"document_id": "TEST", "content": "B" * 200},  # 200 chars
        ]

        tool = GetDocumentMetadataTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute(document_id="TEST")

        assert result.success is True
        assert result.data["estimated_chars"] == 300
        assert result.data["estimated_words"] == 60  # 300 / 5


# ===== Tool 11: GetChunkContextTool =====


class TestGetChunkContextTool:
    """Test GetChunkContextTool (get chunk with surrounding context)."""

    def test_get_chunk_context_found(self, mock_vector_store, mock_embedder, mock_config):
        """Test getting chunk context."""
        tool = GetChunkContextTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=mock_config,
        )

        result = tool.execute(chunk_id="chunk_002")

        assert result.success is True
        assert "target_chunk" in result.data
        assert "context_before" in result.data
        assert "context_after" in result.data
        assert "context_window" in result.data

        assert result.data["target_chunk"]["chunk_id"] == "chunk_002"
        assert result.data["context_window"] == 2
        assert result.metadata["has_context"] is True

    def test_get_chunk_context_not_found(self, mock_vector_store, mock_embedder, mock_config):
        """Test getting context for nonexistent chunk."""
        tool = GetChunkContextTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=mock_config,
        )

        result = tool.execute(chunk_id="NONEXISTENT")

        assert result.success is False
        assert result.data is None
        assert "not found" in result.error.lower()

    def test_get_chunk_context_with_window(self, mock_embedder):
        """Test context window behavior."""
        store = Mock()
        store.metadata_layer3 = [
            {
                "chunk_id": "chunk_1",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 1",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 2",
            },
            {
                "chunk_id": "chunk_3",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 3",
            },
            {
                "chunk_id": "chunk_4",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 4",
            },
            {
                "chunk_id": "chunk_5",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 5",
            },
        ]

        config = Mock()
        config.context_window = 1  # 1 chunk before and after

        tool = GetChunkContextTool(
            vector_store=store,
            embedder=mock_embedder,
            config=config,
        )

        result = tool.execute(chunk_id="chunk_3")

        assert result.success is True
        assert len(result.data["context_before"]) == 1  # chunk_2
        assert len(result.data["context_after"]) == 1  # chunk_4
        assert result.data["context_before"][0]["chunk_id"] == "chunk_2"
        assert result.data["context_after"][0]["chunk_id"] == "chunk_4"

    def test_get_chunk_context_at_boundary(self, mock_embedder):
        """Test context at document boundaries (first/last chunk)."""
        store = Mock()
        store.metadata_layer3 = [
            {
                "chunk_id": "chunk_1",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 1",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "TEST",
                "section_id": "sec_1",
                "content": "Chunk 2",
            },
        ]

        config = Mock()
        config.context_window = 2

        tool = GetChunkContextTool(
            vector_store=store,
            embedder=mock_embedder,
            config=config,
        )

        # First chunk - no context before
        result = tool.execute(chunk_id="chunk_1")
        assert result.success is True
        assert len(result.data["context_before"]) == 0
        assert len(result.data["context_after"]) == 1

        # Last chunk - no context after
        result = tool.execute(chunk_id="chunk_2")
        assert result.success is True
        assert len(result.data["context_before"]) == 1
        assert len(result.data["context_after"]) == 0


# ===== Tool 12: ListAvailableToolsTool =====


class TestListAvailableToolsTool:
    """Test ListAvailableToolsTool (list all tools)."""

    def test_list_available_tools(self, mock_embedder):
        """Test listing available tools."""
        store = Mock()
        tool = ListAvailableToolsTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert "tools" in result.data
        assert "total_count" in result.data
        assert "best_practices" in result.data

        # Result structure should be correct even if empty
        # (registry may be empty in test environment)
        assert isinstance(result.data["tools"], list)

        # If we have tools, check their structure
        if len(result.data["tools"]) > 0:
            tool_info = result.data["tools"][0]
            assert "name" in tool_info
            assert "description" in tool_info
            assert "parameters" in tool_info
            assert "tier" in tool_info
            assert "when_to_use" in tool_info

    def test_list_available_tools_includes_best_practices(self, mock_embedder):
        """Test that tool list includes best practices."""
        store = Mock()
        tool = ListAvailableToolsTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert "best_practices" in result.data
        assert "general" in result.data["best_practices"]
        assert "selection_strategy" in result.data["best_practices"]

    def test_list_available_tools_metadata(self, mock_embedder):
        """Test tool list metadata (tier counts)."""
        store = Mock()
        tool = ListAvailableToolsTool(
            vector_store=store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        assert result.success is True
        assert "total_tools" in result.metadata
        assert "tier1_count" in result.metadata
        assert "tier2_count" in result.metadata
        assert "tier3_count" in result.metadata


# ===== Input Validation Tests =====


class TestInputValidation:
    """Test Pydantic input validation for all tools."""

    def test_search_invalid_k(self, mock_vector_store, mock_embedder):
        """Test that invalid k parameter is rejected."""
        mock_config = Mock()
        mock_config.tool_config = Mock()
        mock_config.tool_config.query_expansion_provider = "openai"
        mock_config.tool_config.query_expansion_model = "gpt-5-nano"

        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=mock_config,
        )

        # k too small (< 1)
        result = tool.execute(query="test", k=0, num_expands=1)
        assert result.success is False
        assert "Invalid input" in result.error

    def test_search_missing_query(self, mock_vector_store, mock_embedder):
        """Test that missing required parameter is rejected."""
        mock_config = Mock()
        mock_config.tool_config = Mock()
        mock_config.tool_config.query_expansion_provider = "openai"
        mock_config.tool_config.query_expansion_model = "gpt-5-nano"

        tool = SearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=mock_config,
        )

        # Missing required 'query' parameter
        result = tool.execute(k=6, num_expands=1)
        assert result.success is False
        assert "Invalid input" in result.error

    def test_document_search_extra_params_rejected(self, mock_vector_store, mock_embedder):
        """Test that extra parameters are rejected (Pydantic extra='forbid')."""
        tool = DocumentSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(
            query="test", document_id="TEST", k=6, extra_param="should_fail"  # Extra parameter
        )

        assert result.success is False
        assert "Invalid input" in result.error

    def test_k_parameter_clamping(self, mock_vector_store, mock_embedder):
        """Test that k parameter exceeding Pydantic limit is rejected."""
        tool = SimpleSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        # k too large (exceeds Pydantic le=20 constraint)
        result = tool.execute(query="test", k=100)

        # Should fail validation (Pydantic rejects before internal clamping)
        assert result.success is False
        assert "Invalid input" in result.error


# ===== Error Handling Tests =====


class TestErrorHandling:
    """Test error handling across tools."""

    def test_embedder_failure(self, mock_vector_store):
        """Test handling of embedder failures."""
        bad_embedder = Mock()
        bad_embedder.embed_texts = Mock(side_effect=RuntimeError("Embedding failed"))
        bad_embedder.dimensions = 3072

        tool = SimpleSearchTool(
            vector_store=mock_vector_store,
            embedder=bad_embedder,
        )

        result = tool.execute(query="test", k=6)

        assert result.success is False
        assert "System error" in result.error
        assert result.metadata["error_type"] == "system"

    def test_vector_store_failure(self, mock_embedder):
        """Test handling of vector store failures."""
        bad_store = Mock()
        bad_store.hierarchical_search = Mock(side_effect=RuntimeError("Search failed"))

        tool = SimpleSearchTool(
            vector_store=bad_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="test", k=6)

        assert result.success is False
        assert result.metadata["error_type"] == "system"

    def test_missing_metadata_attribute(self, mock_embedder):
        """Test graceful handling when metadata attributes are missing."""
        incomplete_store = Mock()
        # Tool implementation uses hasattr() which returns False for missing attrs
        # This causes the tool to skip that branch and return empty results
        del incomplete_store.metadata_layer1
        del incomplete_store.faiss_store

        tool = GetDocumentListTool(
            vector_store=incomplete_store,
            embedder=mock_embedder,
        )

        result = tool.execute()

        # Tool gracefully handles missing attributes by returning empty list
        # This is the actual implementation behavior (uses hasattr checks)
        assert result.success is True
        assert result.data["count"] == 0
        assert result.data["documents"] == []


# ===== ToolResult Structure Tests =====


class TestToolResultStructure:
    """Test ToolResult structure consistency."""

    def test_successful_result_structure(self, mock_vector_store, mock_embedder):
        """Test that successful results have correct structure."""
        tool = SimpleSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="test", k=6)

        assert hasattr(result, "success")
        assert hasattr(result, "data")
        assert hasattr(result, "error")
        assert hasattr(result, "metadata")
        assert hasattr(result, "citations")
        assert hasattr(result, "execution_time_ms")

        assert result.success is True
        assert result.error is None
        assert isinstance(result.metadata, dict)
        assert isinstance(result.citations, list)
        assert result.execution_time_ms >= 0

    def test_failed_result_structure(self, mock_vector_store):
        """Test that failed results have correct structure."""
        bad_embedder = Mock()
        bad_embedder.embed_texts = Mock(side_effect=RuntimeError("Test error"))
        bad_embedder.dimensions = 3072

        tool = SimpleSearchTool(
            vector_store=mock_vector_store,
            embedder=bad_embedder,
        )

        result = tool.execute(query="test", k=6)

        assert result.success is False
        assert result.error is not None
        assert isinstance(result.error, str)
        assert len(result.error) > 0
        assert result.execution_time_ms >= 0

    def test_metadata_contains_tool_info(self, mock_vector_store, mock_embedder):
        """Test that metadata always contains tool information."""
        tool = SimpleSearchTool(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        result = tool.execute(query="test", k=6)

        assert "tool_name" in result.metadata
        assert "tier" in result.metadata
        assert result.metadata["tool_name"] == "simple_search"
        assert result.metadata["tier"] == 1
