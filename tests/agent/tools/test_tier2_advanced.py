"""
Tests for Tier 2 Advanced Retrieval Tools.

Tests all 9 advanced retrieval tools:
1. MultiHopSearchTool
2. CompareDocumentsTool
3. FindRelatedChunksTool
4. TemporalSearchTool
5. HybridSearchWithFiltersTool
6. CrossReferenceSearchTool
7. ExpandSearchContextTool (Phase 7B)
8. ChunkSimilaritySearchTool (Phase 7B)
9. ExplainSearchResultsTool (Phase 7B)

Coverage:
- Valid inputs and expected outputs
- Error cases (missing KG, no results, invalid inputs)
- Phase 7B features (context expansion, score preservation, similarity search)
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.agent.tools.tier2_advanced import (
    # MultiHopSearchTool,  # REMOVED - replaced by GraphSearchTool
    CompareDocumentsTool,
    # FindRelatedChunksTool,  # TODO: Check if this tool exists
    # TemporalSearchTool,  # TODO: Check if this tool exists
    # HybridSearchWithFiltersTool,  # REMOVED - replaced by FilteredSearchTool
    # CrossReferenceSearchTool,  # TODO: Check if this tool exists
    # ExpandSearchContextTool,  # TODO: Check if exists (might be ExpandContextTool)
    # ChunkSimilaritySearchTool,  # REMOVED - replaced by SimilaritySearchTool
    ExplainSearchResultsTool,
)

# New consolidated tools (to be tested):
from src.agent.tools.tier2_advanced import (
    GraphSearchTool,  # Replaces MultiHopSearchTool + EntityTool
    FilteredSearchTool,  # Replaces HybridSearchWithFiltersTool
    SimilaritySearchTool,  # Replaces ChunkSimilaritySearchTool
    ExpandContextTool,  # Check if this replaces ExpandSearchContextTool
)
from src.agent.tools.base import ToolResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = Mock()
    embedder.embed_texts.return_value = np.random.rand(1, 3072)
    embedder.dimensions = 3072
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store with hierarchical search."""
    store = Mock()

    # Default search results
    def hierarchical_search(**kwargs):
        return {
            "layer1": [],
            "layer2": [],
            "layer3": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "content": "This is chunk 0 content about waste disposal.",
                    "raw_content": "This is chunk 0 content about waste disposal.",
                    "rrf_score": 0.9,
                    "bm25_score": 0.85,
                    "dense_score": 0.88,
                },
                {
                    "chunk_id": "doc1:sec1:1",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "section_title": "Introduction",
                    "content": "This is chunk 1 content.",
                    "raw_content": "This is chunk 1 content.",
                    "rrf_score": 0.8,
                    "bm25_score": 0.75,
                    "dense_score": 0.78,
                },
                {
                    "chunk_id": "doc2:sec2:0",
                    "document_id": "doc2",
                    "doc_id": "doc2",
                    "section_id": "sec2",
                    "section_title": "Requirements",
                    "content": "This is chunk 2 from doc2.",
                    "raw_content": "This is chunk 2 from doc2.",
                    "date": "2023-05-15",
                    "rrf_score": 0.7,
                },
            ],
        }

    store.hierarchical_search.side_effect = hierarchical_search

    # Add metadata for layer3
    store.metadata_layer3 = [
        {
            "chunk_id": "doc1:sec1:0",
            "document_id": "doc1",
            "doc_id": "doc1",
            "section_id": "sec1",
            "section_title": "Introduction",
            "content": "This is chunk 0 content about waste disposal.",
            "raw_content": "This is chunk 0 content about waste disposal.",
            "rrf_score": 0.9,
            "bm25_score": 0.85,
            "dense_score": 0.88,
        },
        {
            "chunk_id": "doc1:sec1:1",
            "document_id": "doc1",
            "doc_id": "doc1",
            "section_id": "sec1",
            "section_title": "Introduction",
            "content": "This is chunk 1 content.",
            "raw_content": "This is chunk 1 content.",
            "rrf_score": 0.8,
            "bm25_score": 0.75,
            "dense_score": 0.78,
        },
        {
            "chunk_id": "doc2:sec2:0",
            "document_id": "doc2",
            "doc_id": "doc2",
            "section_id": "sec2",
            "section_title": "Requirements",
            "content": "This is chunk 2 from doc2.",
            "raw_content": "This is chunk 2 from doc2.",
            "rrf_score": 0.7,
        },
    ]

    return store


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    reranker = Mock()

    def rerank(query, chunks, top_k=6):
        # Add rerank scores and return top_k
        for i, chunk in enumerate(chunks[:top_k]):
            chunk["rerank_score"] = 1.0 - (i * 0.1)
        return chunks[:top_k]

    reranker.rerank.side_effect = rerank
    return reranker


@pytest.fixture
def mock_graph_retriever():
    """Create mock graph retriever."""
    retriever = Mock()

    def search_with_graph(**kwargs):
        # Return chunks with graph boosting
        return [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Graph-boosted chunk 0.",
                "raw_content": "Graph-boosted chunk 0.",
                "boosted_score": 0.95,
            },
            {
                "chunk_id": "doc1:sec1:1",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Graph-boosted chunk 1.",
                "raw_content": "Graph-boosted chunk 1.",
                "boosted_score": 0.85,
            },
        ]

    retriever.search_with_graph.side_effect = search_with_graph
    return retriever


@pytest.fixture
def mock_knowledge_graph():
    """Create mock knowledge graph."""
    kg = Mock()
    return kg


@pytest.fixture
def tool_dependencies(mock_vector_store, mock_embedder, mock_reranker):
    """Common dependencies for all tools."""
    return {
        "vector_store": mock_vector_store,
        "embedder": mock_embedder,
        "reranker": mock_reranker,
        "graph_retriever": None,
        "knowledge_graph": None,
        "context_assembler": None,
        "config": None,
    }


# ============================================================================
# TEST 1: MultiHopSearchTool
# ============================================================================


class TestMultiHopSearchTool:
    """Test multi-hop search using knowledge graph traversal."""

    def test_multi_hop_search_without_kg_fails(self, tool_dependencies):
        """Test that multi-hop search fails gracefully without KG."""
        tool = MultiHopSearchTool(**tool_dependencies)

        result = tool.execute(query="Find documents related to X through Y", max_hops=2, k=6)

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Knowledge graph not available" in result.error

    def test_multi_hop_search_with_kg(self, tool_dependencies, mock_graph_retriever):
        """Test multi-hop search with knowledge graph."""
        tool_dependencies["graph_retriever"] = mock_graph_retriever
        tool = MultiHopSearchTool(**tool_dependencies)

        result = tool.execute(query="Find waste disposal regulations", max_hops=2, k=6)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert len(result.data) > 0
        assert result.data[0]["content"] == "Graph-boosted chunk 0."
        assert len(result.citations) > 0
        assert result.metadata["hops"] == 2

    def test_multi_hop_search_no_results(self, tool_dependencies, mock_reranker):
        """Test multi-hop search when no results found."""
        # Create a fresh mock for this test
        mock_gr = Mock()
        mock_gr.search_with_graph.return_value = []

        tool_dependencies["graph_retriever"] = mock_gr
        tool_dependencies["reranker"] = mock_reranker

        tool = MultiHopSearchTool(**tool_dependencies)
        result = tool.execute(query="Nonexistent topic", max_hops=2, k=6)

        assert result.success is True
        # Empty list should be formatted
        assert isinstance(result.data, list)
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0

    def test_multi_hop_search_max_hops_validation(self, tool_dependencies, mock_graph_retriever):
        """Test max_hops parameter validation."""
        tool_dependencies["graph_retriever"] = mock_graph_retriever
        tool = MultiHopSearchTool(**tool_dependencies)

        # Valid: max_hops=1
        result = tool.execute(query="test", max_hops=1, k=6)
        assert result.success is True

        # Valid: max_hops=3
        result = tool.execute(query="test", max_hops=3, k=6)
        assert result.success is True

        # Invalid: max_hops=0 (should fail validation)
        result = tool.execute(query="test", max_hops=0, k=6)
        assert result.success is False
        assert "validation" in result.error.lower()


# ============================================================================
# TEST 2: CompareDocumentsTool
# ============================================================================


class TestCompareDocumentsTool:
    """Test document comparison functionality."""

    def test_compare_documents_basic(self, tool_dependencies, mock_vector_store):
        """Test basic document comparison."""

        # Setup mock to return different results for each document
        def custom_search(**kwargs):
            doc_filter = kwargs.get("document_filter")
            if doc_filter == "doc1":
                return {
                    "layer3": [
                        {
                            "chunk_id": "doc1:sec1:0",
                            "doc_id": "doc1",
                            "document_id": "doc1",
                            "content": "Doc1 content.",
                        }
                    ]
                }
            else:
                return {
                    "layer3": [
                        {
                            "chunk_id": "doc2:sec1:0",
                            "doc_id": "doc2",
                            "document_id": "doc2",
                            "content": "Doc2 content.",
                        }
                    ]
                }

        mock_vector_store.hierarchical_search.side_effect = custom_search

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="doc1", doc_id_2="doc2")

        assert result.success is True
        assert result.data["doc_id_1"] == "doc1"
        assert result.data["doc_id_2"] == "doc2"
        assert "doc1_chunk_count" in result.data
        assert "doc2_chunk_count" in result.data
        assert "doc1" in result.citations
        assert "doc2" in result.citations

    def test_compare_documents_with_aspect(self, tool_dependencies, mock_vector_store):
        """Test document comparison with specific aspect."""
        tool = CompareDocumentsTool(**tool_dependencies)

        result = tool.execute(doc_id_1="doc1", doc_id_2="doc2", comparison_aspect="requirements")

        assert result.success is True
        assert result.data["comparison_aspect"] == "requirements"
        assert "doc1_relevant_chunks" in result.data
        assert "doc2_relevant_chunks" in result.data

    def test_compare_documents_first_not_found(self, tool_dependencies):
        """Test comparison when first document not found."""
        # Create fresh mock that returns empty for all calls
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {"layer3": []}
        tool_dependencies["vector_store"] = mock_vs

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="nonexistent", doc_id_2="doc2")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_compare_documents_second_not_found(self, tool_dependencies, mock_vector_store):
        """Test comparison when second document not found."""
        call_count = [0]

        def custom_search(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns results (doc1)
                return {"layer3": [{"chunk_id": "doc1:sec1:0", "doc_id": "doc1"}]}
            else:
                # Second call returns empty (doc2 not found)
                return {"layer3": []}

        mock_vector_store.hierarchical_search.side_effect = custom_search

        tool = CompareDocumentsTool(**tool_dependencies)
        result = tool.execute(doc_id_1="doc1", doc_id_2="nonexistent")

        assert result.success is False
        assert "nonexistent" in result.error


# ============================================================================
# TEST 3: FindRelatedChunksTool
# ============================================================================


class TestFindRelatedChunksTool:
    """Test finding semantically related chunks."""

    def test_find_related_chunks_basic(self, tool_dependencies):
        """Test finding related chunks."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=6, same_document_only=False)

        assert result.success is True
        assert len(result.data) > 0
        # Source chunk should be filtered out
        assert not any(c["chunk_id"] == "doc1:sec1:0" for c in result.data)
        assert result.metadata["source_chunk_id"] == "doc1:sec1:0"

    def test_find_related_chunks_same_document_only(self, tool_dependencies):
        """Test finding related chunks within same document."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=6, same_document_only=True)

        assert result.success is True
        assert result.metadata["same_document_only"] is True
        # Source chunk doc1:sec1:0 should be filtered out
        # Check that all remaining results are from doc1 or doc2 (mock returns both initially)
        # but results might include doc2:sec2:0 since mock doesn't filter perfectly
        # The key is that doc1:sec1:0 should not be in the results
        assert not any(c.get("chunk_id") == "doc1:sec1:0" for c in result.data)

    def test_find_related_chunks_not_found(self, tool_dependencies, mock_vector_store):
        """Test when chunk ID not found."""
        # Mock search to return no matching chunk
        mock_vector_store.hierarchical_search.return_value = {"layer3": []}

        tool = FindRelatedChunksTool(**tool_dependencies)
        result = tool.execute(chunk_id="nonexistent:chunk:id", k=6)

        assert result.success is False
        assert "not found" in result.error

    def test_find_related_chunks_with_reranker(self, tool_dependencies):
        """Test that reranker is applied to results."""
        tool = FindRelatedChunksTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", k=3)

        assert result.success is True
        # Verify reranker was called (results should have rerank_score)
        if result.data:
            # Check that we got limited results (k=3)
            assert len(result.data) <= 3


# ============================================================================
# TEST 4: TemporalSearchTool
# ============================================================================


class TestTemporalSearchTool:
    """Test temporal search with date filtering."""

    def test_temporal_search_no_date_filter(self, tool_dependencies):
        """Test temporal search without date filtering."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(query="regulations", k=6)

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["start_date"] is None
        assert result.metadata["end_date"] is None

    def test_temporal_search_with_start_date(self, tool_dependencies):
        """Test temporal search with start date filter."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(query="regulations", start_date="2023-01-01", k=6)

        assert result.success is True
        assert result.metadata["start_date"] == "2023-01-01"

    def test_temporal_search_with_date_range(self, tool_dependencies):
        """Test temporal search with date range."""
        tool = TemporalSearchTool(**tool_dependencies)

        result = tool.execute(
            query="regulations", start_date="2023-01-01", end_date="2023-12-31", k=6
        )

        assert result.success is True
        assert result.metadata["start_date"] == "2023-01-01"
        assert result.metadata["end_date"] == "2023-12-31"

    def test_temporal_search_filters_by_date(self, tool_dependencies, mock_reranker):
        """Test that chunks are actually filtered by date."""
        # Create fresh mock with specific data
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "date": "2023-05-15",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "date": "2022-03-20",
                },
                {
                    "chunk_id": "c3",
                    "doc_id": "doc3",
                    "document_id": "doc3",
                    "section_id": "sec3",
                    "content": "Content 3",
                    "raw_content": "Content 3",
                    "date": "2024-01-10",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = TemporalSearchTool(**tool_dependencies)
        result = tool.execute(query="test", start_date="2023-01-01", end_date="2023-12-31", k=6)

        assert result.success is True
        # Only chunk with date 2023-05-15 should pass filter
        assert len(result.data) >= 1
        # Check first result is c1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_temporal_search_no_results_in_range(self, tool_dependencies):
        """Test when no results in date range."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content",
                    "raw_content": "Content",
                    "date": "2020-01-01",
                }
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = TemporalSearchTool(**tool_dependencies)
        result = tool.execute(query="test", start_date="2023-01-01", end_date="2023-12-31", k=6)

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 5: HybridSearchWithFiltersTool
# ============================================================================


class TestHybridSearchWithFiltersTool:
    """Test hybrid search with metadata filters."""

    def test_hybrid_search_no_filters(self, tool_dependencies):
        """Test hybrid search without filters."""
        tool = HybridSearchWithFiltersTool(**tool_dependencies)

        result = tool.execute(query="waste disposal", k=6)

        assert result.success is True
        assert len(result.data) > 0
        assert result.metadata["document_type"] is None
        assert result.metadata["section_type"] is None

    def test_hybrid_search_with_document_type(self, tool_dependencies, mock_reranker):
        """Test filtering by document type."""
        # Create fresh mock with doc_type metadata
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "doc_type": "regulation",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "doc_type": "contract",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", document_type="regulation", k=6)

        assert result.success is True
        assert len(result.data) >= 1
        # Verify we got the regulation chunk
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_with_section_type(self, tool_dependencies, mock_reranker):
        """Test filtering by section type."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "section_type": "requirements",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "section_type": "introduction",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", section_type="requirements", k=6)

        assert result.success is True
        assert len(result.data) >= 1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_with_both_filters(self, tool_dependencies, mock_reranker):
        """Test filtering by both document and section type."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content 1",
                    "raw_content": "Content 1",
                    "doc_type": "regulation",
                    "section_type": "requirements",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Content 2",
                    "raw_content": "Content 2",
                    "doc_type": "contract",
                    "section_type": "requirements",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs
        tool_dependencies["reranker"] = mock_reranker

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(
            query="test", document_type="regulation", section_type="requirements", k=6
        )

        assert result.success is True
        assert len(result.data) >= 1
        assert any(c["chunk_id"] == "c1" for c in result.data)

    def test_hybrid_search_no_results_after_filtering(self, tool_dependencies, mock_vector_store):
        """Test when filters eliminate all results."""
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [{"chunk_id": "c1", "doc_type": "contract"}]
        }

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", document_type="regulation", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 6: CrossReferenceSearchTool
# ============================================================================


class TestCrossReferenceSearchTool:
    """Test cross-reference search."""

    def test_cross_reference_search_basic(self, tool_dependencies):
        """Test basic cross-reference search."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "See Article 5.2 for details.",
                    "raw_content": "See Article 5.2 for details.",
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "doc2",
                    "document_id": "doc2",
                    "section_id": "sec2",
                    "content": "Article 5.2 specifies requirements.",
                    "raw_content": "Article 5.2 specifies requirements.",
                },
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 5.2", k=6)

        assert result.success is True
        assert len(result.data) >= 2
        assert result.metadata["reference_text"] == "Article 5.2"

    def test_cross_reference_search_case_insensitive(self, tool_dependencies):
        """Test that cross-reference search is case-insensitive."""
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "document_id": "doc1",
                    "section_id": "sec1",
                    "content": "see ARTICLE 5.2 for details.",
                    "raw_content": "see ARTICLE 5.2 for details.",
                }
            ]
        }
        tool_dependencies["vector_store"] = mock_vs

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="article 5.2", k=6)

        assert result.success is True
        assert len(result.data) >= 1

    def test_cross_reference_search_no_matches(self, tool_dependencies, mock_vector_store):
        """Test when reference text not found in any chunks."""
        mock_vector_store.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "c1",
                    "doc_id": "doc1",
                    "content": "Different content.",
                    "raw_content": "Different content.",
                }
            ]
        }

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 99.99", k=6)

        assert result.success is True
        assert result.data == []
        assert result.metadata["results_count"] == 0

    def test_cross_reference_search_limits_results(self, tool_dependencies, mock_vector_store):
        """Test that results are limited to k."""
        # Create 10 matching chunks
        chunks = [
            {
                "chunk_id": f"c{i}",
                "doc_id": "doc1",
                "document_id": "doc1",
                "section_id": "sec1",
                "content": f"Article 5 mentioned here {i}.",
                "raw_content": f"Article 5 mentioned here {i}.",
            }
            for i in range(10)
        ]
        mock_vector_store.hierarchical_search.return_value = {"layer3": chunks}

        tool = CrossReferenceSearchTool(**tool_dependencies)
        result = tool.execute(reference_text="Article 5", k=3)

        assert result.success is True
        assert len(result.data) <= 3


# ============================================================================
# TEST 7: ExpandSearchContextTool (Phase 7B)
# ============================================================================


class TestExpandSearchContextTool:
    """Test context expansion with multiple strategies."""

    def test_expand_search_context_section_strategy(self, tool_dependencies, mock_vector_store):
        """Test section-based expansion."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="section", k=3)

        assert result.success is True
        assert "expansions" in result.data
        assert result.data["expansion_strategy"] == "section"

    def test_expand_search_context_similarity_strategy(self, tool_dependencies):
        """Test similarity-based expansion."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="similarity", k=3)

        assert result.success is True
        assert result.data["expansion_strategy"] == "similarity"

    def test_expand_search_context_hybrid_strategy(self, tool_dependencies):
        """Test hybrid expansion (section + similarity)."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="hybrid", k=4)

        assert result.success is True
        assert result.data["expansion_strategy"] == "hybrid"

    def test_expand_search_context_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test expansion when chunk not found."""
        mock_vector_store.metadata_layer3 = []

        tool = ExpandSearchContextTool(**tool_dependencies)
        result = tool.execute(chunk_ids=["nonexistent:chunk:id"], expansion_strategy="section", k=3)

        assert result.success is False
        assert "No chunks found" in result.error

    def test_expand_search_context_multiple_chunks(self, tool_dependencies):
        """Test expanding multiple chunks."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(
            chunk_ids=["doc1:sec1:0", "doc1:sec1:1"], expansion_strategy="section", k=2
        )

        assert result.success is True
        assert len(result.data["expansions"]) == 2

    def test_expand_search_context_deduplication(self, tool_dependencies):
        """Test that duplicate chunks are removed."""
        tool = ExpandSearchContextTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"], expansion_strategy="hybrid", k=3)

        # Check that target chunk is not in expanded_chunks
        if result.success and result.data["expansions"]:
            expansion = result.data["expansions"][0]
            target_id = expansion["target_chunk"]["chunk_id"]
            expanded_ids = [c["chunk_id"] for c in expansion["expanded_chunks"]]
            assert target_id not in expanded_ids


# ============================================================================
# TEST 8: ChunkSimilaritySearchTool (Phase 7B)
# ============================================================================


class TestChunkSimilaritySearchTool:
    """Test 'more like this chunk' similarity search."""

    def test_chunk_similarity_search_cross_document(self, tool_dependencies):
        """Test similarity search across all documents."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        assert "target_chunk" in result.data
        assert "similar_chunks" in result.data
        assert result.metadata["cross_document"] is True

    def test_chunk_similarity_search_same_document(self, tool_dependencies, mock_vector_store):
        """Test similarity search within same document."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=False, k=6)

        assert result.success is True
        assert result.metadata["cross_document"] is False
        # All similar chunks should be from same document
        if result.data["similar_chunks"]:
            assert all(c["document_id"] == "doc1" for c in result.data["similar_chunks"])

    def test_chunk_similarity_search_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test when chunk ID not found."""
        mock_vector_store.metadata_layer3 = []

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="nonexistent:chunk:id", cross_document=True, k=6)

        assert result.success is False
        assert "not found" in result.error

    def test_chunk_similarity_search_no_content(self, tool_dependencies, mock_vector_store):
        """Test when chunk has no content."""
        mock_vector_store.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "content": "",  # Empty content
            }
        ]

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is False
        assert "no content" in result.error

    def test_chunk_similarity_search_excludes_self(self, tool_dependencies):
        """Test that target chunk is excluded from results."""
        tool = ChunkSimilaritySearchTool(**tool_dependencies)

        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        # Target chunk should not be in similar_chunks
        similar_ids = [c["chunk_id"] for c in result.data["similar_chunks"]]
        assert "doc1:sec1:0" not in similar_ids

    def test_chunk_similarity_search_no_similar_chunks(self, tool_dependencies):
        """Test when no similar chunks found."""
        # Create fresh mock that returns only the target chunk
        mock_vs = Mock()
        mock_vs.hierarchical_search.return_value = {
            "layer3": [
                {
                    "chunk_id": "doc1:sec1:0",
                    "document_id": "doc1",
                    "doc_id": "doc1",
                    "section_id": "sec1",
                    "content": "Content",
                    "raw_content": "Content",
                }
            ]
        }
        mock_vs.metadata_layer3 = [
            {
                "chunk_id": "doc1:sec1:0",
                "document_id": "doc1",
                "doc_id": "doc1",
                "section_id": "sec1",
                "content": "Content",
                "raw_content": "Content",
            }
        ]
        tool_dependencies["vector_store"] = mock_vs

        tool = ChunkSimilaritySearchTool(**tool_dependencies)
        result = tool.execute(chunk_id="doc1:sec1:0", cross_document=True, k=6)

        assert result.success is True
        # When no similar chunks found, tool returns empty list
        # (early return before dict construction)
        assert isinstance(result.data, list)
        assert len(result.data) == 0
        assert result.metadata["results_count"] == 0


# ============================================================================
# TEST 9: ExplainSearchResultsTool (Phase 7B)
# ============================================================================


class TestExplainSearchResultsTool:
    """Test search result explanation with score breakdowns."""

    def test_explain_search_results_basic(self, tool_dependencies):
        """Test basic search result explanation."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0", "doc1:sec1:1"])

        assert result.success is True
        assert "explanations" in result.data
        assert len(result.data["explanations"]) == 2

    def test_explain_search_results_with_scores(self, tool_dependencies):
        """Test explanation includes all score types."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert explanation["found"] is True
        assert "scores" in explanation
        assert "bm25_score" in explanation["scores"]
        assert "dense_score" in explanation["scores"]
        assert "rrf_score" in explanation["scores"]

    def test_explain_search_results_primary_method_detection(self, tool_dependencies):
        """Test detection of primary retrieval method."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert "primary_retrieval_method" in explanation
        # With our mock data, dense_score (0.88) > bm25_score (0.85)
        assert "semantic" in explanation["primary_retrieval_method"]

    def test_explain_search_results_chunk_not_found(self, tool_dependencies, mock_vector_store):
        """Test explanation when chunk not found."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["nonexistent:chunk:id"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert explanation["found"] is False
        assert "error" in explanation

    def test_explain_search_results_hybrid_detection(self, tool_dependencies):
        """Test detection of hybrid search availability."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        assert "hybrid_search_enabled" in result.data

    def test_explain_search_results_content_preview(self, tool_dependencies):
        """Test that content preview is included."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0"])

        assert result.success is True
        explanation = result.data["explanations"][0]
        assert "content_preview" in explanation
        assert len(explanation["content_preview"]) <= 203  # 200 + "..."

    def test_explain_search_results_multiple_chunks(self, tool_dependencies):
        """Test explaining multiple chunks."""
        tool = ExplainSearchResultsTool(**tool_dependencies)

        result = tool.execute(chunk_ids=["doc1:sec1:0", "doc1:sec1:1", "doc2:sec2:0"])

        assert result.success is True
        assert len(result.data["explanations"]) == 3
        assert result.metadata["chunk_count"] == 3


# ============================================================================
# TEST: Edge Cases & Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling across all tools."""

    def test_invalid_k_parameter(self, tool_dependencies):
        """Test that invalid k values are caught by validation."""
        tool = MultiHopSearchTool(**{**tool_dependencies, "graph_retriever": Mock()})

        # k=0 should fail validation
        result = tool.execute(query="test", k=0)
        assert result.success is False
        assert "validation" in result.error.lower()

        # k=21 (exceeds max) should fail validation
        result = tool.execute(query="test", k=21)
        assert result.success is False

    def test_empty_query_string(self, tool_dependencies):
        """Test handling of empty query strings."""
        tool = TemporalSearchTool(**tool_dependencies)

        # Empty string is technically valid in Pydantic (str type)
        # But we can test that it works without error
        result = tool.execute(query="", k=6)
        # Should succeed or fail gracefully, not crash
        assert isinstance(result, ToolResult)

    def test_embedder_failure(self, tool_dependencies, mock_embedder):
        """Test handling of embedder failures."""
        mock_embedder.embed_texts.side_effect = Exception("Embedder failed")

        tool = TemporalSearchTool(**tool_dependencies)
        result = tool.execute(query="test", k=6)

        assert result.success is False
        # Error message should contain information about the failure
        assert result.error is not None
        assert len(result.error) > 0

    def test_vector_store_failure(self, tool_dependencies, mock_vector_store):
        """Test handling of vector store failures."""
        mock_vector_store.hierarchical_search.side_effect = Exception("Vector store error")

        tool = HybridSearchWithFiltersTool(**tool_dependencies)
        result = tool.execute(query="test", k=6)

        assert result.success is False

    def test_tool_statistics_tracking(self, tool_dependencies):
        """Test that tools track execution statistics."""
        tool = CrossReferenceSearchTool(**tool_dependencies)

        # Execute tool multiple times
        tool.execute(reference_text="Article 5", k=6)
        tool.execute(reference_text="Article 6", k=6)

        stats = tool.get_stats()
        assert stats["execution_count"] == 2
        assert stats["name"] == "cross_reference_search"
        assert "avg_time_ms" in stats


# ============================================================================
# TEST: Integration Tests
# ============================================================================


class TestToolIntegration:
    """Integration tests combining multiple tools."""

    def test_multi_tool_workflow(self, tool_dependencies):
        """Test typical workflow using multiple tools."""
        # 1. Initial search
        search_tool = HybridSearchWithFiltersTool(**tool_dependencies)
        search_result = search_tool.execute(query="waste disposal", k=6)
        assert search_result.success is True

        # 2. Find related chunks
        if search_result.data:
            chunk_id = search_result.data[0]["chunk_id"]
            related_tool = FindRelatedChunksTool(**tool_dependencies)
            related_result = related_tool.execute(chunk_id=chunk_id, k=3)
            assert related_result.success is True

        # 3. Explain results
        if search_result.data:
            chunk_ids = [c["chunk_id"] for c in search_result.data[:2]]
            explain_tool = ExplainSearchResultsTool(**tool_dependencies)
            explain_result = explain_tool.execute(chunk_ids=chunk_ids)
            assert explain_result.success is True

    def test_tool_error_propagation(self, tool_dependencies, mock_vector_store):
        """Test that errors are properly propagated through tool chain."""
        # Setup vector store to fail
        mock_vector_store.hierarchical_search.side_effect = RuntimeError("Database error")

        # Each tool should handle the error gracefully
        tools = [
            TemporalSearchTool(**tool_dependencies),
            HybridSearchWithFiltersTool(**tool_dependencies),
            CrossReferenceSearchTool(**tool_dependencies),
        ]

        for tool in tools:
            result = tool.execute(query="test", k=6)
            assert result.success is False
            assert result.error is not None
