"""
Tests for PHASE 5C: Cross-Encoder Reranking

Tests cover:
1. Reranker initialization
2. Reranking with different models
3. Score calculation and ranking changes
4. Statistics and performance monitoring
5. Integration with hybrid search
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


# Test data fixtures
@pytest.fixture
def sample_candidates():
    """Sample candidate chunks from hybrid search."""
    return [
        {
            "chunk_id": "chunk_1",
            "content": "The waste disposal requirements specify proper handling of hazardous materials.",
            "rrf_score": 0.031,
            "document_id": "doc_1",
        },
        {
            "chunk_id": "chunk_2",
            "content": "Safety equipment must be worn at all times in the facility.",
            "rrf_score": 0.029,
            "document_id": "doc_1",
        },
        {
            "chunk_id": "chunk_3",
            "content": "Proper waste disposal procedures include segregation, labeling, and secure storage.",
            "rrf_score": 0.028,
            "document_id": "doc_1",
        },
        {
            "chunk_id": "chunk_4",
            "content": "The company provides training on environmental regulations.",
            "rrf_score": 0.025,
            "document_id": "doc_1",
        },
        {
            "chunk_id": "chunk_5",
            "content": "Equipment maintenance schedules are updated quarterly.",
            "rrf_score": 0.022,
            "document_id": "doc_1",
        },
    ]


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for testing without loading actual model."""
    with patch("src.reranker.CrossEncoder") as mock:
        # Mock predict to return fake scores
        mock_instance = Mock()
        mock_instance.predict.return_value = np.array([0.85, 0.40, 0.92, 0.55, 0.20])
        mock.return_value = mock_instance
        yield mock


# Test: Reranker Initialization
def test_reranker_initialization(mock_cross_encoder):
    """Test reranker loads model correctly."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.device == "cpu"
    assert reranker.batch_size == 32
    assert reranker.total_reranks == 0


def test_reranker_model_alias_resolution(mock_cross_encoder):
    """Test model alias resolution."""
    from src.reranker import CrossEncoderReranker, RERANKER_MODELS

    # Test alias
    reranker = CrossEncoderReranker(model_name="fast")
    assert reranker.model_name == RERANKER_MODELS["fast"]

    # Test full model name
    reranker2 = CrossEncoderReranker(model_name="custom-model/my-reranker")
    assert reranker2.model_name == "custom-model/my-reranker"


# Test: Basic Reranking
def test_basic_reranking(mock_cross_encoder, sample_candidates):
    """Test basic reranking functionality."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    query = "waste disposal requirements"
    reranked = reranker.rerank(query=query, candidates=sample_candidates, top_k=3)

    # Should return 3 results
    assert len(reranked) == 3

    # Should add rerank_score
    assert "rerank_score" in reranked[0]
    assert "original_score" in reranked[0]

    # Should preserve original fields
    assert "chunk_id" in reranked[0]
    assert "content" in reranked[0]
    assert "rrf_score" in reranked[0]


def test_reranking_changes_order(mock_cross_encoder, sample_candidates):
    """Test that reranking actually changes the order."""
    from src.reranker import CrossEncoderReranker

    # Mock scores favor chunk_3 (0.92) over chunk_1 (0.85)
    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    query = "waste disposal requirements"
    reranked = reranker.rerank(query=query, candidates=sample_candidates, top_k=3)

    # Chunk 3 should be ranked first (highest rerank score 0.92)
    assert reranked[0]["chunk_id"] == "chunk_3"
    assert reranked[0]["rerank_score"] == 0.92

    # Chunk 1 should be second (score 0.85)
    assert reranked[1]["chunk_id"] == "chunk_1"
    assert reranked[1]["rerank_score"] == 0.85


# Test: Empty Candidates
def test_empty_candidates(mock_cross_encoder):
    """Test reranking with empty candidates."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    reranked = reranker.rerank(query="test", candidates=[], top_k=5)

    assert reranked == []


# Test: Statistics
def test_reranking_with_statistics(mock_cross_encoder, sample_candidates):
    """Test reranking returns statistics."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    query = "waste disposal requirements"
    reranked, stats = reranker.rerank(
        query=query, candidates=sample_candidates, top_k=3, return_stats=True
    )

    # Check stats
    assert stats.candidates == 5
    assert stats.final_results == 3
    assert stats.rerank_time_ms > 0
    assert stats.rank_changes >= 0
    assert stats.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_rank_changes_calculation(mock_cross_encoder, sample_candidates):
    """Test rank changes are calculated correctly."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    query = "waste disposal requirements"
    _, stats = reranker.rerank(
        query=query, candidates=sample_candidates, top_k=3, return_stats=True
    )

    # With mock scores [0.85, 0.40, 0.92, 0.55, 0.20]:
    # Original top 3: chunk_1, chunk_2, chunk_3
    # Reranked top 3: chunk_3, chunk_1, chunk_4
    # Rank changes: chunk_4 is new (chunk_2 dropped)
    assert stats.rank_changes == 1


# Test: Threshold Filtering
def test_reranking_with_threshold(mock_cross_encoder, sample_candidates):
    """Test threshold-based filtering."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    query = "waste disposal requirements"
    reranked = reranker.rerank_with_threshold(
        query=query, candidates=sample_candidates, min_score=0.5, top_k=5  # Only keep scores >= 0.5
    )

    # With mock scores [0.85, 0.40, 0.92, 0.55, 0.20]
    # Should keep: chunk_3 (0.92), chunk_1 (0.85), chunk_4 (0.55)
    assert len(reranked) == 3
    assert all(c["rerank_score"] >= 0.5 for c in reranked)


# Test: Batch Size Handling
def test_batch_size_configuration(mock_cross_encoder):
    """Test custom batch size."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini", batch_size=16)

    assert reranker.batch_size == 16


# Test: Statistics Aggregation
def test_stats_aggregation(mock_cross_encoder, sample_candidates):
    """Test statistics are aggregated correctly over multiple reranks."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    # Perform multiple reranks
    for i in range(3):
        reranker.rerank(query=f"query {i}", candidates=sample_candidates, top_k=2)

    stats = reranker.get_stats()

    assert stats["total_reranks"] == 3
    assert stats["total_time_ms"] > 0
    assert stats["avg_time_ms"] == stats["total_time_ms"] / 3


def test_reset_stats(mock_cross_encoder, sample_candidates):
    """Test statistics can be reset."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    # Rerank once
    reranker.rerank(query="test", candidates=sample_candidates, top_k=2)
    assert reranker.total_reranks == 1

    # Reset
    reranker.reset_stats()
    assert reranker.total_reranks == 0
    assert reranker.total_time_ms == 0.0


# Test: Error Handling
def test_reranking_with_missing_content(mock_cross_encoder):
    """Test reranking handles missing 'content' field gracefully."""
    from src.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="ms-marco-mini")

    candidates = [
        {"chunk_id": "chunk_1"},  # Missing 'content'
        {"chunk_id": "chunk_2", "content": "Valid content"},
    ]

    # Should not crash
    reranked = reranker.rerank(query="test", candidates=candidates, top_k=2)

    # Should still process both (with empty string for missing content)
    assert len(reranked) == 2


def test_cross_encoder_prediction_failure(sample_candidates):
    """Test fallback when cross-encoder prediction fails."""
    from src.reranker import CrossEncoderReranker

    with patch("src.reranker.CrossEncoder") as mock:
        mock_instance = Mock()
        mock_instance.predict.side_effect = Exception("Model inference failed")
        mock.return_value = mock_instance

        reranker = CrossEncoderReranker(model_name="ms-marco-mini")

        # Should fallback to returning original candidates
        reranked = reranker.rerank(query="test", candidates=sample_candidates, top_k=3)

        # Should return top 3 from original
        assert len(reranked) == 3
        assert reranked[0]["chunk_id"] == "chunk_1"  # Original order


# Test: Convenience Function
def test_convenience_function(mock_cross_encoder, sample_candidates):
    """Test the convenience rerank_results function."""
    from src.reranker import rerank_results

    reranked = rerank_results(
        query="waste disposal", candidates=sample_candidates, top_k=2, model_name="fast"
    )

    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]


# Test: Integration with HybridVectorStore
@pytest.mark.integration
def test_reranking_with_hybrid_search():
    """
    Integration test: Hybrid search → Reranking

    This test requires actual models, so it's marked as integration.
    Skipped in unit tests.
    """
    pytest.skip("Integration test - requires actual models")

    # Example integration flow:
    # 1. hybrid_store.hierarchical_search(k=50)
    # 2. reranker.rerank(candidates, top_k=6)
    # 3. Return top 6 reranked results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
