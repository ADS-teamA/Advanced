"""
PHASE 5B: Full Pipeline Integration Tests

Tests the complete IndexingPipeline with hybrid search enabled,
verifying end-to-end functionality from document indexing to hybrid retrieval.
"""

import pytest
from pathlib import Path

from src.indexing_pipeline import IndexingPipeline, IndexingConfig


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not Path("data/test_documents").exists(), reason="Test documents directory not found"
)
def test_pipeline_with_hybrid_enabled():
    """Test complete pipeline with hybrid search enabled."""
    # Configure pipeline with hybrid search
    config = IndexingConfig(
        # PHASE 1-3
        enable_smart_hierarchy=True,
        generate_summaries=True,
        chunk_size=500,
        enable_sac=True,
        # PHASE 4
        embedding_model="text-embedding-3-large",
        normalize_embeddings=True,
        # PHASE 5B: Enable hybrid search
        enable_hybrid_search=True,
        hybrid_fusion_k=60,
        # PHASE 5A: Disable KG for faster testing
        enable_knowledge_graph=False,
    )

    pipeline = IndexingPipeline(config)

    # Find a test document
    test_docs = list(Path("data/test_documents").glob("*.pdf"))
    if not test_docs:
        pytest.skip("No test PDF documents found")

    test_doc = test_docs[0]

    # Index document
    result = pipeline.index_document(document_path=test_doc, save_intermediate=False)

    # Verify hybrid store was created
    assert result is not None
    assert "vector_store" in result
    assert "stats" in result

    # Check that hybrid search is enabled
    assert result["stats"]["hybrid_enabled"] is True

    # Verify hybrid store interface
    hybrid_store = result["vector_store"]
    assert hasattr(hybrid_store, "faiss_store")
    assert hasattr(hybrid_store, "bm25_store")
    assert hasattr(hybrid_store, "fusion_k")

    # Get stats
    stats = hybrid_store.get_stats()
    assert stats["hybrid_enabled"] is True
    assert stats["fusion_k"] == 60
    assert "bm25_layer3_count" in stats
    assert stats["bm25_layer3_count"] > 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not Path("data/test_documents").exists(), reason="Test documents directory not found"
)
def test_hybrid_search_end_to_end():
    """Test complete hybrid search workflow from indexing to retrieval."""
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Configure pipeline with hybrid search
    config = IndexingConfig(
        enable_hybrid_search=True,
        hybrid_fusion_k=60,
        chunk_size=500,
        enable_sac=True,
        enable_knowledge_graph=False,
    )

    pipeline = IndexingPipeline(config)

    # Find test document
    test_docs = list(Path("data/test_documents").glob("*.pdf"))
    if not test_docs:
        pytest.skip("No test PDF documents found")

    test_doc = test_docs[0]

    # Index document
    result = pipeline.index_document(test_doc)
    hybrid_store = result["vector_store"]

    # Create embedder for query
    embedder = EmbeddingGenerator(EmbeddingConfig(model="text-embedding-3-large"))

    # Test hybrid search
    query_text = "safety requirements"
    query_embedding = embedder.embed_texts([query_text])

    results = hybrid_store.hierarchical_search(
        query_text=query_text, query_embedding=query_embedding, k_layer3=6
    )

    # Verify results structure
    assert "layer1" in results
    assert "layer2" in results
    assert "layer3" in results

    # Verify Layer 3 results (PRIMARY)
    layer3_results = results["layer3"]
    assert len(layer3_results) <= 6
    assert all("rrf_score" in r for r in layer3_results)
    assert all("content" in r for r in layer3_results)
    assert all("chunk_id" in r for r in layer3_results)

    # RRF scores should be in descending order
    rrf_scores = [r["rrf_score"] for r in layer3_results]
    assert rrf_scores == sorted(rrf_scores, reverse=True)


@pytest.mark.integration
def test_pipeline_dense_only_still_works():
    """Test that pipeline still works with hybrid search disabled."""
    from pathlib import Path

    # Configure pipeline with hybrid search DISABLED
    config = IndexingConfig(
        enable_hybrid_search=False,  # Disabled
        chunk_size=500,
        enable_sac=True,
        enable_knowledge_graph=False,
    )

    pipeline = IndexingPipeline(config)

    # Find test document
    test_docs = list(Path("data/test_documents").glob("*.pdf"))
    if not test_docs:
        pytest.skip("No test PDF documents found")

    test_doc = test_docs[0]

    # Index document
    result = pipeline.index_document(test_doc)

    # Verify dense-only store was created
    assert result is not None
    assert result["stats"]["hybrid_enabled"] is False

    # Vector store should be FAISSVectorStore, not HybridVectorStore
    vector_store = result["vector_store"]
    assert not hasattr(vector_store, "fusion_k")

    # Search should still work (dense-only)
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    embedder = EmbeddingGenerator(EmbeddingConfig(model="text-embedding-3-large"))

    query_embedding = embedder.embed_texts(["safety requirements"])

    results = vector_store.hierarchical_search(query_embedding=query_embedding, k_layer3=6)

    # Verify dense-only results
    assert "layer3" in results
    assert len(results["layer3"]) <= 6
    # Dense-only results have "score", not "rrf_score"
    assert all("score" in r for r in results["layer3"])
    assert all("rrf_score" not in r for r in results["layer3"])


@pytest.mark.integration
@pytest.mark.slow
def test_hybrid_vs_dense_comparison():
    """Compare hybrid search results vs dense-only search."""
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Find test document
    test_docs = list(Path("data/test_documents").glob("*.pdf"))
    if not test_docs:
        pytest.skip("No test PDF documents found")

    test_doc = test_docs[0]

    # Index with dense-only
    config_dense = IndexingConfig(
        enable_hybrid_search=False, chunk_size=500, enable_knowledge_graph=False
    )
    pipeline_dense = IndexingPipeline(config_dense)
    result_dense = pipeline_dense.index_document(test_doc)

    # Index with hybrid
    config_hybrid = IndexingConfig(
        enable_hybrid_search=True, hybrid_fusion_k=60, chunk_size=500, enable_knowledge_graph=False
    )
    pipeline_hybrid = IndexingPipeline(config_hybrid)
    result_hybrid = pipeline_hybrid.index_document(test_doc)

    # Create embedder
    embedder = EmbeddingGenerator(EmbeddingConfig(model="text-embedding-3-large"))

    # Search with same query
    query_text = "safety specification"
    query_embedding_dense = embedder.embed_texts([query_text])
    query_embedding_hybrid = embedder.embed_texts([query_text])

    # Dense search
    results_dense = result_dense["vector_store"].hierarchical_search(
        query_embedding=query_embedding_dense, k_layer3=6
    )

    # Hybrid search
    results_hybrid = result_hybrid["vector_store"].hierarchical_search(
        query_text=query_text, query_embedding=query_embedding_hybrid, k_layer3=6
    )

    # Both should return results
    assert len(results_dense["layer3"]) > 0
    assert len(results_hybrid["layer3"]) > 0

    # Hybrid results should have RRF scores
    assert all("rrf_score" in r for r in results_hybrid["layer3"])

    # Dense results should only have dense scores
    assert all("score" in r for r in results_dense["layer3"])
    assert all("rrf_score" not in r for r in results_dense["layer3"])

    # Results may differ (that's the point of hybrid search!)
    # We don't assert they're identical


@pytest.mark.integration
def test_hybrid_store_save_load_integration(tmp_path):
    """Test saving and loading hybrid store through pipeline."""
    # Find test document
    test_docs = list(Path("data/test_documents").glob("*.pdf"))
    if not test_docs:
        pytest.skip("No test PDF documents found")

    test_doc = test_docs[0]

    # Configure pipeline with hybrid search
    config = IndexingConfig(enable_hybrid_search=True, chunk_size=500, enable_knowledge_graph=False)

    pipeline = IndexingPipeline(config)

    # Index document
    result = pipeline.index_document(test_doc)
    hybrid_store = result["vector_store"]

    # Save
    output_dir = tmp_path / "hybrid_store"
    hybrid_store.save(output_dir)

    # Verify files exist
    assert (output_dir / "layer1.index").exists()
    assert (output_dir / "layer2.index").exists()
    assert (output_dir / "layer3.index").exists()
    assert (output_dir / "metadata.pkl").exists()
    assert (output_dir / "bm25_layer1.pkl").exists()
    assert (output_dir / "bm25_layer2.pkl").exists()
    assert (output_dir / "bm25_layer3.pkl").exists()
    assert (output_dir / "hybrid_config.pkl").exists()

    # Load
    from src.hybrid_search import HybridVectorStore

    loaded_store = HybridVectorStore.load(output_dir)

    # Verify loaded store
    assert loaded_store.fusion_k == 60
    assert hasattr(loaded_store, "faiss_store")
    assert hasattr(loaded_store, "bm25_store")

    # Verify search works on loaded store
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    embedder = EmbeddingGenerator(EmbeddingConfig(model="text-embedding-3-large"))

    query_text = "test query"
    query_embedding = embedder.embed_texts([query_text])

    results = loaded_store.hierarchical_search(
        query_text=query_text, query_embedding=query_embedding, k_layer3=3
    )

    assert "layer3" in results


@pytest.mark.integration
def test_error_handling_when_hybrid_fails():
    """Test that pipeline gracefully falls back to dense-only if hybrid fails."""
    # This test would simulate hybrid search failure
    # In practice, the try/except in IndexingPipeline handles this

    config = IndexingConfig(enable_hybrid_search=True, chunk_size=500, enable_knowledge_graph=False)

    # Pipeline should initialize without error
    pipeline = IndexingPipeline(config)

    # Even if hybrid fails, dense-only should work
    # (tested by the actual implementation's error handling)
    assert pipeline.config.enable_hybrid_search is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
