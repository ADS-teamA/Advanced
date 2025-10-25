"""
PHASE 5B: Hybrid Search Tests

Tests for BM25 sparse retrieval, RRF fusion, and hybrid vector store.

Test Structure:
1. Unit tests for BM25Index
2. Unit tests for BM25Store
3. Unit tests for RRF fusion
4. Integration tests for HybridVectorStore
5. Full pipeline integration tests
"""

import pytest
from pathlib import Path
import numpy as np

from src.multi_layer_chunker import Chunk, ChunkMetadata
from src.hybrid_search import BM25Index, BM25Store, HybridVectorStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = {
        "layer1": [
            Chunk(
                chunk_id="doc1_L1",
                content="safety specification requirements for equipment manufacturing",
                raw_content="safety specification requirements for equipment manufacturing",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L1",
                    layer=1,
                    document_id="doc1",
                    section_title="Document Summary",
                ),
            )
        ],
        "layer2": [
            Chunk(
                chunk_id="doc1_L2_s1",
                content="safety requirements chapter covering equipment standards",
                raw_content="safety requirements chapter covering equipment standards",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L2_s1",
                    layer=2,
                    document_id="doc1",
                    section_id="s1",
                    section_title="Safety Section",
                ),
            ),
            Chunk(
                chunk_id="doc1_L2_s2",
                content="testing procedures for compliance verification",
                raw_content="testing procedures for compliance verification",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L2_s2",
                    layer=2,
                    document_id="doc1",
                    section_id="s2",
                    section_title="Testing Section",
                ),
            ),
        ],
        "layer3": [
            Chunk(
                chunk_id="doc1_L3_s1_c1",
                content="equipment must meet safety standards including electrical and mechanical specifications",
                raw_content="equipment must meet safety standards including electrical and mechanical specifications",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L3_s1_c1",
                    layer=3,
                    document_id="doc1",
                    section_id="s1",
                    section_title="Safety Section",
                ),
            ),
            Chunk(
                chunk_id="doc1_L3_s1_c2",
                content="testing procedures for safety compliance must be documented and verified",
                raw_content="testing procedures for safety compliance must be documented and verified",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L3_s1_c2",
                    layer=3,
                    document_id="doc1",
                    section_id="s1",
                    section_title="Safety Section",
                ),
            ),
            Chunk(
                chunk_id="doc1_L3_s2_c1",
                content="verification methods include inspection and certification procedures",
                raw_content="verification methods include inspection and certification procedures",
                metadata=ChunkMetadata(
                    chunk_id="doc1_L3_s2_c1",
                    layer=3,
                    document_id="doc1",
                    section_id="s2",
                    section_title="Testing Section",
                ),
            ),
        ],
    }
    return chunks


# ============================================================================
# Unit Tests: BM25Index
# ============================================================================


def test_bm25_index_initialization():
    """Test BM25Index initialization."""
    index = BM25Index()

    assert index.bm25 is None
    assert len(index.corpus) == 0
    assert len(index.tokenized_corpus) == 0
    assert len(index.chunk_ids) == 0
    assert len(index.metadata) == 0
    assert len(index.doc_id_map) == 0


def test_bm25_index_build_from_chunks(sample_chunks):
    """Test building BM25 index from chunks."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks["layer3"])

    assert index.bm25 is not None
    assert len(index.corpus) == 3
    assert len(index.tokenized_corpus) == 3
    assert len(index.chunk_ids) == 3
    assert len(index.metadata) == 3
    assert "doc1" in index.doc_id_map
    assert len(index.doc_id_map["doc1"]) == 3


def test_bm25_index_tokenization():
    """Test tokenization logic."""
    index = BM25Index()

    tokens = index._tokenize("Safety Specification Requirements")
    assert tokens == ["safety", "specification", "requirements"]

    tokens = index._tokenize("Test with punctuation!")
    assert "punctuation!" in tokens  # Simple tokenizer keeps punctuation


def test_bm25_index_search(sample_chunks):
    """Test BM25 search functionality."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks["layer3"])

    results = index.search(query="safety equipment", k=3)

    assert len(results) <= 3
    assert all("score" in r for r in results)
    assert all("chunk_id" in r for r in results)
    assert all("content" in r for r in results)

    # First result should be most relevant (contains both terms)
    assert "safety" in results[0]["content"].lower()


def test_bm25_index_document_filtering(sample_chunks):
    """Test document filtering in BM25 search."""
    index = BM25Index()
    index.build_from_chunks(sample_chunks["layer3"])

    results = index.search(query="safety", k=10, document_filter="doc1")

    assert len(results) > 0
    assert all(r["document_id"] == "doc1" for r in results)


def test_bm25_index_empty_search():
    """Test search on empty index."""
    index = BM25Index()
    results = index.search(query="test", k=5)

    assert len(results) == 0


def test_bm25_index_save_load(sample_chunks, tmp_path):
    """Test saving and loading BM25 index."""
    # Build index
    index = BM25Index()
    index.build_from_chunks(sample_chunks["layer3"])

    # Save
    save_path = tmp_path / "bm25_test.pkl"
    index.save(save_path)
    assert save_path.exists()

    # Load
    loaded_index = BM25Index.load(save_path)

    assert len(loaded_index.corpus) == len(index.corpus)
    assert len(loaded_index.metadata) == len(index.metadata)
    assert loaded_index.bm25 is not None

    # Search should work on loaded index
    results = loaded_index.search(query="safety", k=1)
    assert len(results) == 1


# ============================================================================
# Unit Tests: BM25Store
# ============================================================================


def test_bm25_store_initialization():
    """Test BM25Store initialization."""
    store = BM25Store()

    assert store.index_layer1 is not None
    assert store.index_layer2 is not None
    assert store.index_layer3 is not None


def test_bm25_store_build_from_chunks(sample_chunks):
    """Test building all 3 layers from chunks dict."""
    store = BM25Store()
    store.build_from_chunks(sample_chunks)

    assert store.index_layer1.bm25 is not None
    assert store.index_layer2.bm25 is not None
    assert store.index_layer3.bm25 is not None

    assert len(store.index_layer1.corpus) == 1
    assert len(store.index_layer2.corpus) == 2
    assert len(store.index_layer3.corpus) == 3


def test_bm25_store_search_all_layers(sample_chunks):
    """Test searching all 3 layers."""
    store = BM25Store()
    store.build_from_chunks(sample_chunks)

    # Layer 1
    l1_results = store.search_layer1("safety requirements", k=1)
    assert len(l1_results) == 1

    # Layer 2
    l2_results = store.search_layer2("testing procedures", k=2)
    assert len(l2_results) <= 2

    # Layer 3
    l3_results = store.search_layer3("safety equipment", k=3)
    assert len(l3_results) <= 3


def test_bm25_store_save_load(sample_chunks, tmp_path):
    """Test saving and loading BM25 store."""
    # Build store
    store = BM25Store()
    store.build_from_chunks(sample_chunks)

    # Save
    output_dir = tmp_path / "bm25_store"
    store.save(output_dir)

    assert (output_dir / "bm25_layer1.pkl").exists()
    assert (output_dir / "bm25_layer2.pkl").exists()
    assert (output_dir / "bm25_layer3.pkl").exists()

    # Load
    loaded_store = BM25Store.load(output_dir)

    assert len(loaded_store.index_layer1.corpus) == 1
    assert len(loaded_store.index_layer2.corpus) == 2
    assert len(loaded_store.index_layer3.corpus) == 3

    # Search should work
    results = loaded_store.search_layer3("safety", k=1)
    assert len(results) == 1


# ============================================================================
# Unit Tests: RRF Fusion
# ============================================================================


@pytest.fixture
def mock_dense_results():
    """Mock dense retrieval results."""
    return [
        {"chunk_id": "c1", "score": 0.95, "content": "text1", "document_id": "doc1"},
        {"chunk_id": "c2", "score": 0.90, "content": "text2", "document_id": "doc1"},
        {"chunk_id": "c3", "score": 0.85, "content": "text3", "document_id": "doc1"},
    ]


@pytest.fixture
def mock_sparse_results():
    """Mock sparse retrieval results."""
    return [
        {"chunk_id": "c2", "score": 15.5, "content": "text2", "document_id": "doc1"},
        {"chunk_id": "c4", "score": 12.0, "content": "text4", "document_id": "doc1"},
        {"chunk_id": "c1", "score": 10.0, "content": "text1", "document_id": "doc1"},
    ]


@pytest.fixture
def mock_hybrid_store():
    """Create a mock hybrid store for RRF testing."""
    # We'll use real stores but with mock data
    from src.faiss_vector_store import FAISSVectorStore

    # Create minimal FAISS store
    faiss_store = FAISSVectorStore(dimensions=128)

    # Create minimal BM25 store
    bm25_store = BM25Store()

    # Create hybrid store
    hybrid_store = HybridVectorStore(faiss_store=faiss_store, bm25_store=bm25_store, fusion_k=60)

    return hybrid_store


def test_rrf_fusion_basic(mock_hybrid_store, mock_dense_results, mock_sparse_results):
    """Test basic RRF fusion logic."""
    results = mock_hybrid_store._rrf_fusion(
        dense_results=mock_dense_results, sparse_results=mock_sparse_results, k=3
    )

    assert len(results) == 3
    assert all("rrf_score" in r for r in results)

    # c2 should rank highest (appears in both lists at top positions)
    # Dense: rank 2, Sparse: rank 1
    assert results[0]["chunk_id"] == "c2"


def test_rrf_fusion_math(mock_hybrid_store):
    """Test RRF score calculation with known values."""
    # Simple case: c1 at rank 1 in dense, rank 3 in sparse
    dense = [{"chunk_id": "c1", "score": 0.9, "content": "text1"}]
    sparse = [
        {"chunk_id": "c2", "score": 10.0, "content": "text2"},
        {"chunk_id": "c3", "score": 8.0, "content": "text3"},
        {"chunk_id": "c1", "score": 5.0, "content": "text1"},
    ]

    results = mock_hybrid_store._rrf_fusion(dense, sparse, k=3)

    # c1: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    c1_result = next(r for r in results if r["chunk_id"] == "c1")
    expected_score = 1 / (60 + 1) + 1 / (60 + 3)
    assert abs(c1_result["rrf_score"] - expected_score) < 0.0001


def test_rrf_fusion_dense_only(mock_hybrid_store):
    """Test RRF with only dense results (sparse empty)."""
    dense = [{"chunk_id": "c1", "score": 0.9, "content": "text1"}]
    sparse = []

    results = mock_hybrid_store._rrf_fusion(dense, sparse, k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    # Score should be 1/(60+1) = 0.0164
    expected = 1 / (60 + 1)
    assert abs(results[0]["rrf_score"] - expected) < 0.0001


def test_rrf_fusion_sparse_only(mock_hybrid_store):
    """Test RRF with only sparse results (dense empty)."""
    dense = []
    sparse = [{"chunk_id": "c1", "score": 10.0, "content": "text1"}]

    results = mock_hybrid_store._rrf_fusion(dense, sparse, k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_rrf_fusion_k_parameter(mock_hybrid_store):
    """Test effect of different k values on RRF."""
    dense = [{"chunk_id": "c1", "score": 0.9, "content": "text1"}]
    sparse = [{"chunk_id": "c1", "score": 10.0, "content": "text1"}]

    # Create stores with different k values
    from src.faiss_vector_store import FAISSVectorStore

    faiss = FAISSVectorStore(dimensions=128)
    bm25 = BM25Store()

    hybrid_low = HybridVectorStore(faiss, bm25, fusion_k=10)
    hybrid_high = HybridVectorStore(faiss, bm25, fusion_k=100)

    results_low = hybrid_low._rrf_fusion(dense, sparse, k=1)
    results_high = hybrid_high._rrf_fusion(dense, sparse, k=1)

    # Lower k gives higher scores (1/(10+1) > 1/(100+1))
    assert results_low[0]["rrf_score"] > results_high[0]["rrf_score"]


# ============================================================================
# Integration Tests: HybridVectorStore
# ============================================================================


def test_hybrid_store_initialization(sample_chunks):
    """Test HybridVectorStore initialization."""
    from src.faiss_vector_store import FAISSVectorStore

    # Create stores
    faiss_store = FAISSVectorStore(dimensions=128)
    bm25_store = BM25Store()
    bm25_store.build_from_chunks(sample_chunks)

    # Create hybrid store
    hybrid_store = HybridVectorStore(faiss_store=faiss_store, bm25_store=bm25_store, fusion_k=60)

    assert hybrid_store.faiss_store is faiss_store
    assert hybrid_store.bm25_store is bm25_store
    assert hybrid_store.fusion_k == 60


def test_hybrid_store_get_stats(sample_chunks):
    """Test get_stats method."""
    from src.faiss_vector_store import FAISSVectorStore

    faiss_store = FAISSVectorStore(dimensions=128)
    bm25_store = BM25Store()
    bm25_store.build_from_chunks(sample_chunks)

    hybrid_store = HybridVectorStore(faiss_store, bm25_store)
    stats = hybrid_store.get_stats()

    assert stats["hybrid_enabled"] is True
    assert stats["fusion_k"] == 60
    assert "bm25_layer1_count" in stats
    assert "bm25_layer2_count" in stats
    assert "bm25_layer3_count" in stats


def test_hybrid_store_save_load(sample_chunks, tmp_path):
    """Test saving and loading hybrid store."""
    from src.faiss_vector_store import FAISSVectorStore
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Create embedder
    embedder = EmbeddingGenerator(
        EmbeddingConfig(provider="huggingface", model="BAAI/bge-small-en-v1.5", batch_size=32)
    )

    # Generate embeddings for sample chunks
    embeddings = {
        "layer1": embedder.embed_chunks(sample_chunks["layer1"], layer=1),
        "layer2": embedder.embed_chunks(sample_chunks["layer2"], layer=2),
        "layer3": embedder.embed_chunks(sample_chunks["layer3"], layer=3),
    }

    # Create FAISS store
    faiss_store = FAISSVectorStore(dimensions=embedder.dimensions)
    faiss_store.add_chunks(sample_chunks, embeddings)

    # Create BM25 store
    bm25_store = BM25Store()
    bm25_store.build_from_chunks(sample_chunks)

    # Create hybrid store
    hybrid_store = HybridVectorStore(faiss_store, bm25_store, fusion_k=60)

    # Save
    output_dir = tmp_path / "hybrid_store"
    hybrid_store.save(output_dir)

    # Check files exist
    assert (output_dir / "layer1.index").exists()
    assert (output_dir / "bm25_layer1.pkl").exists()
    assert (output_dir / "hybrid_config.pkl").exists()

    # Load
    loaded_store = HybridVectorStore.load(output_dir)

    assert loaded_store.fusion_k == 60
    assert len(loaded_store.bm25_store.index_layer3.corpus) == 3


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_hybrid_pipeline_integration(sample_chunks, tmp_path):
    """Test complete hybrid search pipeline with real embeddings."""
    from src.faiss_vector_store import FAISSVectorStore
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Create embedder (use small model for speed)
    embedder = EmbeddingGenerator(
        EmbeddingConfig(provider="huggingface", model="BAAI/bge-small-en-v1.5", batch_size=32)
    )

    # Generate embeddings
    embeddings = {
        "layer1": embedder.embed_chunks(sample_chunks["layer1"], layer=1),
        "layer2": embedder.embed_chunks(sample_chunks["layer2"], layer=2),
        "layer3": embedder.embed_chunks(sample_chunks["layer3"], layer=3),
    }

    # Create FAISS store
    faiss_store = FAISSVectorStore(dimensions=embedder.dimensions)
    faiss_store.add_chunks(sample_chunks, embeddings)

    # Create BM25 store
    bm25_store = BM25Store()
    bm25_store.build_from_chunks(sample_chunks)

    # Create hybrid store
    hybrid_store = HybridVectorStore(faiss_store=faiss_store, bm25_store=bm25_store, fusion_k=60)

    # Test hybrid search
    query_text = "safety equipment requirements"
    query_embedding = embedder.embed_texts([query_text])

    results = hybrid_store.hierarchical_search(
        query_text=query_text, query_embedding=query_embedding, k_layer3=2
    )

    # Assertions
    assert "layer3" in results
    assert len(results["layer3"]) <= 2
    assert all("rrf_score" in r for r in results["layer3"])
    assert all("content" in r for r in results["layer3"])


@pytest.mark.integration
def test_hybrid_backward_compatibility(sample_chunks):
    """Test that dense-only still works when hybrid is disabled."""
    from src.faiss_vector_store import FAISSVectorStore
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig

    # Create embedder
    embedder = EmbeddingGenerator(
        EmbeddingConfig(provider="huggingface", model="BAAI/bge-small-en-v1.5")
    )

    # Generate embeddings
    embeddings = {
        "layer1": embedder.embed_chunks(sample_chunks["layer1"], layer=1),
        "layer2": embedder.embed_chunks(sample_chunks["layer2"], layer=2),
        "layer3": embedder.embed_chunks(sample_chunks["layer3"], layer=3),
    }

    # Create FAISS store (dense-only)
    faiss_store = FAISSVectorStore(dimensions=embedder.dimensions)
    faiss_store.add_chunks(sample_chunks, embeddings)

    # Test dense-only search still works
    query_embedding = embedder.embed_texts(["safety requirements"])

    results = faiss_store.hierarchical_search(query_embedding=query_embedding, k_layer3=2)

    # Should return results without RRF scores
    assert "layer3" in results
    assert len(results["layer3"]) <= 2
    # Dense-only results don't have rrf_score
    assert "score" in results["layer3"][0]
    assert "rrf_score" not in results["layer3"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
