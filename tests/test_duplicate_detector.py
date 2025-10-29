"""
Unit and integration tests for DuplicateDetector.

Tests document duplicate detection using semantic similarity.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.duplicate_detector import DuplicateDetector, DuplicateDetectionConfig


@pytest.fixture
def duplicate_config():
    """Create standard duplicate detection config."""
    return DuplicateDetectionConfig(
        enabled=True,
        similarity_threshold=0.98,
        sample_pages=1,
        cache_size=100,
    )


@pytest.fixture
def mock_embedder():
    """Create mock embedding generator."""
    embedder = Mock()
    # Return consistent 3-dimensional embeddings
    embedder.generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3])]
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock()
    store.search_layer.return_value = []  # No results by default
    return store


def test_config_validation():
    """Test DuplicateDetectionConfig validation."""
    # Valid config
    config = DuplicateDetectionConfig(similarity_threshold=0.95, sample_pages=2)
    config.validate()  # Should not raise

    # Invalid threshold (> 1.0)
    with pytest.raises(ValueError, match="similarity_threshold must be 0-1"):
        config = DuplicateDetectionConfig(similarity_threshold=1.5)
        config.validate()

    # Invalid threshold (< 0.0)
    with pytest.raises(ValueError, match="similarity_threshold must be 0-1"):
        config = DuplicateDetectionConfig(similarity_threshold=-0.1)
        config.validate()

    # Invalid sample_pages
    with pytest.raises(ValueError, match="sample_pages must be >= 1"):
        config = DuplicateDetectionConfig(sample_pages=0)
        config.validate()


def test_detector_initialization(duplicate_config):
    """Test DuplicateDetector initialization."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_path")

    assert detector.config.similarity_threshold == 0.98
    assert detector.config.sample_pages == 1
    assert detector.vector_store_path == "test_path"
    assert detector._embedder is None  # Lazy loading
    assert detector._vector_store is None  # Lazy loading


def test_check_duplicate_disabled():
    """Test that duplicate detection can be disabled."""
    config = DuplicateDetectionConfig(enabled=False)
    detector = DuplicateDetector(config)

    is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is False
    assert sim == 0.0
    assert match is None


@patch("src.duplicate_detector.fitz")
def test_extract_text_sample(mock_fitz, duplicate_config):
    """Test text extraction from PDF."""
    # Mock PyMuPDF document
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample text from page 1"
    mock_doc.__len__.return_value = 5  # 5 pages
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config)

    # Extract text
    text = detector._extract_text_sample("test.pdf")

    assert text == "Sample text from page 1"
    mock_fitz.open.assert_called_once_with("test.pdf")
    mock_doc.close.assert_called_once()


@patch("src.duplicate_detector.fitz")
def test_extract_text_multiple_pages(mock_fitz):
    """Test text extraction from multiple pages."""
    config = DuplicateDetectionConfig(sample_pages=2)

    # Mock PyMuPDF document with multiple pages
    mock_doc = Mock()
    mock_page1 = Mock()
    mock_page1.get_text.return_value = "Page 1 text"
    mock_page2 = Mock()
    mock_page2.get_text.return_value = "Page 2 text"
    mock_doc.__len__.return_value = 5
    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(config)

    # Extract text
    text = detector._extract_text_sample("test.pdf")

    assert text == "Page 1 text\nPage 2 text"


def test_compute_file_hash(duplicate_config, tmp_path):
    """Test file hash computation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config)

    # Compute hash
    hash1 = detector._compute_file_hash(str(test_file))

    # Hash should be consistent
    hash2 = detector._compute_file_hash(str(test_file))
    assert hash1 == hash2

    # Different content should produce different hash
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_text("different content")
    hash3 = detector._compute_file_hash(str(test_file2))
    assert hash1 != hash3


@patch("src.duplicate_detector.fitz")
def test_check_duplicate_no_match(mock_fitz, duplicate_config, mock_embedder, mock_vector_store):
    """Test duplicate check when no match found."""
    # Mock text extraction
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample document text"
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # No matches from vector store
    mock_vector_store.search_layer.return_value = []

    # Check duplicate
    is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is False
    assert sim == 0.0
    assert match is None


@patch("src.duplicate_detector.fitz")
def test_check_duplicate_below_threshold(
    mock_fitz, duplicate_config, mock_embedder, mock_vector_store
):
    """Test duplicate check when similarity is below threshold."""
    # Mock text extraction
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample document text"
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return match with similarity below threshold (0.95 < 0.98)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 0.95,
            "metadata": {"document_id": "doc1"},
        }
    ]

    # Check duplicate
    is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is False
    assert sim == 0.95
    assert match == "doc1"


@patch("src.duplicate_detector.fitz")
def test_check_duplicate_above_threshold(
    mock_fitz, duplicate_config, mock_embedder, mock_vector_store
):
    """Test duplicate check when similarity is above threshold."""
    # Mock text extraction
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample document text"
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return match with similarity above threshold (0.99 >= 0.98)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 0.99,
            "metadata": {"document_id": "doc1"},
        }
    ]

    # Check duplicate
    is_dup, sim, match = detector.check_duplicate("test.pdf")

    assert is_dup is True
    assert sim == 0.99
    assert match == "doc1"


@patch("src.duplicate_detector.fitz")
def test_check_duplicate_excludes_self(
    mock_fitz, duplicate_config, mock_embedder, mock_vector_store
):
    """Test that duplicate check excludes the document itself."""
    # Mock text extraction
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample document text"
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store

    # Return self as match (should be excluded)
    mock_vector_store.search_layer.return_value = [
        {
            "score": 1.0,  # Perfect match
            "metadata": {"document_id": "test_doc"},
        },
        {
            "score": 0.85,  # Different document
            "metadata": {"document_id": "other_doc"},
        },
    ]

    # Check duplicate with document_id
    is_dup, sim, match = detector.check_duplicate("test.pdf", document_id="test_doc")

    # Should skip self and use second match
    assert is_dup is False  # 0.85 < 0.98
    assert sim == 0.85
    assert match == "other_doc"


@patch("src.duplicate_detector.fitz")
def test_embedding_cache(mock_fitz, duplicate_config, mock_embedder, mock_vector_store, tmp_path):
    """Test that embeddings are cached."""
    # Mock text extraction
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample text"
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    # Create real test file for hashing
    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")
    detector._embedder = mock_embedder
    detector._vector_store = mock_vector_store
    mock_vector_store.search_layer.return_value = []

    # First check - should call embedder
    detector.check_duplicate(str(test_file))
    assert mock_embedder.generate_embeddings.call_count == 1

    # Second check on same file - should use cache
    detector.check_duplicate(str(test_file))
    assert mock_embedder.generate_embeddings.call_count == 1  # Not called again


@patch("src.duplicate_detector.fitz")
def test_text_too_short(mock_fitz, duplicate_config):
    """Test handling of documents with very short text."""
    # Mock text extraction with short text
    mock_doc = Mock()
    mock_page = Mock()
    mock_page.get_text.return_value = "short"  # Only 5 chars
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz.open.return_value = mock_doc

    detector = DuplicateDetector(duplicate_config)

    # Check duplicate
    is_dup, sim, match = detector.check_duplicate("test.pdf")

    # Should skip due to short text
    assert is_dup is False
    assert sim == 0.0
    assert match is None


def test_get_stats(duplicate_config):
    """Test getting detector statistics."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")

    stats = detector.get_stats()

    assert stats["enabled"] is True
    assert stats["threshold"] == 0.98
    assert stats["cache_size"] == 0
    assert stats["cache_limit"] == 100
    assert stats["embedder_loaded"] is False
    assert stats["vector_store_loaded"] is False


def test_cache_eviction(duplicate_config, tmp_path):
    """Test that cache evicts oldest entries when full."""
    config = DuplicateDetectionConfig(cache_size=2)  # Small cache
    detector = DuplicateDetector(config)

    # Add 3 entries (should evict oldest)
    for i in range(3):
        file_hash = f"hash{i}"
        embedding = np.array([i, i, i])
        detector._cache_embedding(file_hash, embedding, f"doc{i}")

    # Only last 2 should remain
    assert len(detector._embedding_cache) == 2
    assert "hash0" not in detector._embedding_cache  # Evicted
    assert "hash1" in detector._embedding_cache
    assert "hash2" in detector._embedding_cache


def test_lazy_loading_embedder(duplicate_config):
    """Test that embedder is lazily loaded."""
    detector = DuplicateDetector(duplicate_config)

    assert detector._embedder is None

    # Access embedder (would load it)
    with patch("src.duplicate_detector.EmbeddingGenerator") as mock_gen:
        detector._get_embedder()
        mock_gen.assert_called_once()


def test_lazy_loading_vector_store(duplicate_config):
    """Test that vector store is lazily loaded."""
    detector = DuplicateDetector(duplicate_config, vector_store_path="test_db")

    assert detector._vector_store is None

    # Access vector store (would load it)
    with patch("src.duplicate_detector.HybridVectorStore") as mock_store:
        detector._get_vector_store()
        mock_store.load.assert_called_once_with("test_db")


def test_no_vector_store_path(duplicate_config, mock_embedder):
    """Test behavior when no vector store path is provided."""
    detector = DuplicateDetector(duplicate_config, vector_store_path=None)
    detector._embedder = mock_embedder

    # Get vector store should return None
    store = detector._get_vector_store()
    assert store is None

    # _find_similar_document should handle None gracefully
    is_dup, sim, match = detector._find_similar_document(
        embedding=np.array([0.1, 0.2, 0.3]),
        exclude_doc_id=None,
    )

    assert is_dup is False
    assert sim == 0.0
    assert match is None
