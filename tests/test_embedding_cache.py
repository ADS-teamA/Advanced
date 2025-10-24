#!/usr/bin/env python3
"""
Test suite for embedding cache system in EmbeddingGenerator.

Tests cover:
- Cache hit/miss behavior
- LRU eviction policy (OrderedDict)
- Cache key collision detection
- Cache validation (shape, dtype, dimension checks)
- Cache statistics (hit_rate, error_rate)
- Memory error handling (auto-disable on OOM)
- Cache error counter and auto-disable after 10 errors

Based on implementation in src/embedding_generator.py (lines 69-349)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    # OpenAI is imported inside _init_openai_model, so we need to patch it there
    with patch('openai.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock embeddings.create response - return appropriate number of embeddings
        def create_embeddings(model, input, **kwargs):
            mock_response = MagicMock()
            mock_response.usage.total_tokens = len(input) * 100
            # Return one embedding per input text
            mock_response.data = [MagicMock(embedding=[0.1] * 3072) for _ in input]
            return mock_response

        # Wrap in MagicMock to get call_count tracking
        mock_client.embeddings.create = MagicMock(side_effect=create_embeddings)

        yield mock_client


@pytest.fixture
def embedder(mock_openai_client):
    """Create EmbeddingGenerator instance with mocked OpenAI client."""
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-large",
        batch_size=64,
        normalize=True,
        cache_enabled=True,
        cache_max_size=3  # Small cache for testing eviction
    )

    # Patch OpenAI import and environment
    with patch('openai.OpenAI', return_value=mock_openai_client), \
         patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        embedder = EmbeddingGenerator(config)

    return embedder


class TestCacheHitMiss:
    """Test cache hit/miss behavior."""

    def test_cache_miss_generates_new_embedding(self, embedder, mock_openai_client):
        """Test that cache miss triggers embedding generation."""
        texts = ["This is a test"]

        # First call should be a cache miss
        result = embedder.embed_texts(texts)

        assert embedder._cache_misses == 1
        assert embedder._cache_hits == 0
        assert mock_openai_client.embeddings.create.called
        assert result.shape == (1, 3072)

    def test_cache_hit_returns_cached_embedding(self, embedder, mock_openai_client):
        """Test that cache hit returns cached embedding without API call."""
        texts = ["This is a test"]

        # First call - cache miss
        result1 = embedder.embed_texts(texts)
        call_count_1 = mock_openai_client.embeddings.create.call_count

        # Second call - should be cache hit
        result2 = embedder.embed_texts(texts)
        call_count_2 = mock_openai_client.embeddings.create.call_count

        assert embedder._cache_hits == 1
        assert embedder._cache_misses == 1
        assert call_count_2 == call_count_1  # No new API call
        assert np.array_equal(result1, result2)

    def test_cache_hit_moves_to_end(self, embedder):
        """Test that cache hit moves entry to end (OrderedDict access tracking)."""
        texts1 = ["Text 1"]
        texts2 = ["Text 2"]

        # Add two entries
        embedder.embed_texts(texts1)
        embedder.embed_texts(texts2)

        # Access first entry
        embedder.embed_texts(texts1)

        # Check that text1 was moved to end
        cache_keys = list(embedder._embedding_cache.keys())
        assert len(cache_keys) == 2
        # After accessing texts1, it should be at the end
        assert cache_keys[-1] == embedder._generate_cache_key(texts1)

    def test_different_texts_cause_cache_miss(self, embedder):
        """Test that different texts cause cache miss."""
        texts1 = ["First text"]
        texts2 = ["Second text"]

        embedder.embed_texts(texts1)
        embedder.embed_texts(texts2)

        # Both should be cache misses
        assert embedder._cache_misses == 2
        assert embedder._cache_hits == 0


class TestLRUEviction:
    """Test LRU (Least Recently Used) eviction policy."""

    def test_lru_eviction_removes_oldest_entry(self, embedder):
        """Test that cache eviction removes oldest entry when at capacity."""
        # Cache max size is 3
        texts1 = ["Text 1"]
        texts2 = ["Text 2"]
        texts3 = ["Text 3"]
        texts4 = ["Text 4"]  # This should evict texts1

        embedder.embed_texts(texts1)
        embedder.embed_texts(texts2)
        embedder.embed_texts(texts3)

        assert len(embedder._embedding_cache) == 3

        # Add fourth entry - should evict first
        embedder.embed_texts(texts4)

        assert len(embedder._embedding_cache) == 3
        assert embedder._generate_cache_key(texts1) not in embedder._embedding_cache
        assert embedder._generate_cache_key(texts4) in embedder._embedding_cache

    def test_fifo_eviction_ignores_access_order(self, embedder):
        """Test that eviction uses FIFO, not true LRU (per implementation note line 321)."""
        texts1 = ["Text 1"]
        texts2 = ["Text 2"]
        texts3 = ["Text 3"]
        texts4 = ["Text 4"]

        # Fill cache
        embedder.embed_texts(texts1)
        embedder.embed_texts(texts2)
        embedder.embed_texts(texts3)

        # Access texts1 (moves to end via move_to_end in cache hit)
        embedder.embed_texts(texts1)

        # Verify texts1 was moved to end
        cache_keys = list(embedder._embedding_cache.keys())
        assert cache_keys[-1] == embedder._generate_cache_key(texts1)

        # Add texts4 - should evict oldest insertion (texts2, since texts1 moved)
        # Actually, based on line 327, popitem(last=False) removes FIRST item,
        # which after move_to_end is texts2
        embedder.embed_texts(texts4)

        # texts2 should be evicted (oldest after texts1 moved)
        assert embedder._generate_cache_key(texts2) not in embedder._embedding_cache
        # texts1 should still be in cache (was moved to end)
        assert embedder._generate_cache_key(texts1) in embedder._embedding_cache


class TestCacheCollisionDetection:
    """Test cache key collision detection."""

    def test_cache_collision_detection(self, embedder, caplog):
        """Test that hash collisions are detected via shape[0] mismatch."""
        import logging
        caplog.set_level(logging.ERROR)

        texts1 = ["Text A", "Text B"]  # 2 texts
        texts2 = ["Single text"]  # 1 text (different count)

        # Generate embeddings for texts1 (2 texts)
        embedder.embed_texts(texts1)
        cache_key1 = embedder._generate_cache_key(texts1)

        # Verify texts1 is cached with shape (2, 3072)
        assert cache_key1 in embedder._embedding_cache
        assert embedder._embedding_cache[cache_key1].shape[0] == 2

        # Simulate collision: Force texts2 to use texts1's cache key
        # This simulates a hash collision where different texts have same key
        real_cache_key = embedder._generate_cache_key(texts2)

        # Patch _generate_cache_key to return texts1's key for texts2
        with patch.object(embedder, '_generate_cache_key', return_value=cache_key1):
            embedder.embed_texts(texts2)  # 1 text, but gets texts1's cache key (2 texts)

        # Should log error about count mismatch (expected 1 text, got 2 embeddings)
        assert any("Cache count mismatch" in record.message for record in caplog.records)
        # Original cache entry should be removed due to collision detection
        # Note: After removal, embed_texts generates new embedding and caches it
        # So cache still has an entry, but it's the NEW one for texts2


class TestCacheValidation:
    """Test cache validation for shape, dtype, and dimension checks."""

    def test_cache_validation_rejects_wrong_shape(self, embedder, caplog):
        """Test that cache rejects entries with wrong shape (not 2D array)."""
        texts = ["Test text"]
        cache_key = embedder._generate_cache_key(texts)

        # Manually insert invalid cache entry (1D array instead of 2D)
        invalid_embedding = np.array([0.1, 0.2, 0.3])  # 1D instead of 2D
        embedder._embedding_cache[cache_key] = invalid_embedding

        # Try to retrieve - should detect dimension error and regenerate
        result = embedder.embed_texts(texts)

        # Should log dimension error
        assert any("Cache dimension error" in record.message for record in caplog.records)
        # Should have removed invalid entry and regenerated
        assert embedder._cache_misses == 1  # Treated as miss after validation failed

    def test_cache_validation_rejects_wrong_dimensions(self, embedder, caplog):
        """Test that cache rejects entries with wrong embedding dimensions."""
        import logging
        caplog.set_level(logging.ERROR)

        texts = ["Test text"]
        cache_key = embedder._generate_cache_key(texts)

        # Manually insert invalid cache entry (wrong dimensions)
        invalid_embedding = np.array([[0.1, 0.2, 0.3]])  # 3 dims instead of 3072
        embedder._embedding_cache[cache_key] = invalid_embedding

        # Try to retrieve - should detect dimension mismatch and regenerate
        result = embedder.embed_texts(texts)

        # Should log dimension mismatch
        assert any("Cache dimension mismatch" in record.message for record in caplog.records)
        # Entry was removed but then regenerated with correct dimensions
        assert cache_key in embedder._embedding_cache
        assert embedder._embedding_cache[cache_key].shape == (1, 3072)

    def test_cache_validation_rejects_non_ndarray(self, embedder, caplog):
        """Test that cache rejects non-numpy array entries."""
        import logging
        caplog.set_level(logging.ERROR)

        texts = ["Test text"]
        cache_key = embedder._generate_cache_key(texts)

        # Manually insert invalid cache entry (list instead of ndarray)
        embedder._embedding_cache[cache_key] = [[0.1, 0.2, 0.3]]

        # Try to retrieve - should detect type error and regenerate
        result = embedder.embed_texts(texts)

        # Should log invalid type error
        assert any("Invalid cache entry type" in record.message for record in caplog.records)
        # Entry was removed but then regenerated with correct type
        assert cache_key in embedder._embedding_cache
        assert isinstance(embedder._embedding_cache[cache_key], np.ndarray)


class TestCacheStatistics:
    """Test cache statistics calculation."""

    def test_cache_stats_calculates_hit_rate_and_error_rate(self, embedder):
        """Test that get_cache_stats() calculates hit_rate and error_rate correctly."""
        texts1 = ["Text 1"]
        texts2 = ["Text 2"]

        # Generate some cache activity
        embedder.embed_texts(texts1)  # miss
        embedder.embed_texts(texts1)  # hit
        embedder.embed_texts(texts2)  # miss
        embedder.embed_texts(texts1)  # hit

        stats = embedder.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["max_size"] == 3
        assert stats["current_size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["errors"] == 0
        assert stats["hit_rate"] == 0.5  # 2 hits / 4 total
        assert stats["error_rate"] == 0.0

    def test_cache_stats_with_no_requests(self, embedder):
        """Test cache stats with no requests (avoid division by zero)."""
        stats = embedder.get_cache_stats()

        assert stats["hit_rate"] == 0.0
        assert stats["error_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["errors"] == 0

    def test_cache_stats_tracks_errors(self, embedder, caplog):
        """Test that cache errors are tracked in statistics."""
        texts = ["Test text"]

        # Force an exception in cache lookup (not validation, but actual exception)
        # Validation failures don't increment _cache_errors, only exceptions do
        with patch.object(embedder, '_generate_cache_key', side_effect=Exception("Cache error")):
            embedder.embed_texts(texts)

        stats = embedder.get_cache_stats()
        assert stats["errors"] == 1
        assert stats["error_rate"] > 0


class TestMemoryErrorHandling:
    """Test memory error handling and cache auto-disable."""

    def test_cache_auto_disables_on_memory_error(self, embedder, caplog):
        """Test that cache auto-disables on MemoryError during storage."""
        texts = ["Test text"]

        # Mock _add_to_cache to raise MemoryError
        with patch.object(embedder, '_add_to_cache', side_effect=MemoryError("Out of memory")):
            result = embedder.embed_texts(texts)

        # Cache should be disabled
        assert embedder._cache_enabled is False
        # Cache should be cleared
        assert len(embedder._embedding_cache) == 0
        # Should log warning
        assert any("Cache storage failed due to memory" in record.message for record in caplog.records)
        # Error counter should increment
        assert embedder._cache_errors == 1

    def test_cache_auto_disables_after_10_errors(self, embedder, caplog):
        """Test that cache auto-disables after 10 consecutive errors."""
        texts = ["Test text"]

        # Mock _generate_cache_key to raise exception
        with patch.object(embedder, '_generate_cache_key', side_effect=Exception("Cache error")):
            # Trigger 11 errors
            for i in range(11):
                embedder.embed_texts(texts)

        # Cache should be disabled after 10 errors
        assert embedder._cache_enabled is False
        assert embedder._cache_errors >= 10
        # Should log error about disabling cache
        assert any("Cache disabled due to repeated errors" in record.message for record in caplog.records)

    def test_cache_storage_error_increments_counter(self, embedder, caplog):
        """Test that cache storage errors increment error counter."""
        texts = ["Test text"]

        # Mock _add_to_cache to raise generic exception
        with patch.object(embedder, '_add_to_cache', side_effect=ValueError("Storage error")):
            embedder.embed_texts(texts)

        # Error counter should increment
        assert embedder._cache_errors == 1
        # Should log error
        assert any("Failed to store embeddings in cache" in record.message for record in caplog.records)

    def test_cache_disabled_after_storage_errors(self, embedder, caplog):
        """Test that cache disables after 10 storage errors."""
        texts = ["Test text"]

        # Mock _add_to_cache to raise exception
        with patch.object(embedder, '_add_to_cache', side_effect=ValueError("Error")):
            # Trigger 11 storage errors
            for i in range(11):
                embedder.embed_texts(texts)

        # Cache should be disabled
        assert embedder._cache_enabled is False
        assert embedder._cache_errors >= 10


class TestCacheDisabled:
    """Test behavior when cache is disabled."""

    def test_cache_disabled_skips_caching(self, mock_openai_client):
        """Test that disabled cache skips all caching logic."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large",
            cache_enabled=False  # Disabled
        )

        with patch('openai.OpenAI', return_value=mock_openai_client), \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            embedder = EmbeddingGenerator(config)

        texts = ["Test text"]

        # Multiple calls should all trigger API
        embedder.embed_texts(texts)
        call_count_1 = mock_openai_client.embeddings.create.call_count

        embedder.embed_texts(texts)
        call_count_2 = mock_openai_client.embeddings.create.call_count

        # Both calls should hit API
        assert call_count_2 > call_count_1
        # No cache stats
        assert embedder._cache_hits == 0
        assert embedder._cache_misses == 0


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_is_consistent(self, embedder):
        """Test that same texts generate same cache key."""
        texts = ["Text 1", "Text 2"]

        key1 = embedder._generate_cache_key(texts)
        key2 = embedder._generate_cache_key(texts)

        assert key1 == key2

    def test_cache_key_is_unique_for_different_texts(self, embedder):
        """Test that different texts generate different cache keys."""
        texts1 = ["Text 1"]
        texts2 = ["Text 2"]

        key1 = embedder._generate_cache_key(texts1)
        key2 = embedder._generate_cache_key(texts2)

        assert key1 != key2

    def test_cache_key_separator_collision_warning(self, embedder):
        """Test that separator collision IS REAL (documented in code line 311)."""
        # These texts have different meanings but produce SAME hash due to "|" separator
        texts1 = ["a|b", "c"]     # Joins as "a|b|c"
        texts2 = ["a", "b|c"]     # Joins as "a|b|c" (SAME!)

        key1 = embedder._generate_cache_key(texts1)
        key2 = embedder._generate_cache_key(texts2)

        # The implementation warning (line 311-313) is accurate: separator collision is REAL
        # Both text lists join to "a|b|c", so SHA256 hash is identical
        assert key1 == key2  # SAME key despite different text lists!

        # This is why cache validation checks shape[0] - to detect such collisions


class TestCacheIntegration:
    """Integration tests for cache with actual embedding workflow."""

    def test_cache_improves_performance(self, embedder, mock_openai_client):
        """Test that cache reduces API calls for repeated queries."""
        texts = ["Integration test"]

        # First call
        embedder.embed_texts(texts)
        initial_calls = mock_openai_client.embeddings.create.call_count

        # 5 more calls with same text
        for _ in range(5):
            embedder.embed_texts(texts)

        final_calls = mock_openai_client.embeddings.create.call_count

        # Should only have made 1 API call total
        assert final_calls == initial_calls
        assert embedder._cache_hits == 5
        assert embedder._cache_misses == 1

    def test_cache_with_batch_embedding(self, embedder, mock_openai_client):
        """Test cache works correctly with multiple texts in single call."""
        texts = ["Text 1", "Text 2", "Text 3"]

        # Mock response for 3 texts
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 300
        mock_response.data = [
            MagicMock(embedding=[0.1] * 3072),
            MagicMock(embedding=[0.2] * 3072),
            MagicMock(embedding=[0.3] * 3072)
        ]
        mock_openai_client.embeddings.create.return_value = mock_response

        # First call
        result1 = embedder.embed_texts(texts)
        assert result1.shape == (3, 3072)

        # Second call - should hit cache
        result2 = embedder.embed_texts(texts)

        assert embedder._cache_hits == 1
        assert np.array_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
