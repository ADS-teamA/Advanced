"""
Integration tests for Contextual Retrieval in the full pipeline.

Tests end-to-end integration of:
- Multi-layer chunking with contextual retrieval
- Fallback mechanisms
- Error handling in real pipeline scenarios
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata
from config import ChunkingConfig, ContextGenerationConfig


# Mock ExtractedDocument structure
@dataclass
class MockSection:
    """Mock section for testing."""
    section_id: str
    title: str
    path: str
    content: str
    summary: Optional[str]
    level: int
    depth: int
    page_number: int
    char_start: int
    char_end: int


@dataclass
class MockExtractedDocument:
    """Mock extracted document for testing."""
    document_id: str
    document_summary: str
    full_text: str
    sections: List[MockSection]


class TestMultiLayerChunkerIntegration:
    """Integration tests for MultiLayerChunker with Contextual Retrieval."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample extracted document for testing."""
        sections = [
            MockSection(
                section_id="sec1",
                title="Introduction",
                path="Ch1 > Introduction",
                content="This is a long introduction section. " * 50,  # ~250 chars
                summary="Introduction to the document",
                level=1,
                depth=1,
                page_number=1,
                char_start=0,
                char_end=250
            ),
            MockSection(
                section_id="sec2",
                title="Safety Parameters",
                path="Ch2 > Safety Parameters",
                content="The primary cooling circuit operates at 15.7 MPa. " * 20,  # ~1000 chars
                summary="Safety parameters specification",
                level=1,
                depth=1,
                page_number=2,
                char_start=250,
                char_end=1250
            )
        ]

        return MockExtractedDocument(
            document_id="test_doc_001",
            document_summary="Test nuclear reactor safety document",
            full_text="Full document text here",
            sections=sections
        )

    def test_chunker_with_contextual_retrieval_enabled(self, sample_document):
        """Test chunker with contextual retrieval enabled."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=0,
            enable_contextual=True,
            context_config=ContextGenerationConfig(
                provider="anthropic",
                model="haiku"
            )
        )

        # Mock the contextual retrieval
        with patch("multi_layer_chunker.ContextualRetrieval") as MockContextRetrieval:
            # Setup mock context generator
            mock_generator = Mock()

            # Mock successful context generation
            def mock_batch_generate(chunks):
                from contextual_retrieval import ChunkContext
                return [
                    ChunkContext(
                        chunk_text=chunk_text,
                        context=f"Context for: {chunk_text[:30]}...",
                        success=True
                    )
                    for chunk_text, _ in chunks
                ]

            mock_generator.generate_contexts_batch = Mock(side_effect=mock_batch_generate)
            MockContextRetrieval.return_value = mock_generator

            # Create chunker
            chunker = MultiLayerChunker(config=config)

            # Process document
            result = chunker.chunk_document(sample_document)

            # Verify structure
            assert "layer1" in result
            assert "layer2" in result
            assert "layer3" in result

            # Verify Layer 1 (document level)
            assert len(result["layer1"]) == 1
            assert result["layer1"][0].metadata.layer == 1

            # Verify Layer 2 (section level)
            assert len(result["layer2"]) == 2
            for chunk in result["layer2"]:
                assert chunk.metadata.layer == 2

            # Verify Layer 3 (chunk level with contextual retrieval)
            assert len(result["layer3"]) > 0
            for chunk in result["layer3"]:
                assert chunk.metadata.layer == 3
                # Verify context was added (content != raw_content)
                if chunk.content != chunk.raw_content:
                    assert "Context for:" in chunk.content

            # Verify batch generation was called
            assert mock_generator.generate_contexts_batch.called

    def test_chunker_fallback_to_basic_on_error(self, sample_document):
        """Test that chunker falls back to basic mode when context generation fails."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=0,
            enable_contextual=True,
            context_config=ContextGenerationConfig(
                provider="anthropic",
                model="haiku",
                fallback_to_basic=True
            )
        )

        # Mock contextual retrieval to raise exception
        with patch("multi_layer_chunker.ContextualRetrieval") as MockContextRetrieval:
            mock_generator = Mock()
            mock_generator.generate_contexts_batch = Mock(
                side_effect=Exception("API error - rate limit exceeded")
            )
            MockContextRetrieval.return_value = mock_generator

            # Create chunker
            chunker = MultiLayerChunker(config=config)

            # Process document - should fallback to basic chunking
            result = chunker.chunk_document(sample_document)

            # Should still produce chunks
            assert len(result["layer3"]) > 0

            # Chunks should be basic (content == raw_content)
            for chunk in result["layer3"]:
                assert chunk.content == chunk.raw_content

    def test_chunker_basic_mode_disabled_contextual(self, sample_document):
        """Test chunker with contextual retrieval disabled (basic mode)."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=0,
            enable_contextual=False
        )

        # Create chunker
        chunker = MultiLayerChunker(config=config)

        # Process document
        result = chunker.chunk_document(sample_document)

        # Verify chunks created
        assert len(result["layer3"]) > 0

        # All chunks should be basic (no augmentation)
        for chunk in result["layer3"]:
            assert chunk.content == chunk.raw_content
            assert chunk.metadata.layer == 3

    def test_chunking_stats(self, sample_document):
        """Test that chunking stats are correctly calculated."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=0,
            enable_contextual=False
        )

        chunker = MultiLayerChunker(config=config)
        result = chunker.chunk_document(sample_document)

        # Get stats
        stats = chunker.get_chunking_stats(result)

        # Verify stats structure
        assert "layer1_count" in stats
        assert "layer2_count" in stats
        assert "layer3_count" in stats
        assert "total_chunks" in stats

        assert stats["layer1_count"] == 1
        assert stats["layer2_count"] == 2
        assert stats["layer3_count"] > 0
        assert stats["total_chunks"] == stats["layer1_count"] + stats["layer2_count"] + stats["layer3_count"]

        # Verify Layer 3 stats
        assert "layer3_avg_size" in stats
        assert "layer3_min_size" in stats
        assert "layer3_max_size" in stats
        assert stats["layer3_avg_size"] > 0

    def test_empty_sections_handling(self):
        """Test that empty sections are handled gracefully."""
        empty_doc = MockExtractedDocument(
            document_id="empty_doc",
            document_summary="Empty document",
            full_text="",
            sections=[
                MockSection(
                    section_id="empty_sec",
                    title="Empty Section",
                    path="Ch1",
                    content="",  # Empty content
                    summary=None,
                    level=1,
                    depth=1,
                    page_number=1,
                    char_start=0,
                    char_end=0
                )
            ]
        )

        config = ChunkingConfig(enable_contextual=False)
        chunker = MultiLayerChunker(config=config)
        result = chunker.chunk_document(empty_doc)

        # Should handle empty sections without crashing
        assert "layer1" in result
        assert "layer2" in result
        assert "layer3" in result

        # Layer 3 should have no chunks (empty content)
        assert len(result["layer3"]) == 0


class TestContextualRetrievalErrorScenarios:
    """Test error scenarios in contextual retrieval integration."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return MockExtractedDocument(
            document_id="test_doc",
            document_summary="Test document",
            full_text="Test",
            sections=[
                MockSection(
                    section_id="sec1",
                    title="Section 1",
                    path="Ch1",
                    content="This is test content. " * 30,
                    summary="Test section",
                    level=1,
                    depth=1,
                    page_number=1,
                    char_start=0,
                    char_end=600
                )
            ]
        )

    def test_partial_context_generation_failures(self, sample_document):
        """Test handling when some chunks fail context generation."""
        config = ChunkingConfig(
            chunk_size=200,
            enable_contextual=True,
            context_config=ContextGenerationConfig(
                provider="anthropic",
                model="haiku",
                fallback_to_basic=False  # Don't fallback completely
            )
        )

        with patch("multi_layer_chunker.ContextualRetrieval") as MockContextRetrieval:
            mock_generator = Mock()

            # Mock partial failures
            def mock_batch_generate(chunks):
                from contextual_retrieval import ChunkContext
                results = []
                for i, (chunk_text, _) in enumerate(chunks):
                    if i % 2 == 0:  # Every other chunk fails
                        results.append(ChunkContext(
                            chunk_text=chunk_text,
                            context="",
                            success=False,
                            error="API error"
                        ))
                    else:
                        results.append(ChunkContext(
                            chunk_text=chunk_text,
                            context=f"Context for chunk {i}",
                            success=True
                        ))
                return results

            mock_generator.generate_contexts_batch = Mock(side_effect=mock_batch_generate)
            MockContextRetrieval.return_value = mock_generator

            chunker = MultiLayerChunker(config=config)
            result = chunker.chunk_document(sample_document)

            # Should have chunks
            assert len(result["layer3"]) > 0

            # Some should have context, some should not
            with_context = [c for c in result["layer3"] if c.content != c.raw_content]
            without_context = [c for c in result["layer3"] if c.content == c.raw_content]

            # Should have both
            assert len(with_context) > 0
            assert len(without_context) > 0

    def test_rate_limit_recovery(self, sample_document):
        """Test that rate limiting is handled in batch processing."""
        config = ChunkingConfig(
            chunk_size=200,
            enable_contextual=True,
            context_config=ContextGenerationConfig(
                provider="anthropic",
                model="haiku",
                batch_size=2,  # Small batches
                max_workers=1  # Sequential to test rate limiting
            )
        )

        with patch("multi_layer_chunker.ContextualRetrieval") as MockContextRetrieval:
            mock_generator = Mock()

            # Simulate rate limit then success
            call_count = 0

            def mock_batch_generate(chunks):
                from contextual_retrieval import ChunkContext
                nonlocal call_count
                call_count += 1

                # First batch fails with rate limit, subsequent succeed
                if call_count == 1:
                    raise Exception("Rate limit exceeded (429)")

                return [
                    ChunkContext(
                        chunk_text=chunk_text,
                        context=f"Context {i}",
                        success=True
                    )
                    for i, (chunk_text, _) in enumerate(chunks)
                ]

            mock_generator.generate_contexts_batch = Mock(side_effect=mock_batch_generate)
            MockContextRetrieval.return_value = mock_generator

            chunker = MultiLayerChunker(config=config)

            # This should trigger fallback due to exception in batch generation
            with pytest.raises(Exception):
                # Without fallback_to_basic, this should raise
                config.context_config.fallback_to_basic = False
                chunker = MultiLayerChunker(config=config)
                chunker.chunk_document(sample_document)


class TestChunkMetadataIntegrity:
    """Test that chunk metadata is correctly preserved."""

    def test_metadata_propagation(self):
        """Test that metadata is correctly propagated through layers."""
        doc = MockExtractedDocument(
            document_id="metadata_test",
            document_summary="Metadata test doc",
            full_text="Test",
            sections=[
                MockSection(
                    section_id="meta_sec",
                    title="Metadata Section",
                    path="Ch1 > Sec1.1 > Subsec1.1.1",
                    content="Content here. " * 100,
                    summary="Section summary",
                    level=3,
                    depth=3,
                    page_number=5,
                    char_start=1000,
                    char_end=2400
                )
            ]
        )

        config = ChunkingConfig(enable_contextual=False)
        chunker = MultiLayerChunker(config=config)
        result = chunker.chunk_document(doc)

        # Check Layer 3 metadata
        for chunk in result["layer3"]:
            assert chunk.metadata.document_id == "metadata_test"
            assert chunk.metadata.section_id == "meta_sec"
            assert chunk.metadata.section_title == "Metadata Section"
            assert chunk.metadata.section_path == "Ch1 > Sec1.1 > Subsec1.1.1"
            assert chunk.metadata.section_level == 3
            assert chunk.metadata.section_depth == 3
            assert chunk.metadata.page_number == 5

        # Check Layer 2 metadata
        for chunk in result["layer2"]:
            assert chunk.metadata.section_title == "Metadata Section"
            assert chunk.metadata.parent_chunk_id == "metadata_test_L1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
