"""
Tests for PHASE 6: Context Assembly

Tests cover:
1. Basic context assembly
2. SAC summary stripping
3. Citation format variations (inline, simple, detailed, footnote)
4. Provenance extraction
5. Token and chunk limits
6. Edge cases (empty chunks, missing metadata)
"""

import pytest
from src.context_assembly import (
    ContextAssembler,
    CitationFormat,
    ChunkProvenance,
    AssembledContext,
    assemble_context
)


# Test fixtures
@pytest.fixture
def sample_chunks_with_sac():
    """Sample chunks with SAC summaries (context + raw_content)."""
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "GRI_306.pdf",
            "document_name": "GRI 306",
            "section_title": "Disclosure 306-3",
            "page_number": 15,
            "raw_content": "Organizations shall report waste generated in metric tonnes.",
            "content": "This chunk discusses waste reporting requirements.\n\nOrganizations shall report waste generated in metric tonnes.",
            "rerank_score": 0.92
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "GRI_306.pdf",
            "document_name": "GRI 306",
            "section_title": "Disclosure 306-4",
            "page_number": 17,
            "raw_content": "Waste diverted from disposal shall be categorized by composition.",
            "content": "This section covers waste diversion categories.\n\nWaste diverted from disposal shall be categorized by composition.",
            "rerank_score": 0.88
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "GRI_401.pdf",
            "document_name": "GRI 401",
            "section_title": "Management Approach",
            "page_number": 5,
            "raw_content": "The organization should describe its approach to employment practices.",
            "rerank_score": 0.75
        }
    ]


@pytest.fixture
def sample_chunks_without_metadata():
    """Sample chunks with minimal metadata."""
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "unknown.pdf",
            "content": "This is some content without metadata.",
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_123",
            "text": "Another chunk with minimal info.",
        }
    ]


# Test: Basic Assembly
def test_basic_context_assembly(sample_chunks_with_sac):
    """Test basic context assembly."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    assert isinstance(result, AssembledContext)
    assert result.chunks_used == 2
    assert result.total_length > 0
    assert len(result.provenances) == 2
    assert result.context != ""


def test_assembly_with_all_chunks(sample_chunks_with_sac):
    """Test assembly with all available chunks."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac)

    assert result.chunks_used == 3
    assert len(result.provenances) == 3


def test_empty_chunks_list():
    """Test assembly with empty chunks list."""
    assembler = ContextAssembler()

    result = assembler.assemble([])

    assert result.chunks_used == 0
    assert result.context == ""
    assert result.provenances == []


# Test: SAC Summary Stripping
def test_sac_summary_stripping(sample_chunks_with_sac):
    """Test that SAC summaries are stripped correctly."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=1)

    # Context should contain raw_content, not SAC summary
    assert "Organizations shall report waste generated" in result.context
    # Should NOT contain SAC summary
    assert "This chunk discusses waste reporting" not in result.context


def test_raw_content_preferred():
    """Test that raw_content is used when available."""
    chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "test.pdf",
            "raw_content": "This is the raw content.",
            "content": "SAC summary\n\nThis is the raw content."
        }
    ]

    assembler = ContextAssembler()
    result = assembler.assemble(chunks)

    # Should use raw_content directly
    assert "This is the raw content." in result.context
    assert "SAC summary" not in result.context


def test_fallback_to_content_field():
    """Test fallback when raw_content is missing."""
    chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "test.pdf",
            "content": "Only content field available."
        }
    ]

    assembler = ContextAssembler()
    result = assembler.assemble(chunks)

    assert "Only content field available." in result.context


# Test: Citation Formats
def test_inline_citation_format(sample_chunks_with_sac):
    """Test INLINE citation format."""
    assembler = ContextAssembler(citation_format=CitationFormat.INLINE)

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    assert "[Chunk 1]" in result.context
    assert "[Chunk 2]" in result.context


def test_simple_citation_format(sample_chunks_with_sac):
    """Test SIMPLE citation format."""
    assembler = ContextAssembler(citation_format=CitationFormat.SIMPLE)

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    assert "[1]" in result.context
    assert "[2]" in result.context


def test_detailed_citation_format(sample_chunks_with_sac):
    """Test DETAILED citation format."""
    assembler = ContextAssembler(citation_format=CitationFormat.DETAILED)

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=1)

    # Should contain document, section, page info
    assert "GRI 306" in result.context
    assert "Disclosure 306-3" in result.context
    assert "Page: 15" in result.context


def test_footnote_citation_format(sample_chunks_with_sac):
    """Test FOOTNOTE citation format."""
    assembler = ContextAssembler(citation_format=CitationFormat.FOOTNOTE)

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    # Should have numbered references in text
    assert "[1]" in result.context
    assert "[2]" in result.context

    # Should have Sources section at end
    assert "**Sources:**" in result.context
    assert "GRI 306" in result.context


# Test: Provenance Extraction
def test_provenance_extraction(sample_chunks_with_sac):
    """Test provenance information is extracted correctly."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=1)

    prov = result.provenances[0]
    assert prov.chunk_id == "chunk_001"
    assert prov.document_id == "GRI_306.pdf"
    assert prov.document_name == "GRI 306"
    assert prov.section_title == "Disclosure 306-3"
    assert prov.page_number == 15


def test_provenance_with_missing_metadata(sample_chunks_without_metadata):
    """Test provenance extraction with missing metadata."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_without_metadata, max_chunks=1)

    prov = result.provenances[0]
    assert prov.chunk_id == "chunk_001"
    assert prov.document_id == "unknown.pdf"
    # Document name should be extracted from document_id
    assert prov.document_name == "unknown"


def test_document_name_extraction_from_id():
    """Test that document name is extracted from document_id when missing."""
    chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "data/docs/GRI_306.pdf",
            "content": "Test content"
        }
    ]

    assembler = ContextAssembler()
    result = assembler.assemble(chunks)

    prov = result.provenances[0]
    assert prov.document_name == "GRI_306"


# Test: Chunk and Token Limits
def test_max_chunks_limit(sample_chunks_with_sac):
    """Test max_chunks parameter limits output."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    assert result.chunks_used == 2
    assert len(result.provenances) == 2


def test_max_tokens_limit():
    """Test max_tokens parameter truncates output."""
    # Create chunks with known length
    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "document_id": "test.pdf",
            "raw_content": "A" * 100  # 100 chars = ~25 tokens
        }
        for i in range(10)
    ]

    assembler = ContextAssembler()

    # Limit to ~100 tokens (400 chars)
    result = assembler.assemble(chunks, max_tokens=100)

    # Should include fewer than all chunks
    assert result.chunks_used < 10
    # Should be within token limit (with some tolerance)
    estimated_tokens = result.total_length // 4
    assert estimated_tokens <= 150  # Some tolerance for formatting


def test_max_chunk_length_truncation():
    """Test max_chunk_length truncates individual chunks."""
    chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "test.pdf",
            "raw_content": "A" * 500  # Very long chunk
        }
    ]

    assembler = ContextAssembler(max_chunk_length=100)

    result = assembler.assemble(chunks)

    # Content should be truncated
    assert "..." in result.context
    # Should not exceed max length by much
    assert result.total_length < 200  # 100 + formatting


# Test: Formatting Options
def test_chunk_separator_custom():
    """Test custom chunk separator."""
    chunks = [
        {"chunk_id": "c1", "document_id": "test.pdf", "raw_content": "Content 1"},
        {"chunk_id": "c2", "document_id": "test.pdf", "raw_content": "Content 2"}
    ]

    assembler = ContextAssembler(chunk_separator="\n\n***\n\n")

    result = assembler.assemble(chunks)

    assert "\n\n***\n\n" in result.context


def test_without_chunk_headers():
    """Test assembly without chunk headers."""
    chunks = [
        {"chunk_id": "c1", "document_id": "test.pdf", "raw_content": "Content 1"}
    ]

    assembler = ContextAssembler(add_chunk_headers=False)

    result = assembler.assemble(chunks)

    # Should not have citation headers
    assert "[Chunk 1]" not in result.context
    # Should just have content
    assert "Content 1" in result.context


# Test: Metadata
def test_metadata_included(sample_chunks_with_sac):
    """Test that metadata is populated correctly."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    assert "citation_format" in result.metadata
    assert result.metadata["chunks_requested"] == 3
    assert result.metadata["chunks_included"] == 2
    assert result.metadata["avg_chunk_length"] > 0


def test_get_citations_method(sample_chunks_with_sac):
    """Test get_citations() method."""
    assembler = ContextAssembler()

    result = assembler.assemble(sample_chunks_with_sac, max_chunks=2)

    citations = result.get_citations()

    assert len(citations) == 2
    assert "[1] GRI 306" in citations[0]
    assert "[2] GRI 306" in citations[1]


# Test: Convenience Function
def test_convenience_function(sample_chunks_with_sac):
    """Test assemble_context convenience function."""
    context = assemble_context(
        sample_chunks_with_sac,
        max_chunks=2,
        citation_format="inline"
    )

    assert isinstance(context, str)
    assert len(context) > 0
    assert "[Chunk 1]" in context


def test_convenience_function_with_different_formats(sample_chunks_with_sac):
    """Test convenience function with different citation formats."""
    # Test each format
    for format_name in ["inline", "simple", "detailed", "footnote"]:
        context = assemble_context(
            sample_chunks_with_sac,
            max_chunks=1,
            citation_format=format_name
        )

        assert isinstance(context, str)
        assert len(context) > 0


# Test: ChunkProvenance
def test_chunk_provenance_to_citation():
    """Test ChunkProvenance.to_citation() method."""
    prov = ChunkProvenance(
        chunk_id="chunk_001",
        document_id="GRI_306.pdf",
        document_name="GRI 306",
        section_title="Disclosure 306-3",
        page_number=15
    )

    # Test different formats
    assert prov.to_citation(CitationFormat.INLINE, 1) == "[Chunk 1]"
    assert prov.to_citation(CitationFormat.SIMPLE, 1) == "[1]"
    assert "GRI 306" in prov.to_citation(CitationFormat.DETAILED, 1)
    assert "Page: 15" in prov.to_citation(CitationFormat.DETAILED, 1)


# Integration Test
@pytest.mark.integration
def test_full_assembly_pipeline(sample_chunks_with_sac):
    """
    Integration test: Full context assembly pipeline.

    Simulates: Retrieval → Context Assembly → LLM Prompt
    """
    # Step 1: Assemble context
    assembler = ContextAssembler(
        citation_format=CitationFormat.INLINE,
        include_metadata=True
    )

    result = assembler.assemble(
        chunks=sample_chunks_with_sac,
        max_chunks=3,
        max_tokens=2000
    )

    # Step 2: Verify assembled context
    assert result.chunks_used == 3
    assert result.total_length > 0

    # Step 3: Create LLM prompt
    query = "What are the waste reporting requirements?"
    prompt = f"""Context:
{result.context}

Question: {query}

Answer (with citations):"""

    # Step 4: Verify prompt structure
    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "[Chunk" in prompt  # Has citations

    # Step 5: Verify citations are extractable
    citations = result.get_citations()
    assert len(citations) == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
