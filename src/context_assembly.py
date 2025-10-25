"""
PHASE 6: Context Assembly

Assembles retrieved chunks into LLM-ready context with:
1. SAC summary stripping (keep only raw_content)
2. Chunk concatenation with proper formatting
3. Provenance tracking (citations, document references)

Based on best practices:
- Strip context used for embeddings (SAC summaries)
- Add citations for source attribution
- Include document metadata for verification
- Format for optimal LLM comprehension

Usage:
    from context_assembly import ContextAssembler, CitationFormat

    assembler = ContextAssembler(
        citation_format=CitationFormat.INLINE,
        include_metadata=True
    )

    assembled = assembler.assemble(
        chunks=retrieved_chunks,
        max_chunks=6,
        max_tokens=4000
    )

    # Use assembled.context for LLM
    prompt = f"Context:\n{assembled.context}\n\nQuestion: {question}"
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class CitationFormat(Enum):
    """Citation format styles for assembled context."""

    INLINE = "inline"  # [Chunk 1] inline citations
    FOOTNOTE = "footnote"  # Numbered footnotes at end
    SIMPLE = "simple"  # Simple [1], [2] markers
    DETAILED = "detailed"  # [Doc: GRI_306.pdf, Section: 3.2, Page: 15]


@dataclass
class ChunkProvenance:
    """Provenance information for a chunk."""

    chunk_id: str
    document_id: str
    document_name: Optional[str] = None
    section_title: Optional[str] = None
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None

    def to_citation(self, format: CitationFormat, chunk_number: int) -> str:
        """Generate citation string based on format."""

        if format == CitationFormat.INLINE:
            return f"[Chunk {chunk_number}]"

        elif format == CitationFormat.SIMPLE:
            return f"[{chunk_number}]"

        elif format == CitationFormat.DETAILED:
            parts = []
            if self.document_name:
                parts.append(f"Doc: {self.document_name}")
            if self.section_title:
                parts.append(f"Section: {self.section_title}")
            if self.page_number:
                parts.append(f"Page: {self.page_number}")

            citation = ", ".join(parts) if parts else f"Chunk {chunk_number}"
            return f"[{citation}]"

        elif format == CitationFormat.FOOTNOTE:
            return f"[{chunk_number}]"

        return f"[{chunk_number}]"


@dataclass
class AssembledContext:
    """Assembled context ready for LLM consumption."""

    context: str  # Full assembled context string
    chunks_used: int  # Number of chunks included
    total_length: int  # Total character length
    provenances: List[ChunkProvenance]  # Provenance for each chunk
    metadata: Dict  # Additional metadata

    def get_citations(self) -> List[str]:
        """Get list of all citations."""
        return [
            f"[{i+1}] {prov.document_name or prov.document_id}"
            for i, prov in enumerate(self.provenances)
        ]


class ContextAssembler:
    """
    Assembles retrieved chunks into LLM-ready context.

    Key operations:
    1. Strip SAC summaries (use raw_content only)
    2. Format chunks with citations
    3. Add provenance tracking
    4. Respect token limits
    """

    def __init__(
        self,
        citation_format: CitationFormat = CitationFormat.INLINE,
        include_metadata: bool = True,
        chunk_separator: str = "\n\n---\n\n",
        add_chunk_headers: bool = True,
        max_chunk_length: Optional[int] = None,
    ):
        """
        Initialize context assembler.

        Args:
            citation_format: Citation style to use
            include_metadata: Whether to include document metadata
            chunk_separator: Separator between chunks
            add_chunk_headers: Add headers like "Chunk 1:"
            max_chunk_length: Max length per chunk (truncate if exceeded)
        """
        self.citation_format = citation_format
        self.include_metadata = include_metadata
        self.chunk_separator = chunk_separator
        self.add_chunk_headers = add_chunk_headers
        self.max_chunk_length = max_chunk_length

        logger.info(
            f"ContextAssembler initialized: "
            f"format={citation_format.value}, "
            f"metadata={include_metadata}"
        )

    def assemble(
        self,
        chunks: List[Dict],
        max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None,
        query: Optional[str] = None,
    ) -> AssembledContext:
        """
        Assemble chunks into LLM-ready context.

        Args:
            chunks: List of retrieved chunk dicts
            max_chunks: Maximum number of chunks to include
            max_tokens: Maximum total tokens (approx 4 chars = 1 token)
            query: Optional query for context (not used yet)

        Returns:
            AssembledContext with formatted text and metadata
        """
        if not chunks:
            logger.warning("No chunks provided for assembly")
            return AssembledContext(
                context="", chunks_used=0, total_length=0, provenances=[], metadata={}
            )

        # Limit number of chunks
        if max_chunks:
            chunks = chunks[:max_chunks]

        logger.info(f"Assembling context from {len(chunks)} chunks")

        # Process each chunk
        assembled_parts = []
        provenances = []
        total_length = 0

        for i, chunk in enumerate(chunks, start=1):
            # Extract provenance
            provenance = self._extract_provenance(chunk, i)
            provenances.append(provenance)

            # Strip SAC summary and get raw content
            content = self._strip_sac_summary(chunk)

            # Truncate if needed
            if self.max_chunk_length and len(content) > self.max_chunk_length:
                content = content[: self.max_chunk_length] + "..."
                logger.debug(f"Truncated chunk {i} to {self.max_chunk_length} chars")

            # Format chunk with citation
            formatted = self._format_chunk(content=content, provenance=provenance, chunk_number=i)

            # Check token limit
            if max_tokens:
                estimated_tokens = (total_length + len(formatted)) // 4
                if estimated_tokens > max_tokens:
                    logger.info(
                        f"Reached token limit at chunk {i}/{len(chunks)} "
                        f"(~{estimated_tokens} tokens)"
                    )
                    break

            assembled_parts.append(formatted)
            total_length += len(formatted)

        # Join all parts
        context = self.chunk_separator.join(assembled_parts)

        # Add footnotes if needed
        if self.citation_format == CitationFormat.FOOTNOTE:
            footnotes = self._generate_footnotes(provenances)
            context += f"\n\n{footnotes}"

        # Create metadata
        metadata = {
            "citation_format": self.citation_format.value,
            "chunks_requested": len(chunks),
            "chunks_included": len(assembled_parts),
            "avg_chunk_length": total_length // len(assembled_parts) if assembled_parts else 0,
        }

        logger.info(
            f"Context assembled: {len(assembled_parts)} chunks, "
            f"{total_length} chars (~{total_length // 4} tokens)"
        )

        return AssembledContext(
            context=context,
            chunks_used=len(assembled_parts),
            total_length=total_length,
            provenances=provenances,
            metadata=metadata,
        )

    def _strip_sac_summary(self, chunk: Dict) -> str:
        """
        Strip SAC summary and return only raw content.

        During embedding (PHASE 3), chunks are formatted as:
            context + "\n\n" + raw_content

        During retrieval, we want ONLY raw_content (without SAC context).

        Args:
            chunk: Chunk dict from retrieval

        Returns:
            Raw content without SAC summary
        """
        # Try to get raw_content directly
        if "raw_content" in chunk:
            return chunk["raw_content"]

        # Fallback: use 'content' or 'text'
        content = chunk.get("content") or chunk.get("text", "")

        # If content has SAC format (context + raw), try to strip
        # SAC format: "Context summary\n\n<original text>"
        # Simple heuristic: if there's a double newline, take text after it
        if "\n\n" in content:
            parts = content.split("\n\n", 1)
            if len(parts) == 2:
                # Check if first part looks like SAC summary (short, generic)
                first_part = parts[0]
                if len(first_part) < 200 and not first_part.startswith("#"):
                    # Likely SAC summary, return second part
                    logger.debug("Stripped SAC summary from chunk")
                    return parts[1]

        # No SAC detected, return as-is
        return content

    def _extract_provenance(self, chunk: Dict, chunk_number: int) -> ChunkProvenance:
        """
        Extract provenance information from chunk.

        Args:
            chunk: Chunk dict with metadata
            chunk_number: Position in results (1-indexed)

        Returns:
            ChunkProvenance with extracted metadata
        """
        # Extract document info
        document_id = chunk.get("document_id", "unknown")
        document_name = chunk.get("document_name")

        # If document_name not provided, try to extract from document_id
        if not document_name and document_id:
            # Remove path and extension: "data/docs/GRI_306.pdf" -> "GRI_306"
            import os

            document_name = os.path.splitext(os.path.basename(document_id))[0]

        # Extract section info
        section_title = chunk.get("section_title")
        section_id = chunk.get("section_id")

        # Extract page info
        page_number = chunk.get("page_number") or chunk.get("page")

        # Chunk info
        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_number}")
        chunk_index = chunk.get("chunk_index")

        return ChunkProvenance(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name=document_name,
            section_title=section_title,
            section_id=section_id,
            page_number=page_number,
            chunk_index=chunk_index,
        )

    def _format_chunk(self, content: str, provenance: ChunkProvenance, chunk_number: int) -> str:
        """
        Format a single chunk with citations.

        Args:
            content: Raw chunk content
            provenance: Provenance metadata
            chunk_number: Chunk position (1-indexed)

        Returns:
            Formatted chunk string
        """
        parts = []

        # Add header
        if self.add_chunk_headers:
            citation = provenance.to_citation(self.citation_format, chunk_number)
            header = f"**{citation}**"

            # Add metadata in header if detailed format
            if self.include_metadata and self.citation_format == CitationFormat.DETAILED:
                header = citation

            parts.append(header)

        # Add content
        parts.append(content)

        return "\n".join(parts)

    def _generate_footnotes(self, provenances: List[ChunkProvenance]) -> str:
        """
        Generate footnotes section for FOOTNOTE format.

        Args:
            provenances: List of provenance metadata

        Returns:
            Formatted footnotes string
        """
        footnotes = ["**Sources:**"]

        for i, prov in enumerate(provenances, start=1):
            parts = [f"[{i}]"]

            if prov.document_name:
                parts.append(prov.document_name)

            if prov.section_title:
                parts.append(f"Section: {prov.section_title}")

            if prov.page_number:
                parts.append(f"Page {prov.page_number}")

            footnotes.append(" ".join(parts))

        return "\n".join(footnotes)


# Convenience function
def assemble_context(
    chunks: List[Dict],
    max_chunks: int = 6,
    citation_format: str = "inline",
    include_metadata: bool = True,
) -> str:
    """
    Convenience function for quick context assembly.

    Args:
        chunks: Retrieved chunks
        max_chunks: Maximum chunks to include
        citation_format: Citation style ("inline", "simple", "detailed", "footnote")
        include_metadata: Include document metadata

    Returns:
        Assembled context string
    """
    format_map = {
        "inline": CitationFormat.INLINE,
        "simple": CitationFormat.SIMPLE,
        "detailed": CitationFormat.DETAILED,
        "footnote": CitationFormat.FOOTNOTE,
    }

    assembler = ContextAssembler(
        citation_format=format_map.get(citation_format, CitationFormat.INLINE),
        include_metadata=include_metadata,
    )

    result = assembler.assemble(chunks, max_chunks=max_chunks)

    return result.context


# Example usage
if __name__ == "__main__":
    print("=== PHASE 6: Context Assembly Example ===\n")

    print("1. Create sample retrieved chunks:")
    sample_chunks = [
        {
            "chunk_id": "chunk_001",
            "document_id": "GRI_306.pdf",
            "document_name": "GRI 306",
            "section_title": "Disclosure 306-3",
            "page_number": 15,
            "raw_content": "Organizations shall report waste generated in metric tonnes.",
            "content": "This chunk discusses waste reporting.\n\nOrganizations shall report waste generated in metric tonnes.",
            "rerank_score": 0.92,
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "GRI_306.pdf",
            "document_name": "GRI 306",
            "section_title": "Disclosure 306-4",
            "page_number": 17,
            "raw_content": "Waste diverted from disposal shall be categorized by type.",
            "rerank_score": 0.88,
        },
    ]

    print("2. Assemble with INLINE citations:")
    assembler = ContextAssembler(citation_format=CitationFormat.INLINE)
    result = assembler.assemble(sample_chunks, max_chunks=2)

    print(f"\nAssembled Context ({result.chunks_used} chunks):")
    print("-" * 60)
    print(result.context)
    print("-" * 60)

    print("\n3. Assemble with DETAILED citations:")
    assembler_detailed = ContextAssembler(citation_format=CitationFormat.DETAILED)
    result_detailed = assembler_detailed.assemble(sample_chunks, max_chunks=2)

    print(f"\nDetailed Format:")
    print("-" * 60)
    print(result_detailed.context)
    print("-" * 60)

    print("\n4. Use convenience function:")
    context = assemble_context(sample_chunks, max_chunks=2, citation_format="simple")
    print(f"\nSimple Format:")
    print("-" * 60)
    print(context)
    print("-" * 60)

    print("\n=== Implementation complete! ===")
