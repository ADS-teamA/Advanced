"""
PHASE 3: Multi-Layer Chunking with Summary-Augmented Chunking (SAC)

Based on research:
- LegalBench-RAG: RecursiveCharacterTextSplitter > Fixed-size (Prec@1: 6.41% vs 2.40%)
- Reuter et al., 2024: SAC reduces DRM by 58%
- Lima, 2024: Multi-layer embeddings improve essential chunks by 2.3x

Implementation:
- Layer 1: Document level (1 chunk per document)
- Layer 2: Section level (1 chunk per section)
- Layer 3: Chunk level (RCTS 500 chars + SAC)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: str
    layer: int  # 1=document, 2=section, 3=chunk
    document_id: str
    section_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    # Position info
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0

    # Hierarchy context
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    section_level: int = 0
    section_depth: int = 0


@dataclass
class Chunk:
    """
    A single chunk with content and metadata.

    For Layer 3 (chunk level), content includes SAC summary prepended.
    For embedding, use 'content'.
    For generation, use 'raw_content' (without SAC summary).
    """
    chunk_id: str
    content: str  # For embedding (with SAC if Layer 3)
    raw_content: str  # For generation (without SAC)
    metadata: ChunkMetadata

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "raw_content": self.raw_content,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "layer": self.metadata.layer,
                "document_id": self.metadata.document_id,
                "section_id": self.metadata.section_id,
                "parent_chunk_id": self.metadata.parent_chunk_id,
                "page_number": self.metadata.page_number,
                "char_start": self.metadata.char_start,
                "char_end": self.metadata.char_end,
                "section_title": self.metadata.section_title,
                "section_path": self.metadata.section_path,
                "section_level": self.metadata.section_level,
                "section_depth": self.metadata.section_depth
            }
        }


class MultiLayerChunker:
    """
    Multi-layer chunker with Summary-Augmented Chunking (SAC).

    Creates 3 layers:
    - Layer 1: Document level (summary only)
    - Layer 2: Section level (section summaries)
    - Layer 3: Chunk level (RCTS 500 chars + SAC)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Summary-Augmented Chunking (Reuter et al., 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        enable_sac: bool = True
    ):
        """
        Initialize multi-layer chunker.

        Args:
            chunk_size: Characters per chunk (default: 500, optimal per research)
            chunk_overlap: Overlap between chunks (default: 0, RCTS handles naturally)
            enable_sac: Enable Summary-Augmented Chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_sac = enable_sac

        # Initialize RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "; ",    # Clause separators
                ", ",    # Sub-clause separators
                " ",     # Word boundaries
                ""       # Character fallback
            ]
        )

        logger.info(
            f"MultiLayerChunker initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, SAC={enable_sac}"
        )

    def chunk_document(
        self,
        extracted_doc
    ) -> Dict[str, List[Chunk]]:
        """
        Create multi-layer chunks from ExtractedDocument.

        Args:
            extracted_doc: ExtractedDocument from DoclingExtractorV2

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing chunks
        """

        logger.info(f"Chunking document: {extracted_doc.document_id}")

        # Layer 1: Document level
        layer1_chunks = self._create_layer1_document(extracted_doc)

        # Layer 2: Section level
        layer2_chunks = self._create_layer2_sections(extracted_doc)

        # Layer 3: Chunk level with SAC
        layer3_chunks = self._create_layer3_chunks(extracted_doc)

        logger.info(
            f"Created {len(layer1_chunks)} L1, "
            f"{len(layer2_chunks)} L2, "
            f"{len(layer3_chunks)} L3 chunks"
        )

        return {
            "layer1": layer1_chunks,
            "layer2": layer2_chunks,
            "layer3": layer3_chunks,
            "total_chunks": len(layer1_chunks) + len(layer2_chunks) + len(layer3_chunks)
        }

    def _create_layer1_document(self, extracted_doc) -> List[Chunk]:
        """
        Layer 1: Document-level chunk (summary only).

        Purpose:
        - Global filtering during retrieval
        - Document identification
        - DRM prevention
        """

        # Use document summary if available, else truncate full text
        content = extracted_doc.document_summary or extracted_doc.full_text[:500]

        chunk = Chunk(
            chunk_id=f"{extracted_doc.document_id}_L1",
            content=content,
            raw_content=content,
            metadata=ChunkMetadata(
                chunk_id=f"{extracted_doc.document_id}_L1",
                layer=1,
                document_id=extracted_doc.document_id,
                section_id=None,
                parent_chunk_id=None,
                page_number=0,
                char_start=0,
                char_end=len(content)
            )
        )

        return [chunk]

    def _create_layer2_sections(self, extracted_doc) -> List[Chunk]:
        """
        Layer 2: Section-level chunks.

        Purpose:
        - Mid-level context
        - Section-specific queries
        - Context expansion when needed
        """

        chunks = []

        for section in extracted_doc.sections:
            # Use section summary if available, else section content
            content = section.summary or section.content

            chunk = Chunk(
                chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                content=content,
                raw_content=section.content,  # Always use raw content
                metadata=ChunkMetadata(
                    chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                    layer=2,
                    document_id=extracted_doc.document_id,
                    section_id=section.section_id,
                    parent_chunk_id=f"{extracted_doc.document_id}_L1",
                    page_number=section.page_number,
                    char_start=section.char_start,
                    char_end=section.char_end,
                    section_title=section.title,
                    section_path=section.path,
                    section_level=section.level,
                    section_depth=section.depth
                )
            )

            chunks.append(chunk)

        return chunks

    def _create_layer3_chunks(self, extracted_doc) -> List[Chunk]:
        """
        Layer 3: Chunk-level with Hierarchical Summary-Augmented Chunking (H-SAC).

        CRITICAL: This is the PRIMARY chunking layer!

        Process:
        1. Split each section into 500-char chunks using RCTS
        2. If SAC enabled: Prepend BOTH document + section summaries for embedding
        3. Keep raw content (without summaries) for generation

        Hierarchical SAC (H-SAC) provides TWO levels of context:
        - Document summary: Global context (what document is this?)
        - Section summary: Local context (what section are we in?)
        - Raw chunk: Specific detail

        Based on:
        - Reuter et al., 2024: SAC reduces DRM by 58%
        - LegalBench-RAG: RCTS outperforms fixed-size
        - Enhanced with hierarchical context for better precision
        """

        chunks = []
        chunk_counter = 0

        # Get document summary for H-SAC
        doc_summary = extracted_doc.document_summary or ""

        for section in extracted_doc.sections:
            # Skip empty sections
            if not section.content.strip():
                continue

            # Get section summary for H-SAC
            section_summary = section.summary or ""

            # Split section into raw chunks using RCTS
            raw_chunks = self.text_splitter.split_text(section.content)

            for idx, raw_chunk in enumerate(raw_chunks):
                chunk_counter += 1

                # Apply Hierarchical SAC: Prepend document + section summaries
                if self.enable_sac:
                    # Build hierarchical context
                    context_parts = []

                    if doc_summary:
                        context_parts.append(doc_summary)

                    if section_summary:
                        context_parts.append(section_summary)

                    if context_parts:
                        # CRITICAL: Summaries prepended for embedding ONLY
                        context = " ".join(context_parts)
                        augmented_content = f"{context} {raw_chunk}"
                    else:
                        augmented_content = raw_chunk
                else:
                    augmented_content = raw_chunk

                chunk = Chunk(
                    chunk_id=f"{extracted_doc.document_id}_L3_{section.section_id}_chunk_{idx}",
                    content=augmented_content,  # For embedding (with SAC)
                    raw_content=raw_chunk,      # For generation (without SAC)
                    metadata=ChunkMetadata(
                        chunk_id=f"{extracted_doc.document_id}_L3_{section.section_id}_chunk_{idx}",
                        layer=3,
                        document_id=extracted_doc.document_id,
                        section_id=section.section_id,
                        parent_chunk_id=f"{extracted_doc.document_id}_L2_{section.section_id}",
                        page_number=section.page_number,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        section_title=section.title,
                        section_path=section.path,
                        section_level=section.level,
                        section_depth=section.depth
                    )
                )

                chunks.append(chunk)

        return chunks

    def get_chunking_stats(self, chunks_dict: Dict[str, List[Chunk]]) -> Dict:
        """
        Get statistics about the chunking results.

        Args:
            chunks_dict: Output from chunk_document()

        Returns:
            Dict with chunking statistics
        """

        stats = {
            "layer1_count": len(chunks_dict["layer1"]),
            "layer2_count": len(chunks_dict["layer2"]),
            "layer3_count": len(chunks_dict["layer3"]),
            "total_chunks": chunks_dict["total_chunks"]
        }

        # Layer 3 statistics (PRIMARY layer)
        if chunks_dict["layer3"]:
            layer3_sizes = [len(c.raw_content) for c in chunks_dict["layer3"]]
            stats["layer3_avg_size"] = sum(layer3_sizes) / len(layer3_sizes)
            stats["layer3_min_size"] = min(layer3_sizes)
            stats["layer3_max_size"] = max(layer3_sizes)

            # SAC statistics
            sac_sizes = [
                len(c.content) - len(c.raw_content)
                for c in chunks_dict["layer3"]
            ]
            stats["sac_avg_overhead"] = sum(sac_sizes) / len(sac_sizes) if sac_sizes else 0

        return stats


# Example usage
if __name__ == "__main__":
    # This would be used after DoclingExtractorV2
    from extraction import DoclingExtractorV2, ExtractionConfig

    # Extract document
    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=True  # Required for SAC
    )

    extractor = DoclingExtractorV2(config)
    result = extractor.extract("document.pdf")

    # Create multi-layer chunks
    chunker = MultiLayerChunker(
        chunk_size=500,
        chunk_overlap=0,
        enable_sac=True
    )

    chunks = chunker.chunk_document(result)
    stats = chunker.get_chunking_stats(chunks)

    print(f"Layer 1: {stats['layer1_count']} chunks")
    print(f"Layer 2: {stats['layer2_count']} chunks")
    print(f"Layer 3: {stats['layer3_count']} chunks (PRIMARY)")
    print(f"Total: {stats['total_chunks']} chunks")
