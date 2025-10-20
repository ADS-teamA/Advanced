"""
Multi-layer chunking orchestrator.

Based on Lima 2024 (Multi-Layer Embeddings paper):
- 3-layer architecture: Document → Section → Chunk
- 2.3x improvement in essential chunks (37.86% vs 16.39%)
- Handles semantic overload (e.g., articles with 70+ concepts)
"""

from typing import List

from src.core.models import (
    Document, Summary, Chunk, ChunkMetadata, ChunkType, Section
)
from src.core.config import ChunkingConfig
from src.chunking.chunking_strategy import ChunkingStrategy
from src.chunking.sac_augmenter import SACaugmenter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MultiLayerChunker:
    """
    Multi-layer chunking orchestrator.

    Creates 3 layers of chunks:
    1. Document-level (1 chunk per document) - Global context
    2. Section-level (N chunks, 1 per section) - Mid-level context
    3. Chunk-level (M chunks with RCTS + SAC) - PRIMARY for retrieval

    Evidence from Lima 2024:
    - Essential chunks: 37.86% vs 16.39% baseline (2.3x improvement!)
    - Unnecessary chunks: 62% vs 75% baseline
    - Handles semantic overload effectively
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        sac_augmenter: SACaugmenter,
        config: ChunkingConfig
    ):
        """
        Initialize multi-layer chunker.

        Args:
            chunking_strategy: Chunking strategy (e.g., RCTS)
            sac_augmenter: SAC augmenter
            config: Chunking configuration
        """
        self.strategy = chunking_strategy
        self.sac = sac_augmenter
        self.config = config

        logger.info(
            f"Initialized MultiLayerChunker with multi_layer={config.enable_multi_layer}, "
            f"SAC={config.enable_sac}"
        )

    def create_chunks(
        self,
        document: Document,
        doc_summary: Summary
    ) -> List[Chunk]:
        """
        Create multi-layer chunks from document.

        Args:
            document: Processed document
            doc_summary: Document summary

        Returns:
            List of all chunks (document + section + text chunks)
        """
        all_chunks = []

        # Layer 1: Document-level chunk (if multi-layer enabled)
        if self.config.enable_multi_layer:
            doc_chunk = self._create_document_chunk(document, doc_summary)
            all_chunks.append(doc_chunk)
            logger.info("Created document-level chunk (Layer 1)")

        # Layer 2: Section-level chunks (if multi-layer enabled)
        if self.config.enable_multi_layer and document.structure.sections:
            section_chunks = self._create_section_chunks(document, doc_summary)
            all_chunks.extend(section_chunks)
            logger.info(f"Created {len(section_chunks)} section-level chunks (Layer 2)")

        # Layer 3: Text chunks with RCTS + SAC (PRIMARY)
        text_chunks = self._create_text_chunks(document, doc_summary)
        all_chunks.extend(text_chunks)
        logger.info(f"Created {len(text_chunks)} text chunks with SAC (Layer 3 - PRIMARY)")

        logger.info(
            f"Total chunks created: {len(all_chunks)} "
            f"(doc: {1 if self.config.enable_multi_layer else 0}, "
            f"section: {len(document.structure.sections) if self.config.enable_multi_layer else 0}, "
            f"text: {len(text_chunks)})"
        )

        return all_chunks

    def _create_document_chunk(
        self,
        document: Document,
        doc_summary: Summary
    ) -> Chunk:
        """
        Create document-level chunk (Layer 1).

        Uses document summary as content.
        Purpose: Global filtering, document identification, DRM prevention.
        """
        metadata = ChunkMetadata(
            chunk_id=f"{document.metadata.document_id}_doc",
            document_id=document.metadata.document_id,
            section_id=None,
            chunk_index=0,
            hierarchy_level=0,
            char_count=doc_summary.char_count,
            document_summary=doc_summary.text
        )

        # For document-level, content = summary (no augmentation needed)
        chunk = Chunk(
            chunk_id=metadata.chunk_id,
            content=doc_summary.text,
            raw_content=doc_summary.text,
            chunk_type=ChunkType.DOCUMENT,
            metadata=metadata
        )

        return chunk

    def _create_section_chunks(
        self,
        document: Document,
        doc_summary: Summary
    ) -> List[Chunk]:
        """
        Create section-level chunks (Layer 2).

        One chunk per section with section summary.
        Purpose: Mid-level context, section-specific queries.
        """
        section_chunks = []

        for section in document.structure.sections:
            # For MVP, use first 100 chars as section summary
            # Future: Generate proper section summaries
            section_summary = section.text[:100].strip() + "..."

            metadata = ChunkMetadata(
                chunk_id=f"{section.section_id}_summary",
                document_id=document.metadata.document_id,
                section_id=section.section_id,
                chunk_index=0,
                hierarchy_level=1,
                char_count=len(section_summary),
                document_summary=doc_summary.text,
                section_summary=section_summary
            )

            # Apply SAC augmentation to section summary
            chunk = self.sac.create_chunk(
                raw_content=section_summary,
                summary=doc_summary,
                chunk_type=ChunkType.SECTION,
                metadata=metadata
            )

            section_chunks.append(chunk)

        return section_chunks

    def _create_text_chunks(
        self,
        document: Document,
        doc_summary: Summary
    ) -> List[Chunk]:
        """
        Create text chunks with RCTS + SAC (Layer 3 - PRIMARY).

        This is the primary layer for retrieval.
        Uses RCTS for chunking and SAC for augmentation.
        """
        text_chunks = []

        # If document has sections, chunk each section separately
        if document.structure.sections:
            for section in document.structure.sections:
                # Split section text using RCTS
                raw_chunks = self.strategy.split_text(section.text)

                # Create chunks with SAC augmentation
                section_text_chunks = self.sac.create_chunks_batch(
                    raw_chunks=raw_chunks,
                    summary=doc_summary,
                    document_id=document.metadata.document_id,
                    section_id=section.section_id,
                    chunk_type=ChunkType.CHUNK
                )

                text_chunks.extend(section_text_chunks)

        else:
            # No sections detected, chunk entire document
            raw_chunks = self.strategy.split_text(document.text)

            # Create chunks with SAC augmentation
            text_chunks = self.sac.create_chunks_batch(
                raw_chunks=raw_chunks,
                summary=doc_summary,
                document_id=document.metadata.document_id,
                section_id=None,
                chunk_type=ChunkType.CHUNK
            )

        return text_chunks
