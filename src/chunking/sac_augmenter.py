"""
Summary-Augmented Chunking (SAC) implementation.

Based on Reuter et al., 2024:
- SAC reduces Document-Level Retrieval Mismatch (DRM) by 58%
- Prepends document summary to each chunk for global context
- Critical for distinguishing similar chunks from different documents
"""

from src.core.models import Chunk, ChunkMetadata, ChunkType, Summary
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SACaugmenter:
    """
    Summary-Augmented Chunking implementation.

    SAC solves the Document-Level Retrieval Mismatch (DRM) problem:
    - Baseline DRM rate: 67%
    - SAC DRM rate: 28%
    - Reduction: 58% (dramatic improvement!)

    How it works:
    1. Prepend document summary to each chunk
    2. Embedding captures BOTH local content + global context
    3. Retriever can distinguish similar chunks from different documents
    4. Example: "termination clause" in Contract A ≠ Contract B

    IMPORTANT: Summary is prepended for EMBEDDING (Phase 4),
    but STRIPPED for generation (Phase 7).
    """

    def __init__(self, enable_sac: bool = True):
        """
        Initialize SAC augmenter.

        Args:
            enable_sac: If True, apply SAC. If False, use raw chunks.
        """
        self.enable_sac = enable_sac

        if enable_sac:
            logger.info("SAC augmentation ENABLED (58% DRM reduction)")
        else:
            logger.warning("SAC augmentation DISABLED (not recommended)")

    def augment_chunk(
        self,
        raw_content: str,
        summary: Summary
    ) -> str:
        """
        Augment chunk with document summary (prepend).

        Args:
            raw_content: Original chunk text
            summary: Document summary

        Returns:
            Augmented content (summary + chunk)
        """
        if not self.enable_sac:
            return raw_content

        # Prepend summary with space separator
        augmented = f"{summary.text} {raw_content}"

        return augmented

    def create_chunk(
        self,
        raw_content: str,
        summary: Summary,
        chunk_type: ChunkType,
        metadata: ChunkMetadata
    ) -> Chunk:
        """
        Create chunk with SAC augmentation.

        Args:
            raw_content: Original chunk text
            summary: Document summary
            chunk_type: Type of chunk (document/section/chunk)
            metadata: Chunk metadata

        Returns:
            Chunk object with dual content (augmented + raw)
        """
        # Apply SAC augmentation
        augmented_content = self.augment_chunk(raw_content, summary)

        # Update metadata char count
        metadata.char_count = len(raw_content)
        metadata.document_summary = summary.text

        chunk = Chunk(
            chunk_id=metadata.chunk_id,
            content=augmented_content,  # For embedding in Phase 4
            raw_content=raw_content,     # For generation in Phase 7
            chunk_type=chunk_type,
            metadata=metadata
        )

        logger.debug(
            f"Created chunk {chunk.chunk_id}: "
            f"{len(raw_content)} chars -> {len(augmented_content)} chars (with SAC)"
        )

        return chunk

    def create_chunks_batch(
        self,
        raw_chunks: list[str],
        summary: Summary,
        document_id: str,
        section_id: str = None,
        chunk_type: ChunkType = ChunkType.CHUNK
    ) -> list[Chunk]:
        """
        Create multiple chunks with SAC augmentation.

        Args:
            raw_chunks: List of raw chunk texts
            summary: Document summary
            document_id: Document ID
            section_id: Optional section ID
            chunk_type: Type of chunks

        Returns:
            List of Chunk objects
        """
        chunks = []

        for idx, raw_chunk in enumerate(raw_chunks):
            # Create metadata
            chunk_id = f"{document_id}_chunk_{idx}"
            if section_id:
                chunk_id = f"{section_id}_chunk_{idx}"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                section_id=section_id,
                chunk_index=idx,
                hierarchy_level=2,  # Chunk level
                char_count=len(raw_chunk),
                document_summary=summary.text
            )

            # Create augmented chunk
            chunk = self.create_chunk(
                raw_content=raw_chunk,
                summary=summary,
                chunk_type=chunk_type,
                metadata=metadata
            )

            chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks with SAC augmentation "
            f"(summary: {summary.char_count} chars)"
        )

        return chunks
