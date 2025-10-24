"""
Metadata utilities for MY_SUJBOT pipeline.

Provides standardized metadata structures across:
- context_assembly.py (provenance extraction)
- faiss_vector_store.py (chunk metadata)
- hybrid_search.py (BM25 + FAISS metadata)

Features:
- Standardized ChunkMetadata dataclass
- Helper functions for metadata extraction
- Validation and type conversion

Usage:
    from src.utils import ChunkMetadata

    metadata = ChunkMetadata.from_chunk(chunk_dict)
    provenance = metadata.format_provenance()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """
    Standardized chunk metadata structure.

    This dataclass provides a unified representation of chunk metadata
    across the entire pipeline, replacing scattered dictionary patterns.

    Attributes:
        chunk_id: Unique chunk identifier
        document_id: Source document identifier
        section_id: Optional section identifier
        section_title: Optional section title
        section_path: Optional hierarchical section path (e.g., "1.2.3")
        page_number: Optional page number in source document
        layer: Indexing layer (1=document, 2=section, 3=chunk)
        custom: Optional custom metadata (free-form dict)

    Example:
        >>> metadata = ChunkMetadata(
        >>>     chunk_id="chunk_0",
        >>>     document_id="GRI_306",
        >>>     section_title="Waste disposal requirements",
        >>>     page_number=15,
        >>>     layer=3
        >>> )
        >>> print(metadata.format_provenance())
        '[Doc: GRI_306, Section: Waste disposal requirements, Page: 15]'
    """

    chunk_id: str
    document_id: str
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    page_number: Optional[int] = None
    layer: int = 3
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metadata fields."""
        # Validate layer
        if self.layer not in [1, 2, 3]:
            raise ValueError(f"Invalid layer: {self.layer}. Must be 1, 2, or 3.")

        # Validate chunk_id and document_id
        if not self.chunk_id or not isinstance(self.chunk_id, str):
            raise ValueError(f"chunk_id must be non-empty string: {self.chunk_id}")
        if not self.document_id or not isinstance(self.document_id, str):
            raise ValueError(f"document_id must be non-empty string: {self.document_id}")

        # Validate page_number if provided
        if self.page_number is not None and not isinstance(self.page_number, int):
            try:
                self.page_number = int(self.page_number)
            except (ValueError, TypeError):
                logger.warning(f"Invalid page_number: {self.page_number}, setting to None")
                self.page_number = None

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any]) -> "ChunkMetadata":
        """
        Create ChunkMetadata from chunk dictionary.

        Supports various dictionary formats from different pipeline components:
        - FAISS vector store format
        - Hybrid search format
        - Context assembly format

        Args:
            chunk: Chunk dictionary with metadata

        Returns:
            ChunkMetadata instance

        Example:
            >>> chunk = {
            >>>     "chunk_id": "chunk_0",
            >>>     "document_id": "GRI_306",
            >>>     "section_title": "Waste disposal",
            >>>     "page_number": 15,
            >>>     "layer": 3
            >>> }
            >>> metadata = ChunkMetadata.from_chunk(chunk)
        """
        # Extract standard fields
        chunk_id = chunk.get("chunk_id") or chunk.get("id", "unknown")
        document_id = chunk.get("document_id") or chunk.get("doc_id", "unknown")

        # Extract section information
        section_id = chunk.get("section_id")
        section_title = chunk.get("section_title") or chunk.get("section_name")
        section_path = chunk.get("section_path") or chunk.get("section_number")

        # Extract page number
        page_number = chunk.get("page_number") or chunk.get("page")

        # Extract layer
        layer = chunk.get("layer", 3)

        # Extract custom metadata
        # Exclude standard fields to avoid duplication
        standard_fields = {
            "chunk_id", "id", "document_id", "doc_id",
            "section_id", "section_title", "section_name",
            "section_path", "section_number",
            "page_number", "page", "layer",
            "content", "raw_content", "context",  # Content fields
            "embedding", "score", "distance"  # Vector store fields
        }
        custom = {k: v for k, v in chunk.items() if k not in standard_fields}

        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            section_id=section_id,
            section_title=section_title,
            section_path=section_path,
            page_number=page_number,
            layer=layer,
            custom=custom
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (standardized format).

        Returns:
            Dictionary with non-None fields

        Example:
            >>> metadata.to_dict()
            {
                "chunk_id": "chunk_0",
                "document_id": "GRI_306",
                "section_title": "Waste disposal",
                "page_number": 15,
                "layer": 3
            }
        """
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "layer": self.layer
        }

        # Add optional fields if present
        if self.section_id:
            result["section_id"] = self.section_id
        if self.section_title:
            result["section_title"] = self.section_title
        if self.section_path:
            result["section_path"] = self.section_path
        if self.page_number is not None:
            result["page_number"] = self.page_number

        # Merge custom metadata
        result.update(self.custom)

        return result

    def format_provenance(
        self,
        format_style: str = "detailed",
        include_chunk_id: bool = False
    ) -> str:
        """
        Format metadata as provenance string for citations.

        Args:
            format_style: Citation format style
                - "detailed": [Doc: X, Section: Y, Page: Z]
                - "simple": [X, Y, Page Z]
                - "minimal": [X]
            include_chunk_id: Include chunk_id in citation

        Returns:
            Formatted provenance string

        Example:
            >>> metadata.format_provenance("detailed")
            '[Doc: GRI_306, Section: Waste disposal requirements, Page: 15]'

            >>> metadata.format_provenance("simple")
            '[GRI_306, Waste disposal requirements, Page 15]'

            >>> metadata.format_provenance("minimal")
            '[GRI_306]'
        """
        if format_style == "minimal":
            return f"[{self.document_id}]"

        # Build components
        components = []

        if format_style == "detailed":
            components.append(f"Doc: {self.document_id}")
        else:
            components.append(self.document_id)

        # Add section if available
        if self.section_title:
            if format_style == "detailed":
                components.append(f"Section: {self.section_title}")
            else:
                components.append(self.section_title)
        elif self.section_path:
            if format_style == "detailed":
                components.append(f"Section: {self.section_path}")
            else:
                components.append(self.section_path)

        # Add page number if available
        if self.page_number is not None:
            if format_style == "detailed":
                components.append(f"Page: {self.page_number}")
            else:
                components.append(f"Page {self.page_number}")

        # Add chunk_id if requested
        if include_chunk_id:
            if format_style == "detailed":
                components.append(f"Chunk: {self.chunk_id}")
            else:
                components.append(self.chunk_id)

        return f"[{', '.join(components)}]"

    def merge(self, other: "ChunkMetadata") -> "ChunkMetadata":
        """
        Merge two metadata instances (useful for combining information).

        This creates a new ChunkMetadata with fields from both instances,
        with `other` taking precedence for conflicts.

        Args:
            other: Another ChunkMetadata instance

        Returns:
            New merged ChunkMetadata

        Example:
            >>> base = ChunkMetadata(chunk_id="chunk_0", document_id="GRI_306")
            >>> extra = ChunkMetadata(chunk_id="chunk_0", document_id="GRI_306", page_number=15)
            >>> merged = base.merge(extra)
            >>> merged.page_number
            15
        """
        return ChunkMetadata(
            chunk_id=other.chunk_id or self.chunk_id,
            document_id=other.document_id or self.document_id,
            section_id=other.section_id or self.section_id,
            section_title=other.section_title or self.section_title,
            section_path=other.section_path or self.section_path,
            page_number=other.page_number if other.page_number is not None else self.page_number,
            layer=other.layer,
            custom={**self.custom, **other.custom}
        )


def extract_document_id(chunk: Dict[str, Any]) -> str:
    """
    Extract document ID from chunk (handles various formats).

    Args:
        chunk: Chunk dictionary

    Returns:
        Document ID or "unknown"

    Example:
        >>> extract_document_id({"document_id": "GRI_306"})
        'GRI_306'

        >>> extract_document_id({"doc_id": "GRI_306"})
        'GRI_306'
    """
    return chunk.get("document_id") or chunk.get("doc_id", "unknown")


def extract_section_title(chunk: Dict[str, Any]) -> Optional[str]:
    """
    Extract section title from chunk (handles various formats).

    Args:
        chunk: Chunk dictionary

    Returns:
        Section title or None

    Example:
        >>> extract_section_title({"section_title": "Waste disposal"})
        'Waste disposal'

        >>> extract_section_title({"section_name": "Waste disposal"})
        'Waste disposal'
    """
    return chunk.get("section_title") or chunk.get("section_name")


def extract_page_number(chunk: Dict[str, Any]) -> Optional[int]:
    """
    Extract page number from chunk (handles various formats).

    Args:
        chunk: Chunk dictionary

    Returns:
        Page number or None

    Example:
        >>> extract_page_number({"page_number": 15})
        15

        >>> extract_page_number({"page": "15"})
        15
    """
    page = chunk.get("page_number") or chunk.get("page")
    if page is None:
        return None

    try:
        return int(page)
    except (ValueError, TypeError):
        logger.warning(f"Invalid page number: {page}")
        return None


# Example usage
if __name__ == "__main__":
    print("=== Metadata Utilities Examples ===\n")

    # Example 1: Create from chunk dictionary
    print("1. Create ChunkMetadata from chunk dictionary...")
    chunk = {
        "chunk_id": "chunk_0",
        "document_id": "GRI_306",
        "section_title": "Waste disposal requirements",
        "page_number": 15,
        "layer": 3,
        "custom_field": "custom_value"
    }
    metadata = ChunkMetadata.from_chunk(chunk)
    print(f"   Metadata: {metadata}")

    # Example 2: Format provenance
    print("\n2. Format provenance (different styles)...")
    print(f"   Detailed: {metadata.format_provenance('detailed')}")
    print(f"   Simple:   {metadata.format_provenance('simple')}")
    print(f"   Minimal:  {metadata.format_provenance('minimal')}")

    # Example 3: Convert to dictionary
    print("\n3. Convert to dictionary...")
    meta_dict = metadata.to_dict()
    print(f"   Dict: {meta_dict}")

    # Example 4: Merge metadata
    print("\n4. Merge metadata...")
    base = ChunkMetadata(chunk_id="chunk_0", document_id="GRI_306", layer=3)
    extra = ChunkMetadata(
        chunk_id="chunk_0",
        document_id="GRI_306",
        page_number=15,
        section_title="Waste disposal",
        layer=3
    )
    merged = base.merge(extra)
    print(f"   Merged: {merged.format_provenance('detailed')}")

    # Example 5: Helper functions
    print("\n5. Helper functions...")
    test_chunk = {
        "doc_id": "GRI_305",
        "section_name": "Emissions",
        "page": "10"
    }
    print(f"   Document ID: {extract_document_id(test_chunk)}")
    print(f"   Section:     {extract_section_title(test_chunk)}")
    print(f"   Page:        {extract_page_number(test_chunk)}")

    print("\n=== All examples completed ===")
