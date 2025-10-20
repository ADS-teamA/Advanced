"""
Core data models for the RAG pipeline.

These models define the type-safe data structures used throughout
the pipeline, based on the specifications in PIPELINE.md.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class DocumentType(str, Enum):
    """Types of legal documents supported."""
    LEGISLATION = "legislation"
    CONTRACT = "contract"
    NDA = "nda"
    ESG_REPORT = "esg_report"
    POLICY = "policy"
    REGULATION = "regulation"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Types of chunks in multi-layer architecture."""
    DOCUMENT = "document"  # Layer 1: Document-level
    SECTION = "section"    # Layer 2: Section-level
    CHUNK = "chunk"        # Layer 3: Text chunk (primary)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a legal document."""
    document_id: str
    document_type: DocumentType
    source_path: str
    extraction_date: datetime = field(default_factory=datetime.now)
    parties: Optional[List[str]] = None
    effective_date: Optional[datetime] = None
    total_sections: int = 0
    total_chars: int = 0
    language: str = "en"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "source_path": self.source_path,
            "extraction_date": self.extraction_date.isoformat(),
            "parties": self.parties,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "total_sections": self.total_sections,
            "total_chars": self.total_chars,
            "language": self.language
        }


@dataclass
class Section:
    """Represents a section in a hierarchical document structure."""
    section_id: str
    title: str
    text: str
    level: int  # Hierarchy level (0=top, 1=subsection, etc.)
    parent_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class DocumentStructure:
    """Hierarchical structure of a legal document."""
    sections: List[Section]
    hierarchy_type: str  # 'legislation', 'contract', 'nda', etc.
    doc_type: DocumentType = DocumentType.UNKNOWN  # Detected document type
    total_levels: int = 1

    def get_section_by_id(self, section_id: str) -> Optional[Section]:
        """Retrieve a section by ID."""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None

    def get_top_level_sections(self) -> List[Section]:
        """Get all top-level sections."""
        return [s for s in self.sections if s.level == 0]


@dataclass
class Document:
    """Represents a processed legal document."""
    text: str
    structure: DocumentStructure
    metadata: DocumentMetadata

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Summary:
    """Represents a document or section summary."""
    text: str
    char_count: int
    model: str  # e.g., "gpt-4o-mini"
    generation_date: datetime = field(default_factory=datetime.now)

    def is_valid(self, max_chars: int, tolerance: int) -> bool:
        """Check if summary meets length constraints."""
        return self.char_count <= (max_chars + tolerance)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: str
    document_id: str
    section_id: Optional[str] = None
    chunk_index: int = 0
    hierarchy_level: int = 2  # 0=document, 1=section, 2=chunk
    char_count: int = 0
    document_summary: Optional[str] = None
    section_summary: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "chunk_index": self.chunk_index,
            "hierarchy_level": self.hierarchy_level,
            "char_count": self.char_count,
            "document_summary": self.document_summary,
            "section_summary": self.section_summary
        }


@dataclass
class Chunk:
    """
    Represents a text chunk with SAC (Summary-Augmented Chunking).

    The 'content' field contains the augmented text (with summary prepended)
    used for embedding in Phase 4. The 'raw_content' field contains the
    original text without the summary, used for generation in Phase 7.
    """
    chunk_id: str
    content: str          # Augmented content (with summary for embedding)
    raw_content: str      # Original content (without summary for generation)
    chunk_type: ChunkType
    metadata: ChunkMetadata

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "raw_content": self.raw_content,
            "chunk_type": self.chunk_type.value,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class ProcessingResult:
    """Result of processing a document through Phases 1-3."""
    document: Document
    summary: Summary
    chunks: List[Chunk]
    metrics: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "document": {
                "text_length": len(self.document.text),
                "metadata": self.document.metadata.to_dict()
            },
            "summary": {
                "text": self.summary.text,
                "char_count": self.summary.char_count,
                "model": self.summary.model
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metrics": self.metrics
        }

    def get_chunk_by_type(self, chunk_type: ChunkType) -> List[Chunk]:
        """Get all chunks of a specific type."""
        return [c for c in self.chunks if c.chunk_type == chunk_type]

    def get_document_chunk(self) -> Optional[Chunk]:
        """Get the document-level chunk (Layer 1)."""
        doc_chunks = self.get_chunk_by_type(ChunkType.DOCUMENT)
        return doc_chunks[0] if doc_chunks else None

    def get_section_chunks(self) -> List[Chunk]:
        """Get all section-level chunks (Layer 2)."""
        return self.get_chunk_by_type(ChunkType.SECTION)

    def get_text_chunks(self) -> List[Chunk]:
        """Get all text chunks (Layer 3 - primary)."""
        return self.get_chunk_by_type(ChunkType.CHUNK)
