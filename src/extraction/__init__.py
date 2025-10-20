"""
Docling-based document extraction module.

This module provides advanced document structure extraction using IBM Docling,
with specialized support for legal and technical documents.
"""

# V2 extractor with smart hierarchy and summaries
from .docling_extractor_v2 import DoclingExtractorV2, ExtractionConfig
from .summary_generator import SummaryGenerator

# PHASE 3: Multi-layer chunking with SAC
from .multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata

# PHASE 4: Embedding & FAISS indexing
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .faiss_vector_store import FAISSVectorStore
from .indexing_pipeline import IndexingPipeline, IndexingConfig

# Legacy extractors
try:
    from .docling_extractor import DoclingExtractor
    from .document_processor import DocumentProcessor
    from .legal_analyzer import LegalDocumentAnalyzer
    _legacy_available = True
except ImportError:
    _legacy_available = False

if _legacy_available:
    __all__ = [
        'DoclingExtractorV2',
        'ExtractionConfig',
        'SummaryGenerator',
        'MultiLayerChunker',
        'Chunk',
        'ChunkMetadata',
        'EmbeddingGenerator',
        'EmbeddingConfig',
        'FAISSVectorStore',
        'IndexingPipeline',
        'IndexingConfig',
        'DoclingExtractor',
        'DocumentProcessor',
        'LegalDocumentAnalyzer'
    ]
else:
    __all__ = [
        'DoclingExtractorV2',
        'ExtractionConfig',
        'SummaryGenerator',
        'MultiLayerChunker',
        'Chunk',
        'ChunkMetadata',
        'EmbeddingGenerator',
        'EmbeddingConfig',
        'FAISSVectorStore',
        'IndexingPipeline',
        'IndexingConfig'
    ]
