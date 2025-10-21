"""
Docling-based document extraction module.

This module provides advanced document structure extraction using IBM Docling,
with specialized support for legal and technical documents.
"""

# V2 extractor with smart hierarchy and summaries
from docling_extractor_v2 import DoclingExtractorV2, ExtractionConfig
from summary_generator import SummaryGenerator

# PHASE 3: Multi-layer chunking with SAC
from multi_layer_chunker import MultiLayerChunker, Chunk, ChunkMetadata

# PHASE 4: Embedding & FAISS indexing
from embedding_generator import EmbeddingGenerator, EmbeddingConfig
from faiss_vector_store import FAISSVectorStore
from indexing_pipeline import IndexingPipeline, IndexingConfig

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
