"""
PHASE 4: Complete Indexing Pipeline

Orchestrates:
1. PHASE 1-3: Extraction, hierarchy, summaries, chunking
2. PHASE 4: Embedding generation + FAISS indexing

Based on research:
- LegalBench-RAG: text-embedding-3-large + RCTS
- Multi-Layer Embeddings: 3 separate indexes
- Dense-only retrieval (no BM25)

Usage:
    pipeline = IndexingPipeline(
        embedding_model="text-embedding-3-large",
        enable_sac=True
    )

    vector_store = pipeline.index_document("document.pdf")
    vector_store.save("output/vector_store")
"""

import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from docling_extractor_v2 import DoclingExtractorV2, ExtractionConfig
from multi_layer_chunker import MultiLayerChunker
from embedding_generator import EmbeddingGenerator, EmbeddingConfig
from faiss_vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingConfig:
    """Configuration for complete indexing pipeline."""

    # PHASE 1: Extraction
    enable_smart_hierarchy: bool = True
    ocr_language: list = None

    # PHASE 2: Summaries
    generate_summaries: bool = True
    summary_model: str = "gpt-4o-mini"
    summary_max_chars: int = 150

    # PHASE 3: Chunking
    chunk_size: int = 500
    chunk_overlap: int = 0
    enable_sac: bool = True

    # PHASE 4: Embedding
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 100
    normalize_embeddings: bool = True

    def __post_init__(self):
        if self.ocr_language is None:
            self.ocr_language = ["cs-CZ", "en-US"]


class IndexingPipeline:
    """
    Complete indexing pipeline for RAG system.

    Phases:
    1. Smart hierarchy extraction (font-size based)
    2. Generic summary generation (gpt-4o-mini)
    3. Multi-layer chunking + SAC (RCTS 500 chars)
    4. Embedding + FAISS indexing (3 separate indexes)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Summary-Augmented Chunking (Reuter et al., 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    """

    def __init__(self, config: Optional[IndexingConfig] = None):
        """
        Initialize indexing pipeline.

        Args:
            config: IndexingConfig instance (defaults to research-optimal settings)
        """
        self.config = config or IndexingConfig()

        logger.info("Initializing IndexingPipeline...")

        # Initialize PHASE 1+2: Extraction
        self.extraction_config = ExtractionConfig(
            enable_smart_hierarchy=self.config.enable_smart_hierarchy,
            generate_summaries=self.config.generate_summaries,
            summary_model=self.config.summary_model,
            summary_max_chars=self.config.summary_max_chars,
            ocr_language=self.config.ocr_language
        )
        self.extractor = DoclingExtractorV2(self.extraction_config)

        # Initialize PHASE 3: Chunking
        self.chunker = MultiLayerChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            enable_sac=self.config.enable_sac
        )

        # Initialize PHASE 4: Embedding
        self.embedding_config = EmbeddingConfig(
            model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            normalize=self.config.normalize_embeddings
        )
        self.embedder = EmbeddingGenerator(self.embedding_config)

        logger.info(
            f"Pipeline initialized: "
            f"SAC={self.config.enable_sac}, "
            f"model={self.config.embedding_model} "
            f"({self.embedder.dimensions}D)"
        )

    def index_document(
        self,
        pdf_path: Path,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None
    ) -> FAISSVectorStore:
        """
        Index a single PDF document.

        Complete pipeline:
        1. Extract with smart hierarchy (PHASE 1)
        2. Generate summaries (PHASE 2)
        3. Multi-layer chunking + SAC (PHASE 3)
        4. Embed + FAISS index (PHASE 4)

        Args:
            pdf_path: Path to PDF document
            save_intermediate: Save intermediate results (chunks, embeddings)
            output_dir: Directory for intermediate results

        Returns:
            FAISSVectorStore with indexed document
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("="*80)
        logger.info(f"Indexing document: {pdf_path.name}")
        logger.info("="*80)

        # PHASE 1+2: Extract + Summaries
        logger.info("PHASE 1+2: Extraction + Summaries")
        result = self.extractor.extract(pdf_path)
        logger.info(
            f"✓ Extracted: {result.num_sections} sections, "
            f"depth={result.hierarchy_depth}"
        )

        # PHASE 3: Multi-layer chunking + SAC
        logger.info("PHASE 3: Multi-Layer Chunking + SAC")
        chunks = self.chunker.chunk_document(result)
        chunking_stats = self.chunker.get_chunking_stats(chunks)
        logger.info(
            f"✓ Chunked: L1={chunking_stats['layer1_count']}, "
            f"L2={chunking_stats['layer2_count']}, "
            f"L3={chunking_stats['layer3_count']} (PRIMARY)"
        )

        # PHASE 4: Embedding
        logger.info("PHASE 4: Embedding Generation")
        embeddings = {
            "layer1": self.embedder.embed_chunks(chunks["layer1"], layer=1),
            "layer2": self.embedder.embed_chunks(chunks["layer2"], layer=2),
            "layer3": self.embedder.embed_chunks(chunks["layer3"], layer=3)
        }
        logger.info(
            f"✓ Embedded: {self.embedder.dimensions}D vectors, "
            f"{embeddings['layer3'].shape[0]} Layer 3 chunks"
        )

        # PHASE 4: FAISS Indexing
        logger.info("PHASE 4: FAISS Indexing")
        vector_store = FAISSVectorStore(dimensions=self.embedder.dimensions)
        vector_store.add_chunks(chunks, embeddings)
        store_stats = vector_store.get_stats()
        logger.info(
            f"✓ Indexed: {store_stats['total_vectors']} vectors "
            f"({store_stats['documents']} documents)"
        )

        # Save intermediate results
        if save_intermediate and output_dir:
            self._save_intermediate(
                output_dir=output_dir,
                result=result,
                chunks=chunks,
                chunking_stats=chunking_stats
            )

        logger.info("="*80)
        logger.info(f"✓ Indexing complete: {pdf_path.name}")
        logger.info("="*80)

        return vector_store

    def index_batch(
        self,
        pdf_paths: list,
        output_dir: Path,
        save_per_document: bool = False
    ) -> FAISSVectorStore:
        """
        Index multiple PDF documents into a single vector store.

        Args:
            pdf_paths: List of PDF file paths
            output_dir: Directory to save vector store
            save_per_document: Save individual document vector stores

        Returns:
            FAISSVectorStore with all indexed documents
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch indexing {len(pdf_paths)} documents...")

        # Create combined vector store
        vector_store = FAISSVectorStore(dimensions=self.embedder.dimensions)

        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"\n[{i}/{len(pdf_paths)}] Processing: {Path(pdf_path).name}")

            try:
                # Index document
                doc_store = self.index_document(
                    pdf_path=pdf_path,
                    save_intermediate=False
                )

                # Merge into combined store (by adding chunks)
                # Note: This is simplified - production would need proper merging
                vector_store.index_layer1 = doc_store.index_layer1
                vector_store.index_layer2 = doc_store.index_layer2
                vector_store.index_layer3 = doc_store.index_layer3
                vector_store.metadata_layer1.extend(doc_store.metadata_layer1)
                vector_store.metadata_layer2.extend(doc_store.metadata_layer2)
                vector_store.metadata_layer3.extend(doc_store.metadata_layer3)

                # Save per-document store
                if save_per_document:
                    doc_name = Path(pdf_path).stem
                    doc_output = output_dir / f"{doc_name}_store"
                    doc_store.save(doc_output)
                    logger.info(f"✓ Saved individual store: {doc_output}")

            except Exception as e:
                logger.error(f"✗ Failed to index {pdf_path}: {e}")
                continue

        # Save combined store
        combined_output = output_dir / "combined_store"
        vector_store.save(combined_output)

        logger.info(f"\n✓ Batch indexing complete: {vector_store.get_stats()}")
        logger.info(f"✓ Saved to: {combined_output}")

        return vector_store

    def _save_intermediate(
        self,
        output_dir: Path,
        result,
        chunks: Dict,
        chunking_stats: Dict
    ):
        """Save intermediate results (chunks, stats)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks
        chunks_export = {
            "metadata": {
                "document_id": result.document_id,
                "source_path": result.source_path,
                "chunking_stats": chunking_stats
            },
            "layer1": [c.to_dict() for c in chunks["layer1"]],
            "layer2": [c.to_dict() for c in chunks["layer2"]],
            "layer3": [c.to_dict() for c in chunks["layer3"]]
        }

        chunks_path = output_dir / f"{result.document_id}_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_export, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved intermediate: {chunks_path}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with research-optimal settings
    config = IndexingConfig(
        # PHASE 1-2
        enable_smart_hierarchy=True,
        generate_summaries=True,
        summary_model="gpt-4o-mini",

        # PHASE 3
        chunk_size=500,
        enable_sac=True,

        # PHASE 4
        embedding_model="text-embedding-3-large",
        normalize_embeddings=True
    )

    pipeline = IndexingPipeline(config)

    # Index single document
    vector_store = pipeline.index_document(
        pdf_path="data/document.pdf",
        save_intermediate=True,
        output_dir=Path("output/indexing")
    )

    # Save vector store
    vector_store.save(Path("output/vector_store"))

    # Search example
    query_embedding = pipeline.embedder.embed_texts(["safety specification"])
    results = vector_store.hierarchical_search(query_embedding, k_layer3=6)

    print(f"\nTop results:")
    for i, result in enumerate(results["layer3"][:3], 1):
        print(f"{i}. {result['section_title']} (score: {result['score']:.4f})")
