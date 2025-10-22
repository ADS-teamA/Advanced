"""
PHASE 4-5A: Complete Indexing Pipeline

Orchestrates:
1. PHASE 1-3: Extraction, hierarchy, summaries, chunking
2. PHASE 4: Embedding generation + FAISS indexing
3. PHASE 5A: Knowledge Graph construction (optional)

Supported formats: PDF, DOCX, PPTX, XLSX, HTML

Based on research:
- LegalBench-RAG: text-embedding-3-large + RCTS
- Multi-Layer Embeddings: 3 separate indexes
- Dense-only retrieval (no BM25)
- Knowledge Graph: Entity & relationship extraction

Usage:
    pipeline = IndexingPipeline(
        embedding_model="text-embedding-3-large",
        enable_sac=True,
        enable_knowledge_graph=True
    )

    result = pipeline.index_document("document.pdf")
    result["vector_store"].save("output/vector_store")
    result["knowledge_graph"].save_json("output/knowledge_graph.json")
"""

import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from config import ExtractionConfig
from docling_extractor_v2 import DoclingExtractorV2
from multi_layer_chunker import MultiLayerChunker
from embedding_generator import EmbeddingGenerator, EmbeddingConfig
from faiss_vector_store import FAISSVectorStore

# Knowledge Graph imports (optional)
try:
    from graph import (
        KnowledgeGraphPipeline,
        KnowledgeGraphConfig,
        EntityExtractionConfig as KGEntityConfig,
        RelationshipExtractionConfig as KGRelationshipConfig,
        GraphStorageConfig,
        GraphBackend,
    )
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

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

    # PHASE 5A: Knowledge Graph (optional)
    enable_knowledge_graph: bool = False
    kg_llm_provider: str = "openai"
    kg_llm_model: str = "gpt-4o-mini"
    kg_backend: str = "simple"  # simple, neo4j, networkx
    kg_min_entity_confidence: float = 0.6
    kg_min_relationship_confidence: float = 0.5
    kg_batch_size: int = 10
    kg_max_workers: int = 5

    # PHASE 5B: Hybrid Search (optional)
    enable_hybrid_search: bool = False  # BM25 + dense with RRF fusion
    hybrid_fusion_k: int = 60  # RRF k parameter (research: k=60 optimal)

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

        # Initialize PHASE 5A: Knowledge Graph (optional)
        self.kg_pipeline = None
        if self.config.enable_knowledge_graph:
            if not KG_AVAILABLE:
                logger.warning(
                    "Knowledge Graph requested but not available. "
                    "Install with: pip install openai anthropic"
                )
            else:
                self._initialize_kg_pipeline()

        logger.info(
            f"Pipeline initialized: "
            f"SAC={self.config.enable_sac}, "
            f"model={self.config.embedding_model} "
            f"({self.embedder.dimensions}D), "
            f"KG={self.config.enable_knowledge_graph}, "
            f"Hybrid={self.config.enable_hybrid_search}"
        )

    def _initialize_kg_pipeline(self):
        """Initialize Knowledge Graph pipeline."""
        import os

        # Map backend string to enum
        backend_map = {
            "simple": GraphBackend.SIMPLE,
            "neo4j": GraphBackend.NEO4J,
            "networkx": GraphBackend.NETWORKX,
        }
        backend = backend_map.get(self.config.kg_backend, GraphBackend.SIMPLE)

        # Get API key
        api_key = None
        if self.config.kg_llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif self.config.kg_llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            logger.warning(
                f"No API key found for {self.config.kg_llm_provider}. "
                f"Set {self.config.kg_llm_provider.upper()}_API_KEY environment variable."
            )
            self.kg_pipeline = None
            return

        # Create KG config
        kg_config = KnowledgeGraphConfig(
            entity_extraction=KGEntityConfig(
                llm_provider=self.config.kg_llm_provider,
                llm_model=self.config.kg_llm_model,
                min_confidence=self.config.kg_min_entity_confidence,
                batch_size=self.config.kg_batch_size,
                max_workers=self.config.kg_max_workers,
            ),
            relationship_extraction=KGRelationshipConfig(
                llm_provider=self.config.kg_llm_provider,
                llm_model=self.config.kg_llm_model,
                min_confidence=self.config.kg_min_relationship_confidence,
                batch_size=self.config.kg_batch_size,
                max_workers=self.config.kg_max_workers,
            ),
            graph_storage=GraphStorageConfig(
                backend=backend,
                export_json=True,
            ),
            openai_api_key=api_key if self.config.kg_llm_provider == "openai" else None,
            anthropic_api_key=api_key if self.config.kg_llm_provider == "anthropic" else None,
        )

        self.kg_pipeline = KnowledgeGraphPipeline(kg_config)
        logger.info(
            f"Knowledge Graph initialized: "
            f"model={self.config.kg_llm_model}, "
            f"backend={self.config.kg_backend}"
        )

    def index_document(
        self,
        document_path: Path,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Index a single document.

        Supported formats: PDF, DOCX, PPTX, XLSX, HTML

        Complete pipeline:
        1. Extract with smart hierarchy (PHASE 1)
        2. Generate summaries (PHASE 2)
        3. Multi-layer chunking + SAC (PHASE 3)
        4. Embed + FAISS index (PHASE 4)
        5. Build knowledge graph (PHASE 5A - optional)

        Args:
            document_path: Path to document file (PDF, DOCX, PPTX, XLSX, HTML)
            save_intermediate: Save intermediate results (chunks, embeddings)
            output_dir: Directory for intermediate results

        Returns:
            Dict with:
                - vector_store: FAISSVectorStore with indexed document
                - knowledge_graph: KnowledgeGraph (if enabled, else None)
                - chunks: Dict of chunks (if save_intermediate)
                - stats: Pipeline statistics
        """
        document_path = Path(document_path)

        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        # Validate format
        supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]
        if document_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format: {document_path.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        logger.info("="*80)
        logger.info(f"Indexing document: {document_path.name}")
        logger.info("="*80)

        # PHASE 1+2: Extract + Summaries
        logger.info("PHASE 1+2: Extraction + Summaries")
        result = self.extractor.extract(document_path)
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

        # PHASE 5B: Hybrid Search (optional)
        if self.config.enable_hybrid_search:
            logger.info("PHASE 5B: Hybrid Search (BM25 + Dense + RRF)")

            try:
                from src.hybrid_search import BM25Store, HybridVectorStore

                # Build BM25 indexes for all 3 layers
                bm25_store = BM25Store()
                bm25_store.build_from_chunks(chunks)

                logger.info(
                    f"✓ BM25 Indexed: "
                    f"L1={len(bm25_store.index_layer1.corpus)}, "
                    f"L2={len(bm25_store.index_layer2.corpus)}, "
                    f"L3={len(bm25_store.index_layer3.corpus)}"
                )

                # Wrap FAISS + BM25 into HybridVectorStore
                hybrid_store = HybridVectorStore(
                    faiss_store=vector_store,
                    bm25_store=bm25_store,
                    fusion_k=self.config.hybrid_fusion_k
                )

                # Replace vector_store with hybrid_store for return
                vector_store = hybrid_store
                store_stats = vector_store.get_stats()

                logger.info(f"✓ Hybrid Search enabled: RRF k={self.config.hybrid_fusion_k}")

            except Exception as e:
                logger.error(f"✗ Hybrid Search failed: {e}")
                logger.warning("Continuing with dense-only retrieval...")
                import traceback
                logger.debug(traceback.format_exc())

        # PHASE 5A: Knowledge Graph (optional)
        knowledge_graph = None
        if self.kg_pipeline:
            logger.info("PHASE 5A: Knowledge Graph Construction")

            try:
                # Prepare chunks for KG (use Layer 3 primary chunks)
                kg_chunks = [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "raw_content": chunk.raw_content,
                        "metadata": {
                            "document_id": chunk.metadata.document_id,
                            "section_path": chunk.metadata.section_path,
                            "section_title": chunk.metadata.section_title,
                        }
                    }
                    for chunk in chunks["layer3"]
                ]

                # Build knowledge graph
                knowledge_graph = self.kg_pipeline.build_from_chunks(
                    chunks=kg_chunks,
                    document_id=result.document_id
                )

                logger.info(
                    f"✓ Knowledge Graph: {len(knowledge_graph.entities)} entities, "
                    f"{len(knowledge_graph.relationships)} relationships"
                )

                # Save KG if output_dir specified
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    kg_path = output_dir / f"{result.document_id}_kg.json"
                    knowledge_graph.save_json(str(kg_path))
                    logger.info(f"✓ Saved Knowledge Graph: {kg_path}")

            except Exception as e:
                logger.error(f"✗ Knowledge Graph construction failed: {e}")
                logger.warning("Continuing without Knowledge Graph...")
                knowledge_graph = None

        # Save intermediate results
        if save_intermediate and output_dir:
            self._save_intermediate(
                output_dir=output_dir,
                result=result,
                chunks=chunks,
                chunking_stats=chunking_stats
            )

        logger.info("="*80)
        logger.info(f"✓ Indexing complete: {document_path.name}")
        logger.info("="*80)

        # Prepare result
        result_dict = {
            "vector_store": vector_store,
            "knowledge_graph": knowledge_graph,
            "stats": {
                "document_id": result.document_id,
                "source_path": str(document_path),
                "vector_store": store_stats,
                "chunking": chunking_stats,
                "hybrid_enabled": self.config.enable_hybrid_search,
                "kg_enabled": self.config.enable_knowledge_graph,
                "kg_entities": len(knowledge_graph.entities) if knowledge_graph else 0,
                "kg_relationships": len(knowledge_graph.relationships) if knowledge_graph else 0,
            }
        }

        if save_intermediate:
            result_dict["chunks"] = chunks

        return result_dict

    def index_batch(
        self,
        document_paths: list,
        output_dir: Path,
        save_per_document: bool = False
    ) -> Dict:
        """
        Index multiple documents into a single vector store.

        Supported formats: PDF, DOCX, PPTX, XLSX, HTML

        Args:
            document_paths: List of document file paths
            output_dir: Directory to save vector store
            save_per_document: Save individual document vector stores

        Returns:
            Dict with:
                - vector_store: Combined FAISSVectorStore
                - knowledge_graphs: List of KnowledgeGraphs (if enabled)
                - stats: Batch statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch indexing {len(document_paths)} documents...")

        # Create combined vector store
        vector_store = FAISSVectorStore(dimensions=self.embedder.dimensions)

        # Collect knowledge graphs
        knowledge_graphs = []

        for i, document_path in enumerate(document_paths, 1):
            logger.info(f"\n[{i}/{len(document_paths)}] Processing: {Path(document_path).name}")

            try:
                # Index document
                result = self.index_document(
                    document_path=document_path,
                    save_intermediate=False,
                    output_dir=output_dir if save_per_document else None
                )

                # Extract components
                doc_store = result["vector_store"]
                doc_kg = result.get("knowledge_graph")

                # Merge into combined store (by adding chunks)
                # Note: This is simplified - production would need proper merging
                vector_store.index_layer1 = doc_store.index_layer1
                vector_store.index_layer2 = doc_store.index_layer2
                vector_store.index_layer3 = doc_store.index_layer3
                vector_store.metadata_layer1.extend(doc_store.metadata_layer1)
                vector_store.metadata_layer2.extend(doc_store.metadata_layer2)
                vector_store.metadata_layer3.extend(doc_store.metadata_layer3)

                # Collect knowledge graph
                if doc_kg:
                    knowledge_graphs.append(doc_kg)

                # Save per-document store
                if save_per_document:
                    doc_name = Path(document_path).stem
                    doc_output = output_dir / f"{doc_name}_store"
                    doc_store.save(doc_output)
                    logger.info(f"✓ Saved individual store: {doc_output}")

            except Exception as e:
                logger.error(f"✗ Failed to index {document_path}: {e}")
                continue

        # Save combined store
        combined_output = output_dir / "combined_store"
        vector_store.save(combined_output)

        # Save combined knowledge graph (if any)
        if knowledge_graphs and self.kg_pipeline:
            try:
                # Merge all KGs
                from graph import KnowledgeGraph
                combined_kg = KnowledgeGraph(
                    entities=[e for kg in knowledge_graphs for e in kg.entities],
                    relationships=[r for kg in knowledge_graphs for r in kg.relationships],
                )
                combined_kg.compute_stats()

                kg_output = output_dir / "combined_kg.json"
                combined_kg.save_json(str(kg_output))
                logger.info(f"✓ Saved combined Knowledge Graph: {kg_output}")
                logger.info(
                    f"  Total: {len(combined_kg.entities)} entities, "
                    f"{len(combined_kg.relationships)} relationships"
                )
            except Exception as e:
                logger.error(f"✗ Failed to save combined KG: {e}")

        logger.info(f"\n✓ Batch indexing complete: {vector_store.get_stats()}")
        logger.info(f"✓ Saved to: {combined_output}")

        return {
            "vector_store": vector_store,
            "knowledge_graphs": knowledge_graphs,
            "stats": {
                "total_documents": len(document_paths),
                "successful": len(knowledge_graphs) if self.kg_pipeline else len(document_paths),
                "vector_store": vector_store.get_stats(),
                "kg_enabled": self.config.enable_knowledge_graph,
                "total_entities": sum(len(kg.entities) for kg in knowledge_graphs),
                "total_relationships": sum(len(kg.relationships) for kg in knowledge_graphs),
            }
        }

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
    # Initialize pipeline with research-optimal settings + Knowledge Graph
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
        normalize_embeddings=True,

        # PHASE 5A: Knowledge Graph (NEW!)
        enable_knowledge_graph=True,
        kg_llm_provider="openai",
        kg_llm_model="gpt-4o-mini",
        kg_backend="simple",  # simple, neo4j, or networkx
    )

    pipeline = IndexingPipeline(config)

    # Index single document
    result = pipeline.index_document(
        document_path=Path("data/document.pdf"),
        save_intermediate=True,
        output_dir=Path("output/indexing")
    )

    # Extract components
    vector_store = result["vector_store"]
    knowledge_graph = result["knowledge_graph"]

    # Save vector store
    vector_store.save(Path("output/vector_store"))

    # Save knowledge graph
    if knowledge_graph:
        knowledge_graph.save_json("output/knowledge_graph.json")
        print(f"\nKnowledge Graph:")
        print(f"  Entities: {len(knowledge_graph.entities)}")
        print(f"  Relationships: {len(knowledge_graph.relationships)}")

    # Search example
    query_embedding = pipeline.embedder.embed_texts(["safety specification"])
    results = vector_store.hierarchical_search(query_embedding, k_layer3=6)

    print(f"\nTop results:")
    for i, result in enumerate(results["layer3"][:3], 1):
        print(f"{i}. {result['section_title']} (score: {result['score']:.4f})")
