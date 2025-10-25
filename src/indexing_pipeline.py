"""
PHASE 1-6: Complete Indexing Pipeline

Orchestrates:
1. PHASE 1-3: Extraction, hierarchy, summaries, chunking
2. PHASE 4: Embedding generation + FAISS indexing
3. PHASE 5A: Knowledge Graph construction (optional)
4. PHASE 5B: Hybrid Search with BM25 + RRF (optional)
5. PHASE 5C: Cross-Encoder Reranking (optional)
6. PHASE 5D: Graph-Vector Integration (optional)
7. PHASE 6: Context Assembly for LLM (optional)

Supported formats: PDF, DOCX, PPTX, XLSX, HTML

Based on research:
- LegalBench-RAG: text-embedding-3-large + RCTS
- Multi-Layer Embeddings: 3 separate indexes
- Hybrid Search: BM25 + Dense + RRF fusion
- Cross-Encoder Reranking: Two-stage retrieval
- Graph Integration: Entity-aware boosting
- Context Assembly: SAC stripping + citations

Usage:
    pipeline = IndexingPipeline(
        embedding_model="text-embedding-3-large",
        enable_sac=True,
        enable_knowledge_graph=True,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_context_assembly=True
    )

    result = pipeline.index_document("document.pdf")
    result["vector_store"].save("output/vector_store")
    result["knowledge_graph"].save_json("output/knowledge_graph.json")
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field

from src.config import (
    ExtractionConfig,
    SummarizationConfig,
    ChunkingConfig,
    EmbeddingConfig,
)
from src.docling_extractor_v2 import DoclingExtractorV2
from src.multi_layer_chunker import MultiLayerChunker
from src.embedding_generator import EmbeddingGenerator
from src.faiss_vector_store import FAISSVectorStore
from src.cost_tracker import get_global_tracker, reset_global_tracker

# Knowledge Graph imports (optional)
try:
    from src.graph import (
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
    """
    Configuration for complete indexing pipeline.

    Uses nested config objects for clean architecture. All sub-configs
    can be customized or loaded from environment via from_env().
    """

    # Speed/Cost Mode (determines Batch API usage)
    # "fast" = completions (2-3 min, full price) | "eco" = Batch API (15-30 min, 50% cheaper)
    speed_mode: str = "fast"  # "fast" or "eco"

    # Sub-configs (nested config objects)
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)
    summarization_config: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # PHASE 5A: Knowledge Graph (enabled by default for SOTA 2025)
    enable_knowledge_graph: bool = True
    kg_config: Optional[KnowledgeGraphConfig] = None

    # PHASE 5B: Hybrid Search (enabled by default for SOTA 2025)
    enable_hybrid_search: bool = True  # BM25 + dense with RRF fusion (+23% precision)
    hybrid_fusion_k: int = 60  # RRF k parameter (research: k=60 optimal)

    # PHASE 5C: Cross-Encoder Reranking (optional)
    enable_reranking: bool = False  # Two-stage retrieval: hybrid → rerank
    reranker_model: str = "bge-reranker-large"  # SOTA accuracy (aliases: "accurate", "sota")
    reranker_candidates: int = 50  # Retrieve 50 via hybrid search
    reranker_top_k: int = 6  # Rerank to top 6

    # PHASE 5D: Graph-Vector Integration (optional)
    enable_graph_retrieval: bool = False  # Graph-enhanced retrieval
    graph_boost_weight: float = 0.3  # Boost weight for graph matches
    enable_multi_hop: bool = False  # Multi-hop graph traversal

    # PHASE 6: Context Assembly (optional)
    enable_context_assembly: bool = False  # Assemble retrieved chunks for LLM
    citation_format: str = "inline"  # Citation style: inline, simple, detailed, footnote
    max_context_chunks: int = 6  # Max chunks to include in assembled context
    max_context_tokens: int = 4000  # Max tokens in assembled context (~16K chars)
    include_chunk_metadata: bool = True  # Include document/section/page metadata

    def __post_init__(self):
        """Configure speed mode and validate settings."""
        # Validate speed mode
        if self.speed_mode not in ["fast", "eco"]:
            raise ValueError(f"speed_mode must be 'fast' or 'eco', got: {self.speed_mode}")

        # Configure summarization based on speed mode
        if self.speed_mode == "eco":
            # Eco mode: Use Batch API (50% cheaper, slower)
            self.summarization_config.use_batch_api = True
            self.summarization_config.batch_api_timeout = 43200  # 12 hours
            logger.info(f"💰 ECO MODE: Using Batch API (50% cost savings, 15-30 min latency)")
        else:  # fast mode
            # Fast mode: Use completions (full price, fast)
            self.summarization_config.use_batch_api = False
            logger.info(f"⚡ FAST MODE: Using completions (full price, 2-3 min latency)")

        # Initialize KG config if enabled
        if self.enable_knowledge_graph and self.kg_config is None:
            try:
                from src.graph.config import KnowledgeGraphConfig

                self.kg_config = KnowledgeGraphConfig.from_env()
            except ImportError:
                logger.warning("Knowledge Graph enabled but graph module not available")

    @classmethod
    def from_env(cls, **overrides) -> "IndexingConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            SPEED_MODE: "fast" or "eco" (default: "fast")
            ENABLE_KNOWLEDGE_GRAPH: Enable KG construction (default: "true")
            ENABLE_HYBRID_SEARCH: Enable hybrid search (default: "true")

        Args:
            **overrides: Override specific fields

        Returns:
            IndexingConfig instance loaded from environment
        """
        # Load speed mode from env
        speed_mode = os.getenv("SPEED_MODE", "fast")

        # Create config with sub-configs loaded from env
        config = cls(
            speed_mode=speed_mode,
            extraction_config=ExtractionConfig.from_env(),
            summarization_config=SummarizationConfig.from_env(),
            chunking_config=ChunkingConfig.from_env(),
            embedding_config=EmbeddingConfig.from_env(),
            enable_knowledge_graph=os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true",
            enable_hybrid_search=os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
            **overrides,
        )

        return config


class IndexingPipeline:
    """
    Complete indexing pipeline for RAG system.

    Phases:
    1. Smart hierarchy extraction (font-size based)
    2. Generic summary generation (gpt-4o-mini)
    3. Multi-layer chunking + SAC (RCTS 500 chars)
    4. Embedding + FAISS indexing (3 separate indexes)
    5A. Knowledge Graph construction (optional)
    5B. Hybrid search with BM25 + RRF (optional)
    5C. Cross-encoder reranking (optional)
    5D. Graph-vector integration (optional)
    6. Context assembly for LLM (optional)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Summary-Augmented Chunking (Reuter et al., 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    - Contextual Retrieval (Anthropic, 2024)
    - HybridRAG (2024): Graph + Vector integration
    - Context Assembly: SAC stripping + citations
    """

    def __init__(self, config: Optional[IndexingConfig] = None):
        """
        Initialize indexing pipeline.

        Args:
            config: IndexingConfig instance (defaults loaded from .env)
        """
        self.config = config or IndexingConfig.from_env()

        logger.info("Initializing IndexingPipeline...")

        # Initialize PHASE 1: Extraction (uses nested config)
        self.extractor = DoclingExtractorV2(self.config.extraction_config)

        # Initialize PHASE 3: Chunking (uses nested config)
        self.chunker = MultiLayerChunker(config=self.config.chunking_config)

        # Initialize PHASE 4: Embedding (uses nested config)
        self.embedder = EmbeddingGenerator(self.config.embedding_config)

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
            f"SAC={self.config.chunking_config.enable_contextual}, "
            f"model={self.config.embedding_config.model} "
            f"({self.embedder.dimensions}D), "
            f"KG={self.config.enable_knowledge_graph}, "
            f"Hybrid={self.config.enable_hybrid_search}, "
            f"Rerank={self.config.enable_reranking}, "
            f"GraphRetrieval={self.config.enable_graph_retrieval}"
        )

    def _initialize_kg_pipeline(self):
        """Initialize Knowledge Graph pipeline using nested kg_config."""
        if self.config.kg_config is None:
            logger.warning("Knowledge Graph enabled but no kg_config provided")
            self.kg_pipeline = None
            return

        # Use the kg_config directly (already initialized in IndexingConfig.__post_init__)
        kg_config = self.config.kg_config

        # Validate API key
        if kg_config.entity_extraction.llm_provider == "openai" and not kg_config.openai_api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.kg_pipeline = None
            return
        elif (
            kg_config.entity_extraction.llm_provider == "anthropic"
            and not kg_config.anthropic_api_key
        ):
            logger.warning(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
            self.kg_pipeline = None
            return

        # Initialize pipeline with config
        self.kg_pipeline = KnowledgeGraphPipeline(kg_config)
        logger.info(
            f"Knowledge Graph initialized: "
            f"model={kg_config.entity_extraction.llm_model}, "
            f"backend={kg_config.graph_storage.backend.value}"
        )

    def index_document(
        self,
        document_path: Path,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None,
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

        logger.info("=" * 80)
        logger.info(f"Indexing document: {document_path.name}")
        logger.info("=" * 80)

        # Reset cost tracker for this document
        reset_global_tracker()

        # PHASE 1+2: Extract + Summaries
        logger.info("PHASE 1+2: Extraction + Summaries")
        result = self.extractor.extract(document_path)
        logger.info(
            f"✓ Extracted: {result.num_sections} sections, " f"depth={result.hierarchy_depth}"
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
            "layer3": self.embedder.embed_chunks(chunks["layer3"], layer=3),
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
                    fusion_k=self.config.hybrid_fusion_k,
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
        kg_error = None
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
                        },
                    }
                    for chunk in chunks["layer3"]
                ]

                # Build knowledge graph
                knowledge_graph = self.kg_pipeline.build_from_chunks(
                    chunks=kg_chunks, document_id=result.document_id
                )

                logger.info(
                    f"✓ Knowledge Graph: {len(knowledge_graph.entities)} entities, "
                    f"{len(knowledge_graph.relationships)} relationships"
                )

            except Exception as e:
                logger.error(f"✗ Knowledge Graph construction failed: {e}", exc_info=True)
                if self.config.enable_knowledge_graph:
                    logger.error(
                        f"ERROR: Knowledge Graph was enabled in config but construction failed.\n"
                        f"Error: {e}\n"
                        f"To disable KG: Set ENABLE_KNOWLEDGE_GRAPH=false in .env"
                    )
                knowledge_graph = None
                kg_error = str(e)

        # Save intermediate results
        if save_intermediate and output_dir:
            self._save_intermediate(
                output_dir=output_dir, result=result, chunks=chunks, chunking_stats=chunking_stats
            )

        # Display cost summary
        tracker = get_global_tracker()
        if tracker.get_total_cost() > 0:
            logger.info("\n" + tracker.get_summary())

        logger.info("=" * 80)
        logger.info(f"✓ Indexing complete: {document_path.name}")
        logger.info("=" * 80)

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
                "kg_construction_failed": self.config.enable_knowledge_graph
                and knowledge_graph is None
                and kg_error is not None,
                "kg_error": kg_error if kg_error else None,
            },
        }

        if save_intermediate:
            result_dict["chunks"] = chunks

        return result_dict

    def index_batch(
        self, document_paths: list, output_dir: Path, save_per_document: bool = False
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
                    output_dir=output_dir if save_per_document else None,
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
            },
        }

    def _save_intermediate(self, output_dir: Path, result, chunks: Dict, chunking_stats: Dict):
        """Save intermediate results from all phases (PHASE 1, 2, 3)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # PHASE 1: Save extraction results (structure & hierarchy)
        phase1_path = output_dir / "phase1_extraction.json"
        phase1_export = {
            "document_id": result.document_id,
            "source_path": str(result.source_path),
            "num_sections": result.num_sections,
            "hierarchy_depth": result.hierarchy_depth,
            "num_roots": result.num_roots,
            "num_tables": result.num_tables,
            "sections": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "level": s.level,
                    "depth": s.depth,
                    "path": s.path,
                    "page_number": s.page_number,
                    "content_length": len(s.content),
                }
                for s in result.sections
            ],
        }
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(phase1_export, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved PHASE 1: {phase1_path}")

        # PHASE 2: Save summaries
        phase2_path = output_dir / "phase2_summaries.json"
        phase2_export = {
            "document_id": result.document_id,
            "document_summary": result.document_summary,
            "section_summaries": [
                {"section_id": s.section_id, "title": s.title, "summary": s.summary}
                for s in result.sections
            ],
        }
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(phase2_export, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved PHASE 2: {phase2_path}")

        # PHASE 3: Save chunks
        phase3_path = output_dir / "phase3_chunks.json"
        phase3_export = {
            "document_id": result.document_id,
            "source_path": str(result.source_path),
            "chunking_stats": chunking_stats,
            "layer1": [c.to_dict() for c in chunks["layer1"]],
            "layer2": [c.to_dict() for c in chunks["layer2"]],
            "layer3": [c.to_dict() for c in chunks["layer3"]],
        }
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(phase3_export, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved PHASE 3: {phase3_path}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with research-optimal settings
    # Knowledge Graph is now enabled by default (PHASE 5A)
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
        # PHASE 5A: Knowledge Graph (enabled by default)
        # enable_knowledge_graph=True,  # ✅ No need to set - default is True
        kg_llm_provider="openai",
        kg_llm_model="gpt-4o-mini",
        kg_backend="simple",  # simple, neo4j, or networkx
    )

    pipeline = IndexingPipeline(config)

    # Index single document
    result = pipeline.index_document(
        document_path=Path("data/document.pdf"),
        save_intermediate=True,
        output_dir=Path("output/indexing"),
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
