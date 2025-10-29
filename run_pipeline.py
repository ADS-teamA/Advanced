#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner - FULL SOTA 2025 SYSTEM

Runs complete indexing pipeline with ALL advanced features:
- PHASE 1: Smart hierarchy extraction (Docling)
- PHASE 2: Generic summaries
- PHASE 3: Multi-layer chunking + SAC (contextual retrieval)
- PHASE 4: Embeddings + FAISS indexing
- PHASE 5A: Knowledge Graph extraction ✅
- PHASE 5B: Hybrid Search (BM25 + Dense + RRF) ✅

Supported formats: PDF, DOCX, PPTX, XLSX, HTML
Input: Single document OR directory (batch processing)

Usage:
    # Single document
    python run_pipeline.py <document_path>
    python run_pipeline.py "data/document.pdf"

    # Batch processing (directory)
    python run_pipeline.py <directory_path>
    python run_pipeline.py "data/documents/"

Outputs saved to: output/<document_name>/<timestamp>/
- phase1_extraction.json - Document structure & hierarchy
- phase2_summaries.json - Generic summaries
- phase3_chunks.json - Multi-layer chunks with SAC
- phase4_vector_store/ - FAISS + BM25 indexes + metadata
- <document_id>_kg.json - Knowledge graph (if enabled)

Configuration: All settings controlled via .env file
- See .env.example for available options
- Hybrid Search: ENABLE_HYBRID_SEARCH=true
- Knowledge Graph: ENABLE_KNOWLEDGE_GRAPH=true
"""

import sys
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from src.cost_tracker import get_global_tracker, reset_global_tracker


def print_header(text: str):
    """Print formatted header."""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
    print()


def print_success(text: str):
    """Print success message."""
    print(f"✓ {text}")


def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


def get_supported_documents(directory: Path) -> list:
    """
    Get list of all supported documents in directory.

    Args:
        directory: Path to directory

    Returns:
        List of document paths
    """
    supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]
    documents = []

    for ext in supported_formats:
        documents.extend(directory.glob(f"*{ext}"))
        documents.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    documents = sorted(set(documents))

    return documents


def run_complete_pipeline(input_path: Path, output_base: Path = None, merge_target: Path = None):
    """
    Run complete SOTA 2025 RAG pipeline.

    Features:
    - Smart hierarchy extraction
    - Generic summaries
    - Contextual retrieval (SAC)
    - Hybrid search (BM25 + Dense + RRF)
    - Knowledge graph extraction
    - Automatic merge with existing vector store

    Args:
        input_path: Path to document file or directory
        output_base: Base output directory (default: output/)
        merge_target: Path to existing vector store to merge into (e.g., vector_db/)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"✗ Error: Path not found: {input_path}")
        sys.exit(1)

    # Check if it's a directory
    if input_path.is_dir():
        documents = get_supported_documents(input_path)
        if not documents:
            print(f"✗ Error: No supported documents found in directory: {input_path}")
            print(f"  Supported formats: PDF, DOCX, PPTX, XLSX, HTML")
            sys.exit(1)

        print_header(f"BATCH PROCESSING - {len(documents)} documents from {input_path.name}")
        print_info(f"Found documents:")
        for doc in documents:
            print_info(f"  - {doc.name}")
        print()

        # Process each document
        for i, document_path in enumerate(documents, 1):
            print()
            print("=" * 80)
            print(f"PROCESSING [{i}/{len(documents)}]: {document_path.name}")
            print("=" * 80)
            print()
            run_single_document(document_path, output_base, merge_target)

        print_header("BATCH PROCESSING COMPLETE")
        print_success(f"Processed {len(documents)} documents")
        print_info(f"Outputs saved to: {output_base or Path('output')}")
        return

    # Single document processing
    run_single_document(input_path, output_base, merge_target)


def run_single_document(document_path: Path, output_base: Path = None, merge_target: Path = None):
    """
    Process single document through complete SOTA 2025 pipeline.

    Args:
        document_path: Path to document file
        output_base: Base output directory (default: output/)
        merge_target: Path to existing vector store to merge into (e.g., vector_db/)
    """
    document_path = Path(document_path)

    # Validate format
    supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]
    if document_path.suffix.lower() not in supported_formats:
        print(f"✗ Error: Unsupported format: {document_path.suffix}")
        print(f"  Supported formats: {', '.join(supported_formats)}")
        return

    # Setup output directory
    if output_base is None:
        output_base = Path(__file__).parent / "output"

    doc_name = document_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
    output_dir = output_base / doc_name  # No timestamp - enables resume functionality
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"SOTA 2025 RAG PIPELINE - {document_path.name}")

    # Reset cost tracker for this document
    reset_global_tracker()

    # Create pipeline with FULL SOTA configuration from .env
    print_info("Initializing pipeline with SOTA 2025 configuration...")
    config = IndexingConfig.from_env()  # Loads all settings from .env

    # Print active configuration
    print_info(f"LLM Model: {config.summarization_config.model}")
    print_info(f"Embedding Model: {config.embedding_config.model}")
    print_info(f"Chunk Size: {config.chunking_config.chunk_size} chars")
    print_info(f"SAC (Contextual Retrieval): {'ON' if config.chunking_config.enable_contextual else 'OFF'}")
    print_info(f"Hybrid Search (BM25+Dense): {'ON ✅' if config.enable_hybrid_search else 'OFF'}")
    print_info(f"Knowledge Graph: {'ON ✅' if config.enable_knowledge_graph else 'OFF'}")
    print_info(f"Output: {output_dir}")
    print()

    pipeline = IndexingPipeline(config)

    # Run pipeline with intermediate saves
    try:
        print_header("RUNNING PIPELINE (ALL PHASES)")

        result = pipeline.index_document(
            document_path=document_path,
            save_intermediate=True,  # Save phase1-3 JSON outputs
            output_dir=output_dir
        )

        # Check if indexing was skipped due to duplicate detection
        if result is None:
            print_header("DOCUMENT SKIPPED")
            print_info("Document was identified as duplicate and skipped")
            print_info("No indexing or merging performed")
            return

        vector_store = result["vector_store"]
        knowledge_graph = result["knowledge_graph"]
        stats = result["stats"]

        # Save PHASE 4: Vector store (FAISS + BM25 if hybrid)
        print_info("Saving vector store...")
        vs_path = output_dir / "phase4_vector_store"
        vector_store.save(vs_path)
        print_success(f"Vector store saved: {vs_path}")

        # Save PHASE 5A: Knowledge Graph (if enabled)
        kg_path = None
        if knowledge_graph:
            kg_path = output_dir / f"{doc_name}_kg.json"
            knowledge_graph.save_json(str(kg_path))
            print_success(f"Knowledge graph saved: {kg_path}")

        # MERGE with existing vector store (if --merge flag provided)
        if merge_target:
            merge_target = Path(merge_target)

            if not merge_target.exists():
                print()
                print_info(f"Creating new vector store at: {merge_target}")
                merge_target.mkdir(parents=True, exist_ok=True)
                # Copy new store to merge target
                import shutil
                for item in vs_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, merge_target / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, merge_target / item.name, dirs_exist_ok=True)
                print_success(f"Initialized vector store at: {merge_target}")
            else:
                print()
                print_header("MERGING WITH EXISTING VECTOR STORE")
                print_info(f"Target: {merge_target}")

                try:
                    # Load existing vector store
                    from src.hybrid_search import HybridVectorStore
                    from src.faiss_vector_store import FAISSVectorStore

                    print_info("Loading existing vector store...")
                    existing_store = HybridVectorStore.load(merge_target)
                    existing_stats_before = existing_store.get_stats()

                    print_info(f"Existing store: {existing_stats_before['total_vectors']} vectors, "
                              f"{existing_stats_before['documents']} documents")
                    print_info(f"New store: {stats['vector_store']['total_vectors']} vectors, "
                              f"{stats['vector_store']['documents']} documents")

                    # Merge vector stores with deduplication
                    print_info("Merging FAISS indexes with deduplication...")
                    faiss_merge_stats = existing_store.faiss_store.merge(
                        vector_store if isinstance(vector_store, FAISSVectorStore) else vector_store.faiss_store
                    )

                    print_info("Merging BM25 indexes...")
                    if hasattr(existing_store, 'bm25_store') and hasattr(vector_store, 'bm25_store'):
                        existing_store.bm25_store.merge(vector_store.bm25_store)

                    # Save merged store
                    print_info("Saving merged vector store...")
                    existing_store.save(merge_target)

                    merged_stats = existing_store.get_stats()
                    print_success(f"Merge complete!")
                    print_info(f"Merged store: {merged_stats['total_vectors']} vectors, "
                              f"{merged_stats['documents']} documents")
                    print_info(f"Added: {faiss_merge_stats['added']} vectors, "
                              f"Skipped: {faiss_merge_stats['skipped']} duplicates")

                    # Merge Knowledge Graphs with cross-document relationships
                    if knowledge_graph and config.enable_knowledge_graph:
                        print()
                        print_info("Merging knowledge graphs with cross-document deduplication...")

                        from src.graph import (
                            KnowledgeGraph,
                            UnifiedKnowledgeGraphManager,
                            CrossDocumentRelationshipDetector
                        )

                        # Initialize unified KG manager (uses merge_target directory for storage)
                        storage_dir = merge_target.parent if merge_target.parent.name != 'phase4_vector_store' else merge_target.parent.parent
                        manager = UnifiedKnowledgeGraphManager(storage_dir=str(storage_dir))

                        # Initialize cross-document detector
                        detector = CrossDocumentRelationshipDetector(
                            use_llm_validation=False,  # Fast pattern-based detection
                            confidence_threshold=0.7
                        )

                        # Load or create unified KG
                        unified_kg = manager.load_or_create()

                        print_info(f"Current unified KG: {len(unified_kg.entities)} entities, "
                                  f"{len(unified_kg.relationships)} relationships")
                        print_info(f"New document KG: {len(knowledge_graph.entities)} entities, "
                                  f"{len(knowledge_graph.relationships)} relationships")

                        # Merge with deduplication and cross-doc detection
                        unified_kg = manager.merge_document_graph(
                            unified_kg=unified_kg,
                            document_kg=knowledge_graph,
                            document_id=doc_name,
                            cross_doc_detector=detector
                        )

                        # Save unified KG + per-document backup
                        manager.save(unified_kg, document_id=doc_name)

                        # Get statistics
                        doc_stats = manager.get_document_statistics(unified_kg)

                        print_success(f"KG merge complete with cross-document relationships!")
                        print_info(f"Unified KG: {len(unified_kg.entities)} entities, "
                                  f"{len(unified_kg.relationships)} relationships")
                        print_info(f"Documents in unified KG: {doc_stats['total_documents']}")
                        print_info(f"Cross-document entities: {doc_stats['cross_document_entities']} "
                                  f"({doc_stats['cross_document_entity_percentage']:.1f}%)")
                        print_info(f"Saved: {storage_dir / 'unified_kg.json'}")

                except Exception as e:
                    print()
                    print_info(f"⚠️  Merge failed: {e}")
                    print_info("Continuing without merge...")
                    import traceback
                    logger.debug(traceback.format_exc())

        # Print comprehensive statistics
        print_header("INDEXING COMPLETE")
        print_success(f"Document processed: {document_path.name}")
        print()

        print_info("📊 PIPELINE STATISTICS")
        print_info("-" * 60)

        # Chunking stats
        chunking = stats["chunking"]
        print_info(f"Layer 1 (documents):  {chunking['layer1_count']:4d} chunks")
        print_info(f"Layer 2 (sections):   {chunking['layer2_count']:4d} chunks")
        print_info(f"Layer 3 (paragraphs): {chunking['layer3_count']:4d} chunks (PRIMARY)")
        print_info(f"Total chunks:         {chunking['total_chunks']:4d}")

        if 'layer3_avg_size' in chunking:
            print()
            print_info(f"Layer 3 avg size:     {chunking['layer3_avg_size']:.0f} chars")
            if 'context_avg_overhead' in chunking:
                print_info(f"SAC context overhead: {chunking['context_avg_overhead']:.0f} chars/chunk")

        # Vector store stats
        vs_stats = stats["vector_store"]
        print()
        print_info(f"Vector dimensions:    {vs_stats.get('dense_dimensions', vs_stats.get('dimensions', 'N/A'))}D")
        print_info(f"Total vectors:        {vs_stats['total_vectors']}")
        print_info(f"Documents indexed:    {vs_stats['documents']}")

        # Hybrid search stats
        if stats.get("hybrid_enabled"):
            print()
            print_info("✅ Hybrid Search: ACTIVE")
            print_info(f"   BM25 Layer 1:      {vs_stats.get('bm25_layer1_docs', 0)} docs")
            print_info(f"   BM25 Layer 2:      {vs_stats.get('bm25_layer2_docs', 0)} docs")
            print_info(f"   BM25 Layer 3:      {vs_stats.get('bm25_layer3_docs', 0)} docs")
            print_info(f"   RRF Fusion k:      {config.hybrid_fusion_k}")
        else:
            print()
            print_info("ℹ️  Hybrid Search: DISABLED (dense-only retrieval)")

        # Knowledge graph stats
        if stats.get("kg_construction_failed"):
            print()
            print_info("❌ Knowledge Graph: FAILED")
            print_info(f"   Error: {stats.get('kg_error', 'Unknown error')}")
            print_info("   Continuing with vector search only")
        elif stats.get("kg_enabled") and knowledge_graph:
            print()
            print_info("✅ Knowledge Graph: ACTIVE")
            print_info(f"   Entities:          {stats['kg_entities']}")
            print_info(f"   Relationships:     {stats['kg_relationships']}")
        elif stats.get("kg_enabled"):
            print()
            print_info("⚠️  Knowledge Graph: ENABLED but no graph generated")
        else:
            print()
            print_info("ℹ️  Knowledge Graph: DISABLED")

        print()

    except FileNotFoundError as e:
        logger.error(f"Document not found: {e}", exc_info=True)
        print(f"\n✗ ERROR: Document not found")
        print(f"   {e}")
        print("\nPlease check:")
        print(f"  - Path is correct: {document_path}")
        print("  - File exists and is readable")
        sys.exit(1)

    except PermissionError as e:
        logger.error(f"Permission denied: {e}", exc_info=True)
        print(f"\n✗ ERROR: Permission denied")
        print(f"   {e}")
        print("\nPlease check:")
        print("  - File/directory permissions")
        print("  - Current user has read/write access")
        sys.exit(1)

    except MemoryError as e:
        logger.error(f"Out of memory: {e}", exc_info=True)
        print(f"\n✗ ERROR: Out of memory")
        print(f"   Document may be too large: {document_path}")
        print("\nSolutions:")
        print("  - Process smaller documents")
        print("  - Increase system memory")
        print("  - Reduce batch_size in config")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)

    except Exception as e:
        # Check if it's an API error (works for both anthropic and openai)
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['api', 'authentication', 'auth', 'key', 'quota', 'rate limit']):
            logger.error(f"API error: {e}", exc_info=True)
            print(f"\n✗ ERROR: API call failed")
            print(f"   {e}")
            print("\nPlease check:")
            print("  - API keys are valid (check .env file)")
            print("  - API quota/rate limits")
            print("  - Internet connectivity")
            print("  - API service status")
        else:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n✗ ERROR: {e}")
            print()
            print("Common issues:")
            print("  - Missing API keys (check .env file)")
            print("  - Invalid document format")
            print("  - Insufficient disk space")
            print("\nFor more details, check the log file above.")
        sys.exit(1)

    # Print cost summary
    tracker = get_global_tracker()

    if tracker.total_cost > 0:
        print_header("💰 API COST SUMMARY")
        print_info(f"Total cost:        ${tracker.total_cost:.4f} USD")
        print_info(f"Total calls:       {len(tracker.entries)}")
        print_info(f"Input tokens:      {tracker.total_input_tokens:,}")
        print_info(f"Output tokens:     {tracker.total_output_tokens:,}")
        print()

        # Cost by provider
        if tracker.cost_by_provider:
            print_info("Cost by provider:")
            for provider, cost in tracker.cost_by_provider.items():
                print_info(f"  {provider:12s}  ${cost:.4f}")
            print()

        # Cost by operation
        if tracker.cost_by_operation:
            print_info("Cost by operation:")
            for operation, cost in tracker.cost_by_operation.items():
                print_info(f"  {operation:12s}  ${cost:.4f}")
        print()

    # Final success message
    print("=" * 80)
    print(f"✅ ALL DONE! Outputs saved in: {output_dir}")
    print("=" * 80)
    print()
    print("📁 Output files:")
    print(f"  • phase1_extraction.json    - Document structure")
    print(f"  • phase2_summaries.json      - Section summaries")
    print(f"  • phase3_chunks.json         - Multi-layer chunks")
    print(f"  • phase4_vector_store/       - FAISS + BM25 indexes")
    if knowledge_graph:
        print(f"  • {doc_name}_kg.json         - Knowledge graph")
    print()
    print("🚀 To use with agent:")
    print(f"  uv run python -m src.agent.cli --vector-store {vs_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete SOTA 2025 RAG Pipeline - Index documents with automatic merge support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index single document
  python run_pipeline.py data/document.pdf

  # Index directory (batch)
  python run_pipeline.py data/documents/

  # Index and merge into existing vector_db
  python run_pipeline.py data/document.pdf --merge vector_db

  # Batch index with merge
  python run_pipeline.py data/documents/ --merge vector_db
        """
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to document file or directory to index"
    )

    parser.add_argument(
        "--merge",
        type=str,
        metavar="TARGET",
        help="Merge indexed documents into existing vector store at TARGET path (e.g., vector_db)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    merge_target = Path(args.merge) if args.merge else None

    run_complete_pipeline(input_path, merge_target=merge_target)
