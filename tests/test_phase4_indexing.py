#!/usr/bin/env python3
"""
PHASE 4: Embedding & FAISS Indexing Test

Tests complete indexing pipeline:
- PHASE 1: Smart Hierarchy Extraction
- PHASE 2: Generic Summary Generation
- PHASE 3: Multi-Layer Chunking + SAC
- PHASE 4: Embedding + FAISS Indexing

Based on research:
- text-embedding-3-large (LegalBench-RAG baseline)
- 3 separate FAISS indexes (Multi-Layer Embeddings)
- Dense-only retrieval (no BM25)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from indexing_pipeline import IndexingPipeline, IndexingConfig
from config import ExtractionConfig, SummarizationConfig, ChunkingConfig, EmbeddingConfig


def test_single_document():
    """Test PHASE 4 on a single document."""

    print("="*80)
    print("PHASE 4 TEST: Single Document Indexing")
    print("="*80)
    print()

    # Document path
    pdf_path = Path("data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        print("Please add a test PDF")
        sys.exit(1)

    output_dir = Path("output/phase4_test")

    print(f"Document: {pdf_path.name}")
    print(f"Output:   {output_dir}")
    print()

    # Initialize pipeline with research-optimal settings
    print("Initializing pipeline...")
    print()

    config = IndexingConfig(
        # PHASE 1: Extraction with nested config
        extraction_config=ExtractionConfig(
            enable_smart_hierarchy=True,
            ocr_language=["ces", "eng"]  # Tesseract language codes
        ),

        # PHASE 2: Summarization with nested config
        summarization_config=SummarizationConfig(
            model="gpt-4o-mini",
            max_chars=150
        ),

        # PHASE 3: Chunking with nested config
        chunking_config=ChunkingConfig(
            chunk_size=500,
            chunk_overlap=0,
            enable_contextual=True  # SAC
        ),

        # PHASE 4: Embedding with nested config
        embedding_config=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large",
            batch_size=100,
            normalize=True
        )
    )

    try:
        pipeline = IndexingPipeline(config)

        # Index document
        print("Starting indexing pipeline...")
        print()

        vector_store = pipeline.index_document(
            pdf_path=pdf_path,
            save_intermediate=True,
            output_dir=output_dir
        )

        # Display statistics
        print()
        print("="*80)
        print("VECTOR STORE STATISTICS")
        print("="*80)
        print()

        stats = vector_store.get_stats()
        print(f"Dimensions:        {stats['dimensions']}")
        print(f"Layer 1 (Document): {stats['layer1_count']} vectors")
        print(f"Layer 2 (Section):  {stats['layer2_count']} vectors")
        print(f"Layer 3 (Chunk):    {stats['layer3_count']} vectors (PRIMARY)")
        print(f"Total vectors:      {stats['total_vectors']}")
        print(f"Documents indexed:  {stats['documents']}")
        print()

        # Save vector store
        store_path = output_dir / "vector_store"
        vector_store.save(store_path)
        print(f"✓ Vector store saved: {store_path}")
        print()

        # Test search
        print("="*80)
        print("TESTING RETRIEVAL")
        print("="*80)
        print()

        test_queries = [
            "waste management procedures",
            "environmental impact assessment",
            "disposal methods"
        ]

        for query in test_queries:
            print(f"Query: '{query}'")
            print("-" * 80)

            # Embed query
            query_embedding = pipeline.embedder.embed_texts([query])

            # Hierarchical search
            results = vector_store.hierarchical_search(
                query_embedding=query_embedding,
                k_layer3=6,
                use_doc_filtering=True,
                similarity_threshold_offset=0.25
            )

            # Display results
            print(f"Layer 1 results: {len(results['layer1'])}")
            if results['layer1']:
                doc_result = results['layer1'][0]
                print(f"  Document: {doc_result['document_id']} (score: {doc_result['score']:.4f})")

            print(f"Layer 3 results: {len(results['layer3'])}")
            for i, result in enumerate(results["layer3"][:3], 1):
                print(f"  {i}. {result['section_title']}")
                print(f"     Score: {result['score']:.4f}")
                print(f"     Page:  {result['page_number']}")
                print(f"     Path:  {result['section_path']}")
                print()

        print("="*80)
        print("✓ PHASE 4 TEST COMPLETE")
        print("="*80)
        print()

        print("Next steps:")
        print("  1. Review vector store in output/phase4_test/")
        print("  2. Test with multiple documents (batch indexing)")
        print("  3. PHASE 5: Implement retrieval API")
        print("  4. PHASE 6: Context assembly")
        print("  5. PHASE 7: Answer generation")
        print()

        print("Current status:")
        print("  ✓ PHASE 1: Smart Hierarchy")
        print("  ✓ PHASE 2: Generic Summaries")
        print("  ✓ PHASE 3: Multi-Layer Chunking + SAC")
        print("  ✓ PHASE 4: Embedding + FAISS Indexing")
        print("  ⏳ PHASE 5-7: To be implemented")
        print()

        return vector_store

    except ValueError as e:
        print(f"⚠️  Error: {e}")
        print()
        if "OPENAI_API_KEY" in str(e):
            print("OpenAI API key required for PHASE 2 (summaries) and PHASE 4 (embeddings)")
            print("Set with: export OPENAI_API_KEY='sk-...'")
        return None

    except Exception as e:
        print(f"✗ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_documents():
    """Test PHASE 4 batch indexing (multiple documents)."""

    print()
    print("="*80)
    print("PHASE 4 TEST: Batch Document Indexing")
    print("="*80)
    print()

    # Find all PDFs in data directory
    data_dir = Path("data/regulace/GRI")

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return None

    pdf_paths = list(data_dir.glob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found in data directory")
        return None

    print(f"Found {len(pdf_paths)} PDF files:")
    for pdf_path in pdf_paths:
        print(f"  - {pdf_path.name}")
    print()

    output_dir = Path("output/phase4_batch_test")

    # Initialize pipeline
    config = IndexingConfig(
        extraction_config=ExtractionConfig(
            enable_smart_hierarchy=True,
            ocr_language=["ces", "eng"]
        ),
        summarization_config=SummarizationConfig(
            model="gpt-4o-mini",
            max_chars=150
        ),
        chunking_config=ChunkingConfig(
            chunk_size=500,
            enable_contextual=True  # SAC
        ),
        embedding_config=EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-large"
        )
    )

    try:
        pipeline = IndexingPipeline(config)

        # Batch index
        vector_store = pipeline.index_batch(
            pdf_paths=pdf_paths,
            output_dir=output_dir,
            save_per_document=True
        )

        # Display stats
        print()
        print("="*80)
        print("BATCH INDEXING COMPLETE")
        print("="*80)
        print()

        stats = vector_store.get_stats()
        print(f"Total documents:    {stats['documents']}")
        print(f"Total vectors:      {stats['total_vectors']}")
        print(f"Layer 3 chunks:     {stats['layer3_count']} (PRIMARY)")
        print()

        return vector_store

    except Exception as e:
        print(f"✗ Batch indexing error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_alternative_embedding_model():
    """Test with alternative embedding model (BGE-M3 for multilingual)."""

    print()
    print("="*80)
    print("PHASE 4 TEST: Alternative Embedding Model (BGE-M3)")
    print("="*80)
    print()

    pdf_path = Path("data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf")

    if not pdf_path.exists():
        print("Test document not found")
        return None

    output_dir = Path("output/phase4_bge_test")

    # Configure with BGE-M3 (open-source, multilingual)
    config = IndexingConfig(
        extraction_config=ExtractionConfig(
            enable_smart_hierarchy=True,
            ocr_language=["ces", "eng"]
        ),
        summarization_config=SummarizationConfig(
            enabled=False  # Skip summaries to avoid API dependency
        ),
        chunking_config=ChunkingConfig(
            chunk_size=500,
            enable_contextual=False  # SAC requires summaries
        ),
        embedding_config=EmbeddingConfig(
            provider="huggingface",
            model="bge-m3"  # Open-source alternative
        )
    )

    try:
        print("Note: Using BGE-M3 (open-source, multilingual)")
        print("This will download the model on first run (~2GB)")
        print()

        pipeline = IndexingPipeline(config)

        vector_store = pipeline.index_document(
            pdf_path=pdf_path,
            save_intermediate=True,
            output_dir=output_dir
        )

        stats = vector_store.get_stats()
        print()
        print(f"✓ Indexed with BGE-M3: {stats['total_vectors']} vectors")
        print()

        return vector_store

    except ImportError as e:
        print(f"⚠️  BGE-M3 requires sentence-transformers:")
        print("   Install with: uv pip install sentence-transformers")
        print()
        return None

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test PHASE 4: Embedding & FAISS Indexing")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "bge"],
        default="single",
        help="Test mode: single document, batch, or BGE-M3 model"
    )

    args = parser.parse_args()

    if args.mode == "single":
        test_single_document()
    elif args.mode == "batch":
        test_batch_documents()
    elif args.mode == "bge":
        test_alternative_embedding_model()


if __name__ == "__main__":
    main()
