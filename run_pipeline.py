#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner - PHASE 1-4

Runs the complete indexing pipeline with outputs saved after each phase.

Usage:
    python run_pipeline.py <pdf_path>
    python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"

Outputs are saved to: output/<document_name>/
- phase1_extraction.json - Document structure & hierarchy
- phase2_summaries.json - Generic summaries
- phase3_chunks.json - Multi-layer chunks with SAC
- phase4_vector_store/ - FAISS indexes

Model Configuration (from .env):
- LLM: {LLM_PROVIDER}/{LLM_MODEL} (for summaries)
- Embedding: {EMBEDDING_PROVIDER}/{EMBEDDING_MODEL} (for vectors)
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from extraction import (
    DoclingExtractorV2,
    ExtractionConfig,
    MultiLayerChunker,
    EmbeddingGenerator,
    EmbeddingConfig,
    FAISSVectorStore
)
from extraction.config import get_default_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def run_complete_pipeline(pdf_path: Path, output_base: Path = None):
    """
    Run complete PHASE 1-4 pipeline with outputs after each phase.

    Args:
        pdf_path: Path to PDF document
        output_base: Base output directory (default: output/)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"✗ Error: File not found: {pdf_path}")
        sys.exit(1)

    # Setup output directory
    if output_base is None:
        output_base = Path(__file__).parent / "output"

    doc_name = pdf_path.stem.replace(" ", "_")
    output_dir = output_base / doc_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"RAG PIPELINE - {pdf_path.name}")

    # Load model config
    config = get_default_config()
    print_info(f"LLM: {config.llm_provider}/{config.llm_model}")
    print_info(f"Embedding: {config.embedding_provider}/{config.embedding_model}")
    print_info(f"Output directory: {output_dir}")
    print()

    # =================================================================
    # PHASE 1: Smart Hierarchy Extraction
    # =================================================================
    print_header("PHASE 1: Smart Hierarchy Extraction")

    extraction_config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=False,  # Will do in PHASE 2
        extract_tables=True
    )

    extractor = DoclingExtractorV2(extraction_config)
    result = extractor.extract(pdf_path)

    print_success("Document extracted")
    print_info(f"Sections: {result.num_sections}")
    print_info(f"Hierarchy depth: {result.hierarchy_depth}")
    print_info(f"Root sections: {result.num_roots}")
    print_info(f"Tables: {result.num_tables}")

    # Save PHASE 1 output
    phase1_output = output_dir / "phase1_extraction.json"
    with open(phase1_output, 'w', encoding='utf-8') as f:
        json.dump({
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
                    "content_length": len(s.content)
                }
                for s in result.sections
            ]
        }, f, indent=2, ensure_ascii=False)

    print_success(f"PHASE 1 output saved: {phase1_output}")

    # =================================================================
    # PHASE 2: Generic Summary Generation
    # =================================================================
    print_header("PHASE 2: Generic Summary Generation")

    try:
        from extraction import SummaryGenerator

        llm_config = config.get_llm_config()
        generator = SummaryGenerator(
            model=llm_config["model"],
            max_chars=150,
            api_key=llm_config["api_key"]
        )

        # Generate document summary
        print_info("Generating document summary...")
        result.document_summary = generator.generate_document_summary(result.full_text)

        # Generate section summaries (in parallel for speed)
        print_info(f"Generating summaries for {result.num_sections} sections...")
        section_texts = [(s.content, s.title) for s in result.sections]
        summaries = generator.generate_batch_summaries(section_texts)

        # Assign summaries back to sections
        for section, summary in zip(result.sections, summaries):
            section.summary = summary

        sections_with_summaries = sum(1 for s in result.sections if s.summary)

        print_success("Summaries generated")
        print_info(f"Document summary: {result.document_summary[:80]}...")
        print_info(f"Sections with summaries: {sections_with_summaries}/{result.num_sections}")

        # Save PHASE 2 output
        phase2_output = output_dir / "phase2_summaries.json"
        with open(phase2_output, 'w', encoding='utf-8') as f:
            json.dump({
                "document_id": result.document_id,
                "document_summary": result.document_summary,
                "section_summaries": [
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "summary": s.summary
                    }
                    for s in result.sections
                ]
            }, f, indent=2, ensure_ascii=False)

        print_success(f"PHASE 2 output saved: {phase2_output}")

        summaries_generated = True

    except Exception as e:
        print(f"⚠️  PHASE 2 skipped: {e}")
        print_info("Set appropriate API key in .env to enable summary generation")
        summaries_generated = False

    # =================================================================
    # PHASE 3: Multi-Layer Chunking + SAC
    # =================================================================
    print_header("PHASE 3: Multi-Layer Chunking + SAC")

    chunker = MultiLayerChunker(
        chunk_size=500,
        chunk_overlap=0,
        enable_sac=summaries_generated
    )

    chunks = chunker.chunk_document(result)
    stats = chunker.get_chunking_stats(chunks)

    print_success("Multi-layer chunking completed")
    print_info(f"Layer 1 (Document): {stats['layer1_count']} chunks")
    print_info(f"Layer 2 (Section):  {stats['layer2_count']} chunks")
    print_info(f"Layer 3 (Chunk):    {stats['layer3_count']} chunks (PRIMARY)")
    print_info(f"Total chunks:       {stats['total_chunks']}")

    if 'layer3_avg_size' in stats:
        print()
        print_info(f"Layer 3 avg size: {stats['layer3_avg_size']:.0f} chars")
        print_info(f"SAC overhead: {stats['sac_avg_overhead']:.0f} chars/chunk")

    # Save PHASE 3 output
    phase3_output = output_dir / "phase3_chunks.json"
    with open(phase3_output, 'w', encoding='utf-8') as f:
        json.dump({
            "document_id": result.document_id,
            "chunking_stats": stats,
            "layer1": [c.to_dict() for c in chunks["layer1"]],
            "layer2": [c.to_dict() for c in chunks["layer2"]],
            "layer3": [c.to_dict() for c in chunks["layer3"]]
        }, f, indent=2, ensure_ascii=False)

    print_success(f"PHASE 3 output saved: {phase3_output}")

    # =================================================================
    # PHASE 4: Embedding + FAISS Indexing
    # =================================================================
    print_header("PHASE 4: Embedding + FAISS Indexing")

    try:
        embedding_cfg = config.get_embedding_config()

        embedder = EmbeddingGenerator(
            EmbeddingConfig(
                model=embedding_cfg["model"],
                batch_size=100,
                normalize=True
            )
        )

        print_info(f"Generating embeddings with {embedding_cfg['model']}...")

        # Generate embeddings for all layers
        embeddings = {
            "layer1": embedder.embed_chunks(chunks["layer1"], layer=1),
            "layer2": embedder.embed_chunks(chunks["layer2"], layer=2),
            "layer3": embedder.embed_chunks(chunks["layer3"], layer=3)
        }

        print_success("Embeddings generated")
        print_info(f"Dimensions: {embedder.dimensions}D")
        print_info(f"Layer 1: {embeddings['layer1'].shape}")
        print_info(f"Layer 2: {embeddings['layer2'].shape}")
        print_info(f"Layer 3: {embeddings['layer3'].shape}")

        # Create vector store
        print_info("Creating FAISS vector store...")
        vector_store = FAISSVectorStore(dimensions=embedder.dimensions)
        vector_store.add_chunks(chunks, embeddings)

        store_stats = vector_store.get_stats()

        print_success("FAISS index created")
        print_info(f"Total vectors: {store_stats['total_vectors']}")
        print_info(f"Documents: {store_stats['documents']}")

        # Save PHASE 4 output
        phase4_output = output_dir / "phase4_vector_store"
        vector_store.save(phase4_output)

        print_success(f"PHASE 4 output saved: {phase4_output}")

    except Exception as e:
        print(f"⚠️  PHASE 4 failed: {e}")
        print_info("Check API keys in .env file")
        print_info("For Kanon 2: VOYAGE_API_KEY")
        print_info("For OpenAI: OPENAI_API_KEY")

    # =================================================================
    # SUMMARY
    # =================================================================
    print_header("PIPELINE COMPLETE")

    print_success(f"All outputs saved to: {output_dir}")
    print()
    print_info("Output files:")
    print_info(f"  - phase1_extraction.json   (Document structure)")
    print_info(f"  - phase2_summaries.json    (Generic summaries)")
    print_info(f"  - phase3_chunks.json       (Multi-layer chunks)")
    print_info(f"  - phase4_vector_store/     (FAISS indexes)")
    print()
    print_info("Next steps:")
    print_info("  - PHASE 5: Query & Retrieval API")
    print_info("  - PHASE 6: Context Assembly")
    print_info("  - PHASE 7: Answer Generation")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pdf_path>")
        print()
        print("Example:")
        print('  python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"')
        print()
        print("Model Configuration (edit .env file):")

        # Show current config
        try:
            config = get_default_config()
            print(f"  LLM: {config.llm_provider}/{config.llm_model}")
            print(f"  Embedding: {config.embedding_provider}/{config.embedding_model}")
        except:
            print("  (Configure models in .env file)")

        sys.exit(1)

    pdf_path = sys.argv[1]
    run_complete_pipeline(pdf_path)


if __name__ == "__main__":
    main()
