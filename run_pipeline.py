#!/usr/bin/env python3
"""
Complete RAG Pipeline Runner - PHASE 1-4

Runs the complete indexing pipeline with outputs saved after each phase.

Supported formats: PDF, DOCX, PPTX, XLSX, HTML
Input: Single document OR directory (batch processing)

Usage:
    # Single document
    python run_pipeline.py <document_path>
    python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"
    python run_pipeline.py "data/report.docx"

    # Batch processing (directory)
    python run_pipeline.py <directory_path>
    python run_pipeline.py "data/regulace/GRI"

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

from config import get_default_config, ExtractionConfig
from docling_extractor_v2 import DoclingExtractorV2
from multi_layer_chunker import MultiLayerChunker
from embedding_generator import EmbeddingGenerator, EmbeddingConfig
from faiss_vector_store import FAISSVectorStore
from summary_generator import SummaryGenerator

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


def run_complete_pipeline(input_path: Path, output_base: Path = None):
    """
    Run complete PHASE 1-4 pipeline with outputs after each phase.

    Supported formats: PDF, DOCX, PPTX, XLSX, HTML

    Args:
        input_path: Path to document file or directory (PDF, DOCX, PPTX, XLSX, HTML)
        output_base: Base output directory (default: output/)
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
            run_single_document(document_path, output_base)

        print_header("BATCH PROCESSING COMPLETE")
        print_success(f"Processed {len(documents)} documents")
        print_info(f"Outputs saved to: {output_base or Path('output')}")
        return

    # Single document processing
    run_single_document(input_path, output_base)


def run_single_document(document_path: Path, output_base: Path = None):
    """
    Process a single document through the complete pipeline.

    Args:
        document_path: Path to document file
        output_base: Base output directory (default: output/)
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

    doc_name = document_path.stem.replace(" ", "_")
    output_dir = output_base / doc_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header(f"RAG PIPELINE - {document_path.name}")

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
    result = extractor.extract(document_path)

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
        llm_config = config.get_llm_config()
        generator = SummaryGenerator(
            model=llm_config["model"],
            max_chars=150,
            api_key=llm_config["api_key"]
        )

        # Generate section summaries FIRST (in parallel for speed)
        print_info(f"Generating summaries for {result.num_sections} sections...")
        section_texts = [(s.content, s.title) for s in result.sections]
        summaries = generator.generate_batch_summaries(section_texts)

        # Assign summaries back to sections
        for section, summary in zip(result.sections, summaries):
            section.summary = summary

        # Generate document summary from section summaries (hierarchical)
        print_info("Generating document summary (hierarchical)...")
        valid_section_summaries = [s.summary for s in result.sections if s.summary]
        result.document_summary = generator.generate_document_summary(
            section_summaries=valid_section_summaries,
            document_text=result.full_text  # Fallback if no section summaries
        )

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
        print("Usage: python run_pipeline.py <path>")
        print()
        print("Supported formats: PDF, DOCX, PPTX, XLSX, HTML")
        print("Input: Single document file OR directory containing documents")
        print()
        print("Examples:")
        print("  Single document:")
        print('    python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"')
        print('    python run_pipeline.py "data/report.docx"')
        print()
        print("  Batch processing (directory):")
        print('    python run_pipeline.py "data/regulace/GRI"')
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

    input_path = sys.argv[1]
    run_complete_pipeline(input_path)


if __name__ == "__main__":
    main()
