#!/usr/bin/env python3
"""
Complete RAG Pipeline Test: PHASE 1 + PHASE 2 + PHASE 3

PHASE 1: Smart Hierarchy Extraction (font-size based)
PHASE 2: Generic Summary Generation (gpt-4o-mini, 150 chars)
PHASE 3: Multi-Layer Chunking + SAC (500 chars, prepend summaries)

Based on PIPELINE.md research findings.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from extraction import (
    DoclingExtractorV2,
    ExtractionConfig,
    MultiLayerChunker
)


def test_phase1_only(pdf_path: Path, output_dir: Path):
    """Test PHASE 1: Smart hierarchy extraction."""

    print("="*80)
    print("PHASE 1: Smart Hierarchy Extraction")
    print("="*80)
    print()

    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=False,
        extract_tables=True
    )

    extractor = DoclingExtractorV2(config)
    result = extractor.extract(pdf_path)

    print(f"✓ Document extracted")
    print(f"  Sections: {result.num_sections}")
    print(f"  Hierarchy depth: {result.hierarchy_depth}")
    print(f"  Root sections: {result.num_roots}")
    print(f"  Tables: {result.num_tables}")

    return result


def test_phase1_phase2(pdf_path: Path, output_dir: Path):
    """Test PHASE 1 + PHASE 2: Smart hierarchy + Summaries."""

    print()
    print("="*80)
    print("PHASE 1 + 2: Smart Hierarchy + Generic Summaries")
    print("="*80)
    print()

    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=True,
        summary_model="gpt-4o-mini",
        summary_max_chars=150,
        extract_tables=True
    )

    try:
        extractor = DoclingExtractorV2(config)
        result = extractor.extract(pdf_path)

        print(f"✓ Document extracted with summaries")
        print(f"  Sections: {result.num_sections}")
        print(f"  Document summary: {result.document_summary[:80] if result.document_summary else 'N/A'}...")

        # Count sections with summaries
        sections_with_summaries = sum(1 for s in result.sections if s.summary)
        print(f"  Sections with summaries: {sections_with_summaries}/{result.num_sections}")

        return result

    except ValueError as e:
        print(f"⚠️  PHASE 2 skipped: {e}")
        print("   Set OPENAI_API_KEY to enable summary generation")
        return None


def test_phase3_chunking(result, output_dir: Path):
    """Test PHASE 3: Multi-layer chunking with SAC."""

    if not result:
        print("\n⚠️  Skipping PHASE 3: No extracted document available")
        return None

    print()
    print("="*80)
    print("PHASE 3: Multi-Layer Chunking + SAC")
    print("="*80)
    print()

    # Create chunker
    chunker = MultiLayerChunker(
        chunk_size=500,
        chunk_overlap=0,
        enable_sac=True
    )

    # Chunk document
    chunks_dict = chunker.chunk_document(result)
    stats = chunker.get_chunking_stats(chunks_dict)

    print(f"✓ Multi-layer chunking completed")
    print(f"  Layer 1 (Document): {stats['layer1_count']} chunks")
    print(f"  Layer 2 (Section):  {stats['layer2_count']} chunks")
    print(f"  Layer 3 (Chunk):    {stats['layer3_count']} chunks (PRIMARY)")
    print(f"  Total chunks:       {stats['total_chunks']}")
    print()

    if 'layer3_avg_size' in stats:
        print(f"Layer 3 statistics:")
        print(f"  Average chunk size: {stats['layer3_avg_size']:.0f} chars")
        print(f"  Min chunk size:     {stats['layer3_min_size']} chars")
        print(f"  Max chunk size:     {stats['layer3_max_size']} chars")
        print(f"  SAC avg overhead:   {stats['sac_avg_overhead']:.0f} chars")

    # Save chunks to JSON
    output_path = output_dir / f"{result.document_id}_phase3_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert chunks to dict
    chunks_export = {
        "metadata": {
            "document_id": result.document_id,
            "source_path": result.source_path,
            "chunking_stats": stats
        },
        "layer1": [c.to_dict() for c in chunks_dict["layer1"]],
        "layer2": [c.to_dict() for c in chunks_dict["layer2"]],
        "layer3": [c.to_dict() for c in chunks_dict["layer3"]]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_export, f, indent=2, ensure_ascii=False)

    print()
    print(f"✓ Chunks saved to: {output_path}")

    return chunks_dict


def demonstrate_sac(chunks_dict):
    """Demonstrate how SAC works."""

    print()
    print("="*80)
    print("SAC (Summary-Augmented Chunking) DEMONSTRATION")
    print("="*80)
    print()

    if not chunks_dict or not chunks_dict["layer3"]:
        print("No Layer 3 chunks available")
        return

    # Show first chunk
    chunk = chunks_dict["layer3"][0]

    print("Example Layer 3 chunk:")
    print()
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Section: {chunk.metadata.section_title}")
    print()

    print("RAW CONTENT (for generation):")
    print("-" * 80)
    print(chunk.raw_content[:200] + "...")
    print()

    print("AUGMENTED CONTENT (for embedding, with SAC):")
    print("-" * 80)
    print(chunk.content[:200] + "...")
    print()

    print("SAC Summary Prepended:")
    print("-" * 80)
    sac_summary = chunk.content[:len(chunk.content) - len(chunk.raw_content)]
    print(sac_summary)
    print()

    print("Purpose:")
    print("  - Embedding uses AUGMENTED content (with document summary)")
    print("  - Generation uses RAW content (without summary)")
    print("  - This reduces Document-level Retrieval Mismatch (DRM) by 58%!")


def show_layer_hierarchy(chunks_dict):
    """Show hierarchical relationships between layers."""

    print()
    print("="*80)
    print("LAYER HIERARCHY")
    print("="*80)
    print()

    if not chunks_dict:
        return

    # Layer 1
    if chunks_dict["layer1"]:
        l1 = chunks_dict["layer1"][0]
        print(f"[Layer 1] Document: {l1.chunk_id}")
        print(f"  Content: {l1.content[:100]}...")
        print()

    # Layer 2 (first 3)
    if chunks_dict["layer2"]:
        print("[Layer 2] Sections:")
        for l2 in chunks_dict["layer2"][:3]:
            print(f"  ├─ {l2.chunk_id}")
            print(f"  │  Section: {l2.metadata.section_title}")
            print(f"  │  Parent: {l2.metadata.parent_chunk_id}")
            print()

    # Layer 3 (first 3)
    if chunks_dict["layer3"]:
        print("[Layer 3] Chunks (first 3):")
        for l3 in chunks_dict["layer3"][:3]:
            print(f"  ├─ {l3.chunk_id}")
            print(f"  │  Section: {l3.metadata.section_title}")
            print(f"  │  Parent: {l3.metadata.parent_chunk_id}")
            print(f"  │  Size: {len(l3.raw_content)} chars")
            print()


def main():
    # Test document
    pdf_path = Path("data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        print("Please add a test PDF")
        sys.exit(1)

    output_dir = Path("output/complete_pipeline_test")

    print("="*80)
    print("COMPLETE RAG PIPELINE TEST")
    print("="*80)
    print()
    print(f"Document: {pdf_path.name}")
    print(f"Output:   {output_dir}")
    print()

    # Run all phases
    result_p1 = test_phase1_only(pdf_path, output_dir)
    result_p2 = test_phase1_phase2(pdf_path, output_dir)
    chunks = test_phase3_chunking(result_p2 or result_p1, output_dir)

    # Demonstrations
    if chunks:
        demonstrate_sac(chunks)
        show_layer_hierarchy(chunks)

    print()
    print("="*80)
    print("✓ COMPLETE PIPELINE TEST FINISHED")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review chunked output in JSON")
    print("  2. PHASE 4: Embedding & Indexing (3 FAISS indexes)")
    print("  3. PHASE 5: Query & Retrieval (K=6, DRM prevention)")
    print("  4. PHASE 6: Context Assembly")
    print("  5. PHASE 7: Answer Generation")
    print()
    print("Current status:")
    print("  ✓ PHASE 1: Smart Hierarchy")
    print("  ✓ PHASE 2: Generic Summaries")
    print("  ✓ PHASE 3: Multi-Layer Chunking + SAC")
    print("  ⏳ PHASE 4-7: To be implemented")


if __name__ == "__main__":
    main()
