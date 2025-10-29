"""
Phase Loaders for Resume Functionality

Loads completed phases from JSON files back into Python objects.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class PhaseLoaders:
    """
    Loads phase outputs from JSON files for resume functionality.

    Reconstructs Python objects (ExtractedDocument, Chunks) from saved JSON.
    """

    @staticmethod
    def load_phase1(json_path: Path):
        """
        Load PHASE 1 extraction result from JSON.

        Args:
            json_path: Path to phase1_extraction.json

        Returns:
            Partial ExtractedDocument (without full text/content)

        Note:
            Returns minimal object - enough for phase detection.
            Full content not needed since we skip to later phases.
        """
        from src.docling_extractor_v2 import ExtractedDocument, DocumentSection

        logger.info(f"Loading Phase 1 from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct sections (without full content - not saved)
        sections = [
            DocumentSection(
                section_id=s["section_id"],
                title=s["title"],
                level=s["level"],
                depth=s["depth"],
                path=s["path"],
                page_number=s["page_number"],
                content="",  # Not saved in phase1, will be loaded if needed
                parent_id=None,  # Not needed for resume
                children_ids=[],  # Not needed for resume
                ancestors=[],  # Not needed for resume
                char_start=0,  # Not saved in phase1
                char_end=s.get("content_length", 0),  # Use content_length as char_end
                content_length=s.get("content_length", 0),
                summary=None  # Loaded in phase2
            )
            for s in data["sections"]
        ]

        # Create partial ExtractedDocument
        result = ExtractedDocument(
            document_id=data["document_id"],
            source_path=data["source_path"],
            sections=sections,
            hierarchy_depth=data["hierarchy_depth"],
            num_roots=data["num_roots"],
            num_sections=data["num_sections"],
            num_tables=data.get("num_tables", 0),
            # Fields not saved/needed:
            extraction_time=0.0,
            full_text="",
            markdown="",
            json_content={},
            tables=[],
            num_pages=0,
            total_chars=0,
            document_summary=None  # Added in phase2
        )

        logger.info(f"Loaded Phase 1: {result.document_id}, {result.num_sections} sections")
        return result

    @staticmethod
    def load_phase2(json_path: Path, extraction_result):
        """
        Load PHASE 2 summaries and merge into ExtractedDocument.

        Args:
            json_path: Path to phase2_summaries.json
            extraction_result: ExtractedDocument from phase1

        Returns:
            ExtractedDocument with summaries added
        """
        logger.info(f"Loading Phase 2 from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add document summary
        extraction_result.document_summary = data["document_summary"]

        # Add section summaries
        section_summaries = {s["section_id"]: s["summary"] for s in data["section_summaries"]}

        for section in extraction_result.sections:
            section.summary = section_summaries.get(section.section_id)

        logger.info(
            f"Loaded Phase 2: document summary + {len(section_summaries)} section summaries"
        )

        return extraction_result

    @staticmethod
    def load_phase3(json_path: Path) -> Dict[str, List]:
        """
        Load PHASE 3 chunks from JSON.

        Args:
            json_path: Path to phase3_chunks.json

        Returns:
            Dict with keys 'layer1', 'layer2', 'layer3' containing Chunk objects
        """
        from src.multi_layer_chunker import Chunk, ChunkMetadata

        logger.info(f"Loading Phase 3 from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def reconstruct_chunk(chunk_data: Dict) -> Chunk:
            """Reconstruct Chunk object from dict."""
            meta = chunk_data["metadata"]

            return Chunk(
                chunk_id=chunk_data["chunk_id"],
                content=chunk_data["content"],
                raw_content=chunk_data["raw_content"],
                metadata=ChunkMetadata(
                    chunk_id=meta["chunk_id"],
                    layer=meta["layer"],
                    document_id=meta["document_id"],
                    title=meta.get("title"),
                    section_id=meta.get("section_id"),
                    parent_chunk_id=meta.get("parent_chunk_id"),
                    page_number=meta.get("page_number", 0),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    section_title=meta.get("section_title"),
                    section_path=meta.get("section_path"),
                    section_level=meta.get("section_level", 0),
                    section_depth=meta.get("section_depth", 0),
                )
            )

        # Reconstruct chunks for all 3 layers
        chunks = {
            "layer1": [reconstruct_chunk(c) for c in data["layer1"]],
            "layer2": [reconstruct_chunk(c) for c in data["layer2"]],
            "layer3": [reconstruct_chunk(c) for c in data["layer3"]],
        }

        logger.info(
            f"Loaded Phase 3: "
            f"L1={len(chunks['layer1'])}, "
            f"L2={len(chunks['layer2'])}, "
            f"L3={len(chunks['layer3'])} chunks"
        )

        return chunks
