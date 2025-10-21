"""
IBM Docling-based document extraction with intelligent hierarchy detection.

This module provides high-precision extraction with:
- Font-size based hierarchy classification
- HierarchicalChunker for proper parent-child relationships
- Generic summary generation (PHASE 2 of pipeline)
- 97.9% table extraction accuracy
- 100% text fidelity
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    OcrMacOptions,
    LayoutOptions
)
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_XLARGE,
)
from docling_core.types.doc import DoclingDocument, SectionHeaderItem
from docling_core.transforms.chunker import HierarchicalChunker

# Import summary generator (PHASE 2)
try:
    from .summary_generator import SummaryGenerator
except ImportError:
    from summary_generator import SummaryGenerator

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for Docling extraction."""

    # OCR settings
    enable_ocr: bool = True
    ocr_language: List[str] = field(default_factory=lambda: ["cs-CZ", "en-US"])
    ocr_recognition: str = "accurate"  # "accurate" or "fast"

    # Table extraction
    table_mode: TableFormerMode = TableFormerMode.ACCURATE
    extract_tables: bool = True

    # Hierarchy extraction
    extract_hierarchy: bool = True
    enable_smart_hierarchy: bool = True  # Font-size based classification
    hierarchy_tolerance: float = 0.8  # BBox height clustering tolerance (lower = stricter)

    # Summary generation (PHASE 2)
    generate_summaries: bool = False  # Enable in PHASE 2
    summary_model: str = "gpt-4o-mini"
    summary_max_chars: int = 150
    summary_style: str = "generic"  # "generic" or "expert"

    # Output formats
    generate_markdown: bool = True
    generate_json: bool = True

    # Performance
    layout_model: str = "EGRET_XLARGE"  # Options: HERON, EGRET_LARGE, EGRET_XLARGE (recommended)


@dataclass
class TableData:
    """Extracted table with metadata."""
    table_id: str
    caption: Optional[str]
    num_rows: int
    num_cols: int
    data: List[List[str]]
    bbox: Optional[Dict[str, float]]
    page_number: int

    def to_dict(self) -> Dict:
        return {
            "table_id": self.table_id,
            "caption": self.caption,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "data": self.data,
            "bbox": self.bbox,
            "page_number": self.page_number
        }


@dataclass
class DocumentSection:
    """Hierarchical document section from HierarchicalChunker."""
    section_id: str
    title: str
    content: str
    level: int  # Font-size based level (1=largest, 2=second largest, etc.)
    depth: int  # Depth in hierarchy tree (1=root, 2=child of root, etc.)
    parent_id: Optional[str]
    children_ids: List[str]
    ancestors: List[str]  # List of parent titles
    path: str  # Full path: "Chapter 1 > Section 1.1 > Subsection 1.1.1"
    page_number: int
    char_start: int
    char_end: int
    content_length: int

    # PHASE 2: Summaries
    summary: Optional[str] = None  # 150-char generic summary

    def to_dict(self) -> Dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "ancestors": self.ancestors,
            "path": self.path,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_length": self.content_length,
            "summary": self.summary
        }


@dataclass
class ExtractedDocument:
    """Complete extracted document with hierarchical structure."""

    # Basic info
    document_id: str
    source_path: str
    extraction_time: float  # seconds

    # Content
    full_text: str
    markdown: str
    json_content: Dict

    # Hierarchical structure (PHASE 1)
    sections: List[DocumentSection]
    hierarchy_depth: int
    num_roots: int

    # Tables
    tables: List[TableData]

    # Metadata
    num_pages: int
    num_sections: int
    num_tables: int
    total_chars: int

    # PHASE 2: Document-level summary
    document_summary: Optional[str] = None

    # Extraction metadata
    extraction_method: str = "smart_hierarchy_by_font_size"
    config: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "metadata": {
                "document_id": self.document_id,
                "source_path": self.source_path,
                "extraction_time_seconds": self.extraction_time,
                "extraction_method": self.extraction_method,
                "num_pages": self.num_pages,
                "num_sections": self.num_sections,
                "num_tables": self.num_tables,
                "total_chars": self.total_chars,
                "hierarchy_depth": self.hierarchy_depth,
                "num_roots": self.num_roots,
                "document_summary": self.document_summary,
                "config": self.config
            },
            "hierarchy": {
                "roots": [s.to_dict() for s in self.sections if s.parent_id is None],
                "all_sections": [s.to_dict() for s in self.sections],
                "total_sections": len(self.sections),
                "max_depth": self.hierarchy_depth
            },
            "tables": [t.to_dict() for t in self.tables],
            "full_text": self.full_text,
            "markdown": self.markdown
        }


class DoclingExtractorV2:
    """
    Enhanced Docling extractor with intelligent hierarchy detection.

    Features:
    - Font-size based hierarchy classification (PHASE 1)
    - HierarchicalChunker for proper parent-child relationships
    - Generic summary generation (PHASE 2, optional)
    - 97.9% table extraction accuracy
    - Multi-language support (Czech, English)

    Example:
        >>> config = ExtractionConfig(enable_smart_hierarchy=True)
        >>> extractor = DoclingExtractorV2(config)
        >>> result = extractor.extract("nuclear_doc.pdf")
        >>> print(f"Hierarchy depth: {result.hierarchy_depth}")
        >>> for section in result.sections:
        ...     print(f"  {'  ' * section.depth}{section.title}")
    """

    def __init__(self, config: Optional[ExtractionConfig] = None, openai_api_key: Optional[str] = None):
        """Initialize the enhanced extractor."""
        self.config = config or ExtractionConfig()
        self.converter = self._setup_converter()

        # Initialize summary generator (PHASE 2)
        if self.config.generate_summaries:
            try:
                self.summary_generator = SummaryGenerator(
                    model=self.config.summary_model,
                    max_chars=self.config.summary_max_chars,
                    api_key=openai_api_key
                )
                logger.info("Summary generator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize summary generator: {e}")
                self.summary_generator = None
        else:
            self.summary_generator = None

        logger.info(f"DoclingExtractorV2 initialized with smart_hierarchy={self.config.enable_smart_hierarchy}")

    def _setup_converter(self) -> DocumentConverter:
        """Setup document converter with optimal settings."""

        # Configure OCR
        ocr_options = OcrMacOptions(
            recognition=self.config.ocr_recognition,
            lang=self.config.ocr_language
        )

        # Configure layout model based on config
        layout_model_map = {
            "HERON": DOCLING_LAYOUT_HERON,
            "EGRET_LARGE": DOCLING_LAYOUT_EGRET_LARGE,
            "EGRET_XLARGE": DOCLING_LAYOUT_EGRET_XLARGE,
        }
        layout_model = layout_model_map.get(
            self.config.layout_model,
            DOCLING_LAYOUT_EGRET_XLARGE  # Default to EGRET XLarge (best for hierarchy)
        )
        logger.info(f"Using layout model: {self.config.layout_model}")

        # Configure pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = self.config.extract_tables
        pipeline_options.table_structure_options.mode = self.config.table_mode
        pipeline_options.ocr_options = ocr_options
        pipeline_options.layout_options = LayoutOptions(model_spec=layout_model)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return converter

    def extract(
        self,
        source: Union[str, Path],
        document_id: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Extract complete hierarchical structure from document.

        Args:
            source: Path to the document file
            document_id: Optional document identifier

        Returns:
            ExtractedDocument with intelligent hierarchy and optional summaries
        """
        start_time = datetime.now()
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        doc_id = document_id or source_path.stem
        logger.info(f"Starting extraction of {source_path.name}")

        # Convert document
        result = self.converter.convert(str(source_path))
        docling_doc: DoclingDocument = result.document

        # Extract content
        full_text = docling_doc.export_to_text()
        markdown = docling_doc.export_to_markdown() if self.config.generate_markdown else ""
        json_content = docling_doc.export_to_dict() if self.config.generate_json else {}

        # PHASE 1: Extract hierarchical structure
        sections = self._extract_hierarchical_sections(docling_doc)

        # PHASE 2: Generate summaries (if enabled)
        if self.config.generate_summaries:
            sections = self._generate_section_summaries(sections)
            document_summary = self._generate_document_summary(full_text)
        else:
            document_summary = None

        # Extract tables
        tables = self._extract_tables(docling_doc)

        # Calculate metrics
        num_pages = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 1
        hierarchy_depth = max((s.depth for s in sections), default=0)
        num_roots = sum(1 for s in sections if s.parent_id is None)

        extraction_time = (datetime.now() - start_time).total_seconds()

        extracted_doc = ExtractedDocument(
            document_id=doc_id,
            source_path=str(source_path),
            extraction_time=extraction_time,
            full_text=full_text,
            markdown=markdown,
            json_content=json_content,
            sections=sections,
            hierarchy_depth=hierarchy_depth,
            num_roots=num_roots,
            tables=tables,
            num_pages=num_pages,
            num_sections=len(sections),
            num_tables=len(tables),
            total_chars=len(full_text),
            document_summary=document_summary,
            extraction_method="smart_hierarchy_by_font_size" if self.config.enable_smart_hierarchy else "basic",
            config={
                "enable_smart_hierarchy": self.config.enable_smart_hierarchy,
                "generate_summaries": self.config.generate_summaries,
                "ocr_language": self.config.ocr_language
            }
        )

        logger.info(
            f"Extraction completed in {extraction_time:.2f}s: "
            f"{len(sections)} sections (depth={hierarchy_depth}), "
            f"{len(tables)} tables"
        )

        return extracted_doc

    def _extract_hierarchical_sections(self, docling_doc: DoclingDocument) -> List[DocumentSection]:
        """
        Extract hierarchical sections using HierarchicalChunker.

        If enable_smart_hierarchy=True, reclassifies levels based on font size.
        """

        # Step 1: Extract headers with bbox info (for font-size classification)
        headers = []
        for item, level in docling_doc.iterate_items():
            if isinstance(item, SectionHeaderItem):
                header_info = {
                    'text': item.text,
                    'level': item.level,
                    'bbox': item.prov[0].bbox.as_tuple() if item.prov and item.prov[0].bbox else None,
                    'item': item  # Store reference
                }
                headers.append(header_info)

        logger.info(f"Found {len(headers)} headers")

        # Step 2: Smart hierarchy classification (if enabled)
        if self.config.enable_smart_hierarchy and headers:
            headers = self._classify_header_levels_by_font_size(headers)

            # Update SectionHeaderItem.level in-place
            for h in headers:
                if 'item' in h:
                    h['item'].level = h['level']

            logger.info("Applied smart hierarchy classification")

        # Step 3: Use HierarchicalChunker to build sections
        chunker = HierarchicalChunker()
        chunks = list(chunker.chunk(docling_doc))

        logger.info(f"HierarchicalChunker produced {len(chunks)} chunks")

        # Step 4: Convert chunks to DocumentSection objects
        sections = []
        section_id = 0
        char_position = 0
        heading_to_section = {}

        for chunk in chunks:
            section_id += 1

            headings = chunk.meta.headings if chunk.meta.headings else []
            depth = len(headings) if headings else 0
            title = headings[-1] if headings else "Untitled"

            # Level is depth (HierarchicalChunker preserves our smart levels)
            level = depth

            # Determine parent
            parent_id = None
            if len(headings) > 1:
                parent_heading_path = tuple(headings[:-1])
                parent_id = heading_to_section.get(parent_heading_path)

            content = chunk.text

            section = DocumentSection(
                section_id=f"sec_{section_id}",
                title=title,
                content=content,
                level=level,
                depth=depth,
                parent_id=parent_id,
                children_ids=[],
                ancestors=list(headings[:-1]) if len(headings) > 1 else [],
                path=" > ".join(headings) if headings else title,
                page_number=chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items and chunk.meta.doc_items[0].prov else 0,
                char_start=char_position,
                char_end=char_position + len(content),
                content_length=len(content)
            )

            sections.append(section)

            if headings:
                heading_to_section[tuple(headings)] = section.section_id

            char_position += len(content)

        # Build children lists
        for section in sections:
            if section.parent_id:
                for parent in sections:
                    if parent.section_id == section.parent_id:
                        parent.children_ids.append(section.section_id)
                        break

        return sections

    def _classify_header_levels_by_font_size(self, headers: List[Dict]) -> List[Dict]:
        """
        Classify header levels based on bbox height (font size) with improved clustering.

        Enhanced Strategy:
        - Extract font heights from bbox
        - Use stricter tolerance for clustering (configurable)
        - Consider text length (shorter = likely header)
        - Sort by height (descending) and assign levels
        - Merge very similar clusters
        """

        if not headers:
            return headers

        # Extract heights and text info
        heights_with_index = []
        for i, h in enumerate(headers):
            if h['bbox']:
                height = abs(h['bbox'][3] - h['bbox'][1])
                text_len = len(h.get('text', ''))
                heights_with_index.append((i, height, text_len))

        if not heights_with_index:
            return headers

        # Improved clustering: group by height similarity
        clusters = []
        sorted_heights = sorted(heights_with_index, key=lambda x: x[1], reverse=True)

        for idx, height, text_len in sorted_heights:
            placed = False

            # Try to place in existing cluster
            for cluster in clusters:
                # Calculate cluster statistics
                cluster_heights = [c[1] for c in cluster]
                avg_height = sum(cluster_heights) / len(cluster_heights)
                max_height = max(cluster_heights)
                min_height = min(cluster_heights)

                # Stricter clustering: height must be within tolerance of cluster range
                height_diff = abs(height - avg_height)

                # Use absolute tolerance (pixels) for consistent clustering
                if height_diff <= self.config.hierarchy_tolerance:
                    cluster.append((idx, height, text_len))
                    placed = True
                    break

            if not placed:
                clusters.append([(idx, height, text_len)])

        # Sort clusters by average height (descending)
        clusters.sort(key=lambda c: sum(h for _, h, _ in c) / len(c), reverse=True)

        # Merge clusters that are too similar (within 5% of each other)
        merged_clusters = []
        for cluster in clusters:
            avg_height = sum(h for _, h, _ in cluster) / len(cluster)

            # Try to merge with existing cluster
            merged = False
            for existing_cluster in merged_clusters:
                existing_avg = sum(h for _, h, _ in existing_cluster) / len(existing_cluster)
                relative_diff = abs(avg_height - existing_avg) / existing_avg

                if relative_diff < 0.05:  # Within 5% - merge
                    existing_cluster.extend(cluster)
                    merged = True
                    break

            if not merged:
                merged_clusters.append(cluster)

        # Re-sort after merging
        merged_clusters.sort(key=lambda c: sum(h for _, h, _ in c) / len(c), reverse=True)

        # Limit to max 6 levels (reasonable for most documents)
        max_levels = min(6, len(merged_clusters))
        merged_clusters = merged_clusters[:max_levels]

        # Assign levels
        index_to_level = {}
        for level, cluster in enumerate(merged_clusters, start=1):
            for idx, _, _ in cluster:
                index_to_level[idx] = level

        # Update headers
        updated_headers = []
        for i, h in enumerate(headers):
            h_copy = h.copy()
            if i in index_to_level:
                h_copy['original_level'] = h['level']
                h_copy['level'] = index_to_level[i]
            else:
                # Headers without bbox get level based on position
                h_copy['level'] = max_levels
            updated_headers.append(h_copy)

        # Log reclassification
        level_counts = {}
        for h in updated_headers:
            level = h['level']
            level_counts[level] = level_counts.get(level, 0) + 1

        logger.info(f"Reclassified headers: {level_counts}")

        return updated_headers

    def _extract_tables(self, docling_doc: DoclingDocument) -> List[TableData]:
        """Extract tables with high precision."""
        tables = []
        table_counter = 0

        for item, level in docling_doc.iterate_items():
            if item.__class__.__name__ == 'TableItem':
                table_counter += 1

                try:
                    df = item.export_to_dataframe()

                    if df is not None:
                        data = df.values.tolist()

                        table = TableData(
                            table_id=f"table_{table_counter}",
                            caption=getattr(item, 'caption', None),
                            num_rows=len(data),
                            num_cols=len(data[0]) if data else 0,
                            data=data,
                            bbox=item.prov[0].bbox.as_tuple() if item.prov and item.prov[0].bbox else None,
                            page_number=item.prov[0].page_no if item.prov else 0
                        )

                        tables.append(table)
                except Exception as e:
                    logger.warning(f"Failed to extract table {table_counter}: {e}")
                    continue

        return tables

    def _generate_section_summaries(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """
        PHASE 2: Generate generic summaries for each section.

        Uses gpt-4o-mini to generate 150-char generic summaries.
        """

        if not self.summary_generator:
            logger.warning("Summary generator not initialized")
            return sections

        logger.info(f"Generating summaries for {len(sections)} sections...")

        # Generate summaries in batch for efficiency
        texts_and_titles = [(s.content, s.title) for s in sections]

        try:
            summaries = self.summary_generator.generate_batch_summaries(texts_and_titles)

            # Assign summaries to sections
            for section, summary in zip(sections, summaries):
                section.summary = summary

            logger.info("Section summaries generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate section summaries: {e}")

        return sections

    def _generate_document_summary(self, full_text: str) -> str:
        """
        PHASE 2: Generate document-level generic summary.

        Uses gpt-4o-mini to generate 150-char generic summary.
        """

        if not self.summary_generator:
            logger.warning("Summary generator not initialized")
            return ""

        logger.info("Generating document-level summary...")

        try:
            summary = self.summary_generator.generate_document_summary(full_text)
            logger.info(f"Document summary generated: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate document summary: {e}")
            return ""

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"]
