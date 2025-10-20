"""
Hierarchical structure detection for legal documents.

Based on PIPELINE.md specifications and Lima 2024 (Multi-Layer Embeddings paper).
Detects different hierarchy types for legislation, contracts, NDAs, and ESG reports.
"""

import re
from typing import List, Tuple
from src.core.models import Section, DocumentStructure, DocumentType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StructureDetector:
    """
    Detects hierarchical structure in legal documents.

    Supports multiple document types with different hierarchies:
    - Legislation: Part → Book → Title → Chapter → Article → Paragraph
    - Contract: Recitals → Article → Section → Clause → Subclause
    - NDA: Preamble → Definitions → Obligations → Term → Miscellaneous
    - ESG Report: Sections based on reporting standards
    """

    # Hierarchy patterns for different document types
    LEGAL_HIERARCHY = {
        'legislation': [
            'Part', 'Book', 'Title', 'Chapter', 'Section_Group',
            'Article', 'Paragraph', 'Inciso', 'Alínea', 'Item'
        ],
        'contract': [
            'Recitals', 'Article', 'Section', 'Clause',
            'Subclause', 'Paragraph'
        ],
        'nda': [
            'Preamble', 'Definitions', 'Obligations', 'Exclusions',
            'Term', 'Miscellaneous'
        ],
        'esg_report': [
            'Section', 'Subsection', 'Topic', 'Indicator'
        ]
    }

    # Regex patterns for section detection
    SECTION_PATTERNS = [
        # GRI-specific patterns (must come first for priority)
        (r'^Disclosure\s+(\d+[-\s]\d+)\s+(.+)$', 'disclosure', 2),  # "Disclosure 102-1 Title"
        (r'^(\d+)\.\s+([A-Z][^\n]{15,100})$', 'gri_section', 0),  # "1. Topic management disclosures"

        # Article patterns
        (r'^Article\s+(\d+[.\s:]?\s*[—\-]?\s*.*?)$', 'article', 0),
        (r'^Art\.?\s+(\d+[.\s:]?\s*[—\-]?\s*.*?)$', 'article', 0),
        (r'^ARTICLE\s+([IVXLCDM]+)[.\s:]?\s*(.*?)$', 'article', 0),

        # Section patterns
        (r'^Section\s+(\d+\.?\s*[—\-]?\s*.*?)$', 'section', 1),
        (r'^§\s*(\d+\.?\s*[—\-]?\s*.*?)$', 'section', 1),
        (r'^SECTION\s+([IVXLCDM]+)[.\s:]?\s*(.*?)$', 'section', 1),

        # Numbered sections (1., 1.1, etc.)
        (r'^(\d+\.)\s+([A-Z][^\n]{10,100})$', 'numbered_section', 0),
        (r'^(\d+\.\d+)\s+([A-Z][^\n]{10,100})$', 'numbered_subsection', 1),

        # Chapter patterns
        (r'^Chapter\s+(\d+[.\s:]?\s*[—\-]?\s*.*?)$', 'chapter', 0),
        (r'^CHAPTER\s+([IVXLCDM]+)[.\s:]?\s*(.*?)$', 'chapter', 0),

        # Part patterns
        (r'^Part\s+([IVXLCDM]+)[.\s:]?\s*(.*?)$', 'part', 0),
        (r'^PART\s+(\d+)[.\s:]?\s*(.*?)$', 'part', 0),

        # GRI special sections (Introduction, Glossary, etc.)
        (r'^(Introduction|Glossary|Bibliography|Appendix)$', 'gri_special', 0),

        # Generic headings (uppercase lines) - must be last
        (r'^([A-Z][A-Z\s]{10,80})$', 'heading', 0),
    ]

    def detect_hierarchy(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.UNKNOWN
    ) -> DocumentStructure:
        """
        Detect hierarchical structure in document text.

        Args:
            text: Document text
            doc_type: Type of document (auto-detected if UNKNOWN)

        Returns:
            DocumentStructure with detected sections
        """
        logger.info(f"Detecting structure for document type: {doc_type}")

        # Auto-detect document type if unknown
        if doc_type == DocumentType.UNKNOWN:
            doc_type = self.classify_document_type(text)
            logger.info(f"Auto-detected document type: {doc_type}")

        # Extract sections
        sections = self.extract_sections(text, doc_type)

        # Determine hierarchy type
        hierarchy_type = self._get_hierarchy_type(doc_type)

        # Calculate total hierarchy levels
        total_levels = max([s.level for s in sections], default=1) + 1

        logger.info(
            f"Detected {len(sections)} sections across {total_levels} levels"
        )

        return DocumentStructure(
            sections=sections,
            hierarchy_type=hierarchy_type,
            doc_type=doc_type,
            total_levels=total_levels
        )

    def classify_document_type(self, text: str) -> DocumentType:
        """
        Classify document type based on content patterns.

        Args:
            text: Document text

        Returns:
            Detected DocumentType
        """
        text_lower = text.lower()
        text_sample = text[:5000].lower()

        # Check for NDA patterns
        nda_keywords = [
            'non-disclosure', 'confidential information',
            'receiving party', 'disclosing party', 'nda'
        ]
        if sum(1 for kw in nda_keywords if kw in text_sample) >= 2:
            return DocumentType.NDA

        # Check for contract patterns
        contract_keywords = [
            'whereas', 'parties agree', 'agreement', 'contract',
            'terms and conditions', 'party of the first part'
        ]
        if sum(1 for kw in contract_keywords if kw in text_sample) >= 2:
            return DocumentType.CONTRACT

        # Check for ESG report patterns
        esg_keywords = [
            'sustainability', 'esg report', 'environmental',
            'social', 'governance', 'gri standards', 'csrd', 'esrs'
        ]
        if sum(1 for kw in esg_keywords if kw in text_sample) >= 2:
            return DocumentType.ESG_REPORT

        # Check for legislation patterns
        legislation_keywords = [
            'enacted by', 'legislation', 'statute', 'law',
            'parliament', 'congress', 'section', 'article'
        ]
        if sum(1 for kw in legislation_keywords if kw in text_sample) >= 2:
            return DocumentType.LEGISLATION

        # Default to policy if no specific type detected
        return DocumentType.POLICY

    def extract_sections(
        self,
        text: str,
        doc_type: DocumentType
    ) -> List[Section]:
        """
        Extract sections from document text.

        Args:
            text: Document text
            doc_type: Type of document

        Returns:
            List of Section objects
        """
        sections = []
        lines = text.split('\n')
        current_section = None
        section_counter = 0

        # Skip table of contents - ToC is typically in first ~75 lines for GRI documents
        # Start parsing at line 75 to catch "Introduction" and actual sections
        SKIP_LINES_FOR_TOC = 75

        matched_line = False
        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            # Skip table of contents
            if i < SKIP_LINES_FOR_TOC:
                continue

            # Try to match section patterns
            matched_line = False
            for pattern, section_type, level in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.MULTILINE)

                if match:
                    # Save previous section if exists
                    if current_section:
                        sections.append(current_section)

                    # Create new section
                    section_id = f"{doc_type.value}_{section_type}_{section_counter}"
                    section_counter += 1

                    # Extract title
                    if len(match.groups()) >= 2:
                        title = f"{match.group(1)} {match.group(2)}".strip()
                    else:
                        title = match.group(1).strip() if match.groups() else line

                    current_section = Section(
                        section_id=section_id,
                        title=title,
                        text="",  # Will accumulate text
                        level=level,
                        parent_id=None,  # TODO: Implement parent tracking
                        start_char=sum(len(l) + 1 for l in lines[:i]),
                        end_char=0  # Will be set when section ends
                    )

                    matched_line = True
                    break

            # Accumulate text for current section (skip matched header line)
            if current_section and not matched_line:
                current_section.text += line + "\n"

        # Add last section
        if current_section:
            sections.append(current_section)

        # Filter out sections with minimal content
        # Since we skip ToC, this is just for filtering very short/empty sections
        MIN_SECTION_CHARS = 50  # Sections should have at least 50 chars of content
        substantial_sections = [s for s in sections if len(s.text.strip()) >= MIN_SECTION_CHARS]

        # Remove duplicate sections (keep the one with more content)
        unique_sections = []
        seen_titles = {}

        for section in substantial_sections:
            # Normalize title for comparison
            normalized_title = section.title.lower().strip()

            if normalized_title in seen_titles:
                # Keep the section with more content
                existing_idx = seen_titles[normalized_title]
                if len(section.text) > len(unique_sections[existing_idx].text):
                    unique_sections[existing_idx] = section
            else:
                seen_titles[normalized_title] = len(unique_sections)
                unique_sections.append(section)

        logger.info(f"Filtered sections: {len(sections)} total → {len(unique_sections)} unique with content")

        # If no sections detected, create a single section with all text
        if not unique_sections:
            logger.warning("No structure detected, treating entire document as one section")
            unique_sections.append(Section(
                section_id=f"{doc_type.value}_full_document",
                title="Full Document",
                text=text,
                level=0,
                start_char=0,
                end_char=len(text)
            ))

        return unique_sections

    def _get_hierarchy_type(self, doc_type: DocumentType) -> str:
        """Get hierarchy type string for document type."""
        hierarchy_map = {
            DocumentType.LEGISLATION: 'legislation',
            DocumentType.CONTRACT: 'contract',
            DocumentType.NDA: 'nda',
            DocumentType.ESG_REPORT: 'esg_report',
            DocumentType.POLICY: 'contract',
            DocumentType.REGULATION: 'legislation'
        }
        return hierarchy_map.get(doc_type, 'unknown')
