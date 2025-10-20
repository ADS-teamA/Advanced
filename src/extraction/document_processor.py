"""
Document processor integrating Docling extraction with LawGPT models.

This module bridges IBM Docling's extraction capabilities with the existing
LawGPT pipeline, converting Docling output to LawGPT data models.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime

from .docling_extractor import DoclingExtractor, ExtractionConfig, ExtractedDocument

# Import LawGPT models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.models import (
    Document,
    DocumentStructure,
    DocumentMetadata,
    DocumentType,
    Section
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents using Docling and converts to LawGPT format.

    This processor acts as a bridge between IBM Docling's extraction
    and the LawGPT pipeline, providing seamless integration.

    Example:
        >>> processor = DocumentProcessor()
        >>> doc = processor.process("contract.pdf", doc_type=DocumentType.CONTRACT)
        >>> print(f"Extracted {len(doc.structure.sections)} sections")
        >>> print(doc.metadata.to_dict())
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        detect_document_type: bool = True
    ):
        """
        Initialize the document processor.

        Args:
            config: Docling extraction configuration
            detect_document_type: Whether to auto-detect document type
        """
        self.config = config or ExtractionConfig()
        self.extractor = DoclingExtractor(self.config)
        self.detect_document_type = detect_document_type
        logger.info("DocumentProcessor initialized")

    def process(
        self,
        source: Union[str, Path],
        document_id: Optional[str] = None,
        doc_type: Optional[DocumentType] = None
    ) -> Document:
        """
        Process a document and return LawGPT Document object.

        Args:
            source: Path to the document
            document_id: Optional document identifier
            doc_type: Document type (auto-detected if None and detect_document_type=True)

        Returns:
            Document object compatible with LawGPT pipeline
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        logger.info(f"Processing document: {source_path.name}")

        # Extract using Docling
        extracted = self.extractor.extract(source, document_id)

        # Detect document type if not provided
        if doc_type is None and self.detect_document_type:
            doc_type = self._detect_document_type(extracted)

        # Convert to LawGPT models
        metadata = self._create_metadata(
            extracted=extracted,
            source_path=source_path,
            doc_type=doc_type
        )

        structure = self._create_structure(
            extracted=extracted,
            doc_type=doc_type
        )

        document = Document(
            text=extracted.full_text,
            structure=structure,
            metadata=metadata
        )

        logger.info(
            f"Document processed: {document.metadata.document_id}, "
            f"type={doc_type.value}, sections={len(structure.sections)}"
        )

        return document

    def process_batch(
        self,
        sources: List[Union[str, Path]],
        document_ids: Optional[List[str]] = None,
        doc_types: Optional[List[DocumentType]] = None
    ) -> List[Document]:
        """
        Process multiple documents in batch.

        Args:
            sources: List of document paths
            document_ids: Optional list of document IDs
            doc_types: Optional list of document types

        Returns:
            List of Document objects
        """
        if document_ids and len(document_ids) != len(sources):
            raise ValueError("Length of document_ids must match sources")

        if doc_types and len(doc_types) != len(sources):
            raise ValueError("Length of doc_types must match sources")

        results = []
        for i, source in enumerate(sources):
            doc_id = document_ids[i] if document_ids else None
            doc_type = doc_types[i] if doc_types else None

            try:
                result = self.process(source, doc_id, doc_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                continue

        return results

    def _detect_document_type(self, extracted: ExtractedDocument) -> DocumentType:
        """
        Auto-detect document type based on content and structure.

        Uses heuristics based on:
        - Keywords in content
        - Section structure patterns
        - Document metadata
        """
        text_lower = extracted.full_text.lower()

        # Check for legislation patterns
        legislation_keywords = [
            "zákon", "vyhláška", "nařízení", "směrnice",
            "act", "regulation", "directive", "statute"
        ]
        if any(kw in text_lower for kw in legislation_keywords):
            if "§" in extracted.full_text or "article" in text_lower:
                return DocumentType.LEGISLATION

        # Check for contract patterns
        contract_keywords = [
            "smlouva", "dohoda", "kontrakt",
            "contract", "agreement", "this agreement",
            "parties", "whereas", "witnesseth"
        ]
        if any(kw in text_lower for kw in contract_keywords):
            return DocumentType.CONTRACT

        # Check for NDA patterns
        nda_keywords = [
            "nda", "non-disclosure", "confidentiality agreement",
            "mlčenlivost", "důvěrnost"
        ]
        if any(kw in text_lower for kw in nda_keywords):
            return DocumentType.NDA

        # Check for ESG report patterns
        esg_keywords = [
            "esg", "sustainability", "environmental social governance",
            "carbon", "emissions", "diversity"
        ]
        if any(kw in text_lower for kw in esg_keywords):
            return DocumentType.ESG_REPORT

        # Check for policy patterns
        policy_keywords = [
            "policy", "guideline", "procedure",
            "politika", "směrnice", "postup"
        ]
        if any(kw in text_lower for kw in policy_keywords):
            return DocumentType.POLICY

        logger.info("Could not auto-detect document type, using UNKNOWN")
        return DocumentType.UNKNOWN

    def _create_metadata(
        self,
        extracted: ExtractedDocument,
        source_path: Path,
        doc_type: DocumentType
    ) -> DocumentMetadata:
        """Create DocumentMetadata from extracted document."""

        # Extract parties for contracts/NDAs
        parties = None
        if doc_type in [DocumentType.CONTRACT, DocumentType.NDA]:
            parties = self._extract_parties(extracted.full_text)

        metadata = DocumentMetadata(
            document_id=extracted.document_id,
            document_type=doc_type,
            source_path=str(source_path),
            extraction_date=datetime.now(),
            parties=parties,
            effective_date=None,  # TODO: Extract from content
            total_sections=extracted.num_sections,
            total_chars=extracted.total_chars,
            language=self._detect_language(extracted.full_text)
        )

        return metadata

    def _create_structure(
        self,
        extracted: ExtractedDocument,
        doc_type: DocumentType
    ) -> DocumentStructure:
        """Create DocumentStructure from extracted document."""

        # Convert Docling sections to LawGPT sections
        sections = []
        for doc_sec in extracted.sections:
            section = Section(
                section_id=doc_sec.section_id,
                title=doc_sec.title,
                text=doc_sec.content,
                level=doc_sec.level,
                parent_id=doc_sec.parent_id,
                start_char=doc_sec.char_start,
                end_char=doc_sec.char_end
            )
            sections.append(section)

        # Determine hierarchy type based on document type
        hierarchy_map = {
            DocumentType.LEGISLATION: "legislation",
            DocumentType.CONTRACT: "contract",
            DocumentType.NDA: "nda",
            DocumentType.ESG_REPORT: "esg_report",
            DocumentType.POLICY: "policy",
            DocumentType.REGULATION: "regulation",
            DocumentType.UNKNOWN: "generic"
        }

        hierarchy_type = hierarchy_map.get(doc_type, "generic")

        structure = DocumentStructure(
            sections=sections,
            hierarchy_type=hierarchy_type,
            doc_type=doc_type,
            total_levels=extracted.hierarchy_depth
        )

        return structure

    def _extract_parties(self, text: str) -> Optional[List[str]]:
        """
        Extract parties from contract/NDA text.

        This is a simple heuristic approach. For production use,
        consider using specialized NER models like LexNLP.
        """
        parties = []

        # Look for common party patterns
        patterns = [
            "between",
            "by and between",
            "parties:",
            "strany:"
        ]

        lines = text.split('\n')
        for i, line in enumerate(lines):
            lower_line = line.lower()
            for pattern in patterns:
                if pattern in lower_line:
                    # Extract next few lines as potential party names
                    for j in range(i + 1, min(i + 5, len(lines))):
                        potential_party = lines[j].strip()
                        if potential_party and len(potential_party) < 200:
                            parties.append(potential_party)

        return parties[:10] if parties else None  # Limit to first 10

    def _detect_language(self, text: str) -> str:
        """
        Detect document language.

        Simple heuristic based on common words. For production,
        consider using proper language detection library.
        """
        text_lower = text.lower()

        # Czech indicators
        czech_words = ["zákon", "smlouva", "dohoda", "článek", "odstavec"]
        czech_count = sum(1 for word in czech_words if word in text_lower)

        # English indicators
        english_words = ["the", "and", "agreement", "contract", "section"]
        english_count = sum(1 for word in english_words if word in text_lower)

        if czech_count > english_count:
            return "cs"
        elif english_count > czech_count:
            return "en"
        else:
            return "en"  # Default to English

    def get_extraction_metadata(self, source: Union[str, Path]) -> dict:
        """
        Get extraction metadata without full processing (faster).

        Args:
            source: Path to document

        Returns:
            Dictionary with basic metadata
        """
        extracted = self.extractor.extract(source)

        return {
            "document_id": extracted.document_id,
            "num_pages": extracted.num_pages,
            "num_sections": extracted.num_sections,
            "num_tables": extracted.num_tables,
            "total_chars": extracted.total_chars,
            "extraction_time": extracted.extraction_time.total_seconds()
        }
