"""
Metadata extraction from legal documents.

Extracts document ID, type, dates, parties, and other metadata.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import hashlib

from src.core.models import DocumentMetadata, DocumentType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetadataExtractor:
    """Extracts metadata from legal documents."""

    # Date patterns
    DATE_PATTERNS = [
        r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b',  # YYYY/MM/DD
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
    ]

    # Party identification patterns (for contracts/NDAs)
    PARTY_PATTERNS = [
        r'(?:party|parties):\s*([A-Z][^\n]{5,100})',
        r'between\s+([A-Z][^\n]{5,100})\s+and\s+([A-Z][^\n]{5,100})',
        r'(?:executed by|signed by|entered into by)\s+([A-Z][^\n]{5,100})',
        r'"([A-Z][^\n]{10,100})"\s*\((?:hereinafter|the)\s+"[^"]+"\)',
    ]

    def extract_metadata(
        self,
        text: str,
        source_path: str,
        doc_type: DocumentType,
        total_sections: int = 0
    ) -> DocumentMetadata:
        """
        Extract metadata from document text.

        Args:
            text: Document text
            source_path: Path to source document
            doc_type: Detected document type
            total_sections: Number of sections detected

        Returns:
            DocumentMetadata object
        """
        logger.info(f"Extracting metadata from {source_path}")

        # Generate document ID from file path
        document_id = self._generate_document_id(source_path)

        # Extract dates
        effective_date = self._extract_date(text)

        # Extract parties (for contracts/NDAs)
        parties = None
        if doc_type in [DocumentType.CONTRACT, DocumentType.NDA]:
            parties = self._extract_parties(text)

        # Detect language
        language = self._detect_language(text)

        metadata = DocumentMetadata(
            document_id=document_id,
            document_type=doc_type,
            source_path=source_path,
            extraction_date=datetime.now(),
            parties=parties,
            effective_date=effective_date,
            total_sections=total_sections,
            total_chars=len(text),
            language=language
        )

        logger.info(f"Extracted metadata: {metadata.document_id}, type: {doc_type.value}")
        if parties:
            logger.info(f"Found {len(parties)} parties")

        return metadata

    def _generate_document_id(self, source_path: str) -> str:
        """
        Generate unique document ID from file path.

        Uses combination of filename and hash for uniqueness.
        """
        path = Path(source_path)
        filename = path.stem  # Filename without extension

        # Create hash from full path for uniqueness
        path_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]

        # Combine filename with short hash
        doc_id = f"{filename}_{path_hash}"

        return doc_id

    def _extract_date(self, text: str) -> Optional[datetime]:
        """
        Extract effective date from document text.

        Looks for dates in the first 2000 characters.
        """
        text_sample = text[:2000]

        for pattern in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text_sample, re.IGNORECASE)

            for match in matches:
                try:
                    # Try to parse the date
                    date_str = match.group(0)

                    # Handle different date formats
                    for fmt in [
                        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
                        '%Y/%m/%d', '%Y-%m-%d',
                        '%B %d, %Y', '%d %B %Y',
                        '%b %d, %Y', '%d %b %Y'
                    ]:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            # Only return if date is reasonable (not in future, not too old)
                            if 1990 <= parsed_date.year <= datetime.now().year + 1:
                                return parsed_date
                        except ValueError:
                            continue

                except Exception as e:
                    logger.debug(f"Failed to parse date {match.group(0)}: {e}")
                    continue

        return None

    def _extract_parties(self, text: str) -> Optional[List[str]]:
        """
        Extract party names from contract/NDA text.

        Looks for parties in the first 3000 characters.
        """
        text_sample = text[:3000]
        parties = []

        for pattern in self.PARTY_PATTERNS:
            matches = re.finditer(pattern, text_sample, re.IGNORECASE)

            for match in matches:
                for group in match.groups():
                    if group and len(group.strip()) > 5:
                        party = group.strip()

                        # Clean up party name
                        party = re.sub(r'\s+', ' ', party)  # Normalize whitespace
                        party = party.rstrip(',;.')  # Remove trailing punctuation

                        # Avoid duplicates
                        if party not in parties and len(party) < 200:
                            parties.append(party)

        # Return parties if found, otherwise None
        return parties if parties else None

    def _detect_language(self, text: str) -> str:
        """
        Detect document language (simple heuristic).

        Returns:
            Language code ('en', 'cs', etc.)
        """
        text_sample = text[:1000].lower()

        # Czech language indicators
        czech_indicators = ['ě', 'š', 'č', 'ř', 'ž', 'ý', 'á', 'í', 'é', 'ů', 'ú']
        czech_words = ['který', 'která', 'které', 'jako', 'jsou', 'není', 'nebo', 'jeho']

        czech_char_count = sum(1 for char in text_sample if char in czech_indicators)
        czech_word_count = sum(1 for word in czech_words if word in text_sample)

        if czech_char_count > 5 or czech_word_count > 2:
            return 'cs'

        # Default to English
        return 'en'
