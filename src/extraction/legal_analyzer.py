"""
Legal document analyzer with specialized extraction for legal content.

This module provides advanced analysis for legal documents including:
- Contract clause extraction
- Legal entity recognition
- Citation parsing
- Legal term identification
- Risk assessment
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClauseType(str, Enum):
    """Types of legal clauses."""
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    INDEMNIFICATION = "indemnification"
    LIABILITY = "liability"
    JURISDICTION = "jurisdiction"
    DISPUTE_RESOLUTION = "dispute_resolution"
    PAYMENT = "payment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    WARRANTY = "warranty"
    FORCE_MAJEURE = "force_majeure"
    AMENDMENT = "amendment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    NOTICE = "notice"
    ASSIGNMENT = "assignment"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk levels for legal analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LegalClause:
    """Extracted legal clause with metadata."""
    clause_id: str
    clause_type: ClauseType
    title: str
    content: str
    section_id: Optional[str]
    risk_level: RiskLevel
    keywords: List[str]
    char_start: int
    char_end: int

    def to_dict(self) -> Dict:
        return {
            "clause_id": self.clause_id,
            "clause_type": self.clause_type.value,
            "title": self.title,
            "content": self.content,
            "section_id": self.section_id,
            "risk_level": self.risk_level.value,
            "keywords": self.keywords,
            "char_start": self.char_start,
            "char_end": self.char_end
        }


@dataclass
class LegalEntity:
    """Extracted legal entity (party, organization, person)."""
    entity_id: str
    entity_type: str  # "person", "organization", "court", "authority"
    name: str
    context: str
    mentions: int
    first_occurrence: int

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "context": self.context,
            "mentions": self.mentions,
            "first_occurrence": self.first_occurrence
        }


@dataclass
class LegalCitation:
    """Extracted legal citation (statute, case, regulation)."""
    citation_id: str
    citation_type: str  # "statute", "case", "regulation", "article"
    citation_text: str
    full_context: str
    char_position: int

    def to_dict(self) -> Dict:
        return {
            "citation_id": self.citation_id,
            "citation_type": self.citation_type,
            "citation_text": self.citation_text,
            "full_context": self.full_context,
            "char_position": self.char_position
        }


@dataclass
class LegalDate:
    """Extracted legal date with context."""
    date_text: str
    date_type: str  # "effective", "expiration", "signature", "amendment"
    context: str
    char_position: int

    def to_dict(self) -> Dict:
        return {
            "date_text": self.date_text,
            "date_type": self.date_type,
            "context": self.context,
            "char_position": self.char_position
        }


@dataclass
class LegalAnalysis:
    """Complete legal analysis of a document."""
    document_id: str
    clauses: List[LegalClause]
    entities: List[LegalEntity]
    citations: List[LegalCitation]
    dates: List[LegalDate]
    key_terms: Dict[str, int]  # term -> frequency
    risk_summary: Dict[str, int]  # risk_level -> count
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "clauses": [c.to_dict() for c in self.clauses],
            "entities": [e.to_dict() for e in self.entities],
            "citations": [c.to_dict() for c in self.citations],
            "dates": [d.to_dict() for d in self.dates],
            "key_terms": self.key_terms,
            "risk_summary": self.risk_summary,
            "metadata": self.metadata
        }


class LegalDocumentAnalyzer:
    """
    Advanced analyzer for legal documents.

    Provides specialized extraction and analysis for legal content
    including clauses, entities, citations, and risk assessment.

    Example:
        >>> analyzer = LegalDocumentAnalyzer()
        >>> analysis = analyzer.analyze(document_text, document_id="contract_001")
        >>> print(f"Found {len(analysis.clauses)} clauses")
        >>> for clause in analysis.clauses:
        ...     if clause.risk_level == RiskLevel.HIGH:
        ...         print(f"High risk: {clause.title}")
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the legal analyzer.

        Args:
            language: Document language ("en", "cs")
        """
        self.language = language
        self._setup_patterns()
        logger.info(f"LegalDocumentAnalyzer initialized for language: {language}")

    def _setup_patterns(self):
        """Setup regex patterns for legal content extraction."""

        if self.language == "en":
            self.clause_patterns = {
                ClauseType.CONFIDENTIALITY: [
                    r"confidential(?:ity)?",
                    r"non-disclosure",
                    r"proprietary information",
                    r"trade secret"
                ],
                ClauseType.TERMINATION: [
                    r"termination",
                    r"cancel(?:lation)?",
                    r"end of agreement",
                    r"expire"
                ],
                ClauseType.INDEMNIFICATION: [
                    r"indemnif(?:y|ication)",
                    r"hold harmless",
                    r"defend"
                ],
                ClauseType.LIABILITY: [
                    r"liability",
                    r"liable",
                    r"damages",
                    r"loss(?:es)?"
                ],
                ClauseType.JURISDICTION: [
                    r"jurisdiction",
                    r"governing law",
                    r"applicable law",
                    r"laws of"
                ],
                ClauseType.DISPUTE_RESOLUTION: [
                    r"dispute resolution",
                    r"arbitration",
                    r"mediation",
                    r"litigation"
                ],
                ClauseType.PAYMENT: [
                    r"payment",
                    r"compensation",
                    r"fee",
                    r"price"
                ],
                ClauseType.INTELLECTUAL_PROPERTY: [
                    r"intellectual property",
                    r"patent",
                    r"copyright",
                    r"trademark",
                    r"IP rights"
                ],
                ClauseType.WARRANTY: [
                    r"warrant(?:y|ies)",
                    r"represent(?:ation)?",
                    r"guarantee"
                ],
                ClauseType.FORCE_MAJEURE: [
                    r"force majeure",
                    r"act of god",
                    r"unavoidable"
                ]
            }

            # Citation patterns
            self.citation_patterns = {
                "statute": r"\d+\s+U\.?S\.?C\.?\s+§?\s*\d+",
                "case": r"\d+\s+[A-Z][a-z\.]+\s+\d+",
                "article": r"Article\s+\d+|Art\.\s*\d+"
            }

        elif self.language == "cs":
            self.clause_patterns = {
                ClauseType.CONFIDENTIALITY: [
                    r"důvěrnost",
                    r"mlčenlivost",
                    r"utajení"
                ],
                ClauseType.TERMINATION: [
                    r"ukončení",
                    r"vypovězení",
                    r"zánik"
                ],
                ClauseType.LIABILITY: [
                    r"odpovědnost",
                    r"náhrada škody",
                    r"ručení"
                ],
                ClauseType.JURISDICTION: [
                    r"jurisdikce",
                    r"rozhodné právo",
                    r"příslušnost"
                ]
            }

            self.citation_patterns = {
                "statute": r"zákon\s+č\.\s*\d+/\d+\s+Sb\.",
                "article": r"§\s*\d+|čl\.\s*\d+|článek\s+\d+"
            }

    def analyze(
        self,
        text: str,
        document_id: str,
        sections: Optional[List] = None
    ) -> LegalAnalysis:
        """
        Perform complete legal analysis on document text.

        Args:
            text: Document text
            document_id: Document identifier
            sections: Optional list of document sections for context

        Returns:
            LegalAnalysis object with all extracted information
        """
        logger.info(f"Starting legal analysis for document: {document_id}")

        # Extract clauses
        clauses = self._extract_clauses(text, sections)

        # Extract entities
        entities = self._extract_entities(text)

        # Extract citations
        citations = self._extract_citations(text)

        # Extract dates
        dates = self._extract_dates(text)

        # Extract key terms
        key_terms = self._extract_key_terms(text)

        # Calculate risk summary
        risk_summary = self._calculate_risk_summary(clauses)

        # Gather metadata
        metadata = {
            "num_clauses": len(clauses),
            "num_entities": len(entities),
            "num_citations": len(citations),
            "num_dates": len(dates),
            "text_length": len(text),
            "language": self.language
        }

        analysis = LegalAnalysis(
            document_id=document_id,
            clauses=clauses,
            entities=entities,
            citations=citations,
            dates=dates,
            key_terms=key_terms,
            risk_summary=risk_summary,
            metadata=metadata
        )

        logger.info(
            f"Legal analysis completed: {len(clauses)} clauses, "
            f"{len(entities)} entities, {len(citations)} citations"
        )

        return analysis

    def _extract_clauses(
        self,
        text: str,
        sections: Optional[List] = None
    ) -> List[LegalClause]:
        """Extract legal clauses from text."""
        clauses = []
        clause_counter = 0

        # Split into paragraphs for clause detection
        paragraphs = text.split('\n\n')
        char_position = 0

        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip short paragraphs
                char_position += len(para) + 2
                continue

            # Detect clause type
            clause_type, keywords = self._detect_clause_type(para)

            if clause_type != ClauseType.OTHER or keywords:
                clause_counter += 1

                # Extract title (first line or sentence)
                title = para.split('.')[0][:100]

                # Assess risk
                risk_level = self._assess_clause_risk(para, clause_type)

                clause = LegalClause(
                    clause_id=f"clause_{clause_counter}",
                    clause_type=clause_type,
                    title=title,
                    content=para,
                    section_id=None,  # TODO: Map to section
                    risk_level=risk_level,
                    keywords=keywords,
                    char_start=char_position,
                    char_end=char_position + len(para)
                )

                clauses.append(clause)

            char_position += len(para) + 2

        return clauses

    def _detect_clause_type(self, text: str) -> Tuple[ClauseType, List[str]]:
        """Detect clause type from text content."""
        text_lower = text.lower()
        matched_keywords = []

        for clause_type, patterns in self.clause_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matched_keywords.append(pattern)
                    return clause_type, matched_keywords

        return ClauseType.OTHER, matched_keywords

    def _assess_clause_risk(self, text: str, clause_type: ClauseType) -> RiskLevel:
        """Assess risk level of a clause."""
        text_lower = text.lower()

        # High-risk clause types
        high_risk_types = {
            ClauseType.INDEMNIFICATION,
            ClauseType.LIABILITY,
            ClauseType.TERMINATION
        }

        # High-risk keywords
        high_risk_keywords = [
            "unlimited", "sole discretion", "at any time",
            "without cause", "waive", "forfeit"
        ]

        # Critical risk keywords
        critical_keywords = [
            "irrevocable", "perpetual", "exclusive",
            "automatic renewal", "liquidated damages"
        ]

        if any(kw in text_lower for kw in critical_keywords):
            return RiskLevel.CRITICAL
        elif clause_type in high_risk_types or any(kw in text_lower for kw in high_risk_keywords):
            return RiskLevel.HIGH
        elif clause_type == ClauseType.OTHER:
            return RiskLevel.LOW
        else:
            return RiskLevel.MEDIUM

    def _extract_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities (simplified NER)."""
        entities = []
        entity_counter = 0

        # Pattern for organizations (simplified)
        org_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|GmbH)\.?',
            r'\b[A-Z][A-Z]+\b'  # Acronyms
        ]

        seen_entities = {}

        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                entity_name = match.group(0)

                if entity_name not in seen_entities:
                    entity_counter += 1
                    context = text[max(0, match.start() - 50):match.end() + 50]

                    entity = LegalEntity(
                        entity_id=f"entity_{entity_counter}",
                        entity_type="organization",
                        name=entity_name,
                        context=context,
                        mentions=1,
                        first_occurrence=match.start()
                    )

                    entities.append(entity)
                    seen_entities[entity_name] = entity
                else:
                    seen_entities[entity_name].mentions += 1

        return entities

    def _extract_citations(self, text: str) -> List[LegalCitation]:
        """Extract legal citations."""
        citations = []
        citation_counter = 0

        for citation_type, pattern in self.citation_patterns.items():
            for match in re.finditer(pattern, text):
                citation_counter += 1
                citation_text = match.group(0)
                context = text[max(0, match.start() - 100):match.end() + 100]

                citation = LegalCitation(
                    citation_id=f"cite_{citation_counter}",
                    citation_type=citation_type,
                    citation_text=citation_text,
                    full_context=context,
                    char_position=match.start()
                )

                citations.append(citation)

        return citations

    def _extract_dates(self, text: str) -> List[LegalDate]:
        """Extract legal dates with context."""
        dates = []

        # Date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}\.\s*\d{1,2}\.\s*\d{4}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]

        # Date type keywords
        date_type_patterns = {
            "effective": r"effective\s+date|účinnost",
            "expiration": r"expir(?:ation|es)|platnost do",
            "signature": r"signed|executed|podepsán",
            "amendment": r"amended|změněn"
        }

        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date_text = match.group(0)
                context = text[max(0, match.start() - 100):match.end() + 100]

                # Detect date type from context
                date_type = "other"
                for dtype, dpattern in date_type_patterns.items():
                    if re.search(dpattern, context, re.IGNORECASE):
                        date_type = dtype
                        break

                date = LegalDate(
                    date_text=date_text,
                    date_type=date_type,
                    context=context,
                    char_position=match.start()
                )

                dates.append(date)

        return dates

    def _extract_key_terms(self, text: str) -> Dict[str, int]:
        """Extract key legal terms and their frequencies."""
        text_lower = text.lower()

        # Common legal terms
        legal_terms = [
            "agreement", "contract", "party", "parties",
            "shall", "hereby", "whereas", "obligations",
            "rights", "terms", "conditions", "provisions",
            "covenant", "consideration", "breach",
            "enforceable", "binding", "lawful"
        ]

        term_freq = {}
        for term in legal_terms:
            count = len(re.findall(r'\b' + term + r'\b', text_lower))
            if count > 0:
                term_freq[term] = count

        # Sort by frequency
        term_freq = dict(sorted(term_freq.items(), key=lambda x: x[1], reverse=True))

        return term_freq

    def _calculate_risk_summary(self, clauses: List[LegalClause]) -> Dict[str, int]:
        """Calculate risk summary from clauses."""
        risk_summary = {
            RiskLevel.LOW.value: 0,
            RiskLevel.MEDIUM.value: 0,
            RiskLevel.HIGH.value: 0,
            RiskLevel.CRITICAL.value: 0
        }

        for clause in clauses:
            risk_summary[clause.risk_level.value] += 1

        return risk_summary

    def analyze_clause_coverage(self, analysis: LegalAnalysis) -> Dict[str, any]:
        """
        Analyze which standard clauses are present/missing.

        Returns:
            Dictionary with coverage analysis
        """
        present_types = {clause.clause_type for clause in analysis.clauses}

        # Standard clauses for contracts
        standard_clauses = {
            ClauseType.TERMINATION,
            ClauseType.JURISDICTION,
            ClauseType.DISPUTE_RESOLUTION,
            ClauseType.LIABILITY,
            ClauseType.ENTIRE_AGREEMENT
        }

        missing_clauses = standard_clauses - present_types
        present_standard = standard_clauses & present_types

        coverage = {
            "coverage_percentage": len(present_standard) / len(standard_clauses) * 100,
            "present_clauses": [c.value for c in present_standard],
            "missing_clauses": [c.value for c in missing_clauses],
            "additional_clauses": [c.clause_type.value for c in analysis.clauses
                                   if c.clause_type not in standard_clauses]
        }

        return coverage
