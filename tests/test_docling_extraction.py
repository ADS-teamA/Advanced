"""
Tests for Docling document extraction.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.extraction.docling_extractor import (
    DoclingExtractor,
    ExtractionConfig,
    TableData,
    DocumentSection,
    ExtractedDocument
)
from src.extraction.document_processor import DocumentProcessor
from src.extraction.legal_analyzer import (
    LegalDocumentAnalyzer,
    LegalClause,
    ClauseType,
    RiskLevel
)
from LawGPT.src.core.models import DocumentType


class TestExtractionConfig:
    """Test ExtractionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        assert config.enable_ocr is True
        assert config.extract_tables is True
        assert config.generate_markdown is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExtractionConfig(
            enable_ocr=False,
            extract_tables=False,
            use_gpu=True
        )
        assert config.enable_ocr is False
        assert config.extract_tables is False
        assert config.use_gpu is True


class TestDoclingExtractor:
    """Test DoclingExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create DoclingExtractor instance."""
        config = ExtractionConfig(enable_ocr=False)  # Faster for tests
        return DoclingExtractor(config)

    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.config is not None
        assert extractor.converter is not None

    def test_supported_formats(self, extractor):
        """Test supported file formats."""
        formats = extractor.get_supported_formats()
        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".pptx" in formats
        assert ".xlsx" in formats

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data").exists(),
        reason="data directory not found"
    )
    def test_extract_real_document(self, extractor):
        """Test extraction on real document if available."""
        data_dir = Path(__file__).parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))

        if pdf_files:
            result = extractor.extract(pdf_files[0])
            assert isinstance(result, ExtractedDocument)
            assert result.document_id is not None
            assert len(result.full_text) > 0
            assert result.num_pages > 0


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance."""
        config = ExtractionConfig(enable_ocr=False)
        return DocumentProcessor(config)

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert processor.extractor is not None
        assert processor.detect_document_type is True

    def test_document_type_detection(self, processor):
        """Test document type detection logic."""
        # Test contract detection
        contract_text = "This Agreement is made between parties whereas..."
        extracted = type('obj', (object,), {
            'full_text': contract_text,
            'num_sections': 5
        })()

        doc_type = processor._detect_document_type(extracted)
        assert doc_type == DocumentType.CONTRACT

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data").exists(),
        reason="data directory not found"
    )
    def test_process_real_document(self, processor):
        """Test processing real document if available."""
        data_dir = Path(__file__).parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))

        if pdf_files:
            document = processor.process(pdf_files[0])
            assert document.metadata is not None
            assert document.structure is not None
            assert len(document.text) > 0


class TestLegalAnalyzer:
    """Test LegalDocumentAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create LegalDocumentAnalyzer instance."""
        return LegalDocumentAnalyzer(language="en")

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.language == "en"
        assert analyzer.clause_patterns is not None

    def test_clause_detection(self, analyzer):
        """Test clause type detection."""
        text = "This confidentiality agreement requires non-disclosure of proprietary information."
        clause_type, keywords = analyzer._detect_clause_type(text)

        assert clause_type == ClauseType.CONFIDENTIALITY
        assert len(keywords) > 0

    def test_risk_assessment(self, analyzer):
        """Test clause risk assessment."""
        # High risk text
        high_risk_text = "Party agrees to unlimited liability at any time without cause."
        risk = analyzer._assess_clause_risk(high_risk_text, ClauseType.LIABILITY)
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]

        # Low risk text
        low_risk_text = "This section defines the terms used in this agreement."
        risk = analyzer._assess_clause_risk(low_risk_text, ClauseType.OTHER)
        assert risk == RiskLevel.LOW

    def test_entity_extraction(self, analyzer):
        """Test entity extraction."""
        text = "Acme Corp and XYZ Inc entered into this agreement."
        entities = analyzer._extract_entities(text)

        assert len(entities) > 0
        entity_names = [e.name for e in entities]
        assert any("Acme" in name or "XYZ" in name for name in entity_names)

    def test_citation_extraction(self, analyzer):
        """Test legal citation extraction."""
        text = "As defined in 42 U.S.C. § 1983 and Article 6 of the convention."
        citations = analyzer._extract_citations(text)

        assert len(citations) > 0

    def test_key_terms_extraction(self, analyzer):
        """Test key legal terms extraction."""
        text = """
        This agreement establishes the terms and conditions whereby
        the parties agree to the obligations set forth herein.
        The parties shall be bound by these terms.
        """
        terms = analyzer._extract_key_terms(text)

        assert "agreement" in terms
        assert "parties" in terms
        assert terms["parties"] >= 2  # Appears at least twice

    def test_full_analysis(self, analyzer):
        """Test complete legal analysis."""
        text = """
        CONFIDENTIALITY AGREEMENT

        This Agreement is entered into by and between Acme Corp and XYZ Inc.

        1. Confidentiality
        The parties agree to maintain confidentiality of all proprietary information.

        2. Termination
        This agreement may be terminated with 30 days notice.

        3. Liability
        Each party agrees to unlimited liability for breaches.

        4. Jurisdiction
        This agreement shall be governed by the laws of California.
        """

        analysis = analyzer.analyze(text, document_id="test_doc")

        assert analysis.document_id == "test_doc"
        assert len(analysis.clauses) > 0
        assert len(analysis.key_terms) > 0
        assert analysis.metadata["text_length"] == len(text)

    def test_clause_coverage_analysis(self, analyzer):
        """Test standard clause coverage analysis."""
        # Create mock analysis with some standard clauses
        from src.extraction.legal_analyzer import LegalAnalysis

        clauses = [
            LegalClause(
                clause_id="c1",
                clause_type=ClauseType.TERMINATION,
                title="Termination",
                content="...",
                section_id=None,
                risk_level=RiskLevel.MEDIUM,
                keywords=[],
                char_start=0,
                char_end=100
            ),
            LegalClause(
                clause_id="c2",
                clause_type=ClauseType.LIABILITY,
                title="Liability",
                content="...",
                section_id=None,
                risk_level=RiskLevel.HIGH,
                keywords=[],
                char_start=100,
                char_end=200
            )
        ]

        analysis = LegalAnalysis(
            document_id="test",
            clauses=clauses,
            entities=[],
            citations=[],
            dates=[],
            key_terms={},
            risk_summary={},
            metadata={}
        )

        coverage = analyzer.analyze_clause_coverage(analysis)

        assert "coverage_percentage" in coverage
        assert "present_clauses" in coverage
        assert "missing_clauses" in coverage
        assert coverage["coverage_percentage"] > 0


class TestIntegration:
    """Integration tests for complete workflow."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data").exists(),
        reason="data directory not found"
    )
    def test_complete_workflow(self):
        """Test complete document processing workflow."""
        data_dir = Path(__file__).parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))

        if not pdf_files:
            pytest.skip("No PDF files found in data directory")

        # Setup
        config = ExtractionConfig(enable_ocr=False)
        processor = DocumentProcessor(config)
        analyzer = LegalDocumentAnalyzer(language="en")

        # Process document
        document = processor.process(pdf_files[0])

        # Analyze
        analysis = analyzer.analyze(
            text=document.text,
            document_id=document.metadata.document_id
        )

        # Verify results
        assert document.metadata.document_id is not None
        assert len(document.structure.sections) >= 0
        assert len(analysis.clauses) >= 0
        assert analysis.metadata["text_length"] > 0

        # Check coverage
        coverage = analyzer.analyze_clause_coverage(analysis)
        assert coverage["coverage_percentage"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
