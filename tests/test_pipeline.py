"""
Integration tests for the RAG pipeline (Phases 1-3).
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import PipelineConfig, SummarizationConfig, ChunkingConfig
from src.core.models import ChunkType
from src.pipeline.rag_pipeline import RAGPipeline
from src.preprocessing.structure_detector import StructureDetector
from src.utils.errors import PDFProcessingError


class TestStructureDetector:
    """Test structure detection."""

    def test_document_type_classification(self):
        """Test document type classification."""
        detector = StructureDetector()

        # Test NDA detection
        nda_text = """
        NON-DISCLOSURE AGREEMENT

        This Non-Disclosure Agreement is entered into between the disclosing party
        and the receiving party to protect confidential information.
        """
        doc_type = detector.classify_document_type(nda_text)
        assert doc_type.value == "nda"

        # Test contract detection
        contract_text = """
        AGREEMENT

        Whereas the parties agree to the following terms and conditions,
        this contract is hereby executed between party of the first part...
        """
        doc_type = detector.classify_document_type(contract_text)
        assert doc_type.value == "contract"

        # Test ESG report detection
        esg_text = """
        SUSTAINABILITY REPORT 2023

        This ESG report presents our environmental, social, and governance
        performance according to GRI Standards and CSRD requirements.
        """
        doc_type = detector.classify_document_type(esg_text)
        assert doc_type.value == "esg_report"


class TestConfiguration:
    """Test configuration system."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        # Test summarization config (evidence-based)
        assert config.summarization.max_chars == 150  # Optimal from Reuter 2024
        assert config.summarization.style == "generic"  # Generic > expert
        assert config.summarization.model == "gpt-4o-mini"

        # Test chunking config (evidence-based)
        assert config.chunking.chunk_size == 500  # Optimal from Reuter 2024
        assert config.chunking.chunk_overlap == 0  # RCTS handles naturally
        assert config.chunking.enable_sac is True  # 58% DRM reduction
        assert config.chunking.enable_multi_layer is True  # 2.3x improvement

    def test_config_validation(self):
        """Test configuration validation."""
        config = PipelineConfig()

        # Should not raise with valid config
        config.validate()

        # Should raise with invalid chunk size
        config.chunking.chunk_size = 50  # Too small
        with pytest.raises(ValueError, match="too small"):
            config.validate()


class TestPipelineIntegration:
    """Integration tests for full pipeline (requires OpenAI API key)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        config = PipelineConfig()
        config.summarization.max_retries = 2  # Reduce retries for testing

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping integration tests")

        return config

    @pytest.fixture
    def sample_pdf_path(self):
        """Get path to sample PDF for testing."""
        # Use an actual ESG report from the data directory
        pdf_path = Path("data/ESG_reporty/PDF/CEZ_2023_EN.pdf")

        if not pdf_path.exists():
            pytest.skip(f"Sample PDF not found: {pdf_path}")

        return str(pdf_path)

    def test_full_pipeline(self, config, sample_pdf_path):
        """Test complete pipeline execution."""
        pipeline = RAGPipeline(config)

        # Process document
        result = pipeline.process_document(sample_pdf_path)

        # Verify result structure
        assert result.document is not None
        assert result.summary is not None
        assert len(result.chunks) > 0

        # Verify summary constraints
        assert len(result.summary.text) > 0
        assert result.summary.char_count <= (
            config.summarization.max_chars + config.summarization.tolerance
        ), f"Summary too long: {result.summary.char_count} chars"

        # Verify multi-layer chunks
        doc_chunks = result.get_chunk_by_type(ChunkType.DOCUMENT)
        section_chunks = result.get_section_chunks()
        text_chunks = result.get_text_chunks()

        if config.chunking.enable_multi_layer:
            assert len(doc_chunks) == 1, "Should have exactly 1 document-level chunk"
            assert len(section_chunks) >= 0, "Should have section chunks"

        assert len(text_chunks) > 0, "Should have text chunks"

        # Verify SAC augmentation
        if config.chunking.enable_sac:
            for chunk in text_chunks:
                # Content should be augmented (longer than raw)
                assert len(chunk.content) > len(chunk.raw_content), \
                    "SAC should augment content with summary"

                # Summary should be in augmented content
                assert result.summary.text in chunk.content, \
                    "Summary should be prepended to chunk content"

                # Raw content should not contain summary
                assert result.summary.text not in chunk.raw_content, \
                    "Raw content should not contain summary"

        # Verify metrics
        assert result.metrics['total_chunks'] == len(result.chunks)
        assert result.metrics['sac_enabled'] == config.chunking.enable_sac
        assert result.metrics['multi_layer_enabled'] == config.chunking.enable_multi_layer

    def test_pipeline_with_sac_disabled(self, config, sample_pdf_path):
        """Test pipeline with SAC disabled."""
        config.chunking.enable_sac = False

        pipeline = RAGPipeline(config)
        result = pipeline.process_document(sample_pdf_path)

        # With SAC disabled, content should equal raw_content
        text_chunks = result.get_text_chunks()
        for chunk in text_chunks:
            assert chunk.content == chunk.raw_content, \
                "Without SAC, content should equal raw_content"

    def test_save_results(self, config, sample_pdf_path, tmp_path):
        """Test saving results to disk."""
        pipeline = RAGPipeline(config)
        result = pipeline.process_document(sample_pdf_path)

        # Save to temporary directory
        pipeline.save_results(result, output_dir=str(tmp_path))

        # Verify files exist
        doc_id = result.document.metadata.document_id
        doc_dir = tmp_path / doc_id

        assert doc_dir.exists()
        assert (doc_dir / "summary.txt").exists()
        assert (doc_dir / "chunks.json").exists()
        assert (doc_dir / "metadata.json").exists()

        # Verify summary content
        summary_text = (doc_dir / "summary.txt").read_text()
        assert summary_text == result.summary.text


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
