"""
RAG Pipeline orchestrator for Phases 1-3.

Coordinates document preprocessing, summarization, and chunking.
"""

import json
from pathlib import Path
from typing import Optional

from src.core.models import Document, DocumentStructure, ProcessingResult, DocumentMetadata, DocumentType
from src.core.config import PipelineConfig
from src.preprocessing.pdf_processor import get_pdf_processor
from src.preprocessing.structure_detector import StructureDetector
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.summarization.summarizer import GenericSummarizer
from src.summarization.openai_provider import OpenAIProvider
from src.summarization.claude_provider import ClaudeProvider
from src.chunking.rcts_strategy import RCTSStrategy
from src.chunking.sac_augmenter import SACaugmenter
from src.chunking.multi_layer_chunker import MultiLayerChunker
from src.utils.logger import setup_logger
from src.utils.errors import PipelineExecutionError

logger = setup_logger(__name__)


class RAGPipeline:
    """
    RAG Pipeline for document processing (Phases 1-3).

    Executes:
    - Phase 1: Document preprocessing (PDF → text, structure detection)
    - Phase 2: Generic summary generation (150 chars)
    - Phase 3: Multi-layer chunking with SAC

    Output: ProcessingResult with document, summary, and chunks
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize RAG pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Validate configuration
        try:
            self.config.validate()
        except Exception as e:
            raise PipelineExecutionError(f"Invalid configuration: {e}") from e

        # Setup logger
        self.logger = setup_logger(
            __name__,
            log_level=self.config.log_level,
            log_format=self.config.log_format,
            log_file=self.config.log_file
        )

        # Initialize components
        self._initialize_components()

        self.logger.info("RAG Pipeline initialized successfully")

    def _initialize_components(self):
        """Initialize all pipeline components."""

        # Phase 1 components
        self.pdf_processor = get_pdf_processor(
            use_ocr=self.config.preprocessing.enable_ocr
        )
        self.structure_detector = StructureDetector()
        self.metadata_extractor = MetadataExtractor()

        # Phase 2 components - Create LLM provider based on configuration
        provider_type = self.config.summarization.provider

        if provider_type == 'claude':
            if self.config.anthropic_api_key:
                llm_provider = ClaudeProvider(
                    api_key=self.config.anthropic_api_key,
                    model=self.config.summarization.model
                )
                self.summarizer = GenericSummarizer(
                    llm_provider=llm_provider,
                    config=self.config.summarization
                )
            else:
                self.logger.warning(
                    "No Anthropic API key provided. Summarization will fail. "
                    "Set ANTHROPIC_API_KEY in .env file."
                )
                self.summarizer = None
        elif provider_type == 'openai':
            if self.config.openai_api_key:
                llm_provider = OpenAIProvider(
                    api_key=self.config.openai_api_key,
                    model=self.config.summarization.model
                )
                self.summarizer = GenericSummarizer(
                    llm_provider=llm_provider,
                    config=self.config.summarization
                )
            else:
                self.logger.warning(
                    "No OpenAI API key provided. Summarization will fail. "
                    "Set OPENAI_API_KEY in .env file."
                )
                self.summarizer = None
        else:
            raise PipelineExecutionError(
                f"Unknown LLM provider: {provider_type}. "
                "Supported providers: 'claude', 'openai'"
            )

        # Phase 3 components
        self.chunking_strategy = RCTSStrategy(self.config.chunking)
        self.sac_augmenter = SACaugmenter(
            enable_sac=self.config.chunking.enable_sac
        )
        self.multi_layer_chunker = MultiLayerChunker(
            chunking_strategy=self.chunking_strategy,
            sac_augmenter=self.sac_augmenter,
            config=self.config.chunking
        )

    def process_document(self, pdf_path: str) -> ProcessingResult:
        """
        Process a single document through Phases 1-3.

        Args:
            pdf_path: Path to PDF document

        Returns:
            ProcessingResult with document, summary, and chunks

        Raises:
            PipelineExecutionError: If processing fails
        """
        try:
            self.logger.info(f"=" * 80)
            self.logger.info(f"Processing document: {pdf_path}")
            self.logger.info(f"=" * 80)

            # PHASE 1: Preprocessing
            self.logger.info("PHASE 1: Document Preprocessing")
            document = self._preprocess_document(pdf_path)
            self.logger.info(
                f"✓ Phase 1 complete: {len(document.text)} chars, "
                f"{document.metadata.total_sections} sections"
            )

            # PHASE 2: Summarization
            self.logger.info("\nPHASE 2: Generic Summary Generation")
            if not self.summarizer:
                raise PipelineExecutionError(
                    f"Summarizer not initialized. Check {self.config.summarization.provider.upper()} API key in .env file."
                )

            summary = self.summarizer.summarize(document.text)
            self.logger.info(
                f"✓ Phase 2 complete: {summary.char_count} chars "
                f"(target: {self.config.summarization.max_chars}±"
                f"{self.config.summarization.tolerance})"
            )

            # PHASE 3: Multi-layer Chunking
            self.logger.info("\nPHASE 3: Multi-Layer Chunking with SAC")
            chunks = self.multi_layer_chunker.create_chunks(document, summary)
            self.logger.info(
                f"✓ Phase 3 complete: {len(chunks)} total chunks created"
            )

            # Calculate metrics
            metrics = self._calculate_metrics(document, summary, chunks)

            # Create result
            result = ProcessingResult(
                document=document,
                summary=summary,
                chunks=chunks,
                metrics=metrics
            )

            self.logger.info("\n" + "=" * 80)
            self.logger.info("PROCESSING COMPLETE")
            self.logger.info(f"Total chunks: {len(chunks)}")
            self.logger.info(f"Document chunks: {len(result.get_chunk_by_type('document'))}")
            self.logger.info(f"Section chunks: {len(result.get_section_chunks())}")
            self.logger.info(f"Text chunks: {len(result.get_text_chunks())}")
            self.logger.info("=" * 80 + "\n")

            return result

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise PipelineExecutionError(
                f"Failed to process document {pdf_path}: {e}"
            ) from e

    def _preprocess_document(self, pdf_path: str) -> Document:
        """Execute Phase 1: Preprocessing."""

        # Extract text from PDF
        text = self.pdf_processor.extract_text(pdf_path)

        # Detect hierarchical structure
        structure = self.structure_detector.detect_hierarchy(text)

        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(
            text=text,
            source_path=pdf_path,
            doc_type=structure.doc_type,
            total_sections=len(structure.sections)
        )

        return Document(
            text=text,
            structure=structure,
            metadata=metadata
        )

    def _calculate_metrics(self, document, summary, chunks) -> dict:
        """Calculate processing metrics."""
        text_chunks = [c for c in chunks if c.chunk_type.value == "chunk"]

        avg_chunk_size = (
            sum(c.metadata.char_count for c in text_chunks) / len(text_chunks)
            if text_chunks else 0
        )

        return {
            "total_chars": len(document.text),
            "total_sections": document.metadata.total_sections,
            "summary_chars": summary.char_count,
            "total_chunks": len(chunks),
            "document_chunks": len([c for c in chunks if c.chunk_type.value == "document"]),
            "section_chunks": len([c for c in chunks if c.chunk_type.value == "section"]),
            "text_chunks": len(text_chunks),
            "avg_chunk_size": round(avg_chunk_size, 2),
            "sac_enabled": self.config.chunking.enable_sac,
            "multi_layer_enabled": self.config.chunking.enable_multi_layer
        }

    def save_results(self, result: ProcessingResult, output_dir: Optional[str] = None):
        """
        Save processing results to disk.

        Args:
            result: Processing result
            output_dir: Output directory (uses config if None)
        """
        output_path = Path(output_dir) if output_dir else self.config.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        doc_id = result.document.metadata.document_id
        doc_output_dir = output_path / doc_id
        doc_output_dir.mkdir(exist_ok=True)

        # Save summary
        summary_file = doc_output_dir / "summary.txt"
        summary_file.write_text(result.summary.text)

        # Save chunks
        chunks_file = doc_output_dir / "chunks.json"
        chunks_data = {
            "document_id": doc_id,
            "total_chunks": len(result.chunks),
            "chunks": [chunk.to_dict() for chunk in result.chunks]
        }
        chunks_file.write_text(json.dumps(chunks_data, indent=2))

        # Save metadata
        metadata_file = doc_output_dir / "metadata.json"
        metadata_data = {
            "document": result.document.metadata.to_dict(),
            "summary": {
                "text": result.summary.text,
                "char_count": result.summary.char_count,
                "model": result.summary.model
            },
            "metrics": result.metrics
        }
        metadata_file.write_text(json.dumps(metadata_data, indent=2))

        self.logger.info(f"Results saved to: {doc_output_dir}")
        self.logger.info(f"  - summary.txt ({len(result.summary.text)} chars)")
        self.logger.info(f"  - chunks.json ({len(result.chunks)} chunks)")
        self.logger.info(f"  - metadata.json")
