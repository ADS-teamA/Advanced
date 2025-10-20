"""Custom exceptions for the RAG pipeline."""


class RAGPipelineError(Exception):
    """Base exception for all RAG pipeline errors."""
    pass


class ConfigurationError(RAGPipelineError):
    """Raised when configuration is invalid."""
    pass


class PDFProcessingError(RAGPipelineError):
    """Raised when PDF processing fails."""
    pass


class StructureDetectionError(RAGPipelineError):
    """Raised when structure detection fails."""
    pass


class SummarizationError(RAGPipelineError):
    """Raised when summarization fails."""
    pass


class ChunkingError(RAGPipelineError):
    """Raised when chunking fails."""
    pass


class PipelineExecutionError(RAGPipelineError):
    """Raised when pipeline execution fails."""
    pass
