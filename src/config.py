"""
Unified configuration system for RAG pipeline.

All configuration is centralized here with sensible defaults based on research:
- PHASE 1: Document Extraction (Docling)
- PHASE 2: Summarization (Generic summaries, 150 chars)
- PHASE 3: Chunking (Hierarchical with SAC, 500 chars)
- PHASE 4: Embedding (Multi-layer embeddings)

Environment variables (.env) - API keys and model selections
All configuration classes can be imported by other modules.
"""

import os
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load .env on module import
load_env()


@dataclass
class ModelConfig:
    """Central model configuration loaded from environment variables."""

    # LLM Configuration
    llm_provider: str  # "claude" or "openai"
    llm_model: str     # e.g., "claude-sonnet-4.5", "gpt-4o-mini"

    # Embedding Configuration
    embedding_provider: str  # "voyage", "openai", "huggingface"
    embedding_model: str     # e.g., "kanon-2", "text-embedding-3-large", "bge-m3"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            LLM_PROVIDER: "claude" or "openai" (default: "claude")
            LLM_MODEL: Model name (default: "claude-sonnet-4-5-20250929")

            EMBEDDING_PROVIDER: "voyage", "openai", or "huggingface" (default: "voyage")
            EMBEDDING_MODEL: Model name (default: "kanon-2")

            ANTHROPIC_API_KEY: Claude API key
            OPENAI_API_KEY: OpenAI API key
            VOYAGE_API_KEY: Voyage AI API key
        """
        return cls(
            # LLM Configuration
            llm_provider=os.getenv("LLM_PROVIDER", "claude"),
            llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929"),

            # Embedding Configuration
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),

            # API Keys
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            voyage_api_key=os.getenv("VOYAGE_API_KEY")
        )

    def get_llm_config(self) -> dict:
        """Get LLM configuration for SummaryGenerator."""
        if self.llm_provider == "claude":
            return {
                "provider": "claude",
                "model": self.llm_model,
                "api_key": self.anthropic_api_key
            }
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "model": self.llm_model,
                "api_key": self.openai_api_key
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_embedding_config(self) -> dict:
        """Get embedding configuration for EmbeddingGenerator."""
        if self.embedding_provider == "voyage":
            return {
                "provider": "voyage",
                "model": self.embedding_model,
                "api_key": self.voyage_api_key
            }
        elif self.embedding_provider == "openai":
            return {
                "provider": "openai",
                "model": self.embedding_model,
                "api_key": self.openai_api_key
            }
        elif self.embedding_provider == "huggingface":
            return {
                "provider": "huggingface",
                "model": self.embedding_model,
                "api_key": None  # Local models
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")


# Model aliases for convenience
MODEL_ALIASES = {
    # Claude 4.5 models (latest)
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",

    # OpenAI models
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",

    # Local Legal LLM models (via Ollama or Transformers)
    "saul-7b": "Equall/Saul-7B-Instruct-v1",  # Legal Mistral fine-tune
    "mistral-legal-7b": "Equall/Saul-7B-Instruct-v1",  # Alias
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",

    # Embedding models (Voyage AI)
    "kanon-2": "kanon-2",
    "voyage-3": "voyage-3-large",
    "voyage-law-2": "voyage-law-2",

    # OpenAI embeddings
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small",

    # HuggingFace models
    "bge-m3": "BAAI/bge-m3",
}


def resolve_model_alias(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model_name, model_name)


@dataclass
class ExtractionConfig:
    """
    Configuration for Docling extraction (PHASE 1).

    To customize, create instance with your values:
        config = ExtractionConfig(
            hierarchy_tolerance=0.5,  # Stricter clustering
            enable_smart_hierarchy=True
        )
    """

    # OCR settings
    enable_ocr: bool = True
    ocr_language: List[str] = field(default_factory=lambda: ["cs-CZ", "en-US"])
    ocr_recognition: str = "accurate"  # "accurate" or "fast"

    # Table extraction
    table_mode: str = "ACCURATE"  # Will be converted to TableFormerMode
    extract_tables: bool = True

    # Hierarchy extraction (CRITICAL for hierarchical chunking)
    extract_hierarchy: bool = True
    enable_smart_hierarchy: bool = True  # Font-size based classification
    hierarchy_tolerance: float = 0.8  # BBox height clustering tolerance (pixels, lower = stricter)

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
class SummarizationConfig:
    """Configuration for summarization (PHASE 2)."""

    provider: str = "claude"  # 'claude' or 'openai'
    model: str = "haiku"  # Alias or full model name
    max_chars: int = 150
    tolerance: int = 20
    style: str = "generic"
    temperature: float = 0.3
    max_tokens: int = 500
    retry_on_exceed: bool = True
    max_retries: int = 3
    max_workers: int = 10
    min_text_length: int = 50


@dataclass
class ContextGenerationConfig:
    """
    Configuration for Contextual Retrieval (Anthropic, Sept 2024).

    Generates LLM-based context for each chunk instead of generic summaries.
    Results in 67% reduction in retrieval failures (Anthropic research).
    """

    # Enable contextual retrieval
    enable_contextual: bool = True

    # LLM provider for context generation
    provider: str = "anthropic"  # 'anthropic', 'openai', 'local'

    # Model selection
    model: str = "haiku"  # Fast & cheap for context generation
    # Options:
    #   Anthropic: haiku, sonnet
    #   OpenAI: gpt-4o-mini, gpt-4o
    #   Local: saul-7b, mistral-legal-7b, llama-3-8b

    # API keys (loaded from environment if not provided)
    api_key: Optional[str] = None

    # Generation params
    temperature: float = 0.3
    max_tokens: int = 150  # Context should be 50-100 words

    # Context window params
    include_surrounding_chunks: bool = True  # Include chunks above/below for better context
    num_surrounding_chunks: int = 1  # Number of chunks to include on each side

    # Fallback behavior
    fallback_to_basic: bool = True  # Use basic chunking if context generation fails

    # Batch processing (for performance)
    batch_size: int = 10  # Generate contexts in batches
    max_workers: int = 5  # Parallel context generation

    def __post_init__(self):
        """Load API key from environment if not provided and validate."""
        import logging
        logger = logging.getLogger(__name__)

        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    logger.warning(
                        "ANTHROPIC_API_KEY not set in environment. "
                        "Contextual retrieval will fail unless API key is provided during initialization."
                    )
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning(
                        "OPENAI_API_KEY not set in environment. "
                        "Contextual retrieval will fail unless API key is provided during initialization."
                    )


@dataclass
class ChunkingConfig:
    """Configuration for chunking (PHASE 3)."""

    method: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 500
    chunk_overlap: int = 0

    # Chunking strategy
    enable_contextual: bool = True  # Contextual Retrieval (RECOMMENDED)
    enable_multi_layer: bool = True

    # Context generation config
    context_config: Optional["ContextGenerationConfig"] = None

    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", "; ", ", ", " ", ""])

    def __post_init__(self):
        """Initialize context_config if not provided."""
        if self.context_config is None and self.enable_contextual:
            self.context_config = ContextGenerationConfig()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation (PHASE 4)."""

    provider: str = "huggingface"  # 'voyage', 'openai', or 'huggingface'
    model: str = "bge-m3"
    batch_size: int = 32
    enable_multi_layer: bool = True


@dataclass
class PipelineConfig:
    """General pipeline configuration."""

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/pipeline.log"


@dataclass
class RAGConfig:
    """
    Unified RAG pipeline configuration.

    Combines all configuration with sensible defaults from research.
    """

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    models: ModelConfig = field(default_factory=ModelConfig.from_env)


def get_default_config() -> RAGConfig:
    """
    Get default RAG pipeline configuration.

    Uses sensible defaults from research + environment variables for API keys.

    Returns:
        RAGConfig instance with all default settings
    """
    return RAGConfig()


def get_model_config() -> ModelConfig:
    """
    Get model configuration from environment (legacy compatibility).

    Default models (optimized for M1 Mac):
    - LLM: Claude Sonnet 4.5 (balance of speed and quality)
    - Embeddings: BGE-M3-v2 (multilingual, runs locally on M1 with MPS acceleration)
    """
    return ModelConfig.from_env()


# Example usage
if __name__ == "__main__":
    # Load full pipeline config
    config = get_default_config()

    print("=== RAG Pipeline Configuration ===\n")

    print("PHASE 1: Extraction")
    print(f"  OCR: {config.extraction.enable_ocr}")
    print(f"  Smart Hierarchy: {config.extraction.enable_smart_hierarchy}")
    print(f"  Hierarchy Tolerance: {config.extraction.hierarchy_tolerance}")
    print(f"  Layout Model: {config.extraction.layout_model}")
    print()

    print("PHASE 2: Summarization")
    print(f"  Provider: {config.summarization.provider}")
    print(f"  Model: {config.summarization.model}")
    print(f"  Max Chars: {config.summarization.max_chars}")
    print()

    print("PHASE 3: Chunking")
    print(f"  Method: {config.chunking.method}")
    print(f"  Chunk Size: {config.chunking.chunk_size}")
    print(f"  Enable Contextual: {config.chunking.enable_contextual}")
    print()

    print("PHASE 4: Embedding")
    print(f"  Provider: {config.embedding.provider}")
    print(f"  Model: {config.embedding.model}")
    print()

    print("Models (from .env):")
    print(f"  LLM: {config.models.llm_provider}/{config.models.llm_model}")
    print(f"  Embedding: {config.models.embedding_provider}/{config.models.embedding_model}")
