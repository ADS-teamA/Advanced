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
class LLMTaskConfig:
    """
    Configuration for a specific LLM task.

    Allows fine-grained control over which model/provider to use for each task
    in the RAG pipeline (summaries, entity extraction, relationship extraction, etc.).
    """

    provider: str  # 'claude', 'openai', 'anthropic'
    model: str  # Model name or alias
    temperature: float = 0.3
    max_tokens: int = 500
    api_key: Optional[str] = None

    def __post_init__(self):
        """Resolve model alias and load API key from environment if not provided."""
        # Resolve model alias
        self.model = resolve_model_alias(self.model)

        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider in ["claude", "anthropic"]:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")

    def to_dict(self) -> dict:
        """Convert to dictionary for module consumption."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key
        }


@dataclass
class LLMTasksConfig:
    """
    Centralized configuration for all LLM tasks in the RAG pipeline.

    Allows specifying different models/providers for each task:
    - summary: Document and section summarization (PHASE 2)
    - context: Contextual chunk augmentation (PHASE 3)
    - entity_extraction: Knowledge graph entity extraction (PHASE 5A)
    - relationship_extraction: Knowledge graph relationship extraction (PHASE 5A)
    - agent: RAG agent responses (PHASE 7)
    - query_decomposition: Complex query decomposition (PHASE 7)
    - hyde: Hypothetical document embeddings (PHASE 7)

    Example:
        # Use different models for different tasks
        llm_tasks = LLMTasksConfig(
            summary=LLMTaskConfig(provider="claude", model="haiku"),
            context=LLMTaskConfig(provider="claude", model="haiku"),
            entity_extraction=LLMTaskConfig(provider="openai", model="gpt-4o"),
            relationship_extraction=LLMTaskConfig(provider="openai", model="gpt-4o-mini"),
            agent=LLMTaskConfig(provider="claude", model="sonnet"),
        )
    """

    summary: Optional[LLMTaskConfig] = None
    context: Optional[LLMTaskConfig] = None
    entity_extraction: Optional[LLMTaskConfig] = None
    relationship_extraction: Optional[LLMTaskConfig] = None
    agent: Optional[LLMTaskConfig] = None
    query_decomposition: Optional[LLMTaskConfig] = None
    hyde: Optional[LLMTaskConfig] = None

    def __post_init__(self):
        """Initialize missing configs with sensible defaults."""
        # Summary generation: Fast and cheap (Haiku or GPT-4o-mini)
        if self.summary is None:
            self.summary = LLMTaskConfig(
                provider="claude",
                model="haiku",
                temperature=0.3,
                max_tokens=500
            )

        # Context generation: Fast and cheap (same as summary)
        if self.context is None:
            self.context = LLMTaskConfig(
                provider="claude",
                model="haiku",
                temperature=0.3,
                max_tokens=150
            )

        # Entity extraction: Deterministic, can use faster models
        if self.entity_extraction is None:
            self.entity_extraction = LLMTaskConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=500
            )

        # Relationship extraction: Deterministic, can use faster models
        if self.relationship_extraction is None:
            self.relationship_extraction = LLMTaskConfig(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=500
            )

        # Agent: More capable model for complex reasoning (Sonnet)
        if self.agent is None:
            self.agent = LLMTaskConfig(
                provider="claude",
                model="sonnet",
                temperature=0.3,
                max_tokens=4096
            )

        # Query decomposition: Can use same as agent or faster
        if self.query_decomposition is None:
            self.query_decomposition = LLMTaskConfig(
                provider="claude",
                model="haiku",
                temperature=0.3,
                max_tokens=500
            )

        # HyDE: Hypothetical document generation, can use faster models
        if self.hyde is None:
            self.hyde = LLMTaskConfig(
                provider="claude",
                model="haiku",
                temperature=0.5,
                max_tokens=500
            )

    @classmethod
    def from_env(cls) -> "LLMTasksConfig":
        """
        Load task-specific LLM configs from environment variables.

        Environment variables (all optional, defaults to sensible choices):
            SUMMARY_LLM_PROVIDER, SUMMARY_LLM_MODEL
            CONTEXT_LLM_PROVIDER, CONTEXT_LLM_MODEL
            ENTITY_LLM_PROVIDER, ENTITY_LLM_MODEL
            RELATIONSHIP_LLM_PROVIDER, RELATIONSHIP_LLM_MODEL
            AGENT_LLM_PROVIDER, AGENT_LLM_MODEL
            QUERY_DECOMP_LLM_PROVIDER, QUERY_DECOMP_LLM_MODEL
            HYDE_LLM_PROVIDER, HYDE_LLM_MODEL
        """
        summary = None
        if os.getenv("SUMMARY_LLM_PROVIDER") or os.getenv("SUMMARY_LLM_MODEL"):
            summary = LLMTaskConfig(
                provider=os.getenv("SUMMARY_LLM_PROVIDER", "claude"),
                model=os.getenv("SUMMARY_LLM_MODEL", "haiku"),
                temperature=float(os.getenv("SUMMARY_LLM_TEMP", "0.3")),
                max_tokens=int(os.getenv("SUMMARY_LLM_MAX_TOKENS", "500"))
            )

        context = None
        if os.getenv("CONTEXT_LLM_PROVIDER") or os.getenv("CONTEXT_LLM_MODEL"):
            context = LLMTaskConfig(
                provider=os.getenv("CONTEXT_LLM_PROVIDER", "claude"),
                model=os.getenv("CONTEXT_LLM_MODEL", "haiku"),
                temperature=float(os.getenv("CONTEXT_LLM_TEMP", "0.3")),
                max_tokens=int(os.getenv("CONTEXT_LLM_MAX_TOKENS", "150"))
            )

        entity = None
        if os.getenv("ENTITY_LLM_PROVIDER") or os.getenv("ENTITY_LLM_MODEL"):
            entity = LLMTaskConfig(
                provider=os.getenv("ENTITY_LLM_PROVIDER", "openai"),
                model=os.getenv("ENTITY_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("ENTITY_LLM_TEMP", "0.0")),
                max_tokens=int(os.getenv("ENTITY_LLM_MAX_TOKENS", "500"))
            )

        relationship = None
        if os.getenv("RELATIONSHIP_LLM_PROVIDER") or os.getenv("RELATIONSHIP_LLM_MODEL"):
            relationship = LLMTaskConfig(
                provider=os.getenv("RELATIONSHIP_LLM_PROVIDER", "openai"),
                model=os.getenv("RELATIONSHIP_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("RELATIONSHIP_LLM_TEMP", "0.0")),
                max_tokens=int(os.getenv("RELATIONSHIP_LLM_MAX_TOKENS", "500"))
            )

        agent = None
        if os.getenv("AGENT_LLM_PROVIDER") or os.getenv("AGENT_LLM_MODEL"):
            agent = LLMTaskConfig(
                provider=os.getenv("AGENT_LLM_PROVIDER", "claude"),
                model=os.getenv("AGENT_LLM_MODEL", "sonnet"),
                temperature=float(os.getenv("AGENT_LLM_TEMP", "0.3")),
                max_tokens=int(os.getenv("AGENT_LLM_MAX_TOKENS", "4096"))
            )

        query_decomp = None
        if os.getenv("QUERY_DECOMP_LLM_PROVIDER") or os.getenv("QUERY_DECOMP_LLM_MODEL"):
            query_decomp = LLMTaskConfig(
                provider=os.getenv("QUERY_DECOMP_LLM_PROVIDER", "claude"),
                model=os.getenv("QUERY_DECOMP_LLM_MODEL", "haiku"),
                temperature=float(os.getenv("QUERY_DECOMP_LLM_TEMP", "0.3")),
                max_tokens=int(os.getenv("QUERY_DECOMP_LLM_MAX_TOKENS", "500"))
            )

        hyde = None
        if os.getenv("HYDE_LLM_PROVIDER") or os.getenv("HYDE_LLM_MODEL"):
            hyde = LLMTaskConfig(
                provider=os.getenv("HYDE_LLM_PROVIDER", "claude"),
                model=os.getenv("HYDE_LLM_MODEL", "haiku"),
                temperature=float(os.getenv("HYDE_LLM_TEMP", "0.5")),
                max_tokens=int(os.getenv("HYDE_LLM_MAX_TOKENS", "500"))
            )

        return cls(
            summary=summary,
            context=context,
            entity_extraction=entity,
            relationship_extraction=relationship,
            agent=agent,
            query_decomposition=query_decomp,
            hyde=hyde
        )


@dataclass
class SummarizationConfig:
    """Configuration for summarization (PHASE 2)."""

    # LLM configuration
    llm_config: Optional[LLMTaskConfig] = None

    # Task-specific parameters
    max_chars: int = 150
    tolerance: int = 20
    style: str = "generic"
    retry_on_exceed: bool = True
    max_retries: int = 3
    max_workers: int = 10
    min_text_length: int = 50

    # Legacy compatibility (deprecated)
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        """Initialize LLM config from legacy fields if provided."""
        if self.llm_config is None:
            # Use legacy fields or defaults
            self.llm_config = LLMTaskConfig(
                provider=self.provider or "claude",
                model=self.model or "haiku",
                temperature=self.temperature or 0.3,
                max_tokens=self.max_tokens or 500
            )


@dataclass
class ContextGenerationConfig:
    """
    Configuration for Contextual Retrieval (Anthropic, Sept 2024).

    Generates LLM-based context for each chunk instead of generic summaries.
    Results in 67% reduction in retrieval failures (Anthropic research).
    """

    # LLM configuration
    llm_config: Optional[LLMTaskConfig] = None

    # Enable contextual retrieval
    enable_contextual: bool = True

    # Context window params
    include_surrounding_chunks: bool = True  # Include chunks above/below for better context
    num_surrounding_chunks: int = 1  # Number of chunks to include on each side

    # Fallback behavior
    fallback_to_basic: bool = True  # Use basic chunking if context generation fails

    # Batch processing (for performance)
    batch_size: int = 10  # Generate contexts in batches
    max_workers: int = 5  # Parallel context generation

    # Legacy compatibility (deprecated)
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        """Initialize LLM config from legacy fields if provided."""
        import logging
        logger = logging.getLogger(__name__)

        if self.llm_config is None:
            # Use legacy fields or defaults
            self.llm_config = LLMTaskConfig(
                provider=self.provider or "anthropic",
                model=self.model or "haiku",
                temperature=self.temperature or 0.3,
                max_tokens=self.max_tokens or 150  # Context should be 50-100 words
            )

            # Warn if API key is missing
            if not self.llm_config.api_key:
                logger.warning(
                    f"{self.llm_config.provider.upper()}_API_KEY not set in environment. "
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
