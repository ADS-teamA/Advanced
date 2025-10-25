"""
Agent Configuration

All agent settings are config-driven (no hardcoded values).
Supports environment variable overrides.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _load_agent_system_prompt() -> str:
    """Load agent system prompt from prompts/ directory."""
    try:
        from .prompt_loader import load_prompt

        return load_prompt("agent_system_prompt")
    except Exception as e:
        logger.error(f"Failed to load agent system prompt: {e}")
        # Fallback to minimal prompt
        return "You are a RAG-powered assistant for legal and technical documents."


def _detect_optimal_embedding_model() -> str:
    """
    Detect optimal embedding model based on platform.

    Cross-platform compatibility (per CLAUDE.md):
    - Apple Silicon (MPS): Use bge-m3 (local, FREE, GPU-accelerated)
    - Linux with NVIDIA GPU: Use bge-m3 (local, FREE, GPU-accelerated)
    - Windows or CPU-only: Use text-embedding-3-large (cloud, avoids PyTorch DLL issues)

    Can be overridden via EMBEDDING_MODEL environment variable.

    Returns:
        str: Optimal embedding model identifier
    """
    # Check environment variable override first
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        logger.info(f"Using EMBEDDING_MODEL from environment: {env_model}")
        return env_model

    system = platform.system()

    try:
        import torch

        # Check for GPU acceleration (Apple Silicon MPS or Linux CUDA)
        has_gpu = (
            system == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ) or (system == "Linux" and torch.cuda.is_available())

        if has_gpu:
            gpu_type = "MPS" if system == "Darwin" else "CUDA"
            logger.info(
                f"Detected {system} with {gpu_type} - using bge-m3 (local, GPU-accelerated)"
            )
            return "bge-m3"

    except ImportError:
        # PyTorch not available, fallback to cloud
        logger.info("PyTorch not available - using cloud embeddings")

    # Default: Windows or CPU-only
    logger.info(f"Platform: {system} (CPU) - using text-embedding-3-large (cloud)")
    return "text-embedding-3-large"


@dataclass
class ToolConfig:
    """Configuration for RAG tools."""

    # Retrieval settings
    default_k: int = 6
    enable_reranking: bool = True
    reranker_candidates: int = 50
    reranker_model: str = "bge-reranker-large"  # SOTA accuracy (was: ms-marco-mini)

    # Graph settings
    enable_graph_boost: bool = True
    graph_boost_weight: float = 0.3

    # Analysis settings
    max_document_compare: int = 3
    compliance_threshold: float = 0.7

    # Context expansion settings (for get_chunk_context tool)
    context_window: int = 2  # Number of chunks before/after for context expansion

    # Performance
    lazy_load_reranker: bool = False  # Load reranker at startup for better tool availability
    lazy_load_graph: bool = True
    cache_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.default_k <= 0:
            raise ValueError(f"default_k must be positive, got {self.default_k}")
        if self.reranker_candidates < self.default_k:
            raise ValueError(
                f"reranker_candidates ({self.reranker_candidates}) must be >= default_k ({self.default_k})"
            )
        if not 0.0 <= self.graph_boost_weight <= 1.0:
            raise ValueError(f"graph_boost_weight must be in [0, 1], got {self.graph_boost_weight}")
        if not 0.0 <= self.compliance_threshold <= 1.0:
            raise ValueError(
                f"compliance_threshold must be in [0, 1], got {self.compliance_threshold}"
            )
        if self.max_document_compare <= 0:
            raise ValueError(
                f"max_document_compare must be positive, got {self.max_document_compare}"
            )
        if self.context_window < 0:
            raise ValueError(f"context_window must be non-negative, got {self.context_window}")


@dataclass
class CLIConfig:
    """CLI-specific configuration."""

    # Display settings
    show_citations: bool = True
    citation_format: str = "inline"  # "inline", "footnote", "detailed"
    show_tool_calls: bool = True
    show_timing: bool = True

    # Streaming
    enable_streaming: bool = True

    # History
    save_history: bool = True
    history_file: Path = field(default_factory=lambda: Path(".agent_history"))
    max_history_items: int = 1000

    def __post_init__(self):
        """Validate configuration values."""
        valid_formats = ["inline", "footnote", "detailed", "simple"]
        if self.citation_format not in valid_formats:
            raise ValueError(
                f"citation_format must be one of {valid_formats}, got '{self.citation_format}'"
            )
        if self.max_history_items <= 0:
            raise ValueError(f"max_history_items must be positive, got {self.max_history_items}")


@dataclass
class AgentConfig:
    """
    Main agent configuration.

    All settings are configurable via environment variables or constructor.
    """

    # === Core Settings ===
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Model selection (user can override via CLI)
    model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929")
    )
    max_tokens: int = 4096
    temperature: float = 0.3

    # === Paths ===
    vector_store_path: Path = field(default_factory=lambda: Path("vector_db"))
    knowledge_graph_path: Optional[Path] = None

    # === Embedding Configuration ===
    # Platform-aware embedding model selection (can override via EMBEDDING_MODEL env var)
    embedding_model: str = field(default_factory=_detect_optimal_embedding_model)

    # === Feature Flags ===
    enable_tool_validation: bool = True
    enable_knowledge_graph: bool = False

    # === Prompt Caching (Anthropic) ===
    # Enable prompt caching to reduce costs by 90% and improve latency
    # Caches: system prompt, tools, and initialization messages
    enable_prompt_caching: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
    )

    # === Debug Mode ===
    debug_mode: bool = False

    # === Sub-Configs ===
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    cli_config: CLIConfig = field(default_factory=CLIConfig)

    # === System Prompt ===
    # Loaded from prompts/agent_system_prompt.txt
    system_prompt: str = field(default_factory=lambda: _load_agent_system_prompt())

    def validate(self) -> None:
        """
        Validate configuration.

        Checks:
        - API key presence and format
        - Numeric ranges (max_tokens, temperature)
        - Path existence
        - Model name validity
        - Sub-config validation (automatically via __post_init__)
        """
        # API key validation
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set via environment variable or config.")

        if not self.anthropic_api_key.startswith("sk-ant-"):
            raise ValueError("Anthropic API key has invalid format (should start with sk-ant-)")

        # Numeric range validation
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.max_tokens > 200000:  # Claude's context window limit
            raise ValueError(f"max_tokens exceeds maximum (200000), got {self.max_tokens}")

        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0.0, 1.0], got {self.temperature}")

        # Model name validation
        if "claude" not in self.model.lower():
            raise ValueError(f"Model name should contain 'claude': {self.model}")

        # Path validation
        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {self.vector_store_path}. "
                f"Run indexing pipeline first."
            )

        # Knowledge graph validation
        if self.enable_knowledge_graph and not self.knowledge_graph_path:
            raise ValueError(
                "Knowledge graph enabled but path not specified. "
                "Set knowledge_graph_path in config."
            )

        # Sub-configs are automatically validated via their __post_init__ methods

    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":
        """
        Create config from environment variables with optional overrides.

        Environment variables:
        - ANTHROPIC_API_KEY: Required
        - AGENT_MODEL: Model to use (default: claude-sonnet-4-5-20250929)
        - VECTOR_STORE_PATH: Path to hybrid store
        - KNOWLEDGE_GRAPH_PATH: Path to KG JSON (optional)

        Args:
            **overrides: Override specific config values

        Returns:
            AgentConfig instance
        """
        config = cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "vector_db")),
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Auto-detect KG if path exists
        kg_path_str = os.getenv("KNOWLEDGE_GRAPH_PATH")
        if kg_path_str:
            kg_path = Path(kg_path_str)
            if kg_path.exists():
                config.knowledge_graph_path = kg_path
                config.enable_knowledge_graph = True

        return config
