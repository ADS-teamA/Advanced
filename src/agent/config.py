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

        # Apple Silicon with MPS support
        if system == "Darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon with MPS - using bge-m3 (local, GPU-accelerated)")
            return "bge-m3"

        # Linux with CUDA support
        if system == "Linux" and torch.cuda.is_available():
            logger.info("Detected Linux with CUDA - using bge-m3 (local, GPU-accelerated)")
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
    reranker_model: str = "ms-marco-mini"

    # Graph settings
    enable_graph_boost: bool = True
    graph_boost_weight: float = 0.3

    # Analysis settings
    max_document_compare: int = 3
    compliance_threshold: float = 0.7

    # Performance
    lazy_load_reranker: bool = True
    lazy_load_graph: bool = True
    cache_embeddings: bool = True


@dataclass
class HyDEConfig:
    """Hypothetical Document Embeddings configuration."""

    enable: bool = False
    num_hypothetical_docs: int = 1
    llm_model: str = "claude-haiku-4-5"
    max_tokens: int = 300
    temperature: float = 0.5


@dataclass
class QueryDecompositionConfig:
    """Multi-query decomposition configuration."""

    enable: bool = False
    max_sub_queries: int = 3
    combine_results: str = "rrf"  # "rrf", "concat", "ranked"
    llm_model: str = "claude-haiku-4-5"


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
    vector_store_path: Path = field(default_factory=lambda: Path("output/hybrid_store"))
    knowledge_graph_path: Optional[Path] = None

    # === Embedding Configuration ===
    # Platform-aware embedding model selection (can override via EMBEDDING_MODEL env var)
    embedding_model: str = field(default_factory=_detect_optimal_embedding_model)

    # === Feature Flags ===
    enable_hyde: bool = False
    enable_query_decomposition: bool = False
    enable_tool_validation: bool = True
    enable_knowledge_graph: bool = False

    # === Debug Mode ===
    debug_mode: bool = False

    # === Sub-Configs ===
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    hyde_config: HyDEConfig = field(default_factory=HyDEConfig)
    query_decomp_config: QueryDecompositionConfig = field(default_factory=QueryDecompositionConfig)
    cli_config: CLIConfig = field(default_factory=CLIConfig)

    # === System Prompt ===
    system_prompt: str = field(
        default_factory=lambda: """You are a RAG-powered legal and technical document assistant with access to specialized retrieval tools.

**Available Tools (17 total, organized by tier):**

TIER 1 - Basic Retrieval (fast, 100-300ms):
- simple_search: Hybrid search with reranking (use for most queries)
- entity_search: Find chunks mentioning specific entities
- document_search: Search within specific document(s)
- section_search: Search within document sections
- keyword_search: Pure BM25 keyword search
- get_document_list: List all indexed documents

TIER 2 - Advanced Retrieval (quality, 500-1000ms):
- multi_hop_search: Graph traversal for multi-hop queries
- compare_documents: Compare content across documents
- find_related_chunks: Find chunks related to a given chunk
- temporal_search: Search with date/time filters
- hybrid_search_with_filters: Search with metadata filters
- cross_reference_search: Find cross-references between documents

TIER 3 - Analysis & Insights (deep, 1-3s):
- explain_entity: Get entity details and relationships
- get_entity_relationships: Get all relationships for entity
- timeline_view: Extract temporal information from results
- summarize_section: Summarize a specific section
- get_statistics: Get corpus statistics

**Guidelines:**
1. ALWAYS cite sources using [Doc: X, Section: Y] format
2. Use simple_search for most queries (hybrid + rerank = best quality)
3. Use graph tools (TIER 3) for entity-centric questions
4. For complex queries, decompose into sub-tasks and use multiple tools
5. Be precise and factual - NEVER make up information
6. If you cannot answer confidently, say so explicitly and explain why
7. Start with fast tools (TIER 1), escalate to TIER 2/3 only if needed

**Best Effort Strategy:**
- Try multiple retrieval strategies before giving up
- If simple_search fails, try keyword_search or entity_search
- If single-doc search fails, try cross_reference_search
- Only report "no results" after exhausting relevant tools""".strip()
    )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set via environment variable or config.")

        if not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {self.vector_store_path}. "
                f"Run indexing pipeline first."
            )

        if self.enable_knowledge_graph and not self.knowledge_graph_path:
            raise ValueError(
                "Knowledge graph enabled but path not specified. "
                "Set knowledge_graph_path in config."
            )

    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":
        """
        Create config from environment variables with optional overrides.

        Environment variables:
        - ANTHROPIC_API_KEY: Required
        - AGENT_MODEL: Model to use (default: claude-sonnet-4-5-20250929)
        - VECTOR_STORE_PATH: Path to hybrid store
        - KNOWLEDGE_GRAPH_PATH: Path to KG JSON (optional)
        - ENABLE_HYDE: Enable HyDE (true/false)
        - ENABLE_DECOMPOSITION: Enable query decomposition (true/false)

        Args:
            **overrides: Override specific config values

        Returns:
            AgentConfig instance
        """
        config = cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "output/hybrid_store")),
            enable_hyde=os.getenv("ENABLE_HYDE", "false").lower() == "true",
            enable_query_decomposition=os.getenv("ENABLE_DECOMPOSITION", "false").lower() == "true",
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
