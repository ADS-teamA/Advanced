"""
Environment-based configuration for RAG pipeline.

Loads model selections and API keys from .env file.
Supports:
- Claude Haiku 4.5 / Sonnet 4.5 for LLM operations
- Kanon 2 Embedder / text-embedding-3-large / BGE-M3 for embeddings
"""

import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass


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

    # Embedding models (Voyage AI)
    "kanon-2": "kanon-2",
    "voyage-3": "voyage-3-large",
    "voyage-law-2": "voyage-law-2",

    # OpenAI embeddings
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small",

    # HuggingFace models
    "bge-m3": "BAAI/bge-m3",
    "bge-m3": "BAAI/bge-m3",
}


def resolve_model_alias(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model_name, model_name)


def get_default_config() -> ModelConfig:
    """
    Get default configuration from environment.

    Default models (optimized for M1 Mac):
    - LLM: Claude Sonnet 4.5 (balance of speed and quality)
    - Embeddings: BGE-M3-v2 (multilingual, runs locally on M1 with MPS acceleration)

    BGE-M3-v2 features:
    - 1024 dimensions
    - 100+ languages (including Czech)
    - 8192 token context
    - Apple Silicon optimized (MPS)
    - No API key required
    """
    return ModelConfig.from_env()


# Example usage
if __name__ == "__main__":
    config = get_default_config()

    print("Model Configuration:")
    print(f"  LLM: {config.llm_provider}/{config.llm_model}")
    print(f"  Embedding: {config.embedding_provider}/{config.embedding_model}")
    print()

    print("LLM Config:")
    print(config.get_llm_config())
    print()

    print("Embedding Config:")
    print(config.get_embedding_config())
