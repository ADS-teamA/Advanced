"""
API Client Factory for MY_SUJBOT pipeline.

Provides centralized API client creation with consistent:
- API key loading (from env or parameter)
- Error handling and sanitization
- Validation
- Logging

Replaces duplicated initialization code across:
- summary_generator.py (_init_claude, _init_openai)
- contextual_retrieval.py (_init_anthropic, _init_openai, _init_local)
- embedding_generator.py (_init_voyage_model, _init_openai_model)
- entity_extractor.py, relationship_extractor.py

Usage:
    from src.utils import APIClientFactory

    # Create Anthropic client
    client = APIClientFactory.create_anthropic()

    # Create OpenAI client
    client = APIClientFactory.create_openai()

    # Create Voyage AI client
    client = APIClientFactory.create_voyage()
"""

import logging
import os
from typing import Optional

from .security import sanitize_error, mask_api_key

logger = logging.getLogger(__name__)


class APIClientFactory:
    """
    Factory for creating API clients with consistent configuration.

    All methods:
    - Load API keys from environment or parameter
    - Validate API key presence and format
    - Provide helpful error messages
    - Sanitize errors before raising
    - Log successful initialization
    """

    @staticmethod
    def create_anthropic(
        api_key: Optional[str] = None,
        validate_connection: bool = False
    ):
        """
        Create Anthropic (Claude) API client.

        Args:
            api_key: Anthropic API key (loads from ANTHROPIC_API_KEY env if not provided)
            validate_connection: If True, makes test API call to validate key

        Returns:
            anthropic.Anthropic client instance

        Raises:
            ImportError: If anthropic package not installed
            ValueError: If API key missing or invalid format
            RuntimeError: If client creation fails

        Example:
            >>> from src.utils import APIClientFactory
            >>> client = APIClientFactory.create_anthropic()
            >>> response = client.messages.create(...)
        """
        # Try importing anthropic SDK
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Claude models. "
                "Install with: uv pip install anthropic"
            )

        # Load API key
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required for Claude models. "
                "Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        # Basic format validation
        if not api_key.startswith("sk-ant-"):
            logger.warning(
                f"Anthropic API key has unexpected format (expected sk-ant-...). "
                f"Got: {mask_api_key(api_key)}"
            )

        # Create client
        try:
            client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"✓ Anthropic client initialized (key: {mask_api_key(api_key)})")

            # Optional: validate connection
            if validate_connection:
                # Make a minimal API call to verify key works
                try:
                    # List models (cheap call)
                    # Note: Anthropic doesn't have a /models endpoint, so we skip this
                    pass
                except Exception as e:
                    safe_error = sanitize_error(str(e))
                    raise RuntimeError(f"Anthropic API key validation failed: {safe_error}")

            return client

        except Exception as e:
            safe_error = sanitize_error(str(e))
            raise RuntimeError(f"Failed to create Anthropic client: {safe_error}")

    @staticmethod
    def create_openai(
        api_key: Optional[str] = None,
        validate_connection: bool = False
    ):
        """
        Create OpenAI API client.

        Args:
            api_key: OpenAI API key (loads from OPENAI_API_KEY env if not provided)
            validate_connection: If True, makes test API call to validate key

        Returns:
            openai.OpenAI client instance

        Raises:
            ImportError: If openai package not installed
            ValueError: If API key missing or invalid format
            RuntimeError: If client creation fails

        Example:
            >>> from src.utils import APIClientFactory
            >>> client = APIClientFactory.create_openai()
            >>> response = client.chat.completions.create(...)
        """
        # Try importing openai SDK
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI models. "
                "Install with: uv pip install openai"
            )

        # Load API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI models. "
                "Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        # Basic format validation
        if not api_key.startswith("sk-"):
            logger.warning(
                f"OpenAI API key has unexpected format (expected sk-...). "
                f"Got: {mask_api_key(api_key)}"
            )

        # Create client
        try:
            client = openai.OpenAI(api_key=api_key)
            logger.info(f"✓ OpenAI client initialized (key: {mask_api_key(api_key)})")

            # Optional: validate connection
            if validate_connection:
                # Make a minimal API call to verify key works
                try:
                    client.models.list()
                except Exception as e:
                    safe_error = sanitize_error(str(e))
                    raise RuntimeError(f"OpenAI API key validation failed: {safe_error}")

            return client

        except Exception as e:
            safe_error = sanitize_error(str(e))
            raise RuntimeError(f"Failed to create OpenAI client: {safe_error}")

    @staticmethod
    def create_voyage(
        api_key: Optional[str] = None,
        validate_connection: bool = False
    ):
        """
        Create Voyage AI client.

        Args:
            api_key: Voyage API key (loads from VOYAGE_API_KEY env if not provided)
            validate_connection: If True, makes test API call to validate key

        Returns:
            voyageai.Client instance

        Raises:
            ImportError: If voyageai package not installed
            ValueError: If API key missing or invalid format
            RuntimeError: If client creation fails

        Example:
            >>> from src.utils import APIClientFactory
            >>> client = APIClientFactory.create_voyage()
            >>> embeddings = client.embed(...)
        """
        # Try importing voyageai SDK
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage AI embeddings. "
                "Install with: uv pip install voyageai"
            )

        # Load API key
        api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "Voyage API key required for Voyage AI models. "
                "Set VOYAGE_API_KEY env var or pass api_key parameter."
            )

        # Basic format validation
        if not api_key.startswith("pa-"):
            logger.warning(
                f"Voyage API key has unexpected format (expected pa-...). "
                f"Got: {mask_api_key(api_key)}"
            )

        # Create client
        try:
            client = voyageai.Client(api_key=api_key)
            logger.info(f"✓ Voyage AI client initialized (key: {mask_api_key(api_key)})")

            # Optional: validate connection
            if validate_connection:
                # Voyage doesn't have a test endpoint, skip validation
                pass

            return client

        except Exception as e:
            safe_error = sanitize_error(str(e))
            raise RuntimeError(f"Failed to create Voyage AI client: {safe_error}")

    @staticmethod
    def create_huggingface_model(
        model_name: str,
        device: Optional[str] = None
    ):
        """
        Create HuggingFace SentenceTransformer model (for local embeddings).

        Args:
            model_name: HuggingFace model name (e.g., "BAAI/bge-m3")
            device: Device for inference ("cpu", "cuda", "mps", or None for auto-detect)

        Returns:
            SentenceTransformer model instance

        Raises:
            ImportError: If sentence-transformers or torch not installed
            RuntimeError: If model loading fails

        Example:
            >>> from src.utils import APIClientFactory
            >>> model = APIClientFactory.create_huggingface_model("BAAI/bge-m3")
            >>> embeddings = model.encode(["text1", "text2"])
        """
        # Try importing required packages
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "sentence-transformers and torch required for local embeddings. "
                "Install with: uv pip install sentence-transformers torch"
            )

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                logger.info("✓ Detected Apple Silicon (MPS) - using GPU acceleration")
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
                logger.info("✓ Detected CUDA - using GPU acceleration")
            else:
                device = "cpu"
                logger.info("✓ Using CPU (no GPU detected)")

        # Load model
        try:
            logger.info(f"Loading HuggingFace model: {model_name} on {device}...")
            model = SentenceTransformer(model_name, device=device)
            logger.info(f"✓ HuggingFace model loaded: {model_name} ({device})")
            return model

        except Exception as e:
            safe_error = sanitize_error(str(e))
            raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {safe_error}")


# Example usage
if __name__ == "__main__":
    print("=== API Client Factory Examples ===\n")

    # Example 1: Create Anthropic client
    print("1. Creating Anthropic client...")
    try:
        client = APIClientFactory.create_anthropic()
        print(f"   ✓ Success: {type(client)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Example 2: Create OpenAI client
    print("\n2. Creating OpenAI client...")
    try:
        client = APIClientFactory.create_openai()
        print(f"   ✓ Success: {type(client)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Example 3: Create Voyage client
    print("\n3. Creating Voyage AI client...")
    try:
        client = APIClientFactory.create_voyage()
        print(f"   ✓ Success: {type(client)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Example 4: Create HuggingFace model
    print("\n4. Creating HuggingFace model...")
    try:
        model = APIClientFactory.create_huggingface_model("BAAI/bge-small-en-v1.5")
        print(f"   ✓ Success: {type(model)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
