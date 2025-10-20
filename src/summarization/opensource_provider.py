"""
Open-source LLM provider placeholder.

Future implementation for local/open-source models (Mixtral, Llama, etc.).
"""

from src.summarization.llm_provider import LLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenSourceProvider(LLMProvider):
    """
    Placeholder for open-source LLM providers.

    Future support for:
    - Mixtral 8x7B (shown to outperform GPT-4 on legal tasks!)
    - Llama models
    - LEGAL-BERT for specialized legal tasks
    """

    def __init__(self, model: str = "mixtral-8x7b-instruct"):
        """
        Initialize open-source provider.

        Args:
            model: Model name
        """
        self.model = model
        logger.warning(
            f"OpenSourceProvider for {model} is not yet implemented. "
            "Use OpenAIProvider for now."
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """Not yet implemented."""
        raise NotImplementedError(
            "Open-source model support is not yet implemented. "
            "Future versions will support Mixtral 8x7B, Llama, and other "
            "open-source models via Ollama, vLLM, or HuggingFace."
        )

    def count_tokens(self, text: str) -> int:
        """Not yet implemented."""
        raise NotImplementedError(
            "Token counting for open-source models not yet implemented."
        )
