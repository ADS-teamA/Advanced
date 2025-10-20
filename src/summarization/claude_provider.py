"""
Anthropic Claude LLM provider implementation.

Uses Anthropic Claude API for text generation and summarization.
"""

from typing import Optional
import time
from anthropic import Anthropic, APIError, RateLimitError

from src.summarization.llm_provider import LLMProvider
from src.utils.errors import SummarizationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClaudeProvider(LLMProvider):
    """
    LLM provider using Anthropic Claude API.

    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and other Claude models.
    Includes automatic retry logic for rate limits.

    Model aliases (Anthropic official - auto-update to latest):
    - "claude-haiku-4-5" -> latest Haiku 4.5 (currently claude-haiku-4-5-20251001)
    - "claude-sonnet-4-5" -> latest Sonnet 4.5 (currently claude-sonnet-4-5-20250929)
    - "claude-opus-4-1" -> latest Opus 4.1 (currently claude-opus-4-1-20250805)
    - "claude-3-5-haiku-latest" -> latest 3.5 Haiku (currently claude-3-5-haiku-20241022)

    Custom shortcuts (for convenience):
    - "haiku" -> "claude-haiku-4-5" (latest Haiku 4.5)
    - "haiku-3.5" -> "claude-3-5-haiku-latest" (latest Haiku 3.5)
    - "sonnet" -> "claude-sonnet-4-5" (latest Sonnet 4.5)
    - "opus" -> "claude-opus-4-1" (latest Opus 4.1)
    """

    # Model name mapping
    # 1. Custom shortcuts -> Official Anthropic aliases (recommended for production)
    # 2. Official Anthropic aliases are passed through as-is
    MODEL_SHORTCUTS = {
        # Custom shortcuts -> Anthropic aliases (auto-update to latest)
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-5",
        "opus": "claude-opus-4-1",

        # Version-specific shortcuts
        "haiku-4.5": "claude-haiku-4-5",
        "haiku-3.5": "claude-3-5-haiku-latest",
        "sonnet-4.5": "claude-sonnet-4-5",
        "sonnet-4": "claude-sonnet-4-0",
        "sonnet-3.7": "claude-3-7-sonnet-latest",
        "opus-4.1": "claude-opus-4-1",
        "opus-4": "claude-opus-4-0",

        # Legacy mappings (for backward compatibility)
        "claude-3-5-haiku": "claude-3-5-haiku-latest",
        "claude-3-5-sonnet": "claude-sonnet-4-5",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key
            model: Model name (full name or shortcut like "haiku", "sonnet")
            max_retries: Maximum number of retries for rate limits
            retry_delay: Initial delay between retries (seconds)
        """
        self.client = Anthropic(api_key=api_key)

        # Resolve model shortcuts to full names
        self.model = self._resolve_model_name(model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized Claude provider with model: {self.model}")

    @classmethod
    def _resolve_model_name(cls, model: str) -> str:
        """
        Resolve model shortcuts to full Anthropic model names.

        Args:
            model: Model name or shortcut (e.g., "haiku", "claude-3-5-haiku-20241022")

        Returns:
            Full model name
        """
        # Return mapped name if it's a shortcut, otherwise return as-is
        return cls.MODEL_SHORTCUTS.get(model.lower(), model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """
        Generate text using Claude API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional Claude parameters

        Returns:
            Generated text

        Raises:
            SummarizationError: If generation fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Calling Claude API (attempt {attempt + 1}/{self.max_retries})"
                )

                # Claude API uses a different format than OpenAI
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    **kwargs
                )

                # Extract text from response
                generated_text = message.content[0].text.strip()

                logger.debug(
                    f"Generated {len(generated_text)} characters "
                    f"using {message.usage.input_tokens + message.usage.output_tokens} tokens"
                )

                return generated_text

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limit hit, retrying in {delay}s... (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                else:
                    raise SummarizationError(
                        f"Rate limit exceeded after {self.max_retries} retries"
                    ) from e

            except APIError as e:
                raise SummarizationError(
                    f"Claude API error: {e}"
                ) from e

            except Exception as e:
                raise SummarizationError(
                    f"Unexpected error calling Claude API: {e}"
                ) from e

        raise SummarizationError(
            f"Failed to generate text after {self.max_retries} attempts"
        )

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 characters per token for English.
        For precise counting, use Anthropic's token counting API.

        Args:
            text: Text to count

        Returns:
            Estimated number of tokens
        """
        # Simple estimation: ~4 chars per token
        # For exact counting, use Anthropic's count_tokens API
        return len(text) // 4
