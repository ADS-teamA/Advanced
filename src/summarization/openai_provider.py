"""
OpenAI LLM provider implementation.

Uses OpenAI API for text generation and summarization.
"""

from typing import Optional
import time
from openai import OpenAI, OpenAIError, RateLimitError

from src.summarization.llm_provider import LLMProvider
from src.utils.errors import SummarizationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    LLM provider using OpenAI API.

    Supports GPT-4, GPT-4o-mini, and other OpenAI models.
    Includes automatic retry logic for rate limits.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini", "gpt-4")
            max_retries: Maximum number of retries for rate limits
            retry_delay: Initial delay between retries (seconds)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized OpenAI provider with model: {model}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            Generated text

        Raises:
            SummarizationError: If generation fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Calling OpenAI API (attempt {attempt + 1}/{self.max_retries})"
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

                generated_text = response.choices[0].message.content.strip()

                logger.debug(
                    f"Generated {len(generated_text)} characters "
                    f"using {response.usage.total_tokens} tokens"
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

            except OpenAIError as e:
                raise SummarizationError(
                    f"OpenAI API error: {e}"
                ) from e

            except Exception as e:
                raise SummarizationError(
                    f"Unexpected error calling OpenAI API: {e}"
                ) from e

        raise SummarizationError(
            f"Failed to generate text after {self.max_retries} attempts"
        )

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple heuristic: ~4 characters per token for English.
        For precise counting, would need tiktoken library.

        Args:
            text: Text to count

        Returns:
            Estimated number of tokens
        """
        # Simple estimation: ~4 chars per token
        # For exact counting, use: import tiktoken
        return len(text) // 4
