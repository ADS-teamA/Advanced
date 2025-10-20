"""
Generic summary generation for legal documents.

Based on Reuter et al., 2024 (Summary-Augmented Chunking paper):
- Generic summaries OUTPERFORM expert-guided summaries (counterintuitive!)
- 150 chars optimal summary length
- ±20 tolerance acceptable
"""

from datetime import datetime
from src.core.models import Summary
from src.core.config import SummarizationConfig
from src.summarization.llm_provider import LLMProvider
from src.utils.errors import SummarizationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GenericSummarizer:
    """
    Generates generic summaries optimized for retrieval.

    Evidence from research:
    - Generic summaries achieve better balance between distinctiveness
      and broad semantic alignment compared to expert-guided summaries
    - 150 characters is optimal summary length
    - Used for Summary-Augmented Chunking (SAC) to prevent DRM
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: SummarizationConfig
    ):
        """
        Initialize summarizer.

        Args:
            llm_provider: LLM provider (OpenAI, open-source, etc.)
            config: Summarization configuration
        """
        self.llm = llm_provider
        self.config = config

        logger.info(
            f"Initialized GenericSummarizer with {config.model}, "
            f"target length: {config.max_chars}±{config.tolerance} chars"
        )

    def summarize(self, document_text: str) -> Summary:
        """
        Generate generic summary for document.

        Args:
            document_text: Full document text

        Returns:
            Summary object

        Raises:
            SummarizationError: If summarization fails
        """
        logger.info("Generating generic summary...")

        # Use first 5000 characters for summarization (sufficient per research)
        text_sample = document_text[:5000]

        # Track all attempts to pick best one if all exceed limit
        attempts = []
        target_max_chars = self.config.max_chars  # Use local variable, don't modify config

        # Try to generate summary with retries
        for attempt in range(self.config.max_retries):
            try:
                prompt = self._build_prompt(text_sample, target_max_chars)

                summary_text = self.llm.generate(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )

                # Validate length
                char_count = len(summary_text)
                attempts.append((summary_text, char_count))

                if char_count <= (self.config.max_chars + self.config.tolerance):
                    logger.info(
                        f"Generated summary: {char_count} chars "
                        f"(target: {self.config.max_chars}±{self.config.tolerance})"
                    )

                    return Summary(
                        text=summary_text,
                        char_count=char_count,
                        model=self.config.model,
                        generation_date=datetime.now()
                    )

                else:
                    # Summary too long, retry with tighter constraint
                    logger.warning(
                        f"Summary too long ({char_count} chars), "
                        f"retrying with tighter constraint (attempt {attempt + 1})"
                    )

                    # Reduce target by 10% for retry (don't modify config)
                    target_max_chars = int(target_max_chars * 0.9)

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Summarization attempt {attempt + 1} failed: {e}, retrying..."
                    )
                else:
                    raise SummarizationError(
                        f"Failed to generate summary after {self.config.max_retries} attempts"
                    ) from e

        # Final fallback: use best attempt (closest to target) if all exceeded limit
        if attempts:
            # Find attempt closest to original target
            best_summary, best_count = min(attempts, key=lambda x: abs(x[1] - self.config.max_chars))

            logger.warning(
                f"All attempts exceeded limit. Using best attempt: {best_count} chars "
                f"(target was {self.config.max_chars}±{self.config.tolerance})"
            )

            return Summary(
                text=best_summary,
                char_count=best_count,
                model=self.config.model,
                generation_date=datetime.now()
            )

        # Absolute fallback: should never reach here
        raise SummarizationError("Failed to generate any summary")

    def _build_prompt(self, text: str, max_chars: int) -> str:
        """
        Build generic summary prompt.

        IMPORTANT: Uses GENERIC style, NOT expert-guided.
        Research shows generic summaries perform better for retrieval.

        Args:
            text: Document text (first 5000 chars)
            max_chars: Maximum summary length

        Returns:
            Formatted prompt
        """
        # Enhanced generic prompt based on Reuter et al., 2024 recommendations
        prompt = f"""You are an expert legal document summarizer. Your task is to create an informative, concise summary.

REQUIREMENTS:
- Write a clear, informative description of what this document is about
- Include: key entities (organizations, parties, subjects), core purpose, main legal topics
- Maximum length: {max_chars} characters (strict limit)
- Style: Natural, flowing sentence(s) - NOT a title or header
- Focus: Provide semantic context for retrieval, not just identification

OUTPUT FORMAT:
Write 1-2 complete sentences describing the document's content and purpose.

EXAMPLE:
For an anti-corruption standard: "GRI 205 provides guidelines for organizations to report on corruption risks, anti-corruption policies, training programs, and confirmed incidents across operations."

Now summarize this document:

{text}

Summary (maximum {max_chars} characters):"""

        return prompt
