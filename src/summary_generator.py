"""
PHASE 2: Generic Summary Generation

Based on Reuter et al., 2024 (Summary-Augmented Chunking):
- Generic summaries OUTPERFORM expert-guided summaries
- Optimal length: 150 characters (±20 tolerance)
- Models: Claude Sonnet 4.5 (default), Claude Haiku 4.5, or gpt-4o-mini

Supports:
- Claude: claude-sonnet-4.5, claude-haiku-4.5 (via Anthropic API)
- OpenAI: gpt-4o-mini, gpt-4o (via OpenAI API)

Configuration is loaded from centralized config.py.
"""

import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import centralized configuration
try:
    from .config import SummarizationConfig, resolve_model_alias
    from .cost_tracker import get_global_tracker
except ImportError:
    from config import SummarizationConfig, resolve_model_alias
    from cost_tracker import get_global_tracker

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generate generic summaries for documents and sections.

    Based on research:
    - Reuter et al., 2024: Generic > Expert-guided for retrieval
    - Optimal length: configured in SummarizationConfig (default 150 chars)
    - Style: Broad semantic alignment, not overfitted to narrow features

    Configuration is centralized in config.py.
    """

    def __init__(
        self,
        config: Optional[SummarizationConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize summary generator.

        Args:
            config: SummarizationConfig instance (uses defaults if None)
            api_key: API key (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)
        """
        # Use provided config or create default
        self.config = config or SummarizationConfig()

        # Resolve model alias (e.g., "haiku" -> "claude-haiku-4-5-20251001")
        self.model = resolve_model_alias(self.config.model)

        # Extract config values for convenience
        self.max_chars = self.config.max_chars
        self.tolerance = self.config.tolerance
        self.max_workers = self.config.max_workers
        self.min_text_length = self.config.min_text_length
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        # Detect provider from model name
        if "claude" in self.model.lower():
            self.provider = "claude"
            self._init_claude(api_key)
        elif "gpt" in self.model.lower():
            self.provider = "openai"
            self._init_openai(api_key)
        else:
            raise ValueError(
                f"Unsupported model: {self.model}. "
                f"Supported: claude-sonnet-4.5, claude-haiku-4.5, gpt-4o-mini, gpt-4o"
            )

        logger.info(
            f"SummaryGenerator initialized: provider={self.provider}, "
            f"model={self.model}, max_chars={self.max_chars}"
        )

    def _init_claude(self, api_key: Optional[str]):
        """Initialize Anthropic Claude client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Claude models. "
                "Install with: uv pip install anthropic"
            )

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required for Claude models. "
                "Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        self.client = Anthropic(api_key=api_key)
        logger.info("Claude client initialized")

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI models. "
                "Install with: uv pip install openai"
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI models. "
                "Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")

    def generate_document_summary(
        self,
        document_text: str = None,
        section_summaries: list[str] = None
    ) -> str:
        """
        Generate generic document-level summary.

        Two modes:
        1. Hierarchical (RECOMMENDED): Summarize section summaries
           - Works for documents of ANY size
           - No truncation, uses all content
           - Better coverage and accuracy

        2. Direct (FALLBACK): Summarize document text
           - Limited to first 5000 chars
           - May miss important content
           - Use only if section summaries unavailable

        Args:
            document_text: Full document text (fallback mode)
            section_summaries: List of section summaries (hierarchical mode)

        Returns:
            Generic summary (~150 chars)
        """

        # Mode 1: Hierarchical summarization (RECOMMENDED)
        if section_summaries:
            # Filter out empty summaries
            valid_summaries = [s for s in section_summaries if s and s.strip()]

            if valid_summaries:
                logger.info(f"Using hierarchical summarization ({len(valid_summaries)} section summaries)")
                return self._generate_from_section_summaries(valid_summaries)

        # Mode 2: Direct summarization (FALLBACK)
        if document_text:
            logger.warning("Using direct summarization (limited to first 5000 chars)")
            # Truncate to first 5000 chars for efficiency
            text_preview = document_text[:30000]

            prompt = f"""You are an expert document summarizer.

Summarize the following document text. Focus on extracting the most important entities,
core purpose, and key topics.

CRITICAL: The summary MUST be EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
This is a hard constraint. The summary should be optimized for providing context to smaller text chunks.

Output only the summary text, nothing else.

Document:
{text_preview}

Summary (max {self.max_chars} characters):"""

            try:
                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt)
                else:  # openai
                    summary = self._generate_with_openai(prompt)

                # Check length and retry if too long
                if len(summary) > self.max_chars + self.tolerance:
                    logger.warning(
                        f"Summary too long ({len(summary)} chars), regenerating with stricter limit"
                    )
                    return self.generate_document_summary_strict(text_preview)

                logger.debug(f"Generated document summary: {len(summary)} chars")
                return summary

            except Exception as e:
                logger.error(f"Failed to generate document summary: {e}")
                # Fallback: simple truncation
                return document_text[:self.max_chars].strip() + "..."

        # No input provided
        raise ValueError("Either document_text or section_summaries must be provided")

    def _generate_from_section_summaries(self, section_summaries: list[str]) -> str:
        """
        Generate document summary from section summaries (hierarchical approach).

        This is the RECOMMENDED method for large documents:
        - No truncation needed
        - Covers entire document through section summaries
        - More accurate and comprehensive

        Args:
            section_summaries: List of section-level summaries

        Returns:
            Document-level summary
        """

        # Combine section summaries
        combined_text = "\n".join(f"- {s}" for s in section_summaries)

        prompt = f"""You are an expert document summarizer.

You are given summaries of different sections from a document.
Create a unified document-level summary that captures the main theme and purpose.

CRITICAL: The summary MUST be EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
This is a hard constraint. The summary should be optimized for providing global context to text chunks.

Output only the summary text, nothing else.

Section summaries:
{combined_text}

Document summary (max {self.max_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            # Check length and retry if too long
            if len(summary) > self.max_chars + self.tolerance:
                logger.warning(
                    f"Summary too long ({len(summary)} chars), regenerating with stricter limit"
                )
                # Retry with stricter limit
                target_chars = int(self.max_chars * 0.9)
                prompt_strict = f"""CRITICAL: Summarize in EXACTLY {target_chars} characters or LESS.

{combined_text}

Summary (STRICT LIMIT: {target_chars} characters):"""

                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt_strict)
                else:
                    summary = self._generate_with_openai(prompt_strict)

                summary = summary[:self.max_chars]  # Hard truncate

            logger.debug(f"Generated hierarchical document summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate hierarchical summary: {e}")
            # Fallback: combine first chars of each section summary
            fallback = " ".join(section_summaries)
            return fallback[:self.max_chars].strip() + "..."

    def _generate_with_claude(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using Claude API.

        Uses temperature and max_tokens from config unless overridden.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track cost
        self.tracker.track_llm(
            provider="anthropic",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            operation="summary"
        )

        return response.content[0].text.strip()

    def _generate_with_openai(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text using OpenAI API.

        Uses temperature and max_tokens from config unless overridden.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        # Track cost
        self.tracker.track_llm(
            provider="openai",
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            operation="summary"
        )

        return response.choices[0].message.content.strip()

    def generate_document_summary_strict(self, document_text: str) -> str:
        """
        Generate summary with stricter length constraint.

        Called when first attempt exceeds max_chars.
        """

        text_preview = document_text[:5000]
        target_chars = int(self.max_chars * 0.9)  # Aim for 90% of max

        prompt = f"""CRITICAL: Summarize this document in EXACTLY {target_chars} characters or LESS.
Be extremely concise.

Document:
{text_preview}

Summary (STRICT LIMIT: {target_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            return summary[:self.max_chars]  # Hard truncate if needed

        except Exception as e:
            logger.error(f"Failed to generate strict summary: {e}")
            return document_text[:self.max_chars].strip() + "..."

    def generate_section_summary(
        self,
        section_text: str,
        section_title: str = ""
    ) -> str:
        """
        Generate generic section-level summary.

        Args:
            section_text: Section content
            section_title: Section title (for context)

        Returns:
            Generic summary (~150 chars)
        """

        # For sections, use shorter preview
        text_preview = section_text[:2000]

        title_context = f"Section title: {section_title}\n\n" if section_title else ""

        prompt = f"""Summarize this document section concisely.

{title_context}Section content:
{text_preview}

CRITICAL: Provide a summary that is EXACTLY {self.max_chars} characters or less (current limit: {self.max_chars} chars).
Focus on the main topic and key information. This is a hard constraint.

Summary (max {self.max_chars} characters):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt)
            else:  # openai
                summary = self._generate_with_openai(prompt)

            # Check length
            if len(summary) > self.max_chars + self.tolerance:
                summary = summary[:self.max_chars]

            logger.debug(f"Generated section summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate section summary: {e}")
            # Fallback
            return section_text[:self.max_chars].strip() + "..."

    def generate_batch_summaries(
        self,
        texts: list[tuple[str, str]]  # [(text, title), ...]
    ) -> list[str]:
        """
        Generate summaries for multiple sections in parallel.

        Uses ThreadPoolExecutor for concurrent API requests (much faster than sequential).

        Args:
            texts: List of (text, title) tuples

        Returns:
            List of summaries (in original order)
        """

        # Filter out tiny sections
        filtered_texts = []
        skip_indices = []
        for i, (text, title) in enumerate(texts):
            if len(text.strip()) < self.min_text_length:
                skip_indices.append(i)
                logger.debug(f"Skipping tiny section '{title}' ({len(text)} chars)")
            else:
                filtered_texts.append((i, text, title))

        if not filtered_texts:
            logger.warning("All sections too small, skipping summary generation")
            return [""] * len(texts)

        # Generate summaries in parallel
        summaries_map = {}

        def generate_one(idx: int, text: str, title: str) -> tuple[int, str]:
            try:
                summary = self.generate_section_summary(text, title)
                return (idx, summary)
            except Exception as e:
                logger.error(f"Failed to generate summary for '{title}': {e}")
                return (idx, text[:self.max_chars].strip() + "...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(generate_one, idx, text, title)
                for idx, text, title in filtered_texts
            ]

            for future in as_completed(futures):
                idx, summary = future.result()
                summaries_map[idx] = summary

        # Build result list in original order
        result = []
        for i in range(len(texts)):
            if i in skip_indices:
                result.append("")  # Empty summary for tiny sections
            else:
                result.append(summaries_map[i])

        logger.info(f"Generated {len(summaries_map)} summaries in parallel (skipped {len(skip_indices)} tiny sections)")
        return result


# Example usage
if __name__ == "__main__":
    # Test
    generator = SummaryGenerator()

    test_doc = """
    This is a technical specification document for a nuclear power plant reactor core.
    It describes the primary cooling system, safety features, and operational parameters.
    The document covers emergency shutdown procedures, redundancy systems, and regulatory
    compliance requirements for the VVER-1200 reactor type.
    """

    summary = generator.generate_document_summary(test_doc)
    print(f"Summary ({len(summary)} chars): {summary}")
