"""
PHASE 2: Generic Summary Generation

Based on Reuter et al., 2024 (Summary-Augmented Chunking):
- Generic summaries OUTPERFORM expert-guided summaries
- Optimal length: 150 characters (±20 tolerance)
- Models: Claude Sonnet 4.5 (default), Claude Haiku 4.5, or gpt-4o-mini

Supports:
- Claude: claude-sonnet-4.5, claude-haiku-4.5 (via Anthropic API)
- OpenAI: gpt-4o-mini, gpt-4o (via OpenAI API)
"""

import logging
import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generate generic summaries for documents and sections.

    Based on research:
    - Reuter et al., 2024: Generic > Expert-guided for retrieval
    - Optimal length: 150 chars
    - Style: Broad semantic alignment, not overfitted to narrow features
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        max_chars: int = 150,
        tolerance: int = 20,
        api_key: Optional[str] = None,
        max_workers: int = 10,  # Parallel requests
        min_text_length: int = 50  # Skip summaries for tiny sections
    ):
        """
        Initialize summary generator.

        Args:
            model: Model to use. Options:
                   - Claude: "claude-sonnet-4.5" (default), "claude-haiku-4.5"
                   - OpenAI: "gpt-4o-mini", "gpt-4o"
            max_chars: Maximum summary length in characters
            tolerance: Acceptable overage (±tolerance)
            api_key: API key (or set ANTHROPIC_API_KEY/OPENAI_API_KEY env var)
        """
        self.model = model
        self.max_chars = max_chars
        self.tolerance = tolerance
        self.max_workers = max_workers
        self.min_text_length = min_text_length

        # Detect provider from model name
        if "claude" in model.lower():
            self.provider = "claude"
            self._init_claude(api_key)
        elif "gpt" in model.lower():
            self.provider = "openai"
            self._init_openai(api_key)
        else:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported: claude-sonnet-4.5, claude-haiku-4.5, gpt-4o-mini, gpt-4o"
            )

        logger.info(f"SummaryGenerator initialized: provider={self.provider}, model={model}, max_chars={max_chars}")

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
            text_preview = document_text[:5000]

            prompt = f"""You are an expert document summarizer.

Summarize the following document text. Focus on extracting the most important entities,
core purpose, and key topics.

The summary must be concise, maximum {self.max_chars} characters long, and optimized
for providing context to smaller text chunks.

Output only the summary text, nothing else.

Document:
{text_preview}

Summary:"""

            try:
                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt, max_tokens=500)
                else:  # openai
                    summary = self._generate_with_openai(prompt, max_tokens=500)

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

The summary must be concise, maximum {self.max_chars} characters long, and optimized
for providing global context to text chunks.

Output only the summary text, nothing else.

Section summaries:
{combined_text}

Document summary:"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt, max_tokens=500)
            else:  # openai
                summary = self._generate_with_openai(prompt, max_tokens=500)

            # Check length and retry if too long
            if len(summary) > self.max_chars + self.tolerance:
                logger.warning(
                    f"Summary too long ({len(summary)} chars), regenerating with stricter limit"
                )
                # Retry with stricter limit
                target_chars = int(self.max_chars * 0.9)
                prompt_strict = f"""Summarize in EXACTLY {target_chars} characters or less:

{combined_text}

Summary ({target_chars} chars max):"""

                if self.provider == "claude":
                    summary = self._generate_with_claude(prompt_strict, max_tokens=500)
                else:
                    summary = self._generate_with_openai(prompt_strict, max_tokens=500)

                summary = summary[:self.max_chars]  # Hard truncate

            logger.debug(f"Generated hierarchical document summary: {len(summary)} chars")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate hierarchical summary: {e}")
            # Fallback: combine first chars of each section summary
            fallback = " ".join(section_summaries)
            return fallback[:self.max_chars].strip() + "..."

    def _generate_with_claude(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    def _generate_with_openai(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def generate_document_summary_strict(self, document_text: str) -> str:
        """
        Generate summary with stricter length constraint.

        Called when first attempt exceeds max_chars.
        """

        text_preview = document_text[:5000]
        target_chars = int(self.max_chars * 0.9)  # Aim for 90% of max

        prompt = f"""Summarize this document in EXACTLY {target_chars} characters or less.
Be extremely concise.

Document:
{text_preview}

Summary ({target_chars} chars max):"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt, max_tokens=500)
            else:  # openai
                summary = self._generate_with_openai(prompt, max_tokens=500)

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

Provide a {self.max_chars}-character summary focusing on the main topic and key information.

Summary:"""

        try:
            if self.provider == "claude":
                summary = self._generate_with_claude(prompt, max_tokens=500)
            else:  # openai
                summary = self._generate_with_openai(prompt, max_tokens=500)

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
