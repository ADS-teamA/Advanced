"""
HyDE (Hypothetical Document Embeddings) Implementation

Generates hypothetical documents from queries to improve retrieval.

Based on: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496

Key idea: Generate hypothetical answers to the query, then search using those
instead of the original query. This aligns the query better with document space.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    """Result from HyDE generation."""

    original_query: str
    hypothetical_documents: List[str]
    combined_query: Optional[str] = None

    def __post_init__(self):
        """Validate HyDEResult invariants."""
        if not self.original_query or not self.original_query.strip():
            raise ValueError("Original query cannot be empty")

        # Validate hypothetical documents are non-empty strings
        for i, doc in enumerate(self.hypothetical_documents):
            if not doc or not doc.strip():
                raise ValueError(f"Hypothetical document {i} is empty")

        # If combined_query is None, default to original_query
        if self.combined_query is None:
            object.__setattr__(self, 'combined_query', self.original_query)

        if not self.combined_query or not self.combined_query.strip():
            raise ValueError("Combined query cannot be empty")


class HyDEGenerator:
    """
    Generate hypothetical documents from queries using LLM.

    Usage:
        hyde = HyDEGenerator(anthropic_api_key, model="claude-haiku-4-5")
        result = hyde.generate(query="What are waste disposal requirements?")
        # Use result.hypothetical_documents for retrieval
    """

    # System prompt for hypothetical document generation
    SYSTEM_PROMPT = """You are an expert at generating hypothetical document passages that would answer a given question.

Given a user's query, generate a realistic passage that directly answers the question as if it were from a relevant technical or legal document.

Guidelines:
- Write in formal, technical language appropriate for legal/regulatory documents
- Be specific and detailed
- Include relevant terminology and concepts
- Keep the passage concise (2-4 sentences, ~100-150 words)
- Write as if this passage exists in a real document
- Do NOT add explanations or meta-commentary
- Output ONLY the hypothetical passage"""

    def __init__(
        self,
        anthropic_api_key: str,
        model: str = "claude-haiku-4-5",
        num_documents: int = 1,
        temperature: float = 0.7,
    ):
        """
        Initialize HyDE generator.

        Args:
            anthropic_api_key: Anthropic API key
            model: Claude model to use (default: claude-haiku-4-5 for speed)
            num_documents: Number of hypothetical documents to generate
            temperature: Moderate temperature prevents overfitting to query wording (Gao et al., 2022)
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.num_documents = num_documents
        self.temperature = temperature

        logger.info(
            f"HyDE initialized: model={model}, num_docs={num_documents}, temp={temperature}"
        )

    def generate(self, query: str) -> HyDEResult:
        """
        Generate hypothetical documents for a query.

        Args:
            query: User's search query

        Returns:
            HyDEResult with hypothetical documents
        """
        try:
            hypothetical_docs = []

            # Generate multiple hypothetical documents for diversity
            for i in range(self.num_documents):
                logger.debug(f"Generating hypothetical doc {i+1}/{self.num_documents}")

                # Add variation prompt for multiple generations
                variation_note = ""
                if i > 0:
                    variation_note = (
                        f" (Variation {i+1}: Provide a different perspective or aspect)"
                    )

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,  # ~100-150 words
                    temperature=self.temperature,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Query: {query}{variation_note}\n\nGenerate a hypothetical document passage:",
                        }
                    ],
                )

                # Extract generated text
                hypothetical_doc = message.content[0].text.strip()
                hypothetical_docs.append(hypothetical_doc)

                logger.debug(f"Generated: {hypothetical_doc[:100]}...")

            # Optionally combine original query with hypothetical docs
            combined = self._combine_query_and_hypotheticals(query, hypothetical_docs)

            logger.info(
                f"HyDE generated {len(hypothetical_docs)} hypothetical documents for query: '{query[:50]}...'"
            )

            return HyDEResult(
                original_query=query,
                hypothetical_documents=hypothetical_docs,
                combined_query=combined,
            )

        except anthropic.AuthenticationError as e:
            logger.error(f"HyDE authentication failed: {e}")
            raise RuntimeError(
                "HyDE failed: Invalid Anthropic API key. "
                "Check ANTHROPIC_API_KEY environment variable."
            )
        except (anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.APIError) as e:
            logger.error(f"HyDE API error: {e}")
            # Fallback acceptable for API issues
            logger.warning("Falling back to original query without HyDE optimization")
            return HyDEResult(original_query=query, hypothetical_documents=[], combined_query=query)
        except Exception as e:
            logger.error(f"HyDE unexpected error: {e}", exc_info=True)
            # Don't hide programming bugs - raise them
            raise

    def _combine_query_and_hypotheticals(self, query: str, hypothetical_docs: List[str]) -> str:
        """
        Combine original query with hypothetical documents.

        Args:
            query: Original query
            hypothetical_docs: Generated hypothetical documents

        Returns:
            Combined query string
        """
        if not hypothetical_docs:
            return query

        # Strategy: Concatenate query + all hypothetical docs
        # This gives the embedder multiple perspectives to encode
        combined = f"{query}\n\n" + "\n\n".join(hypothetical_docs)

        return combined

    def generate_for_embedding(self, query: str) -> str:
        """
        Generate hypothetical documents and return text for embedding.

        This is a convenience method that returns the combined query string
        ready for embedding.

        Args:
            query: User's search query

        Returns:
            Combined query string for embedding
        """
        result = self.generate(query)
        return result.combined_query or result.original_query


# Example usage
if __name__ == "__main__":
    import os

    # Example
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    hyde = HyDEGenerator(api_key, num_documents=2)

    query = "What are the waste disposal requirements in GRI 306?"
    result = hyde.generate(query)

    print(f"Original query: {result.original_query}")
    print(f"\nHypothetical documents:")
    for i, doc in enumerate(result.hypothetical_documents, 1):
        print(f"\n{i}. {doc}")
    print(f"\nCombined query length: {len(result.combined_query or '')}")
