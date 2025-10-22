"""
Query Decomposition Implementation

Decomposes complex queries into simpler sub-queries.

Based on: "Least-to-Most Prompting" (Zhou et al., 2022) and
"Decomposed Prompting" (Khot et al., 2022)

Key idea: Break down complex multi-part questions into simpler sub-questions
that can be answered independently, then combine results.
"""

import logging
from typing import List
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class DecomposedQuery:
    """Result from query decomposition."""

    original_query: str
    sub_queries: List[str]
    is_complex: bool  # Whether decomposition was necessary


class QueryDecomposer:
    """
    Decompose complex queries into simpler sub-queries.

    Usage:
        decomposer = QueryDecomposer(anthropic_api_key)
        result = decomposer.decompose("Find X and check if it complies with Y")
        for sub_q in result.sub_queries:
            # Process each sub-query independently
            pass
    """

    # System prompt for query decomposition
    SYSTEM_PROMPT = """You are an expert at decomposing complex queries into simpler sub-queries.

Given a user's complex query, break it down into 2-4 simpler sub-queries that can be answered independently.

Guidelines:
- Only decompose if the query is genuinely complex (has multiple parts, requires multiple steps)
- Each sub-query should be self-contained and answerable on its own
- Sub-queries should be ordered logically (e.g., first find, then analyze)
- Use clear, specific language
- If the query is already simple, output "SIMPLE" followed by the original query

Examples:

Complex query: "Find waste disposal requirements in GRI 306 and check if our contract complies"
Output:
1. What are the waste disposal requirements specified in GRI 306?
2. What waste disposal provisions are in our contract?
3. Do the contract provisions comply with GRI 306 requirements?

Simple query: "What is GRI 306?"
Output:
SIMPLE
What is GRI 306?

Complex query: "Compare the environmental reporting requirements in GRI 305 and GRI 306"
Output:
1. What are the environmental reporting requirements in GRI 305?
2. What are the environmental reporting requirements in GRI 306?
3. What are the similarities and differences between these requirements?

Now decompose the following query:"""

    def __init__(
        self,
        anthropic_api_key: str,
        model: str = "claude-haiku-4-5",
        temperature: float = 0.3,
    ):
        """
        Initialize query decomposer.

        Args:
            anthropic_api_key: Anthropic API key
            model: Claude model to use (default: claude-haiku-4-5 for speed)
            temperature: Lower temperature for consistent decomposition
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.temperature = temperature

        logger.info(f"QueryDecomposer initialized: model={model}, temp={temperature}")

    def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a query into sub-queries.

        Args:
            query: User's complex query

        Returns:
            DecomposedQuery with sub-queries
        """
        try:
            logger.debug(f"Decomposing query: '{query}'")

            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=self.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )

            # Extract generated text
            response = message.content[0].text.strip()

            # Parse response
            if response.startswith("SIMPLE"):
                # Query is simple, no decomposition needed
                logger.info(f"Query marked as SIMPLE: '{query}'")
                return DecomposedQuery(
                    original_query=query, sub_queries=[query], is_complex=False
                )

            # Parse numbered sub-queries
            sub_queries = self._parse_sub_queries(response)

            if not sub_queries:
                # Fallback: treat as simple
                logger.warning(
                    f"Failed to parse sub-queries from response: '{response}'"
                )
                return DecomposedQuery(
                    original_query=query, sub_queries=[query], is_complex=False
                )

            logger.info(
                f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}"
            )

            return DecomposedQuery(
                original_query=query, sub_queries=sub_queries, is_complex=True
            )

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}", exc_info=True)
            # Fallback: return original query
            return DecomposedQuery(
                original_query=query, sub_queries=[query], is_complex=False
            )

    def _parse_sub_queries(self, response: str) -> List[str]:
        """
        Parse numbered sub-queries from LLM response.

        Expected format:
        1. First sub-query
        2. Second sub-query
        3. Third sub-query

        Args:
            response: LLM response text

        Returns:
            List of sub-queries
        """
        sub_queries = []

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for numbered format: "1. Query" or "1) Query"
            if line[0].isdigit():
                # Remove number and punctuation
                # Handle formats: "1. Query", "1) Query", "1 Query"
                if ". " in line:
                    sub_query = line.split(". ", 1)[1].strip()
                elif ") " in line:
                    sub_query = line.split(") ", 1)[1].strip()
                else:
                    # Just remove leading digits
                    sub_query = line.lstrip("0123456789").strip()

                if sub_query:
                    sub_queries.append(sub_query)

        return sub_queries

    def should_decompose(self, query: str) -> bool:
        """
        Quick heuristic check if query should be decomposed.

        This is a fast pre-check before calling the LLM.

        Args:
            query: User's query

        Returns:
            True if query looks complex
        """
        # Simple heuristics for complexity
        complexity_indicators = [
            " and ",  # Multiple parts
            " then ",  # Sequential steps
            "compare",  # Comparison tasks
            "check if",  # Verification tasks
            "find",  # Multi-step: find then analyze
            " or ",  # Multiple alternatives
            "?",  # Multiple questions
        ]

        query_lower = query.lower()

        # Count indicators
        indicator_count = sum(
            1 for indicator in complexity_indicators if indicator in query_lower
        )

        # Heuristic: 2+ indicators suggests complexity
        is_complex = indicator_count >= 2

        # Also check length - very long queries are often complex
        is_long = len(query.split()) > 15

        return is_complex or is_long


# Example usage
if __name__ == "__main__":
    import os

    # Example
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    decomposer = QueryDecomposer(api_key)

    # Test queries
    queries = [
        "What is GRI 306?",
        "Find waste disposal requirements in GRI 306 and check if our contract complies",
        "Compare environmental reporting in GRI 305 and GRI 306 and summarize differences",
    ]

    for query in queries:
        print(f"\nOriginal: {query}")
        print(f"Should decompose? {decomposer.should_decompose(query)}")

        result = decomposer.decompose(query)
        print(f"Complex? {result.is_complex}")
        print(f"Sub-queries:")
        for i, sq in enumerate(result.sub_queries, 1):
            print(f"  {i}. {sq}")
