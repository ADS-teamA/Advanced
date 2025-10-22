"""
Query Optimizer - Coordinates HyDE and Query Decomposition

Decides when and how to apply query optimization techniques based on configuration.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .decomposition import DecomposedQuery, QueryDecomposer
from .hyde import HyDEGenerator, HyDEResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizedQuery:
    """Result from query optimization."""

    original_query: str
    optimized_queries: List[str]  # One or more queries to execute
    metadata: Dict[str, Any]  # Metadata about optimizations applied


class QueryOptimizer:
    """
    Coordinate query optimization strategies.

    Decides whether to:
    - Apply HyDE (hypothetical document generation)
    - Apply query decomposition
    - Use original query as-is

    Usage:
        optimizer = QueryOptimizer(
            anthropic_api_key,
            enable_hyde=True,
            enable_decomposition=True
        )

        result = optimizer.optimize("Complex query here")
        for query in result.optimized_queries:
            # Execute search with this query
            pass
    """

    def __init__(
        self,
        anthropic_api_key: str,
        enable_hyde: bool = False,
        enable_decomposition: bool = False,
        hyde_config: Optional[Dict[str, Any]] = None,
        decomposition_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize query optimizer.

        Args:
            anthropic_api_key: Anthropic API key
            enable_hyde: Enable HyDE optimization
            enable_decomposition: Enable query decomposition
            hyde_config: Optional HyDE configuration dict
            decomposition_config: Optional decomposition configuration dict
        """
        self.enable_hyde = enable_hyde
        self.enable_decomposition = enable_decomposition

        # Initialize HyDE if enabled
        self.hyde_generator = None
        if enable_hyde:
            hyde_params = hyde_config or {}
            self.hyde_generator = HyDEGenerator(
                anthropic_api_key=anthropic_api_key,
                model=hyde_params.get("model", "claude-haiku-4-5"),
                num_documents=hyde_params.get("num_documents", 1),
                temperature=hyde_params.get("temperature", 0.7),
            )
            logger.info("HyDE enabled")

        # Initialize decomposer if enabled
        self.decomposer = None
        if enable_decomposition:
            decomp_params = decomposition_config or {}
            self.decomposer = QueryDecomposer(
                anthropic_api_key=anthropic_api_key,
                model=decomp_params.get("model", "claude-haiku-4-5"),
                temperature=decomp_params.get("temperature", 0.3),
            )
            logger.info("Query decomposition enabled")

        logger.info(
            f"QueryOptimizer initialized: hyde={enable_hyde}, decomposition={enable_decomposition}"
        )

    def optimize(self, query: str) -> OptimizedQuery:
        """
        Optimize a query using enabled strategies.

        Decision logic:
        1. If decomposition enabled and query is complex:
           - Decompose into sub-queries
           - Optionally apply HyDE to each sub-query
        2. If only HyDE enabled:
           - Apply HyDE to original query
        3. Otherwise:
           - Return original query

        Args:
            query: User's original query

        Returns:
            OptimizedQuery with one or more queries to execute
        """
        metadata = {
            "hyde_applied": False,
            "decomposition_applied": False,
            "sub_query_count": 1,
        }

        # STRATEGY 1: Decomposition + optional HyDE
        if self.enable_decomposition and self.decomposer:
            # Check if query should be decomposed
            should_decompose = self.decomposer.should_decompose(query)

            if should_decompose:
                logger.info(f"Applying decomposition to query: '{query[:50]}...'")
                decomposed = self.decomposer.decompose(query)

                if decomposed.is_complex and len(decomposed.sub_queries) > 1:
                    # Successfully decomposed
                    metadata["decomposition_applied"] = True
                    metadata["sub_query_count"] = len(decomposed.sub_queries)

                    # Optionally apply HyDE to each sub-query
                    if self.enable_hyde and self.hyde_generator:
                        logger.info("Applying HyDE to decomposed sub-queries")
                        optimized_queries = []

                        for sub_q in decomposed.sub_queries:
                            hyde_result = self.hyde_generator.generate(sub_q)
                            # Use combined query (original + hypothetical docs)
                            optimized_queries.append(hyde_result.combined_query or sub_q)

                        metadata["hyde_applied"] = True

                        return OptimizedQuery(
                            original_query=query,
                            optimized_queries=optimized_queries,
                            metadata=metadata,
                        )
                    else:
                        # Just use decomposed sub-queries
                        return OptimizedQuery(
                            original_query=query,
                            optimized_queries=decomposed.sub_queries,
                            metadata=metadata,
                        )

        # STRATEGY 2: HyDE only (no decomposition or query is simple)
        if self.enable_hyde and self.hyde_generator:
            logger.info(f"Applying HyDE to query: '{query[:50]}...'")
            hyde_result = self.hyde_generator.generate(query)

            metadata["hyde_applied"] = True

            return OptimizedQuery(
                original_query=query,
                optimized_queries=[hyde_result.combined_query or query],
                metadata=metadata,
            )

        # STRATEGY 3: No optimization
        logger.info(f"No optimization applied to query: '{query[:50]}...'")
        return OptimizedQuery(original_query=query, optimized_queries=[query], metadata=metadata)

    def get_search_queries(self, query: str) -> List[str]:
        """
        Convenience method to get list of queries to search.

        Args:
            query: User's original query

        Returns:
            List of queries to execute (may be 1 or more)
        """
        result = self.optimize(query)
        return result.optimized_queries

    def is_enabled(self) -> bool:
        """Check if any optimization is enabled."""
        return self.enable_hyde or self.enable_decomposition


# Example usage
if __name__ == "__main__":
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    # Test different configurations
    print("=" * 60)
    print("TEST 1: HyDE only")
    print("=" * 60)

    optimizer1 = QueryOptimizer(api_key, enable_hyde=True, enable_decomposition=False)

    query1 = "What are the waste disposal requirements?"
    result1 = optimizer1.optimize(query1)
    print(f"Original: {result1.original_query}")
    print(f"Optimized queries: {len(result1.optimized_queries)}")
    print(f"Metadata: {result1.metadata}")
    print()

    print("=" * 60)
    print("TEST 2: Decomposition only")
    print("=" * 60)

    optimizer2 = QueryOptimizer(api_key, enable_hyde=False, enable_decomposition=True)

    query2 = "Find waste requirements in GRI 306 and check if our contract complies"
    result2 = optimizer2.optimize(query2)
    print(f"Original: {result2.original_query}")
    print(f"Optimized queries: {len(result2.optimized_queries)}")
    for i, q in enumerate(result2.optimized_queries, 1):
        print(f"  {i}. {q[:100]}...")
    print(f"Metadata: {result2.metadata}")
    print()

    print("=" * 60)
    print("TEST 3: Both HyDE + Decomposition")
    print("=" * 60)

    optimizer3 = QueryOptimizer(api_key, enable_hyde=True, enable_decomposition=True)

    query3 = "Compare GRI 305 and GRI 306 environmental requirements"
    result3 = optimizer3.optimize(query3)
    print(f"Original: {result3.original_query}")
    print(f"Optimized queries: {len(result3.optimized_queries)}")
    print(f"Metadata: {result3.metadata}")
