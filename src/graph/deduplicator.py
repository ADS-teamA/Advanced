"""
Entity deduplication for knowledge graph merging.

Supports:
- Exact match by (type, normalized_value) - fast, default
- Configurable similarity threshold (future: embeddings-based)
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Entity
    from .graph_builder import SimpleGraphBuilder

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """
    Lightweight entity deduplicator for graph merging.

    Deduplication strategy:
    - Exact match on (entity_type, normalized_value)
    - Case-insensitive, whitespace-normalized

    Future extensions:
    - Embedding-based similarity (98% threshold)
    - Fuzzy string matching (Levenshtein distance)

    Example:
        >>> deduplicator = EntityDeduplicator()
        >>> duplicate_id = deduplicator.find_duplicate(entity, graph)
        >>> if duplicate_id:
        ...     print(f"Entity is duplicate of {duplicate_id}")
    """

    def __init__(
        self,
        use_exact_match: bool = True,
        similarity_threshold: float = 0.98,
    ):
        """
        Initialize deduplicator.

        Args:
            use_exact_match: Use exact normalized_value matching (fast)
            similarity_threshold: Similarity threshold for fuzzy matching (future)
        """
        self.use_exact_match = use_exact_match
        self.similarity_threshold = similarity_threshold

        logger.info(
            f"EntityDeduplicator initialized: exact_match={use_exact_match}, "
            f"threshold={similarity_threshold}"
        )

    def find_duplicate(
        self,
        entity: "Entity",
        graph: "SimpleGraphBuilder",
    ) -> Optional[str]:
        """
        Find duplicate entity in graph.

        Args:
            entity: Entity to check for duplicates
            graph: Graph to search in

        Returns:
            Entity ID of duplicate if found, None otherwise
        """
        if self.use_exact_match:
            return self._find_exact_duplicate(entity, graph)
        else:
            # Future: embedding-based or fuzzy matching
            raise NotImplementedError("Non-exact matching not yet implemented")

    def _find_exact_duplicate(
        self,
        entity: "Entity",
        graph: "SimpleGraphBuilder",
    ) -> Optional[str]:
        """Find duplicate using exact (type, normalized_value) match."""
        # Lookup in index (O(1) operation)
        key = (entity.type, entity.normalized_value)
        return graph.entity_by_normalized_value.get(key)

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "strategy": "exact_match" if self.use_exact_match else "similarity",
            "threshold": self.similarity_threshold,
        }
