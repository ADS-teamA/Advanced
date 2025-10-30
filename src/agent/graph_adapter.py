"""
Graph Adapter for Agent Tools

Provides transparent Neo4j access for agent tools while maintaining
the same interface as in-memory KnowledgeGraph object.

This allows tools to use Neo4j without code changes.
"""

import logging
from typing import Any, Dict, List, Optional

from src.graph import Neo4jGraphBuilder, GraphStorageConfig, GraphBackend, Neo4jConfig
from src.graph.models import Entity, Relationship

logger = logging.getLogger(__name__)


class GraphAdapter:
    """
    Adapter that provides KnowledgeGraph-like interface over Neo4j.

    Allows agent tools to use Neo4j transparently by providing:
    - entities: Dict-like access to entities (lazy-loaded from Neo4j)
    - relationships: List-like access to relationships
    - Caching for frequently accessed data

    Example:
        # In CLI
        adapter = GraphAdapter.from_neo4j(Neo4jConfig.from_env())

        # In tools (no changes needed!)
        for entity in adapter.entities.values():  # Queries Neo4j
            ...
    """

    def __init__(self, builder: Neo4jGraphBuilder, owns_builder: bool = False):
        """
        Initialize adapter with Neo4jGraphBuilder.

        Args:
            builder: Neo4jGraphBuilder instance connected to Neo4j
            owns_builder: If True, adapter will close builder on cleanup (default: False)
        """
        self.builder = builder
        self._owns_builder = owns_builder
        self._kg_cache: Optional[Any] = None  # KnowledgeGraph object cached once

    @classmethod
    def from_neo4j(cls, neo4j_config: Neo4jConfig) -> "GraphAdapter":
        """
        Create adapter from Neo4j config.

        Args:
            neo4j_config: Neo4j configuration

        Returns:
            GraphAdapter instance
        """
        config = GraphStorageConfig(
            backend=GraphBackend.NEO4J, neo4j_config=neo4j_config
        )
        builder = Neo4jGraphBuilder(config)
        return cls(builder, owns_builder=True)  # We created it, we own it

    def _ensure_loaded(self) -> None:
        """
        Ensure knowledge graph is loaded from Neo4j (lazy loading).

        This loads BOTH entities and relationships in a single query
        to avoid redundant Neo4j round-trips.
        """
        if self._kg_cache is None:
            logger.info("Loading knowledge graph from Neo4j...")
            self._kg_cache = self.builder.export_to_knowledge_graph()
            logger.info(
                f"Loaded {len(self._kg_cache.entities)} entities and "
                f"{len(self._kg_cache.relationships)} relationships"
            )

    @property
    def entities(self) -> Dict[str, Entity]:
        """
        Get all entities as dict (lazy-loaded from Neo4j).

        Returns:
            Dict mapping entity_id -> Entity

        Note: First access queries Neo4j for BOTH entities and relationships
              (single query) and caches both. This prevents redundant queries
              if both properties are accessed.

              Use refresh() to reload from Neo4j.
        """
        self._ensure_loaded()
        return {e.id: e for e in self._kg_cache.entities}

    @property
    def relationships(self) -> List[Relationship]:
        """
        Get all relationships as list (lazy-loaded from Neo4j).

        Returns:
            List of Relationship objects

        Note: First access queries Neo4j for BOTH entities and relationships
              (single query) and caches both. This prevents redundant queries
              if both properties are accessed.
        """
        self._ensure_loaded()
        return self._kg_cache.relationships

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get single entity by ID (direct Neo4j query, bypasses cache).

        Args:
            entity_id: Entity ID

        Returns:
            Entity object or None if not found
        """
        return self.builder.get_entity(entity_id)

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """
        Get relationships for specific entity (direct Neo4j query).

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships where entity is source or target
        """
        return self.builder.get_relationships_for_entity(entity_id)

    def get_outgoing_relationships(self, entity_id: str) -> List[Relationship]:
        """
        Get outgoing relationships where entity is the source.

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships where entity is the source
        """
        all_rels = self.get_relationships_for_entity(entity_id)
        return [r for r in all_rels if r.source_entity_id == entity_id]

    def get_incoming_relationships(self, entity_id: str) -> List[Relationship]:
        """
        Get incoming relationships where entity is the target.

        Args:
            entity_id: Entity ID

        Returns:
            List of relationships where entity is the target
        """
        all_rels = self.get_relationships_for_entity(entity_id)
        return [r for r in all_rels if r.target_entity_id == entity_id]

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0,
        value_contains: Optional[str] = None,
    ) -> List[Entity]:
        """
        Find entities matching criteria (direct Neo4j query).

        This is more efficient than filtering self.entities for large datasets.

        Args:
            entity_type: Filter by entity type (e.g., 'standard', 'organization')
            min_confidence: Minimum confidence score
            value_contains: Substring to search in entity value

        Returns:
            List of matching entities
        """
        # Build Cypher query
        where_clauses = []
        params = {}

        if entity_type:
            where_clauses.append("e.type = $entity_type")
            params["entity_type"] = entity_type

        if min_confidence > 0.0:
            where_clauses.append("e.confidence >= $min_confidence")
            params["min_confidence"] = min_confidence

        if value_contains:
            where_clauses.append(
                "(toLower(e.value) CONTAINS toLower($value_contains) OR "
                "toLower(e.normalized_value) CONTAINS toLower($value_contains))"
            )
            params["value_contains"] = value_contains

        where_clause = " AND ".join(where_clauses) if where_clauses else "true"

        query = f"""
        MATCH (e:Entity)
        WHERE {where_clause}
        RETURN e
        LIMIT 1000
        """

        results = self.builder.manager.execute(query, params)

        # Convert to Entity objects
        entities = []
        for row in results:
            node = row["e"]
            entity = self.builder._node_to_entity(node)
            entities.append(entity)

        return entities

    def refresh(self) -> None:
        """
        Clear cache and force reload from Neo4j on next access.

        Use this if data has changed in Neo4j and you need fresh data.
        """
        logger.info("Clearing graph cache - will reload from Neo4j on next access")
        self._kg_cache = None

    def close(self) -> None:
        """
        Close Neo4j connection if we own the builder.

        Safe to call multiple times (idempotent).
        """
        if self._owns_builder and self.builder:
            self.builder.close()
            logger.info("GraphAdapter closed Neo4j connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.close()
