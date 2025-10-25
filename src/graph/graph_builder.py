"""
Graph builders for storing knowledge graphs.

Provides multiple backend implementations:
- Neo4jGraphBuilder: Production Neo4j database
- SimpleGraphBuilder: JSON-based in-memory graph for development
- NetworkXGraphBuilder: NetworkX for lightweight operations
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .models import Entity, Relationship, KnowledgeGraph, EntityType, RelationshipType
from .config import GraphStorageConfig, Neo4jConfig


logger = logging.getLogger(__name__)


class GraphBuilder(ABC):
    """
    Abstract base class for graph builders.

    Defines interface for storing and querying knowledge graphs.
    """

    @abstractmethod
    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the graph."""
        pass

    @abstractmethod
    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        pass

    @abstractmethod
    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export graph to KnowledgeGraph object."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save graph to file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections and cleanup."""
        pass


class SimpleGraphBuilder(GraphBuilder):
    """
    Simple in-memory graph builder using Python data structures.

    Stores graph in memory and exports to JSON. Fast for development and testing.
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize simple graph builder."""
        self.config = config

        # Storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

        # Indexes
        self.entity_by_type: Dict[EntityType, Set[str]] = {}
        self.relationships_by_source: Dict[str, Set[str]] = {}
        self.relationships_by_target: Dict[str, Set[str]] = {}

        logger.info("Initialized SimpleGraphBuilder")

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the graph."""
        for entity in entities:
            # Store entity
            self.entities[entity.id] = entity

            # Index by type
            if entity.type not in self.entity_by_type:
                self.entity_by_type[entity.type] = set()
            self.entity_by_type[entity.type].add(entity.id)

        logger.info(f"Added {len(entities)} entities to graph (total: {len(self.entities)})")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the graph."""
        for rel in relationships:
            # Store relationship
            self.relationships[rel.id] = rel

            # Index by source entity
            if rel.source_entity_id not in self.relationships_by_source:
                self.relationships_by_source[rel.source_entity_id] = set()
            self.relationships_by_source[rel.source_entity_id].add(rel.id)

            # Index by target entity
            if rel.target_entity_id not in self.relationships_by_target:
                self.relationships_by_target[rel.target_entity_id] = set()
            self.relationships_by_target[rel.target_entity_id].add(rel.id)

        logger.info(
            f"Added {len(relationships)} relationships to graph (total: {len(self.relationships)})"
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity (as source or target)."""
        rel_ids = set()

        # Get relationships where entity is source
        if entity_id in self.relationships_by_source:
            rel_ids.update(self.relationships_by_source[entity_id])

        # Get relationships where entity is target
        if entity_id in self.relationships_by_target:
            rel_ids.update(self.relationships_by_target[entity_id])

        return [self.relationships[rid] for rid in rel_ids]

    def get_outgoing_relationships(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get outgoing relationships from an entity."""
        rel_ids = self.relationships_by_source.get(entity_id, set())
        rels = [self.relationships[rid] for rid in rel_ids]

        if rel_type:
            rels = [r for r in rels if r.type == rel_type]

        return rels

    def get_incoming_relationships(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get incoming relationships to an entity."""
        rel_ids = self.relationships_by_target.get(entity_id, set())
        rels = [self.relationships[rid] for rid in rel_ids]

        if rel_type:
            rels = [r for r in rels if r.type == rel_type]

        return rels

    def get_neighbors(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Entity]:
        """Get neighbor entities connected by relationships."""
        neighbors = []

        # Outgoing relationships
        outgoing = self.get_outgoing_relationships(entity_id, rel_type)
        for rel in outgoing:
            target = self.get_entity(rel.target_entity_id)
            if target:
                neighbors.append(target)

        # Incoming relationships
        incoming = self.get_incoming_relationships(entity_id, rel_type)
        for rel in incoming:
            source = self.get_entity(rel.source_entity_id)
            if source:
                neighbors.append(source)

        return neighbors

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export to KnowledgeGraph object."""
        kg = KnowledgeGraph(
            entities=list(self.entities.values()),
            relationships=list(self.relationships.values()),
        )

        # Compute statistics
        kg.compute_stats()

        return kg

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        kg.save_json(path)
        logger.info(f"Saved graph to {path}")

    def close(self) -> None:
        """Cleanup (no-op for simple builder)."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_type_counts = {}
        for entity_type, entity_ids in self.entity_by_type.items():
            entity_type_counts[entity_type.value] = len(entity_ids)

        relationship_type_counts = {}
        for rel in self.relationships.values():
            rel_type = rel.type.value
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
        }


class Neo4jGraphBuilder(GraphBuilder):
    """
    Neo4j graph database builder for production use.

    Stores entities as nodes and relationships as edges in Neo4j.
    Supports Cypher queries and graph algorithms.
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize Neo4j graph builder."""
        self.config = config

        if not self.config.neo4j_config:
            raise ValueError("Neo4j config required for Neo4jGraphBuilder")

        self.neo4j_config: Neo4jConfig = self.config.neo4j_config

        # Initialize Neo4j driver
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.neo4j_config.uri,
                auth=(self.neo4j_config.username, self.neo4j_config.password),
                max_connection_lifetime=self.neo4j_config.max_connection_lifetime,
                max_connection_pool_size=self.neo4j_config.max_connection_pool_size,
            )
        except ImportError:
            raise ImportError("neo4j package not installed. Install with: pip install neo4j")

        # Create indexes and constraints
        if self.neo4j_config.create_indexes:
            self._create_indexes()

        logger.info(f"Initialized Neo4jGraphBuilder (uri={self.neo4j_config.uri})")

    def _create_indexes(self):
        """Create indexes for fast lookups."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            # Create constraint on Entity.id
            session.run(
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

            # Create indexes on Entity properties
            session.run("CREATE INDEX entity_type_idx IF NOT EXISTS " "FOR (e:Entity) ON (e.type)")

            session.run(
                "CREATE INDEX entity_normalized_value_idx IF NOT EXISTS "
                "FOR (e:Entity) ON (e.normalized_value)"
            )

        logger.info("Created Neo4j indexes and constraints")

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to Neo4j as nodes."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            for entity in entities:
                # Convert entity to node properties
                props = {
                    "id": entity.id,
                    "type": entity.type.value,
                    "value": entity.value,
                    "normalized_value": entity.normalized_value,
                    "confidence": entity.confidence,
                    "source_chunk_ids": entity.source_chunk_ids,
                    "first_mention_chunk_id": entity.first_mention_chunk_id,
                    "document_id": entity.document_id,
                    "section_path": entity.section_path,
                    "extraction_method": entity.extraction_method,
                }

                # Merge entity (create or update)
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e += $props
                    """,
                    id=entity.id,
                    props=props,
                )

        logger.info(f"Added {len(entities)} entities to Neo4j")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to Neo4j as edges."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            for rel in relationships:
                # Convert relationship to edge properties
                props = {
                    "id": rel.id,
                    "type": rel.type.value,
                    "confidence": rel.confidence,
                    "source_chunk_id": rel.source_chunk_id,
                    "evidence_text": rel.evidence_text,
                    "extraction_method": rel.extraction_method,
                }

                # Create relationship between entities
                rel_type_upper = rel.type.value.upper()
                session.run(
                    f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MATCH (target:Entity {{id: $target_id}})
                    MERGE (source)-[r:{rel_type_upper}]->(target)
                    SET r += $props
                    """,
                    source_id=rel.source_entity_id,
                    target_id=rel.target_entity_id,
                    props=props,
                )

        logger.info(f"Added {len(relationships)} relationships to Neo4j")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from Neo4j."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=entity_id,
            )

            record = result.single()
            if not record:
                return None

            return self._node_to_entity(record["e"])

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $id})-[r]-(other)
                RETURN r, startNode(r) as source, endNode(r) as target
                """,
                id=entity_id,
            )

            relationships = []
            for record in result:
                rel = self._edge_to_relationship(
                    record["r"],
                    record["source"]["id"],
                    record["target"]["id"],
                )
                relationships.append(rel)

            return relationships

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export all data from Neo4j to KnowledgeGraph object."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            # Get all entities
            entity_result = session.run("MATCH (e:Entity) RETURN e")
            entities = [self._node_to_entity(record["e"]) for record in entity_result]

            # Get all relationships
            rel_result = session.run(
                """
                MATCH (source:Entity)-[r]->(target:Entity)
                RETURN r, source.id as source_id, target.id as target_id
                """
            )
            relationships = [
                self._edge_to_relationship(record["r"], record["source_id"], record["target_id"])
                for record in rel_result
            ]

        kg = KnowledgeGraph(entities=entities, relationships=relationships)
        kg.compute_stats()

        return kg

    def save(self, path: str) -> None:
        """Export graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        kg.save_json(path)
        logger.info(f"Exported Neo4j graph to {path}")

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def _node_to_entity(self, node) -> Entity:
        """Convert Neo4j node to Entity object."""
        return Entity(
            id=node["id"],
            type=EntityType(node["type"]),
            value=node["value"],
            normalized_value=node["normalized_value"],
            confidence=node["confidence"],
            source_chunk_ids=node.get("source_chunk_ids", []),
            first_mention_chunk_id=node.get("first_mention_chunk_id"),
            document_id=node.get("document_id"),
            section_path=node.get("section_path"),
            metadata={},
            extraction_method=node.get("extraction_method", "unknown"),
        )

    def _edge_to_relationship(self, edge, source_id: str, target_id: str) -> Relationship:
        """Convert Neo4j edge to Relationship object."""
        return Relationship(
            id=edge["id"],
            type=RelationshipType(edge["type"]),
            source_entity_id=source_id,
            target_entity_id=target_id,
            confidence=edge["confidence"],
            source_chunk_id=edge.get("source_chunk_id", ""),
            evidence_text=edge.get("evidence_text", ""),
            extraction_method=edge.get("extraction_method", "unknown"),
        )


class NetworkXGraphBuilder(GraphBuilder):
    """
    NetworkX-based graph builder for lightweight operations.

    Uses NetworkX DiGraph for in-memory graph operations and algorithms.
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize NetworkX graph builder."""
        self.config = config

        try:
            import networkx as nx

            self.nx = nx
            self.graph = nx.DiGraph()
        except ImportError:
            raise ImportError("networkx package not installed. Install with: pip install networkx")

        logger.info("Initialized NetworkXGraphBuilder")

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities as nodes."""
        for entity in entities:
            self.graph.add_node(
                entity.id,
                entity_obj=entity,
                type=entity.type.value,
                value=entity.value,
                normalized_value=entity.normalized_value,
            )

        logger.info(f"Added {len(entities)} entities to NetworkX graph")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships as edges."""
        for rel in relationships:
            self.graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                relationship_obj=rel,
                type=rel.type.value,
                confidence=rel.confidence,
            )

        logger.info(f"Added {len(relationships)} relationships to NetworkX graph")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        if entity_id in self.graph.nodes:
            return self.graph.nodes[entity_id].get("entity_obj")
        return None

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        relationships = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        # Incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        return relationships

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export to KnowledgeGraph object."""
        entities = [
            data["entity_obj"] for _, data in self.graph.nodes(data=True) if "entity_obj" in data
        ]

        relationships = []
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        kg = KnowledgeGraph(entities=entities, relationships=relationships)
        kg.compute_stats()

        return kg

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        kg.save_json(path)

        logger.info(f"Saved NetworkX graph to {path}")

    def close(self) -> None:
        """Cleanup (no-op)."""
        pass


def create_graph_builder(config: GraphStorageConfig) -> GraphBuilder:
    """
    Factory function to create appropriate graph builder.

    Args:
        config: Graph storage configuration

    Returns:
        GraphBuilder instance
    """
    from .config import GraphBackend

    if config.backend == GraphBackend.NEO4J:
        return Neo4jGraphBuilder(config)
    elif config.backend == GraphBackend.SIMPLE:
        return SimpleGraphBuilder(config)
    elif config.backend == GraphBackend.NETWORKX:
        return NetworkXGraphBuilder(config)
    else:
        raise ValueError(f"Unsupported graph backend: {config.backend}")
