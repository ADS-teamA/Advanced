"""
Unit tests for Graph Builders.

Tests SimpleGraphBuilder, NetworkXGraphBuilder, and factory function.
"""

import pytest
from datetime import datetime

from src.graph.models import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
)
from src.graph.config import GraphStorageConfig, GraphBackend
from src.graph.graph_builder import (
    SimpleGraphBuilder,
    NetworkXGraphBuilder,
    create_graph_builder,
)


class TestSimpleGraphBuilder:
    """Tests for SimpleGraphBuilder."""

    def setup_method(self):
        """Setup test data."""
        self.config = GraphStorageConfig(
            backend=GraphBackend.SIMPLE,
            export_path="./test_graph.json",
        )

        self.entity1 = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.95,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        self.entity2 = Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="GSSB",
            confidence=0.9,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        self.rel1 = Relationship(
            id="r1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.9,
            source_chunk_id="chunk-1",
            evidence_text="Test evidence",
            extraction_method="test",
            extracted_at=datetime.now(),
        )

    def test_add_entities(self):
        """Test adding entities to graph."""
        builder = SimpleGraphBuilder(self.config)

        builder.add_entities([self.entity1, self.entity2])

        assert len(builder.entities) == 2
        assert "e1" in builder.entities
        assert "e2" in builder.entities

    def test_add_relationships(self):
        """Test adding relationships to graph."""
        builder = SimpleGraphBuilder(self.config)

        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        assert len(builder.relationships) == 1
        assert "r1" in builder.relationships

    def test_get_entity(self):
        """Test getting entity by ID."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])

        entity = builder.get_entity("e1")
        assert entity is not None
        assert entity.value == "GRI 306"

        not_found = builder.get_entity("e999")
        assert not_found is None

    def test_get_entities_by_type(self):
        """Test getting entities by type."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])

        standards = builder.get_entities_by_type(EntityType.STANDARD)
        assert len(standards) == 1
        assert standards[0].id == "e1"

        orgs = builder.get_entities_by_type(EntityType.ORGANIZATION)
        assert len(orgs) == 1
        assert orgs[0].id == "e2"

    def test_get_relationships_for_entity(self):
        """Test getting relationships for entity."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        # Entity e1 is source
        rels = builder.get_relationships_for_entity("e1")
        assert len(rels) == 1
        assert rels[0].id == "r1"

        # Entity e2 is target
        rels = builder.get_relationships_for_entity("e2")
        assert len(rels) == 1
        assert rels[0].id == "r1"

    def test_get_outgoing_relationships(self):
        """Test getting outgoing relationships."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        outgoing = builder.get_outgoing_relationships("e1")
        assert len(outgoing) == 1
        assert outgoing[0].source_entity_id == "e1"

        # No outgoing from e2
        outgoing = builder.get_outgoing_relationships("e2")
        assert len(outgoing) == 0

    def test_get_incoming_relationships(self):
        """Test getting incoming relationships."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        incoming = builder.get_incoming_relationships("e2")
        assert len(incoming) == 1
        assert incoming[0].target_entity_id == "e2"

        # No incoming to e1
        incoming = builder.get_incoming_relationships("e1")
        assert len(incoming) == 0

    def test_get_neighbors(self):
        """Test getting neighbor entities."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        # e1 has neighbor e2
        neighbors = builder.get_neighbors("e1")
        assert len(neighbors) == 1
        assert neighbors[0].id == "e2"

        # e2 has neighbor e1
        neighbors = builder.get_neighbors("e2")
        assert len(neighbors) == 1
        assert neighbors[0].id == "e1"

    def test_export_to_knowledge_graph(self):
        """Test exporting to KnowledgeGraph."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        kg = builder.export_to_knowledge_graph()

        assert len(kg.entities) == 2
        assert len(kg.relationships) == 1
        assert kg.stats["total_entities"] == 2

    def test_save_json(self, tmp_path):
        """Test saving graph to JSON."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        output_path = tmp_path / "graph.json"
        builder.save(str(output_path))

        assert output_path.exists()

    def test_get_statistics(self):
        """Test getting graph statistics."""
        builder = SimpleGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        stats = builder.get_statistics()

        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 1
        assert stats["entity_type_counts"]["standard"] == 1
        assert stats["entity_type_counts"]["organization"] == 1


class TestNetworkXGraphBuilder:
    """Tests for NetworkXGraphBuilder."""

    def setup_method(self):
        """Setup test data."""
        self.config = GraphStorageConfig(
            backend=GraphBackend.NETWORKX,
        )

        self.entity1 = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.95,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        self.entity2 = Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="GSSB",
            confidence=0.9,
            extraction_method="test",
            extracted_at=datetime.now(),
        )

        self.rel1 = Relationship(
            id="r1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.9,
            source_chunk_id="chunk-1",
            evidence_text="Test evidence",
            extraction_method="test",
            extracted_at=datetime.now(),
        )

    def test_add_entities(self):
        """Test adding entities as nodes."""
        pytest.importorskip("networkx")

        builder = NetworkXGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])

        assert len(builder.graph.nodes) == 2
        assert "e1" in builder.graph.nodes
        assert "e2" in builder.graph.nodes

    def test_add_relationships(self):
        """Test adding relationships as edges."""
        pytest.importorskip("networkx")

        builder = NetworkXGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        assert len(builder.graph.edges) == 1
        assert builder.graph.has_edge("e1", "e2")

    def test_get_entity(self):
        """Test getting entity from graph."""
        pytest.importorskip("networkx")

        builder = NetworkXGraphBuilder(self.config)
        builder.add_entities([self.entity1])

        entity = builder.get_entity("e1")
        assert entity is not None
        assert entity.value == "GRI 306"

    def test_export_to_knowledge_graph(self):
        """Test exporting NetworkX graph to KnowledgeGraph."""
        pytest.importorskip("networkx")

        builder = NetworkXGraphBuilder(self.config)
        builder.add_entities([self.entity1, self.entity2])
        builder.add_relationships([self.rel1])

        kg = builder.export_to_knowledge_graph()

        assert len(kg.entities) == 2
        assert len(kg.relationships) == 1


class TestFactoryFunction:
    """Tests for create_graph_builder factory function."""

    def test_create_simple_builder(self):
        """Test creating SimpleGraphBuilder."""
        config = GraphStorageConfig(backend=GraphBackend.SIMPLE)

        builder = create_graph_builder(config)

        assert isinstance(builder, SimpleGraphBuilder)

    def test_create_networkx_builder(self):
        """Test creating NetworkXGraphBuilder."""
        pytest.importorskip("networkx")

        config = GraphStorageConfig(backend=GraphBackend.NETWORKX)

        builder = create_graph_builder(config)

        assert isinstance(builder, NetworkXGraphBuilder)

    def test_invalid_backend(self):
        """Test creating builder with invalid backend."""
        config = GraphStorageConfig(backend="invalid")

        with pytest.raises(ValueError):
            create_graph_builder(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
