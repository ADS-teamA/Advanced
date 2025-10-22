"""
Unit tests for Knowledge Graph data models.

Tests Entity, Relationship, and KnowledgeGraph classes.
"""

import pytest
from datetime import datetime
import json

from src.graph.models import (
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
    KnowledgeGraph,
)


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            id="test-1",
            type=EntityType.STANDARD,
            value="GRI 306: Waste 2020",
            normalized_value="GRI 306",
            confidence=0.95,
            source_chunk_ids=["chunk-1"],
            first_mention_chunk_id="chunk-1",
            extraction_method="llm",
            extracted_at=datetime.now(),
        )

        assert entity.id == "test-1"
        assert entity.type == EntityType.STANDARD
        assert entity.value == "GRI 306: Waste 2020"
        assert entity.normalized_value == "GRI 306"
        assert entity.confidence == 0.95

    def test_entity_serialization(self):
        """Test entity to_dict and from_dict."""
        entity = Entity(
            id="test-1",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="GSSB",
            confidence=0.9,
            source_chunk_ids=["chunk-1", "chunk-2"],
            extraction_method="llm",
        )

        # Serialize
        entity_dict = entity.to_dict()
        assert entity_dict["id"] == "test-1"
        assert entity_dict["type"] == "organization"
        assert entity_dict["value"] == "GSSB"

        # Deserialize
        entity_restored = Entity.from_dict(entity_dict)
        assert entity_restored.id == entity.id
        assert entity_restored.type == entity.type
        assert entity_restored.value == entity.value
        assert entity_restored.normalized_value == entity.normalized_value

    def test_entity_equality(self):
        """Test entity equality based on type and normalized_value."""
        entity1 = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306: Waste 2020",
            normalized_value="GRI 306",
            confidence=0.95,
            extraction_method="llm",
        )

        entity2 = Entity(
            id="e2",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.9,
            extraction_method="llm",
        )

        # Same type and normalized value → equal
        assert entity1 == entity2
        assert hash(entity1) == hash(entity2)

    def test_entity_inequality(self):
        """Test entity inequality."""
        entity1 = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.95,
            extraction_method="llm",
        )

        entity2 = Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.9,
            extraction_method="llm",
        )

        # Different types → not equal
        assert entity1 != entity2


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test creating a relationship."""
        rel = Relationship(
            id="rel-1",
            type=RelationshipType.SUPERSEDED_BY,
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.85,
            source_chunk_id="chunk-1",
            evidence_text="GRI 306:2020 supersedes GRI 306:2016",
            extraction_method="llm",
            extracted_at=datetime.now(),
        )

        assert rel.id == "rel-1"
        assert rel.type == RelationshipType.SUPERSEDED_BY
        assert rel.source_entity_id == "e1"
        assert rel.target_entity_id == "e2"
        assert rel.confidence == 0.85

    def test_relationship_serialization(self):
        """Test relationship to_dict and from_dict."""
        rel = Relationship(
            id="rel-1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="standard-1",
            target_entity_id="org-1",
            confidence=0.9,
            source_chunk_id="chunk-1",
            evidence_text="Issued by GSSB",
            properties={"key": "value"},
            extraction_method="llm",
        )

        # Serialize
        rel_dict = rel.to_dict()
        assert rel_dict["type"] == "issued_by"
        assert rel_dict["confidence"] == 0.9

        # Deserialize
        rel_restored = Relationship.from_dict(rel_dict)
        assert rel_restored.id == rel.id
        assert rel_restored.type == rel.type
        assert rel_restored.source_entity_id == rel.source_entity_id
        assert rel_restored.target_entity_id == rel.target_entity_id


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    def setup_method(self):
        """Setup test entities and relationships."""
        self.entity1 = Entity(
            id="e1",
            type=EntityType.STANDARD,
            value="GRI 306",
            normalized_value="GRI 306",
            confidence=0.95,
            extraction_method="test",
        )

        self.entity2 = Entity(
            id="e2",
            type=EntityType.ORGANIZATION,
            value="GSSB",
            normalized_value="GSSB",
            confidence=0.9,
            extraction_method="test",
        )

        self.entity3 = Entity(
            id="e3",
            type=EntityType.TOPIC,
            value="waste",
            normalized_value="waste",
            confidence=0.85,
            extraction_method="test",
        )

        self.rel1 = Relationship(
            id="r1",
            type=RelationshipType.ISSUED_BY,
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.9,
            source_chunk_id="chunk-1",
            evidence_text="GRI 306 issued by GSSB",
            extraction_method="test",
        )

        self.rel2 = Relationship(
            id="r2",
            type=RelationshipType.COVERS_TOPIC,
            source_entity_id="e1",
            target_entity_id="e3",
            confidence=0.85,
            source_chunk_id="chunk-1",
            evidence_text="GRI 306 covers waste",
            extraction_method="test",
        )

    def test_knowledge_graph_creation(self):
        """Test creating a knowledge graph."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[self.rel1],
        )

        assert len(kg.entities) == 2
        assert len(kg.relationships) == 1

    def test_get_entity(self):
        """Test getting entity by ID."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[],
        )

        entity = kg.get_entity("e1")
        assert entity is not None
        assert entity.value == "GRI 306"

        not_found = kg.get_entity("e999")
        assert not_found is None

    def test_get_entity_by_value(self):
        """Test getting entity by normalized value and type."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[],
        )

        entity = kg.get_entity_by_value("GRI 306", EntityType.STANDARD)
        assert entity is not None
        assert entity.id == "e1"

        not_found = kg.get_entity_by_value("GRI 999", EntityType.STANDARD)
        assert not_found is None

    def test_get_relationships_for_entity(self):
        """Test getting all relationships for an entity."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2, self.entity3],
            relationships=[self.rel1, self.rel2],
        )

        # Entity e1 has 2 relationships (as source)
        rels = kg.get_relationships_for_entity("e1")
        assert len(rels) == 2

        # Entity e2 has 1 relationship (as target)
        rels = kg.get_relationships_for_entity("e2")
        assert len(rels) == 1

    def test_get_outgoing_relationships(self):
        """Test getting outgoing relationships."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[self.rel1, self.rel2],
        )

        outgoing = kg.get_outgoing_relationships("e1")
        assert len(outgoing) == 2
        assert all(r.source_entity_id == "e1" for r in outgoing)

    def test_get_incoming_relationships(self):
        """Test getting incoming relationships."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[self.rel1],
        )

        incoming = kg.get_incoming_relationships("e2")
        assert len(incoming) == 1
        assert incoming[0].target_entity_id == "e2"

    def test_compute_stats(self):
        """Test computing graph statistics."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2, self.entity3],
            relationships=[self.rel1, self.rel2],
        )

        stats = kg.compute_stats()

        assert stats["total_entities"] == 3
        assert stats["total_relationships"] == 2
        assert stats["entity_type_counts"]["standard"] == 1
        assert stats["entity_type_counts"]["organization"] == 1
        assert stats["entity_type_counts"]["topic"] == 1
        assert stats["relationship_type_counts"]["issued_by"] == 1
        assert stats["relationship_type_counts"]["covers_topic"] == 1

    def test_knowledge_graph_serialization(self):
        """Test KnowledgeGraph to_dict and from_dict."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[self.rel1],
            source_document_id="doc-1",
        )

        kg.compute_stats()

        # Serialize
        kg_dict = kg.to_dict()
        assert len(kg_dict["entities"]) == 2
        assert len(kg_dict["relationships"]) == 1
        assert kg_dict["source_document_id"] == "doc-1"

        # Deserialize
        kg_restored = KnowledgeGraph.from_dict(kg_dict)
        assert len(kg_restored.entities) == 2
        assert len(kg_restored.relationships) == 1
        assert kg_restored.source_document_id == "doc-1"

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading KnowledgeGraph from JSON."""
        kg = KnowledgeGraph(
            entities=[self.entity1, self.entity2],
            relationships=[self.rel1],
        )

        kg.compute_stats()

        # Save to file
        output_path = tmp_path / "test_graph.json"
        kg.save_json(str(output_path))

        assert output_path.exists()

        # Load from file
        kg_loaded = KnowledgeGraph.load_json(str(output_path))

        assert len(kg_loaded.entities) == 2
        assert len(kg_loaded.relationships) == 1
        assert kg_loaded.stats["total_entities"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
