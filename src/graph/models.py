"""
Data models for Knowledge Graph entities and relationships.

Defines the schema for legal document knowledge graphs:
- Entity: Standards, Organizations, Dates, Clauses, Topics
- Relationship: SUPERSEDED_BY, REFERENCES, ISSUED_BY, EFFECTIVE_DATE, etc.
- KnowledgeGraph: Complete graph structure with entities and relationships
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json


class EntityType(Enum):
    """Types of entities extracted from legal documents."""

    STANDARD = "standard"
    ORGANIZATION = "organization"
    DATE = "date"
    CLAUSE = "clause"
    TOPIC = "topic"
    PERSON = "person"
    LOCATION = "location"
    REGULATION = "regulation"
    CONTRACT = "contract"

    # ---- Nuclear/legal domain additions ----
    PERMIT = "permit"  # Formal authorization (e.g., §9 stages)
    LEGAL_PROVISION = "legal_provision"  # § / article / clause in an instrument
    REACTOR = "reactor"  # Nuclear reactor unit (type, power, etc.)
    FACILITY = "facility"  # Site or installation hosting the reactor/systems
    SYSTEM = "system"  # Plant/system level (I&C, cooling, protection, etc.)
    SAFETY_FUNCTION = "safety_function"  # E.g., shutdown, heat removal, confinement
    FUEL_TYPE = "fuel_type"  # E.g., IRT‑4M, enrichment, cladding
    EMERGENCY_CLASSIFICATION = "emergency_classification"  # Event class (incident/accident)
    LIABILITY_REGIME = "liability_regime"  # Operator liability & insurance framework


class RelationshipType(Enum):
    """Types of relationships between entities."""

    # Document relationships
    SUPERSEDED_BY = "superseded_by"  # Old standard → New standard
    SUPERSEDES = "supersedes"  # New standard → Old standard
    REFERENCES = "references"  # Document A → Document B
    REFERENCED_BY = "referenced_by"  # Document B → Document A

    # Organizational relationships
    ISSUED_BY = "issued_by"  # Standard/Permit → Organization
    DEVELOPED_BY = "developed_by"  # Standard → Organization
    PUBLISHED_BY = "published_by"  # Document → Organization

    # Temporal relationships
    EFFECTIVE_DATE = "effective_date"  # Standard → Date
    EXPIRY_DATE = "expiry_date"  # Contract → Date
    SIGNED_ON = "signed_on"  # Contract → Date

    # Content relationships
    COVERS_TOPIC = "covers_topic"  # Standard → Topic
    CONTAINS_CLAUSE = "contains_clause"  # Contract → Clause
    APPLIES_TO = "applies_to"  # Regulation → Location

    # Structural relationships (hierarchy)
    PART_OF = "part_of"  # Section → Document
    CONTAINS = "contains"  # Document → Section

    # Provenance (entity → chunk)
    MENTIONED_IN = "mentioned_in"  # Entity → Chunk
    DEFINED_IN = "defined_in"  # Entity → Chunk (first occurrence)

    # ---- Nuclear/legal domain additions ----
    REGULATED_BY = "regulated_by"  # Facility/Reactor/System → Regulation/Provision
    REQUIRES_PERMIT_STAGE = "requires_permit_stage"  # Legal provision → Permit (stage)
    OPERATED_BY = "operated_by"  # Facility/Reactor → Organization (operator)
    HAS_SYSTEM = "has_system"  # Facility/Reactor → System
    PERFORMS_FUNCTION = "performs_function"  # System → Safety function
    USES_FUEL = "uses_fuel"  # Reactor → Fuel type
    SUBJECT_TO_LIABILITY = "subject_to_liability"  # Facility/Operator → Liability regime

    # Short provenance/citation links
    CITES_PROVISION = "cites_provision"  # Document/SAR → Legal provision (§…)
    DERIVED_FROM = "derived_from"  # Data/Assessment → Source entity/chunk
    VERSION_OF = "version_of"  # Document version → Canonical document



@dataclass
class Entity:
    """
    Represents an entity extracted from legal documents.

    Entities are the nodes in the knowledge graph.
    Each entity has a unique ID, type, and normalized value.
    """

    id: str  # Unique identifier (auto-generated)
    type: EntityType  # Entity type
    value: str  # Original text (e.g., "GRI 306: Effluents and Waste 2016")
    normalized_value: str  # Normalized form (e.g., "GRI 306")
    confidence: float  # Extraction confidence (0-1)

    # Provenance: where this entity was found
    source_chunk_ids: List[str] = field(default_factory=list)  # All chunks mentioning this entity
    first_mention_chunk_id: Optional[str] = None  # First chunk where entity appears

    # Context from source document
    document_id: Optional[str] = None
    section_path: Optional[str] = None

    # Type-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "llm"  # "llm", "regex", "spacy"
    extracted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_chunk_ids": self.source_chunk_ids,
            "first_mention_chunk_id": self.first_mention_chunk_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "metadata": self.metadata,
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create Entity from dictionary."""
        data_copy = data.copy()
        data_copy["type"] = EntityType(data_copy["type"])
        if data_copy.get("extracted_at"):
            data_copy["extracted_at"] = datetime.fromisoformat(data_copy["extracted_at"])
        return cls(**data_copy)

    def __hash__(self):
        """Hash based on normalized value and type for deduplication."""
        return hash((self.type, self.normalized_value))

    def __eq__(self, other):
        """Equality based on normalized value and type."""
        if not isinstance(other, Entity):
            return False
        return self.type == other.type and self.normalized_value == other.normalized_value


@dataclass
class Relationship:
    """
    Represents a relationship between two entities.

    Relationships are the edges in the knowledge graph.
    Each relationship has a source entity, target entity, and type.
    """

    id: str  # Unique identifier (auto-generated)
    type: RelationshipType  # Relationship type
    source_entity_id: str  # Source entity ID
    target_entity_id: str  # Target entity ID
    confidence: float  # Extraction confidence (0-1)

    # Provenance: where this relationship was extracted
    source_chunk_id: str  # Chunk where relationship was found
    evidence_text: str  # Supporting text snippet

    # Relationship-specific properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "llm"  # "llm", "pattern", "heuristic"
    extracted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
            "evidence_text": self.evidence_text,
            "properties": self.properties,
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create Relationship from dictionary."""
        data_copy = data.copy()
        data_copy["type"] = RelationshipType(data_copy["type"])
        if data_copy.get("extracted_at"):
            data_copy["extracted_at"] = datetime.fromisoformat(data_copy["extracted_at"])
        return cls(**data_copy)


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph structure.

    Contains all entities, relationships, and metadata.
    Can be serialized to JSON or loaded into Neo4j/SimpleGraphStore.
    """

    # Graph content
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    # Source metadata
    source_document_id: Optional[str] = None
    source_chunks_file: Optional[str] = None  # Path to phase3_chunks.json

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)

    # Graph metadata
    created_at: Optional[datetime] = None
    extraction_config: Dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_entity_by_value(
        self, normalized_value: str, entity_type: EntityType
    ) -> Optional[Entity]:
        """Get entity by normalized value and type."""
        for entity in self.entities:
            if entity.normalized_value == normalized_value and entity.type == entity_type:
                return entity
        return None

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity (as source or target)."""
        return [
            rel
            for rel in self.relationships
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]

    def get_outgoing_relationships(self, entity_id: str) -> List[Relationship]:
        """Get relationships where entity is the source."""
        return [rel for rel in self.relationships if rel.source_entity_id == entity_id]

    def get_incoming_relationships(self, entity_id: str) -> List[Relationship]:
        """Get relationships where entity is the target."""
        return [rel for rel in self.relationships if rel.target_entity_id == entity_id]

    def compute_stats(self) -> Dict[str, Any]:
        """Compute statistics about the graph."""
        entity_type_counts = {}
        for entity in self.entities:
            entity_type_counts[entity.type.value] = entity_type_counts.get(entity.type.value, 0) + 1

        relationship_type_counts = {}
        for rel in self.relationships:
            relationship_type_counts[rel.type.value] = (
                relationship_type_counts.get(rel.type.value, 0) + 1
            )

        # Average confidence
        avg_entity_confidence = (
            sum(e.confidence for e in self.entities) / len(self.entities) if self.entities else 0
        )
        avg_rel_confidence = (
            sum(r.confidence for r in self.relationships) / len(self.relationships)
            if self.relationships
            else 0
        )

        self.stats = {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
            "avg_entity_confidence": avg_entity_confidence,
            "avg_relationship_confidence": avg_rel_confidence,
        }

        return self.stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "source_document_id": self.source_document_id,
            "source_chunks_file": self.source_chunks_file,
            "stats": self.stats,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "extraction_config": self.extraction_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create KnowledgeGraph from dictionary."""
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        relationships = [Relationship.from_dict(r) for r in data.get("relationships", [])]

        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            entities=entities,
            relationships=relationships,
            source_document_id=data.get("source_document_id"),
            source_chunks_file=data.get("source_chunks_file"),
            stats=data.get("stats", {}),
            created_at=created_at,
            extraction_config=data.get("extraction_config", {}),
        )

    def save_json(self, output_path: str):
        """Save knowledge graph to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, input_path: str) -> "KnowledgeGraph":
        """Load knowledge graph from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
