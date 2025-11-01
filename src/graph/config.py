"""
Configuration for Knowledge Graph pipeline.

Defines settings for entity extraction, relationship extraction,
graph storage backends, and integration with the RAG pipeline.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

from .models import EntityType, RelationshipType


class GraphBackend(Enum):
    """Graph storage backend options."""

    NEO4J = "neo4j"  # Neo4j graph database (production)
    SIMPLE = "simple"  # SimpleGraphStore (development/testing)
    NETWORKX = "networkx"  # NetworkX in-memory (lightweight)


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction."""

    # LLM settings for extraction
    llm_provider: str = "openai"  # "openai", "anthropic"
    llm_model: str = "gpt-4o-mini"  # Fast, cost-effective for extraction
    temperature: float = 0.0  # Deterministic extraction

    # Enabled entity types
    enabled_entity_types: Set[EntityType] = field(
        default_factory=lambda: {
            EntityType.STANDARD,
            EntityType.ORGANIZATION,
            EntityType.DATE,
            EntityType.CLAUSE,
            EntityType.TOPIC,
            EntityType.REGULATION,
            EntityType.LOCATION,  # Added: For jurisdictions, countries, etc.
            EntityType.PERSON,  # Added: For completeness (authors, signatories)
            EntityType.CONTRACT,  # Added: For completeness (NDAs, MSAs)
        }
    )

    # Extraction parameters
    min_confidence: float = 0.6  # Minimum confidence threshold
    extract_definitions: bool = True  # Extract entity definitions
    normalize_entities: bool = True  # Normalize entity values

    # Performance settings
    # OPTIMIZED: Zvýšeno pro rychlejší zpracování (2× rychlejší)
    batch_size: int = 20  # Chunks per batch
    max_workers: int = 10  # Parallel extraction threads
    cache_results: bool = True  # Cache extraction results

    # Prompt settings
    include_examples: bool = True  # Include few-shot examples in prompt
    max_entities_per_chunk: int = 50  # Max entities per chunk


@dataclass
class RelationshipExtractionConfig:
    """Configuration for relationship extraction."""

    # LLM settings for extraction
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Enabled relationship types
    enabled_relationship_types: Set[RelationshipType] = field(
        default_factory=lambda: {
            # Document relationships
            RelationshipType.SUPERSEDED_BY,
            RelationshipType.SUPERSEDES,
            RelationshipType.REFERENCES,
            RelationshipType.REFERENCED_BY,
            # Organizational relationships
            RelationshipType.ISSUED_BY,
            RelationshipType.DEVELOPED_BY,
            RelationshipType.PUBLISHED_BY,
            # Temporal relationships
            RelationshipType.EFFECTIVE_DATE,
            RelationshipType.EXPIRY_DATE,  # Added: For contract expiry dates
            RelationshipType.SIGNED_ON,
            # Content relationships
            RelationshipType.COVERS_TOPIC,
            RelationshipType.CONTAINS_CLAUSE,
            RelationshipType.APPLIES_TO,  # Added: For regulation → location
            # Structural relationships
            RelationshipType.PART_OF,
            RelationshipType.CONTAINS,
            # Provenance relationships
            RelationshipType.MENTIONED_IN,  # Added: Entity → Chunk provenance
            RelationshipType.DEFINED_IN,
        }
    )

    # Extraction parameters
    min_confidence: float = 0.5
    extract_evidence: bool = True  # Extract supporting evidence text
    max_evidence_length: int = 200  # Max characters for evidence

    # Extraction strategies
    extract_within_chunk: bool = True  # Extract relationships within single chunk
    extract_cross_chunk: bool = True  # Extract relationships across chunks
    extract_from_metadata: bool = True  # Extract from chunk metadata (section_path, etc.)

    # Performance settings
    # OPTIMIZED: Zvýšeno pro rychlejší zpracování (2× rychlejší)
    batch_size: int = 10  # Entity pairs per batch
    max_workers: int = 10
    cache_results: bool = True

    # Advanced settings
    max_relationships_per_entity: int = 100  # Limit relationships per entity


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""  # Load from environment
    database: str = "neo4j"

    # Connection settings
    max_connection_lifetime: int = 3600  # seconds
    max_connection_pool_size: int = 50
    connection_timeout: int = 30  # seconds

    # Graph settings
    create_indexes: bool = True  # Auto-create indexes for entity types
    create_constraints: bool = True  # Auto-create uniqueness constraints

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load Neo4j config from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class EntityDeduplicationConfig:
    """
    Configuration for entity deduplication during indexing.

    Three-layer deduplication strategy:
    - Layer 1 (Exact): Fast exact match on (type, normalized_value) - <1ms
    - Layer 2 (Semantic): Embedding similarity for variants - 50-200ms
    - Layer 3 (Acronym): Acronym expansion + fuzzy match - 100-500ms
    """

    # Master enable/disable
    enabled: bool = True

    # Layer 1: Exact match (always enabled if master enabled)
    exact_match_enabled: bool = True

    # Layer 2: Embedding-based similarity
    use_embeddings: bool = False
    embedding_model: str = "text-embedding-3-large"
    similarity_threshold: float = 0.90  # Cosine similarity threshold
    embedding_batch_size: int = 100
    cache_embeddings: bool = True

    # Layer 3: Acronym expansion
    use_acronym_expansion: bool = False
    acronym_fuzzy_threshold: float = 0.85  # Fuzzy match threshold
    custom_acronyms: dict = field(default_factory=dict)  # Custom acronym mappings

    # Neo4j-specific
    apoc_enabled: bool = True  # Try APOC, fallback to Cypher
    create_uniqueness_constraints: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got {self.similarity_threshold}"
            )

        if not (0.0 <= self.acronym_fuzzy_threshold <= 1.0):
            raise ValueError(
                f"acronym_fuzzy_threshold must be in [0.0, 1.0], got {self.acronym_fuzzy_threshold}"
            )

        if self.embedding_batch_size <= 0:
            raise ValueError(
                f"embedding_batch_size must be positive, got {self.embedding_batch_size}"
            )

    @classmethod
    def from_env(cls) -> "EntityDeduplicationConfig":
        """Load deduplication config from environment variables."""
        import os

        custom_acronyms = {}
        acronym_str = os.getenv("KG_DEDUP_CUSTOM_ACRONYMS", "")
        if acronym_str:
            # Parse format: "ACRO1:expansion1,ACRO2:expansion2"
            for pair in acronym_str.split(","):
                if ":" in pair:
                    acro, expansion = pair.split(":", 1)
                    custom_acronyms[acro.strip()] = expansion.strip()

        return cls(
            enabled=os.getenv("KG_DEDUPLICATE_ENTITIES", "true").lower() == "true",
            use_embeddings=os.getenv("KG_DEDUP_USE_EMBEDDINGS", "false").lower() == "true",
            similarity_threshold=float(os.getenv("KG_DEDUP_SIMILARITY_THRESHOLD", "0.90")),
            use_acronym_expansion=os.getenv("KG_DEDUP_USE_ACRONYM_EXPANSION", "false").lower()
            == "true",
            acronym_fuzzy_threshold=float(os.getenv("KG_DEDUP_ACRONYM_FUZZY_THRESHOLD", "0.85")),
            custom_acronyms=custom_acronyms,
            apoc_enabled=os.getenv("KG_DEDUP_APOC_ENABLED", "true").lower() == "true",
        )


@dataclass
class GraphStorageConfig:
    """Configuration for graph storage."""

    backend: GraphBackend = GraphBackend.SIMPLE

    # Backend-specific configs
    neo4j_config: Optional[Neo4jConfig] = None

    # SimpleGraphStore settings
    simple_store_path: str = "./data/graphs/simple_graph.json"

    # Export settings
    export_json: bool = True  # Export to JSON after construction
    export_path: str = "./data/graphs/knowledge_graph.json"

    # Graph optimization (DEPRECATED - use deduplication_config)
    deduplicate_entities: bool = True  # Merge duplicate entities
    merge_similar_entities: bool = False  # Merge similar entities (expensive)
    similarity_threshold: float = 0.9  # For entity merging

    # Deduplication configuration (NEW)
    deduplication_config: Optional["EntityDeduplicationConfig"] = None

    # Provenance
    track_provenance: bool = True  # Track chunk sources for entities


@dataclass
class KnowledgeGraphConfig:
    """
    Complete configuration for Knowledge Graph pipeline.

    Usage:
        # Default configuration
        config = KnowledgeGraphConfig()

        # Custom configuration
        config = KnowledgeGraphConfig(
            entity_extraction=EntityExtractionConfig(
                llm_model="claude-haiku-4-5-20250929"
            ),
            graph_storage=GraphStorageConfig(
                backend=GraphBackend.NEO4J,
                neo4j_config=Neo4jConfig.from_env()
            )
        )
    """

    # Extraction configs
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    relationship_extraction: RelationshipExtractionConfig = field(
        default_factory=RelationshipExtractionConfig
    )

    # Storage config
    graph_storage: GraphStorageConfig = field(default_factory=GraphStorageConfig)

    # API keys (loaded from environment)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Pipeline settings
    enable_entity_extraction: bool = True
    enable_relationship_extraction: bool = True
    enable_cross_document_relationships: bool = False  # Expensive, for multi-doc graphs

    # Performance settings
    max_retries: int = 3  # Retry failed extractions
    retry_delay: float = 1.0  # seconds
    timeout: int = 300  # seconds per extraction batch

    # Logging
    verbose: bool = True
    log_path: Optional[str] = "./logs/kg_extraction.log"

    @classmethod
    def from_env(cls) -> "KnowledgeGraphConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            KG_LLM_PROVIDER: LLM provider for extraction (openai, anthropic)
            KG_LLM_MODEL: Model name for extraction
            KG_BACKEND: Graph backend (neo4j, simple, networkx)
            KG_EXPORT_PATH: Path to export JSON graph
            ANTHROPIC_API_KEY: API key for Claude
            OPENAI_API_KEY: API key for OpenAI
        """
        entity_extraction = EntityExtractionConfig(
            llm_provider=os.getenv("KG_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("KG_LLM_MODEL", "gpt-4o-mini"),
        )

        relationship_extraction = RelationshipExtractionConfig(
            llm_provider=os.getenv("KG_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("KG_LLM_MODEL", "gpt-4o-mini"),
        )

        backend_str = os.getenv("KG_BACKEND", "simple").lower()
        backend = GraphBackend.SIMPLE
        if backend_str == "neo4j":
            backend = GraphBackend.NEO4J
        elif backend_str == "networkx":
            backend = GraphBackend.NETWORKX

        graph_storage = GraphStorageConfig(
            backend=backend,
            neo4j_config=Neo4jConfig.from_env() if backend == GraphBackend.NEO4J else None,
            export_path=os.getenv("KG_EXPORT_PATH", "./data/graphs/knowledge_graph.json"),
        )

        return cls(
            entity_extraction=entity_extraction,
            relationship_extraction=relationship_extraction,
            graph_storage=graph_storage,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=os.getenv("KG_VERBOSE", "true").lower() == "true",
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        # Check API keys
        if self.entity_extraction.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for anthropic provider")

        if self.entity_extraction.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai provider")

        # Check Neo4j config
        if self.graph_storage.backend == GraphBackend.NEO4J and not self.graph_storage.neo4j_config:
            raise ValueError("neo4j_config required when backend is NEO4J")

        # Check thresholds
        if not (0 <= self.entity_extraction.min_confidence <= 1):
            raise ValueError("entity_extraction.min_confidence must be in [0, 1]")

        if not (0 <= self.relationship_extraction.min_confidence <= 1):
            raise ValueError("relationship_extraction.min_confidence must be in [0, 1]")

    def get_model_alias(self, provider: str, model: str) -> str:
        """Convert model name to alias used in existing pipeline."""
        # Map to aliases in src/config.py
        aliases = {
            "claude-haiku-4-5-20250929": "haiku",
            "claude-sonnet-4-5-20250929": "sonnet",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o": "gpt-4o",
        }
        return aliases.get(model, model)


# Preset configurations for common use cases
def get_default_config() -> KnowledgeGraphConfig:
    """Default config: Fast extraction with SimpleGraphStore."""
    return KnowledgeGraphConfig()


def get_production_config() -> KnowledgeGraphConfig:
    """Production config: Neo4j backend with full extraction."""
    return KnowledgeGraphConfig(
        graph_storage=GraphStorageConfig(
            backend=GraphBackend.NEO4J,
            neo4j_config=Neo4jConfig.from_env(),
        ),
        entity_extraction=EntityExtractionConfig(
            llm_model="gpt-4o",  # More accurate for production
            batch_size=20,
            max_workers=10,
        ),
        relationship_extraction=RelationshipExtractionConfig(
            llm_model="gpt-4o",
            batch_size=10,
            max_workers=10,
        ),
    )


def get_development_config() -> KnowledgeGraphConfig:
    """Development config: Fast, minimal extraction for testing."""
    return KnowledgeGraphConfig(
        graph_storage=GraphStorageConfig(
            backend=GraphBackend.SIMPLE,
            simple_store_path="./data/graphs/dev_graph.json",
        ),
        entity_extraction=EntityExtractionConfig(
            llm_model="gpt-4o-mini",
            batch_size=5,
            max_workers=3,
            enabled_entity_types={
                EntityType.STANDARD,
                EntityType.ORGANIZATION,
                EntityType.TOPIC,
                EntityType.LOCATION,  # Added for dev testing
            },
        ),
        relationship_extraction=RelationshipExtractionConfig(
            llm_model="gpt-4o-mini",
            batch_size=3,
            max_workers=3,
            enabled_relationship_types={
                RelationshipType.SUPERSEDED_BY,
                RelationshipType.REFERENCES,
                RelationshipType.COVERS_TOPIC,
                RelationshipType.MENTIONED_IN,  # Added for dev testing
                RelationshipType.APPLIES_TO,  # Added for dev testing
            },
        ),
        verbose=True,
    )
