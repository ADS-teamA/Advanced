"""
Configuration for Knowledge Graph pipeline.

Defines settings for entity extraction, relationship extraction,
graph storage backends, and integration with the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional
from enum import Enum
import os

from .models import EntityType, RelationshipType

# Import LLMTaskConfig from main config
try:
    from src.config import LLMTaskConfig
except ImportError:
    # Fallback if src.config not available
    LLMTaskConfig = None


class GraphBackend(Enum):
    """Graph storage backend options."""

    NEO4J = "neo4j"           # Neo4j graph database (production)
    SIMPLE = "simple"         # SimpleGraphStore (development/testing)
    NETWORKX = "networkx"     # NetworkX in-memory (lightweight)


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction."""

    # LLM configuration
    llm_config: Optional['LLMTaskConfig'] = None

    # Enabled entity types
    enabled_entity_types: Set[EntityType] = field(default_factory=lambda: {
        EntityType.STANDARD,
        EntityType.ORGANIZATION,
        EntityType.DATE,
        EntityType.CLAUSE,
        EntityType.TOPIC,
        EntityType.REGULATION,
    })

    # Extraction parameters
    min_confidence: float = 0.6            # Minimum confidence threshold
    extract_definitions: bool = True       # Extract entity definitions
    normalize_entities: bool = True        # Normalize entity values

    # Performance settings
    batch_size: int = 10                   # Chunks per batch
    max_workers: int = 5                   # Parallel extraction threads
    cache_results: bool = True             # Cache extraction results

    # Prompt settings
    include_examples: bool = True          # Include few-shot examples in prompt
    max_entities_per_chunk: int = 50       # Max entities per chunk

    # Legacy compatibility (deprecated)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = None

    def __post_init__(self):
        """Initialize LLM config from legacy fields if provided."""
        if self.llm_config is None and LLMTaskConfig is not None:
            # Use legacy fields or defaults
            self.llm_config = LLMTaskConfig(
                provider=self.llm_provider or "openai",
                model=self.llm_model or "gpt-4o-mini",
                temperature=self.temperature if self.temperature is not None else 0.0,
                max_tokens=500
            )


@dataclass
class RelationshipExtractionConfig:
    """Configuration for relationship extraction."""

    # LLM configuration
    llm_config: Optional['LLMTaskConfig'] = None

    # Enabled relationship types
    enabled_relationship_types: Set[RelationshipType] = field(default_factory=lambda: {
        RelationshipType.SUPERSEDED_BY,
        RelationshipType.SUPERSEDES,
        RelationshipType.REFERENCES,
        RelationshipType.ISSUED_BY,
        RelationshipType.EFFECTIVE_DATE,
        RelationshipType.COVERS_TOPIC,
        RelationshipType.CONTAINS_CLAUSE,
    })

    # Extraction parameters
    min_confidence: float = 0.5
    extract_evidence: bool = True          # Extract supporting evidence text
    max_evidence_length: int = 200         # Max characters for evidence

    # Extraction strategies
    extract_within_chunk: bool = True      # Extract relationships within single chunk
    extract_cross_chunk: bool = True       # Extract relationships across chunks
    extract_from_metadata: bool = True     # Extract from chunk metadata (section_path, etc.)

    # Performance settings
    batch_size: int = 5                    # Entity pairs per batch
    max_workers: int = 5
    cache_results: bool = True

    # Advanced settings
    max_relationships_per_entity: int = 100  # Limit relationships per entity

    # Legacy compatibility (deprecated)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = None

    def __post_init__(self):
        """Initialize LLM config from legacy fields if provided."""
        if self.llm_config is None and LLMTaskConfig is not None:
            # Use legacy fields or defaults
            self.llm_config = LLMTaskConfig(
                provider=self.llm_provider or "openai",
                model=self.llm_model or "gpt-4o-mini",
                temperature=self.temperature if self.temperature is not None else 0.0,
                max_tokens=500
            )


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""                     # Load from environment
    database: str = "neo4j"

    # Connection settings
    max_connection_lifetime: int = 3600    # seconds
    max_connection_pool_size: int = 50
    connection_timeout: int = 30           # seconds

    # Graph settings
    create_indexes: bool = True            # Auto-create indexes for entity types
    create_constraints: bool = True        # Auto-create uniqueness constraints

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
class GraphStorageConfig:
    """Configuration for graph storage."""

    backend: GraphBackend = GraphBackend.SIMPLE

    # Backend-specific configs
    neo4j_config: Optional[Neo4jConfig] = None

    # SimpleGraphStore settings
    simple_store_path: str = "./data/graphs/simple_graph.json"

    # Export settings
    export_json: bool = True               # Export to JSON after construction
    export_path: str = "./data/graphs/knowledge_graph.json"

    # Graph optimization
    deduplicate_entities: bool = True      # Merge duplicate entities
    merge_similar_entities: bool = False   # Merge similar entities (expensive)
    similarity_threshold: float = 0.9      # For entity merging

    # Provenance
    track_provenance: bool = True          # Track chunk sources for entities


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
    relationship_extraction: RelationshipExtractionConfig = field(default_factory=RelationshipExtractionConfig)

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
    max_retries: int = 3                   # Retry failed extractions
    retry_delay: float = 1.0               # seconds
    timeout: int = 300                     # seconds per extraction batch

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
            KG_ENTITY_LLM_PROVIDER: LLM provider for entity extraction (overrides KG_LLM_PROVIDER)
            KG_ENTITY_LLM_MODEL: Model for entity extraction (overrides KG_LLM_MODEL)
            KG_RELATIONSHIP_LLM_PROVIDER: LLM provider for relationship extraction
            KG_RELATIONSHIP_LLM_MODEL: Model for relationship extraction
            KG_BACKEND: Graph backend (neo4j, simple, networkx)
            KG_EXPORT_PATH: Path to export JSON graph
            ANTHROPIC_API_KEY: API key for Claude
            OPENAI_API_KEY: API key for OpenAI
        """
        # Entity extraction config (with task-specific overrides)
        entity_llm_config = None
        if LLMTaskConfig is not None:
            entity_llm_config = LLMTaskConfig(
                provider=os.getenv("KG_ENTITY_LLM_PROVIDER", os.getenv("KG_LLM_PROVIDER", "openai")),
                model=os.getenv("KG_ENTITY_LLM_MODEL", os.getenv("KG_LLM_MODEL", "gpt-4o-mini")),
                temperature=0.0,
                max_tokens=500
            )

        entity_extraction = EntityExtractionConfig(
            llm_config=entity_llm_config,
        )

        # Relationship extraction config (with task-specific overrides)
        relationship_llm_config = None
        if LLMTaskConfig is not None:
            relationship_llm_config = LLMTaskConfig(
                provider=os.getenv("KG_RELATIONSHIP_LLM_PROVIDER", os.getenv("KG_LLM_PROVIDER", "openai")),
                model=os.getenv("KG_RELATIONSHIP_LLM_MODEL", os.getenv("KG_LLM_MODEL", "gpt-4o-mini")),
                temperature=0.0,
                max_tokens=500
            )

        relationship_extraction = RelationshipExtractionConfig(
            llm_config=relationship_llm_config,
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
        if self.entity_extraction.llm_config:
            provider = self.entity_extraction.llm_config.provider
            if provider in ["anthropic", "claude"] and not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY required for anthropic provider")

            if provider == "openai" and not self.openai_api_key:
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
    entity_llm = None
    relationship_llm = None

    if LLMTaskConfig is not None:
        entity_llm = LLMTaskConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=500
        )
        relationship_llm = LLMTaskConfig(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=500
        )

    return KnowledgeGraphConfig(
        graph_storage=GraphStorageConfig(
            backend=GraphBackend.NEO4J,
            neo4j_config=Neo4jConfig.from_env(),
        ),
        entity_extraction=EntityExtractionConfig(
            llm_config=entity_llm,
            batch_size=20,
            max_workers=10,
        ),
        relationship_extraction=RelationshipExtractionConfig(
            llm_config=relationship_llm,
            batch_size=10,
            max_workers=10,
        ),
    )


def get_development_config() -> KnowledgeGraphConfig:
    """Development config: Fast, minimal extraction for testing."""
    entity_llm = None
    relationship_llm = None

    if LLMTaskConfig is not None:
        entity_llm = LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        )
        relationship_llm = LLMTaskConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=500
        )

    return KnowledgeGraphConfig(
        graph_storage=GraphStorageConfig(
            backend=GraphBackend.SIMPLE,
            simple_store_path="./data/graphs/dev_graph.json",
        ),
        entity_extraction=EntityExtractionConfig(
            llm_config=entity_llm,
            batch_size=5,
            max_workers=3,
            enabled_entity_types={
                EntityType.STANDARD,
                EntityType.ORGANIZATION,
                EntityType.TOPIC,
            },
        ),
        relationship_extraction=RelationshipExtractionConfig(
            llm_config=relationship_llm,
            batch_size=3,
            max_workers=3,
            enabled_relationship_types={
                RelationshipType.SUPERSEDED_BY,
                RelationshipType.REFERENCES,
                RelationshipType.COVERS_TOPIC,
            },
        ),
        verbose=True,
    )
