"""
TIER 2: Advanced Retrieval Tools

Quality tools (500-1000ms) for complex retrieval tasks.
Use when Tier 1 tools are insufficient.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 2 TOOLS: Advanced Retrieval
# ============================================================================


# ----------------------------------------------------------------------------
# Unified Graph Search Tool
# ----------------------------------------------------------------------------


class GraphSearchInput(ToolInput):
    """Input for unified graph_search tool with multiple modes."""

    mode: str = Field(
        ...,
        description=(
            "Search mode:\n"
            "- 'entity_mentions': Find chunks mentioning an entity (fast, ~300ms)\n"
            "- 'entity_details': Get entity details + relationships + mentions (medium, ~500ms)\n"
            "- 'relationships': Query relationships between entities (medium, ~400ms)\n"
            "- 'multi_hop': Multi-hop BFS graph traversal (slow, ~1-2s)"
        ),
    )

    # Entity identification (required for all modes)
    entity_value: str = Field(
        ...,
        description="Entity value to search for (e.g., 'GRI 306', 'GSSB', 'waste management')",
    )

    entity_type: Optional[str] = Field(
        None,
        description=(
            "Entity type filter (optional): 'standard', 'organization', 'date', 'clause', "
            "'topic', 'person', 'location', 'regulation', 'contract'"
        ),
    )

    # Result control
    k: int = Field(
        6,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )

    # Relationship filtering (for 'relationships' and 'multi_hop' modes)
    relationship_types: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by relationship types (optional): 'superseded_by', 'references', 'issued_by', "
            "'effective_date', 'covers_topic', 'contains_clause', 'applies_to', 'part_of', etc."
        ),
    )

    direction: str = Field(
        "both",
        description="Relationship direction: 'outgoing' (from entity), 'incoming' (to entity), 'both'",
    )

    # Multi-hop parameters (for 'multi_hop' mode)
    max_hops: int = Field(
        2,
        description="Maximum hops for multi-hop traversal (1-3)",
        ge=1,
        le=3,
    )

    # Filtering
    min_confidence: float = Field(
        0.0,
        description="Minimum confidence score for entities/relationships (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    cross_document: bool = Field(
        True,
        description="Allow traversal across documents (True) or stay within same document (False)",
    )

    # Output control
    include_metadata: bool = Field(
        True,
        description="Include detailed entity/relationship metadata in results",
    )


@register_tool
class GraphSearchTool(BaseTool):
    """Unified knowledge graph search with multiple modes."""

    name = "graph_search"
    description = "Unified knowledge graph search (entity mentions, details, relationships, multi-hop)"
    detailed_help = """
    Unified tool for searching the knowledge graph with 4 modes:

    **Mode 1: entity_mentions** (~300ms)
    Find all chunks mentioning a specific entity.
    Example: Find all mentions of "GRI 306" standard across documents.
    Use when: You know an entity and want to see where it's discussed.

    **Mode 2: entity_details** (~500ms)
    Get comprehensive entity information: properties, relationships, and chunk mentions.
    Example: Get full details about "GSSB" organization including what it issued, when, and where mentioned.
    Use when: You need complete information about a specific entity.

    **Mode 3: relationships** (~400ms)
    Query relationships between entities with filtering.
    Example: Find all "superseded_by" relationships for standards, or what topics are "covered" by GRI 306.
    Use when: You need to understand connections between entities.

    **Mode 4: multi_hop** (~1-2s)
    Multi-hop BFS traversal following relationships across the graph.
    Example: "What topics are covered by standards issued by GSSB?" (GSSB → issued_by → Standards → covers_topic → Topics)
    Use when: Complex queries requiring following chains of relationships.

    **Parameters:**
    - entity_value: Entity to search for (e.g., "GRI 306", "GSSB")
    - entity_type: Optional type filter (standard, organization, topic, etc.)
    - k: Max results to return (default: 6)
    - relationship_types: Filter relationships (e.g., ["superseded_by", "references"])
    - direction: "outgoing", "incoming", or "both" (default: both)
    - max_hops: For multi_hop mode, how many hops (1-3, default: 2)
    - min_confidence: Filter low-confidence extractions (0.0-1.0, default: 0.0)
    - cross_document: Allow cross-document traversal (default: True)
    - include_metadata: Include detailed metadata (default: True)

    **Entity Types:**
    standard, organization, date, clause, topic, person, location, regulation, contract

    **Relationship Types:**
    superseded_by, supersedes, references, referenced_by, issued_by, developed_by, published_by,
    effective_date, expiry_date, signed_on, covers_topic, contains_clause, applies_to,
    part_of, contains, mentioned_in, defined_in

    **Examples:**
    1. entity_mentions: {"mode": "entity_mentions", "entity_value": "GRI 306", "k": 10}
    2. entity_details: {"mode": "entity_details", "entity_value": "GSSB", "entity_type": "organization"}
    3. relationships: {"mode": "relationships", "entity_value": "GRI 306", "relationship_types": ["superseded_by", "supersedes"]}
    4. multi_hop: {"mode": "multi_hop", "entity_value": "GSSB", "relationship_types": ["issued_by", "covers_topic"], "max_hops": 2}
    """
    tier = 2
    input_schema = GraphSearchInput
    requires_kg = True

    def execute_impl(
        self,
        mode: str,
        entity_value: str,
        entity_type: Optional[str] = None,
        k: int = 6,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        max_hops: int = 2,
        min_confidence: float = 0.0,
        cross_document: bool = True,
        include_metadata: bool = True,
    ) -> ToolResult:
        """Execute graph search based on mode."""

        # Validate knowledge graph availability
        if not self.knowledge_graph:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available. Run indexing with enable_knowledge_graph=True.",
                metadata={"mode": mode, "entity_value": entity_value},
            )

        # Validate mode
        valid_modes = ["entity_mentions", "entity_details", "relationships", "multi_hop"]
        if mode not in valid_modes:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
                metadata={"mode": mode},
            )

        logger.info(
            f"graph_search: mode={mode}, entity={entity_value}, type={entity_type}, k={k}"
        )

        # Dispatch to appropriate mode handler
        try:
            if mode == "entity_mentions":
                return self._entity_mentions_search(
                    entity_value, entity_type, k, min_confidence, include_metadata
                )
            elif mode == "entity_details":
                return self._entity_details_search(
                    entity_value, entity_type, direction, min_confidence, include_metadata
                )
            elif mode == "relationships":
                return self._relationships_search(
                    entity_value,
                    entity_type,
                    relationship_types,
                    direction,
                    k,
                    min_confidence,
                    include_metadata,
                )
            elif mode == "multi_hop":
                return self._multi_hop_bfs(
                    entity_value,
                    entity_type,
                    relationship_types,
                    max_hops,
                    k,
                    min_confidence,
                    cross_document,
                    include_metadata,
                )
        except Exception as e:
            logger.error(f"graph_search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Graph search failed: {str(e)}",
                metadata={"mode": mode, "entity_value": entity_value},
            )

    def _find_entity(
        self, entity_value: str, entity_type: Optional[str] = None, min_confidence: float = 0.0
    ) -> Optional[object]:
        """Find entity in knowledge graph by value and optional type."""
        entity_value_lower = entity_value.lower().strip()

        # Search through all entities
        candidates = []
        for entity in self.knowledge_graph.entities:
            if entity.confidence < min_confidence:
                continue

            # Check type match if specified
            if entity_type and entity.type.value != entity_type:
                continue

            # Check value match (normalized or original)
            entity_norm_lower = entity.normalized_value.lower()
            entity_val_lower = entity.value.lower()

            if entity_value_lower in entity_norm_lower or entity_value_lower in entity_val_lower:
                candidates.append(entity)

        # Return best match (highest confidence)
        if candidates:
            candidates.sort(key=lambda e: e.confidence, reverse=True)
            return candidates[0]

        return None

    def _entity_mentions_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        k: int,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 1: Find chunks mentioning an entity."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=f"Entity '{entity_value}' not found in knowledge graph"
                + (f" with type '{entity_type}'" if entity_type else ""),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get chunk IDs mentioning this entity
        chunk_ids = list(entity.source_chunk_ids)[:k]

        if not chunk_ids:
            return ToolResult(
                success=True,
                data={
                    "entity": {
                        "id": entity.id,
                        "type": entity.type.value,
                        "value": entity.value,
                        "confidence": entity.confidence,
                    },
                    "chunks": [],
                    "count": 0,
                },
                metadata={"mode": "entity_mentions", "entity_id": entity.id},
            )

        # Retrieve actual chunks from vector store
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.vector_store.get_chunk_by_id(chunk_id)
            if chunk:
                formatted = format_chunk_result(chunk, include_score=False)
                chunks.append(formatted)

        # Build result
        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type.value,
                "value": entity.value,
                "normalized_value": entity.normalized_value,
                "confidence": entity.confidence,
            },
            "chunks": chunks,
            "count": len(chunks),
        }

        if include_metadata:
            result_data["entity"]["metadata"] = entity.metadata
            result_data["entity"]["extraction_method"] = entity.extraction_method

        citations = [f"{c['document_id']}:{c['chunk_id']}" for c in chunks]

        return ToolResult(
            success=True,
            data=result_data,
            citations=citations,
            metadata={
                "mode": "entity_mentions",
                "entity_id": entity.id,
                "chunks_found": len(chunks),
            },
        )

    def _entity_details_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        direction: str,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 2: Get comprehensive entity details."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=f"Entity '{entity_value}' not found in knowledge graph"
                + (f" with type '{entity_type}'" if entity_type else ""),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get relationships
        if direction == "outgoing":
            relationships = self.knowledge_graph.get_outgoing_relationships(entity.id)
        elif direction == "incoming":
            relationships = self.knowledge_graph.get_incoming_relationships(entity.id)
        else:  # both
            relationships = self.knowledge_graph.get_relationships_for_entity(entity.id)

        # Filter by confidence
        relationships = [r for r in relationships if r.confidence >= min_confidence]

        # Format relationships
        formatted_relationships = []
        for rel in relationships:
            source = self.knowledge_graph.get_entity(rel.source_entity_id)
            target = self.knowledge_graph.get_entity(rel.target_entity_id)

            rel_data = {
                "type": rel.type.value,
                "source": {"id": source.id, "value": source.value, "type": source.type.value}
                if source
                else None,
                "target": {"id": target.id, "value": target.value, "type": target.type.value}
                if target
                else None,
                "confidence": rel.confidence,
            }

            if include_metadata:
                rel_data["evidence_text"] = rel.evidence_text
                rel_data["source_chunk_id"] = rel.source_chunk_id
                rel_data["properties"] = rel.properties

            formatted_relationships.append(rel_data)

        # Build result
        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type.value,
                "value": entity.value,
                "normalized_value": entity.normalized_value,
                "confidence": entity.confidence,
                "source_chunk_ids": entity.source_chunk_ids,
                "first_mention_chunk_id": entity.first_mention_chunk_id,
            },
            "relationships": formatted_relationships,
            "relationship_count": len(formatted_relationships),
        }

        if include_metadata:
            result_data["entity"]["metadata"] = entity.metadata
            result_data["entity"]["document_id"] = entity.document_id
            result_data["entity"]["section_path"] = entity.section_path
            result_data["entity"]["extraction_method"] = entity.extraction_method

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "mode": "entity_details",
                "entity_id": entity.id,
                "relationships_found": len(formatted_relationships),
            },
        )

    def _relationships_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        relationship_types: Optional[List[str]],
        direction: str,
        k: int,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 3: Query relationships for an entity."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=f"Entity '{entity_value}' not found in knowledge graph"
                + (f" with type '{entity_type}'" if entity_type else ""),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get relationships based on direction
        if direction == "outgoing":
            relationships = self.knowledge_graph.get_outgoing_relationships(entity.id)
        elif direction == "incoming":
            relationships = self.knowledge_graph.get_incoming_relationships(entity.id)
        else:  # both
            relationships = self.knowledge_graph.get_relationships_for_entity(entity.id)

        # Filter by confidence
        relationships = [r for r in relationships if r.confidence >= min_confidence]

        # Filter by relationship types if specified
        if relationship_types:
            rel_types_lower = [rt.lower() for rt in relationship_types]
            relationships = [r for r in relationships if r.type.value in rel_types_lower]

        # Limit to k
        relationships = relationships[:k]

        # Format relationships with full entity details
        formatted_relationships = []
        for rel in relationships:
            source = self.knowledge_graph.get_entity(rel.source_entity_id)
            target = self.knowledge_graph.get_entity(rel.target_entity_id)

            rel_data = {
                "type": rel.type.value,
                "source": {
                    "id": source.id,
                    "value": source.value,
                    "normalized_value": source.normalized_value,
                    "type": source.type.value,
                }
                if source
                else None,
                "target": {
                    "id": target.id,
                    "value": target.value,
                    "normalized_value": target.normalized_value,
                    "type": target.type.value,
                }
                if target
                else None,
                "confidence": rel.confidence,
            }

            if include_metadata:
                rel_data["evidence_text"] = rel.evidence_text
                rel_data["source_chunk_id"] = rel.source_chunk_id
                rel_data["properties"] = rel.properties
                rel_data["extraction_method"] = rel.extraction_method

            formatted_relationships.append(rel_data)

        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type.value,
                "value": entity.value,
                "normalized_value": entity.normalized_value,
            },
            "relationships": formatted_relationships,
            "count": len(formatted_relationships),
            "filters": {
                "relationship_types": relationship_types,
                "direction": direction,
                "min_confidence": min_confidence,
            },
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "mode": "relationships",
                "entity_id": entity.id,
                "relationships_found": len(formatted_relationships),
            },
        )

    def _multi_hop_bfs(
        self,
        entity_value: str,
        entity_type: Optional[str],
        relationship_types: Optional[List[str]],
        max_hops: int,
        k: int,
        min_confidence: float,
        cross_document: bool,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 4: Multi-hop BFS graph traversal."""

        # Find starting entity
        start_entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not start_entity:
            return ToolResult(
                success=False,
                data=None,
                error=f"Starting entity '{entity_value}' not found in knowledge graph"
                + (f" with type '{entity_type}'" if entity_type else ""),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # BFS traversal
        from collections import deque

        visited_entities = {start_entity.id}
        entity_queue = deque([(start_entity.id, 0)])  # (entity_id, hop_distance)
        entity_distances = {start_entity.id: 0}  # Track distances for scoring

        # Track discovered entities and relationships by hop
        entities_by_hop = {0: [start_entity]}  # hop 0 is the starting entity
        relationships_by_hop = {}  # hop N contains relationships from hop N-1 to hop N

        # Limit entities per hop to prevent explosion
        MAX_ENTITIES_PER_HOP = 20
        MAX_TOTAL_ENTITIES = 200

        total_entities = 1

        while entity_queue and total_entities < MAX_TOTAL_ENTITIES:
            current_entity_id, current_hop = entity_queue.popleft()

            # Don't expand beyond max_hops
            if current_hop >= max_hops:
                continue

            # Get outgoing relationships from current entity
            relationships = self.knowledge_graph.get_outgoing_relationships(current_entity_id)

            # Filter by confidence
            relationships = [r for r in relationships if r.confidence >= min_confidence]

            # Filter by relationship types if specified
            if relationship_types:
                rel_types_lower = [rt.lower() for rt in relationship_types]
                relationships = [r for r in relationships if r.type.value in rel_types_lower]

            # Process relationships
            hop_relationships = []
            hop_entities = []
            entities_added_this_hop = 0

            for rel in relationships:
                target_entity = self.knowledge_graph.get_entity(rel.target_entity_id)

                if not target_entity:
                    continue

                # Cross-document filtering
                if not cross_document:
                    source_doc = self.knowledge_graph.get_entity(current_entity_id).document_id
                    target_doc = target_entity.document_id
                    if source_doc != target_doc:
                        continue

                # Track this relationship
                hop_relationships.append(rel)

                # Add target entity if not visited
                if rel.target_entity_id not in visited_entities:
                    visited_entities.add(rel.target_entity_id)
                    entity_distances[rel.target_entity_id] = current_hop + 1

                    # Add to queue for further expansion
                    entity_queue.append((rel.target_entity_id, current_hop + 1))
                    hop_entities.append(target_entity)

                    entities_added_this_hop += 1
                    total_entities += 1

                    # Limit entities per hop
                    if entities_added_this_hop >= MAX_ENTITIES_PER_HOP:
                        break

            # Store discovered entities and relationships for this hop
            next_hop = current_hop + 1
            if hop_relationships:
                relationships_by_hop[next_hop] = relationships_by_hop.get(next_hop, []) + hop_relationships
            if hop_entities:
                entities_by_hop[next_hop] = entities_by_hop.get(next_hop, []) + hop_entities

        # Collect all chunks from discovered entities with distance-based scoring
        chunk_scores = {}  # chunk_id -> score
        chunk_entities = {}  # chunk_id -> list of entity values mentioning it

        for hop, entities in entities_by_hop.items():
            # Distance-based boost: closer entities get higher scores
            # hop 0: +0.5, hop 1: +0.3, hop 2: +0.15, hop 3: +0.05
            hop_boost = 0.5 / (2**hop)

            for entity in entities:
                for chunk_id in entity.source_chunk_ids:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + hop_boost

                    if chunk_id not in chunk_entities:
                        chunk_entities[chunk_id] = []
                    chunk_entities[chunk_id].append(
                        {"value": entity.value, "type": entity.type.value, "hop": hop}
                    )

        # Sort chunks by score and take top k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Retrieve actual chunks
        chunks = []
        for chunk_id, score in sorted_chunks:
            chunk = self.vector_store.get_chunk_by_id(chunk_id)
            if chunk:
                formatted = format_chunk_result(chunk, include_score=True)
                formatted["graph_score"] = round(score, 4)
                formatted["mentioned_entities"] = chunk_entities.get(chunk_id, [])
                chunks.append(formatted)

        # Format traversal summary
        traversal_summary = {
            "start_entity": {
                "id": start_entity.id,
                "value": start_entity.value,
                "type": start_entity.type.value,
            },
            "total_entities_discovered": len(visited_entities),
            "total_relationships_traversed": sum(len(rels) for rels in relationships_by_hop.values()),
            "max_hop_reached": max(entities_by_hop.keys()),
            "entities_by_hop": {hop: len(entities) for hop, entities in entities_by_hop.items()},
        }

        if include_metadata:
            # Add detailed traversal path
            traversal_summary["relationships_by_hop"] = {}
            for hop, rels in relationships_by_hop.items():
                traversal_summary["relationships_by_hop"][hop] = [
                    {
                        "type": r.type.value,
                        "source_id": r.source_entity_id,
                        "target_id": r.target_entity_id,
                        "confidence": r.confidence,
                    }
                    for r in rels[:10]  # Limit to first 10 per hop
                ]

        result_data = {
            "traversal": traversal_summary,
            "chunks": chunks,
            "count": len(chunks),
        }

        citations = [f"{c['document_id']}:{c['chunk_id']}" for c in chunks]

        return ToolResult(
            success=True,
            data=result_data,
            citations=citations,
            metadata={
                "mode": "multi_hop",
                "start_entity_id": start_entity.id,
                "entities_discovered": len(visited_entities),
                "chunks_found": len(chunks),
                "max_hops": max_hops,
            },
        )


class CompareDocumentsInput(ToolInput):
    doc_id_1: str = Field(..., description="First document ID")
    doc_id_2: str = Field(..., description="Second document ID")
    comparison_aspect: Optional[str] = Field(
        None, description="Optional: specific aspect to compare (e.g., 'requirements', 'dates')"
    )


@register_tool
class CompareDocumentsTool(BaseTool):
    """Compare two documents."""

    name = "compare_documents"
    description = "Compare two documents"
    detailed_help = """
    Compare two documents to find similarities, differences, and potential conflicts.
    Uses semantic similarity to identify related sections across documents.

    **When to use:**
    - "Compare contract X with regulation Y"
    - Find similarities/differences between documents
    - Identify conflicts or overlaps

    **Best practices:**
    - Specify comparison_aspect for focused comparison (e.g., "requirements", "dates")
    - Works best with documents of similar type/topic
    - Returns top matching section pairs with similarity scores

    **Method:** Retrieve all chunks from both docs, compare semantically
    **Speed:** ~1-2s (retrieves full documents)
    """
    tier = 2
    input_schema = CompareDocumentsInput

    def execute_impl(
        self, doc_id_1: str, doc_id_2: str, comparison_aspect: Optional[str] = None
    ) -> ToolResult:
        """Compare two documents."""
        try:
            # Retrieve all chunks from both documents
            doc1_results = self.vector_store.hierarchical_search(
                query_text=doc_id_1,
                query_embedding=None,
                k_layer1=1,
                k_layer2=0,
                k_layer3=50,
                document_filter=doc_id_1,
            )

            doc2_results = self.vector_store.hierarchical_search(
                query_text=doc_id_2,
                query_embedding=None,
                k_layer1=1,
                k_layer2=0,
                k_layer3=50,
                document_filter=doc_id_2,
            )

            doc1_chunks = doc1_results.get("layer3", [])
            doc2_chunks = doc2_results.get("layer3", [])

            if not doc1_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Document '{doc_id_1}' not found",
                )

            if not doc2_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Document '{doc_id_2}' not found",
                )

            # If comparison aspect specified, filter chunks
            if comparison_aspect:
                aspect_embedding = self.embedder.embed_texts([comparison_aspect])

                # Find relevant chunks in each document
                doc1_relevant = self._find_relevant_chunks(
                    doc1_chunks, comparison_aspect, aspect_embedding, k=10
                )
                doc2_relevant = self._find_relevant_chunks(
                    doc2_chunks, comparison_aspect, aspect_embedding, k=10
                )

                comparison_data = {
                    "doc_id_1": doc_id_1,
                    "doc_id_2": doc_id_2,
                    "comparison_aspect": comparison_aspect,
                    "doc1_relevant_chunks": [format_chunk_result(c) for c in doc1_relevant],
                    "doc2_relevant_chunks": [format_chunk_result(c) for c in doc2_relevant],
                }

            else:
                # Return summary information
                comparison_data = {
                    "doc_id_1": doc_id_1,
                    "doc_id_2": doc_id_2,
                    "doc1_chunk_count": len(doc1_chunks),
                    "doc2_chunk_count": len(doc2_chunks),
                    "doc1_sample": [format_chunk_result(c) for c in doc1_chunks[:3]],
                    "doc2_sample": [format_chunk_result(c) for c in doc2_chunks[:3]],
                }

            return ToolResult(
                success=True,
                data=comparison_data,
                citations=[doc_id_1, doc_id_2],
                metadata={"comparison_aspect": comparison_aspect or "general"},
            )

        except Exception as e:
            logger.error(f"Document comparison failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _find_relevant_chunks(self, chunks, query, query_embedding, k=10):
        """Find most relevant chunks using reranker or similarity."""
        if self.reranker:
            return self.reranker.rerank(query, chunks, top_k=k)
        else:
            # Simple truncation
            return chunks[:k]


class ExplainSearchResultsInput(ToolInput):
    """Input for explain_search_results tool."""

    chunk_ids: List[str] = Field(..., description="Chunk IDs from search results to explain")


@register_tool
class ExplainSearchResultsTool(BaseTool):
    """Debug search results."""

    name = "explain_search_results"
    description = "Explain search result scores"
    detailed_help = """
    Debug tool to understand why specific chunks were retrieved.
    Shows score breakdowns: BM25, Dense, RRF, and Rerank scores.

    **When to use:**
    - Debugging unexpected search results
    - Understanding why certain chunks appeared
    - Investigating retrieval quality

    **Best practices:**
    - Use AFTER a search (requires chunk IDs from search results)
    - Most useful for debugging, not regular queries
    - Helps identify which retrieval method (BM25 vs Dense) found the chunk

    **Method:** Score analysis from hybrid search metadata
    **Speed:** <100ms (metadata lookup only)
    """
    tier = 2
    input_schema = ExplainSearchResultsInput

    def execute_impl(self, chunk_ids: List[str]) -> ToolResult:
        """Explain search results with score breakdowns."""
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            explanations = []

            for chunk_id in chunk_ids:
                # Find chunk metadata
                chunk = None
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        chunk = meta
                        break

                if not chunk:
                    explanations.append(
                        {"chunk_id": chunk_id, "found": False, "error": "Chunk not found"}
                    )
                    continue

                # Extract scores (from HybridVectorStore modification)
                explanation = {
                    "chunk_id": chunk_id,
                    "found": True,
                    "document_id": chunk.get("document_id"),
                    "section_id": chunk.get("section_id"),
                    "scores": {
                        "bm25_score": chunk.get("bm25_score", None),
                        "dense_score": chunk.get("dense_score", None),
                        "rrf_score": chunk.get("rrf_score", None),
                        "rerank_score": chunk.get("rerank_score", None),
                    },
                    "fusion_method": chunk.get("fusion_method", "unknown"),
                    "content_preview": chunk.get("content", "")[:200] + "...",
                }

                # Determine which retrieval method contributed most
                scores = explanation["scores"]
                if scores["bm25_score"] and scores["dense_score"]:
                    if scores["bm25_score"] > scores["dense_score"]:
                        explanation["primary_retrieval_method"] = "sparse (BM25 keyword match)"
                    else:
                        explanation["primary_retrieval_method"] = "dense (semantic similarity)"
                elif scores["bm25_score"]:
                    explanation["primary_retrieval_method"] = "sparse (BM25 only)"
                elif scores["dense_score"]:
                    explanation["primary_retrieval_method"] = "dense (semantic only)"
                else:
                    explanation["primary_retrieval_method"] = "unknown"

                explanations.append(explanation)

            # Check if hybrid search is enabled
            hybrid_enabled = hasattr(self.vector_store, "bm25_store") or (
                hasattr(self.vector_store, "faiss_store")
                and hasattr(self.vector_store.faiss_store, "bm25_store")
            )

            return ToolResult(
                success=True,
                data={
                    "explanations": explanations,
                    "hybrid_search_enabled": hybrid_enabled,
                    "note": "BM25/Dense scores only available if retrieved via hybrid search with score preservation",
                },
                metadata={"chunk_count": len(chunk_ids), "hybrid_enabled": hybrid_enabled},
            )

        except Exception as e:
            logger.error(f"Explain search results failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================


class FilteredSearchInput(ToolInput):
    """Input for unified filtered_search tool with search method control."""

    query: str = Field(..., description="Search query")
    search_method: str = Field(
        "hybrid",
        description="Search method: 'hybrid' (BM25+Dense+RRF, default, ~200-300ms), 'bm25_only' (keyword only, ~50-100ms), 'dense_only' (semantic only, ~100-200ms)"
    )
    filter_type: Optional[str] = Field(
        None, description="Type of filter to apply: 'document', 'section', 'metadata', 'temporal'. If None, searches entire database"
    )
    filter_value: Optional[str] = Field(
        None, description="Filter value (document_id, section_title, or date range). Required if filter_type is set"
    )
    document_type: Optional[str] = Field(None, description="For metadata filter: document type")
    section_type: Optional[str] = Field(None, description="For metadata filter: section type")
    start_date: Optional[str] = Field(
        None, description="For temporal filter: start date (ISO: YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="For temporal filter: end date (ISO: YYYY-MM-DD)"
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)

    # Legacy compatibility for exact_match_search
    search_type: Optional[str] = Field(
        None, description="DEPRECATED: Use search_method='bm25_only' instead. Maps 'keywords'/'cross_references' to bm25_only"
    )
    document_id: Optional[str] = Field(
        None, description="DEPRECATED: Use filter_type='document', filter_value=<doc_id> instead"
    )
    section_id: Optional[str] = Field(
        None, description="DEPRECATED: Use filter_type='section', filter_value=<section_title> instead"
    )


@register_tool
class FilteredSearchTool(BaseTool):
    """Unified search with filters and search method control."""

    name = "filtered_search"
    description = "Unified search with filters (hybrid/BM25/dense)"
    detailed_help = """
    Unified search combining keyword (BM25) and semantic (dense) retrieval with filtering.
    Consolidates exact_match_search functionality via search_method parameter.

    **Search Methods:**
    - 'hybrid' (default): BM25 + Dense + RRF fusion (~200-300ms, best quality)
    - 'bm25_only': Keyword search only (~50-100ms, fastest, good for exact matches)
    - 'dense_only': Semantic search only (~100-200ms, good for concepts)

    **Filter Types:**
    - 'document': Search within specific document (index-level filtering, fastest)
    - 'section': Search within specific section (post-filter)
    - 'metadata': Filter by document_type/section_type (post-filter)
    - 'temporal': Search within date range (post-filter)
    - None: Search entire database without filtering

    **When to use each method:**
    - Use 'bm25_only' for: Keywords, exact phrases, references (e.g., "article 5")
    - Use 'hybrid' for: General queries, mixed keyword+semantic (best quality)
    - Use 'dense_only' for: Semantic similarity, paraphrased queries, concepts

    **Performance Guide:**
    - Fastest (~50-100ms): search_method='bm25_only' + filter_type='document'
    - Fast (~100-200ms): search_method='bm25_only' without filter
    - Balanced (~200-300ms): search_method='hybrid' (default)

    **Backward compatibility:**
    - Old exact_match_search(search_type='keywords') → filtered_search(search_method='bm25_only')
    - Old exact_match_search(document_id='X') → filtered_search(filter_type='document', filter_value='X')

    **Method:** BM25 + Dense + RRF (configurable via search_method)
    **Speed:** ~50-300ms (depending on search_method and filter_type)
    """
    tier = 2
    input_schema = FilteredSearchInput

    def execute_impl(
        self,
        query: str,
        search_method: str = "hybrid",
        filter_type: Optional[str] = None,
        filter_value: Optional[str] = None,
        document_type: Optional[str] = None,
        section_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = 6,
        # Legacy parameters
        search_type: Optional[str] = None,
        document_id: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> ToolResult:
        from .utils import validate_k_parameter

        # Handle legacy parameter mapping
        if search_type and search_method == "hybrid":
            search_method = "bm25_only"
            logger.debug(f"Legacy search_type='{search_type}' mapped to search_method='bm25_only'")

        if document_id and not filter_type:
            filter_type = "document"
            filter_value = document_id
            logger.debug(f"Legacy document_id mapped to filter_type='document'")

        if section_id and not filter_type:
            filter_type = "section"
            filter_value = section_id
            logger.debug(f"Legacy section_id mapped to filter_type='section'")

        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Validate search_method
        if search_method not in {"hybrid", "bm25_only", "dense_only"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid search_method: {search_method}. Must be 'hybrid', 'bm25_only', or 'dense_only'"
            )

        # Validate filter_type if provided
        if filter_type and filter_type not in {"document", "section", "metadata", "temporal"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid filter_type: {filter_type}. Must be 'document', 'section', 'metadata', or 'temporal'"
            )

        try:
            # Branch by search_method
            if search_method == "bm25_only":
                chunks = self._execute_bm25_only(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k
                )
            elif search_method == "dense_only":
                chunks = self._execute_dense_only(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k
                )
            else:  # hybrid
                chunks = self._execute_hybrid(
                    query, filter_type, filter_value, document_type, section_type,
                    start_date, end_date, k
                )

            if not chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "query": query,
                        "filter_type": filter_type,
                        "filter_value": filter_value,
                        "no_results": True,
                    },
                )

            formatted = [format_chunk_result(c) for c in chunks]
            citations = [
                f"[{i+1}] {c['document_id']}: {c['section_title']}" for i, c in enumerate(formatted)
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "query": query,
                    "search_method": search_method,
                    "filter_type": filter_type,
                    "filter_value": filter_value,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Filtered search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _execute_bm25_only(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[Dict]:
        """BM25-only search (~50-100ms) with optional filtering."""
        retrieval_k = k * 3 if filter_type in {"section", "metadata", "temporal"} else k

        # Document filter: index-level (fast)
        if filter_type == "document":
            if hasattr(self.vector_store, "bm25_store"):
                results = self.vector_store.bm25_store.search_layer3(
                    query=query, k=retrieval_k, document_filter=filter_value
                )
            else:
                # Fallback to hierarchical with dummy embedding
                import numpy as np
                dummy_embedding = np.zeros((1, self.embedder.dimensions))
                results_dict = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=dummy_embedding,
                    k_layer3=retrieval_k,
                    document_filter=filter_value,
                )
                results = results_dict.get("layer3", [])
            return results[:k]

        # No filter or post-filter cases
        if hasattr(self.vector_store, "bm25_store"):
            results = self.vector_store.bm25_store.search_layer3(
                query=query, k=retrieval_k, document_filter=None
            )
        else:
            import numpy as np
            dummy_embedding = np.zeros((1, self.embedder.dimensions))
            results_dict = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=dummy_embedding,
                k_layer3=retrieval_k,
                document_filter=None,
            )
            results = results_dict.get("layer3", [])

        # Apply post-filters
        if filter_type == "section":
            section_lower = filter_value.lower()
            results = [
                c for c in results
                if section_lower in c.get("section_title", "").lower()
            ][:k]
        elif filter_type == "metadata":
            if document_type:
                results = [c for c in results if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                results = [c for c in results if c.get("section_type", "").lower() == section_type.lower()]
            results = results[:k]
        elif filter_type == "temporal":
            results = self._apply_temporal_filter(results, start_date, end_date, k)

        return results

    def _execute_dense_only(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[Dict]:
        """Dense-only search (~100-200ms) with optional filtering."""
        retrieval_k = k * 3 if filter_type else k
        query_embedding = self.embedder.embed_texts([query])

        # Dense search (no document filter support in FAISS for dense-only)
        dense_results = self.vector_store.faiss_store.search_layer3(
            query_embedding=query_embedding, k=retrieval_k, document_filter=None
        )
        results = dense_results

        # Apply all filters as post-filters
        if filter_type == "document":
            results = [c for c in results if c.get("document_id") == filter_value][:k]
        elif filter_type == "section":
            section_lower = filter_value.lower()
            results = [c for c in results if section_lower in c.get("section_title", "").lower()][:k]
        elif filter_type == "metadata":
            if document_type:
                results = [c for c in results if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                results = [c for c in results if c.get("section_type", "").lower() == section_type.lower()]
            results = results[:k]
        elif filter_type == "temporal":
            results = self._apply_temporal_filter(results, start_date, end_date, k)

        return results

    def _execute_hybrid(
        self,
        query: str,
        filter_type: Optional[str],
        filter_value: Optional[str],
        document_type: Optional[str],
        section_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[Dict]:
        """Hybrid search (~200-300ms): BM25 + Dense + RRF fusion."""
        query_embedding = self.embedder.embed_texts([query])
        retrieval_k = k * 3 if filter_type in {"section", "metadata", "temporal"} else k

        # Document filter: index-level (fast path for hybrid)
        if filter_type == "document":
            dense_results = self.vector_store.faiss_store.search_layer3(
                query_embedding=query_embedding, k=retrieval_k, document_filter=filter_value
            )
            sparse_results = self.vector_store.bm25_store.search_layer3(
                query=query, k=retrieval_k, document_filter=filter_value
            )
            chunks = self.vector_store._rrf_fusion(dense_results, sparse_results, k=k)
            return chunks

        # No filter or post-filter cases
        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=retrieval_k,
        )
        chunks = results.get("layer3", [])

        # Apply post-filters
        if filter_type == "section":
            section_lower = filter_value.lower()
            chunks = [c for c in chunks if section_lower in c.get("section_title", "").lower()]
        elif filter_type == "metadata":
            if document_type:
                chunks = [c for c in chunks if c.get("doc_type", "").lower() == document_type.lower()]
            if section_type:
                chunks = [c for c in chunks if c.get("section_type", "").lower() == section_type.lower()]
        elif filter_type == "temporal":
            chunks = self._apply_temporal_filter(chunks, start_date, end_date, k)

        # Rerank if available
        if self.reranker and len(chunks) > k:
            chunks = self.reranker.rerank(query, chunks, top_k=k)
        else:
            chunks = chunks[:k]

        return chunks

    def _apply_temporal_filter(
        self,
        chunks: List[Dict],
        start_date: Optional[str],
        end_date: Optional[str],
        k: int,
    ) -> List[Dict]:
        """Apply temporal filter to chunks."""
        from datetime import datetime

        if not start_date and not end_date:
            return chunks[:k]

        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        filtered_chunks = []
        for chunk in chunks:
            chunk_date_str = chunk.get("date") or chunk.get("timestamp")
            if not chunk_date_str:
                continue
            try:
                chunk_date = datetime.fromisoformat(chunk_date_str)
                if start_dt and chunk_date < start_dt:
                    continue
                if end_dt and chunk_date > end_dt:
                    continue
                filtered_chunks.append(chunk)
            except (ValueError, TypeError):
                continue

        return filtered_chunks[:k]


class SimilaritySearchInput(ToolInput):
    """Input for unified similarity_search tool."""

    chunk_id: str = Field(..., description="Chunk ID to find similar content for")
    search_mode: str = Field(
        ..., description="Search mode: 'related' (semantically related), 'similar' (more like this)"
    )
    cross_document: bool = Field(
        True, description="Search across all documents or within same document"
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)


@register_tool
class SimilaritySearchTool(BaseTool):
    """Find similar chunks."""

    name = "similarity_search"
    description = "Find similar chunks"
    detailed_help = """
    Find semantically similar or related chunks based on embedding similarity.

    **Search modes:**
    - 'related': Semantically related content
    - 'similar': More content like this

    **When to use:**
    - Find content similar to a specific chunk
    - Explore related information
    - "Show me more like this"

    **Best practices:**
    - Requires chunk_id from previous search result
    - Use k=5-10 for best results
    - Set cross_document=false to search within same document only
    - Good for discovering related content

    **Method:** Dense embedding similarity (cosine)
    **Speed:** ~200-500ms
    """
    tier = 2
    input_schema = SimilaritySearchInput

    def execute_impl(
        self, chunk_id: str, search_mode: str, cross_document: bool = True, k: int = 6
    ) -> ToolResult:
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find target chunk
            target_chunk = None
            for meta in layer3_chunks:
                if meta.get("chunk_id") == chunk_id:
                    target_chunk = meta
                    break

            if not target_chunk:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Chunk '{chunk_id}' not found",
                    metadata={"chunk_id": chunk_id},
                )

            # Get chunk content
            content = target_chunk.get("raw_content", target_chunk.get("content", ""))
            if not content:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Chunk '{chunk_id}' has no content",
                    metadata={"chunk_id": chunk_id},
                )

            # Validate embedder
            if (
                not self.embedder
                or not hasattr(self.embedder, "dimensions")
                or not self.embedder.dimensions
            ):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Embedder not properly initialized",
                    metadata={"chunk_id": chunk_id},
                )

            # Embed and search
            query_embedding = self.embedder.embed_texts([content])

            document_filter = None
            if not cross_document:
                document_filter = target_chunk.get("document_id")

            results = self.vector_store.hierarchical_search(
                query_text=content,
                query_embedding=query_embedding,
                k_layer3=k * 2 + 1,
                document_filter=document_filter,
                use_doc_filtering=not cross_document,
            )

            chunks = results.get("layer3", [])

            # Filter out target chunk
            similar_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") != chunk_id]

            # Apply document filter if needed (redundant but safe)
            if not cross_document:
                target_doc_id = target_chunk.get("document_id")
                similar_chunks = [
                    chunk for chunk in similar_chunks if chunk.get("document_id") == target_doc_id
                ]

            # Rerank if available and mode is 'related'
            if search_mode == "related" and self.reranker and len(similar_chunks) > k:
                similar_chunks = self.reranker.rerank(content, similar_chunks, top_k=k)
            else:
                similar_chunks = similar_chunks[:k]

            if not similar_chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "chunk_id": chunk_id,
                        "search_mode": search_mode,
                        "cross_document": cross_document,
                        "results_count": 0,
                    },
                )

            formatted = [format_chunk_result(chunk) for chunk in similar_chunks]
            citations = list(set(chunk.get("document_id", "Unknown") for chunk in similar_chunks))

            return ToolResult(
                success=True,
                data={
                    "target_chunk": format_chunk_result(target_chunk),
                    "similar_chunks": formatted,
                    "similarity_count": len(formatted),
                },
                citations=citations,
                metadata={
                    "chunk_id": chunk_id,
                    "search_mode": search_mode,
                    "cross_document": cross_document,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class ExpandContextInput(ToolInput):
    """Input for unified expand_context tool."""

    chunk_ids: List[str] = Field(..., description="List of chunk IDs to expand")
    expansion_mode: str = Field(
        ...,
        description="Expansion mode: 'adjacent' (before/after chunks), 'section' (same section), 'similarity' (semantically similar), 'hybrid' (section + similarity)",
    )
    k: int = Field(3, description="Number of additional chunks per input chunk", ge=1, le=10)


@register_tool
class ExpandContextTool(BaseTool):
    """Expand chunk context."""

    name = "expand_context"
    description = "Expand chunk context"
    detailed_help = """
    Expand chunks with additional surrounding or related context.

    **Expansion modes:**
    - 'adjacent': Chunks immediately before/after (linear context)
    - 'section': All chunks from same section
    - 'similarity': Semantically similar chunks
    - 'hybrid': Combination of section + similarity

    **When to use:**
    - Need more context around a specific chunk
    - Answer requires surrounding text
    - Chunk alone is insufficient

    **Best practices:**
    - Use 'adjacent' for simple before/after context
    - Use 'section' to see full section around chunk
    - Use 'similarity' to find related content elsewhere
    - Use 'hybrid' for comprehensive context
    - Start with k=3, increase if needed

    **Method:** Metadata lookup + embeddings (similarity mode)
    **Speed:** ~100-500ms (depending on mode)
    """
    tier = 2
    input_schema = ExpandContextInput

    def execute_impl(self, chunk_ids: List[str], expansion_mode: str, k: int = 3) -> ToolResult:
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find target chunks
            target_chunks = []
            for chunk_id in chunk_ids:
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        target_chunks.append(meta)
                        break

            if not target_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No chunks found for IDs: {chunk_ids}",
                    metadata={"chunk_ids": chunk_ids},
                )

            expanded_results = []

            for target_chunk in target_chunks:
                expansion = {
                    "target_chunk": format_chunk_result(target_chunk),
                    "expanded_chunks": [],
                }

                if expansion_mode == "adjacent":
                    # Get adjacent chunks (before/after)
                    context_chunks = self._expand_adjacent(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = context_chunks
                    expansion["context_before"] = [
                        c for c in context_chunks if c.get("position") == "before"
                    ]
                    expansion["context_after"] = [
                        c for c in context_chunks if c.get("position") == "after"
                    ]

                elif expansion_mode == "section":
                    # Get chunks from same section
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = section_chunks

                elif expansion_mode == "similarity":
                    # Get semantically similar chunks
                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k)
                        expansion["expanded_chunks"] = similarity_chunks
                    except Exception as e:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Similarity expansion failed: {e}",
                        )

                elif expansion_mode == "hybrid":
                    # Combine section + similarity
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k // 2)
                    expansion["expanded_chunks"].extend(section_chunks)

                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k // 2)
                        expansion["expanded_chunks"].extend(similarity_chunks)
                    except Exception as e:
                        expansion["expansion_warning"] = f"Similarity expansion failed: {str(e)}"

                    # Remove duplicates
                    seen_ids = {target_chunk.get("chunk_id")}
                    unique_expanded = []
                    for chunk in expansion["expanded_chunks"]:
                        chunk_id = chunk.get("chunk_id")
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            unique_expanded.append(chunk)
                    expansion["expanded_chunks"] = unique_expanded

                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Invalid expansion_mode: {expansion_mode}. Must be 'adjacent', 'section', 'similarity', or 'hybrid'",
                    )

                expansion["expansion_count"] = len(expansion["expanded_chunks"])
                expanded_results.append(expansion)

            # Collect citations
            citations = []
            for result in expanded_results:
                doc_id = result["target_chunk"].get("document_id", "Unknown")
                if doc_id not in citations:
                    citations.append(doc_id)

            return ToolResult(
                success=True,
                data={"expansions": expanded_results, "expansion_mode": expansion_mode},
                citations=citations,
                metadata={
                    "chunk_count": len(target_chunks),
                    "expansion_mode": expansion_mode,
                },
            )

        except Exception as e:
            logger.error(f"Expand context failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _expand_adjacent(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """Get adjacent chunks (before/after)."""
        document_id = target_chunk.get("document_id")
        section_id = target_chunk.get("section_id")
        chunk_id = target_chunk.get("chunk_id")

        # Find all chunks from same section
        same_section = [
            (i, chunk)
            for i, chunk in enumerate(all_chunks)
            if chunk.get("document_id") == document_id and chunk.get("section_id") == section_id
        ]

        # Sort by chunk_id
        same_section.sort(key=lambda x: x[1].get("chunk_id", ""))

        # Find target position
        target_position = None
        for pos, (_, chunk) in enumerate(same_section):
            if chunk.get("chunk_id") == chunk_id:
                target_position = pos
                break

        if target_position is None:
            return []

        # Get context_window from config (default: 2)
        context_window = k

        # Extract context chunks
        start_pos = max(0, target_position - context_window)
        end_pos = min(len(same_section), target_position + context_window + 1)

        context_chunks = []

        # Before chunks
        for i in range(start_pos, target_position):
            chunk = format_chunk_result(same_section[i][1])
            chunk["position"] = "before"
            context_chunks.append(chunk)

        # After chunks
        for i in range(target_position + 1, end_pos):
            chunk = format_chunk_result(same_section[i][1])
            chunk["position"] = "after"
            context_chunks.append(chunk)

        return context_chunks

    def _expand_by_section(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """Expand by finding neighboring chunks from same section."""
        try:
            document_id = target_chunk.get("document_id")
            section_id = target_chunk.get("section_id")
            chunk_id = target_chunk.get("chunk_id")

            if not document_id or not section_id:
                return []

            # Find chunks from same section
            same_section = [
                chunk
                for chunk in all_chunks
                if isinstance(chunk, dict)
                and chunk.get("document_id") == document_id
                and chunk.get("section_id") == section_id
                and chunk.get("chunk_id") != chunk_id
            ]

            # Sort by chunk_id
            same_section.sort(key=lambda x: x.get("chunk_id", ""))

            return [format_chunk_result(chunk) for chunk in same_section[:k]]

        except Exception as e:
            logger.error(f"Section-based expansion failed: {e}", exc_info=True)
            return []

    def _expand_by_similarity(self, target_chunk: Dict, k: int) -> List[Dict]:
        """Expand by finding semantically similar chunks."""
        # Validate embedder
        if (
            not self.embedder
            or not hasattr(self.embedder, "dimensions")
            or not self.embedder.dimensions
        ):
            raise ValueError("Embedder not available for similarity-based expansion")

        content = target_chunk.get("content", "")
        if not content:
            return []

        try:
            # Embed and search
            query_embedding = self.embedder.embed_texts([content])

            results = self.vector_store.hierarchical_search(
                query_text=content,
                query_embedding=query_embedding,
                k_layer3=k + 1,
            )

            chunks = results.get("layer3", [])

            # Filter out target chunk
            target_id = target_chunk.get("chunk_id")
            similar_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") != target_id]

            return [format_chunk_result(chunk) for chunk in similar_chunks[:k]]

        except Exception as e:
            logger.error(f"Similarity expansion failed: {e}", exc_info=True)
            raise
