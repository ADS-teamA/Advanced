"""
TIER 3: Analysis & Insights Tools

Deep analysis tools (1-3s) for specialized tasks.
Use for entity analysis, timeline extraction, statistics.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 3 TOOLS: Analysis & Insights
# ============================================================================


class ExplainEntityInput(ToolInput):
    entity_id: str = Field(..., description="Entity ID or entity name to explain")


@register_tool
class ExplainEntityTool(BaseTool):
    """Get detailed entity information and relationships."""

    name = "explain_entity"
    description = "Get detailed information about an entity including all relationships, mentions, and context. Use for queries like 'explain what is GRI 306'"
    tier = 3
    input_schema = ExplainEntityInput
    requires_kg = True

    def execute_impl(self, entity_id: str) -> ToolResult:
        """Get comprehensive entity information."""
        if not self.knowledge_graph:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available. Use entity_search instead.",
            )

        try:
            # Find entity by ID or name
            entity = None

            # Try exact ID match first
            if entity_id in self.knowledge_graph.entities:
                entity = self.knowledge_graph.entities[entity_id]
            else:
                # Search by name (case-insensitive)
                entity_id_lower = entity_id.lower()
                for ent_id, ent in self.knowledge_graph.entities.items():
                    if ent.name.lower() == entity_id_lower:
                        entity = ent
                        break

            if not entity:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Entity '{entity_id}' not found in knowledge graph",
                )

            # Get all relationships
            outgoing_rels = self.knowledge_graph.get_outgoing_relationships(entity.id)
            incoming_rels = self.knowledge_graph.get_incoming_relationships(entity.id)

            # Get related entities
            related_entities = []
            for rel in outgoing_rels + incoming_rels:
                target_id = rel.target if rel.source == entity.id else rel.source
                if target_id in self.knowledge_graph.entities:
                    related_entities.append(self.knowledge_graph.entities[target_id])

            # Get chunks mentioning this entity
            query_embedding = self.embedder.embed_texts([entity.name])
            chunk_results = self.vector_store.hierarchical_search(
                query_text=entity.name, query_embedding=query_embedding, k_layer3=10
            )

            chunks = chunk_results.get("layer3", [])

            # Build entity explanation
            entity_data = {
                "entity_id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "confidence": entity.confidence,
                "outgoing_relationships": [
                    {
                        "type": rel.type,
                        "target": (
                            self.knowledge_graph.entities[rel.target].name
                            if rel.target in self.knowledge_graph.entities
                            else rel.target
                        ),
                        "confidence": rel.confidence,
                    }
                    for rel in outgoing_rels
                ],
                "incoming_relationships": [
                    {
                        "type": rel.type,
                        "source": (
                            self.knowledge_graph.entities[rel.source].name
                            if rel.source in self.knowledge_graph.entities
                            else rel.source
                        ),
                        "confidence": rel.confidence,
                    }
                    for rel in incoming_rels
                ],
                "related_entities": [
                    {"id": e.id, "name": e.name, "type": e.type} for e in related_entities
                ],
                "mentions": [format_chunk_result(chunk) for chunk in chunks[:5]],
                "mention_count": len(chunks),
            }

            return ToolResult(
                success=True,
                data=entity_data,
                citations=[chunk.get("doc_id", "Unknown") for chunk in chunks[:5]],
                metadata={
                    "entity_id": entity.id,
                    "relationship_count": len(outgoing_rels) + len(incoming_rels),
                },
            )

        except Exception as e:
            logger.error(f"Entity explanation failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class GetEntityRelationshipsInput(ToolInput):
    entity_id: str = Field(..., description="Entity ID or name")
    relationship_type: Optional[str] = Field(
        None, description="Filter by relationship type (e.g., 'REFERENCES', 'ISSUED_BY')"
    )
    direction: str = Field("both", description="Direction: 'outgoing', 'incoming', or 'both'")


@register_tool
class GetEntityRelationshipsTool(BaseTool):
    """Get entity relationships with filtering."""

    name = "get_entity_relationships"
    description = "Get relationships for an entity with optional filtering by type and direction. Use for queries like 'what documents reference GRI 306?'"
    tier = 3
    input_schema = GetEntityRelationshipsInput
    requires_kg = True

    def execute_impl(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> ToolResult:
        """Get filtered entity relationships."""
        if not self.knowledge_graph:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available.",
            )

        try:
            # Find entity
            entity = None
            if entity_id in self.knowledge_graph.entities:
                entity = self.knowledge_graph.entities[entity_id]
            else:
                # Search by name
                for ent_id, ent in self.knowledge_graph.entities.items():
                    if ent.name.lower() == entity_id.lower():
                        entity = ent
                        break

            if not entity:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Entity '{entity_id}' not found",
                )

            # Get relationships
            relationships = []

            if direction in ["outgoing", "both"]:
                outgoing = self.knowledge_graph.get_outgoing_relationships(entity.id)
                if relationship_type:
                    outgoing = [r for r in outgoing if r.type == relationship_type]
                relationships.extend(
                    [
                        {
                            "direction": "outgoing",
                            "type": r.type,
                            "target": (
                                self.knowledge_graph.entities[r.target].name
                                if r.target in self.knowledge_graph.entities
                                else r.target
                            ),
                            "target_id": r.target,
                            "confidence": r.confidence,
                            "properties": r.properties,
                        }
                        for r in outgoing
                    ]
                )

            if direction in ["incoming", "both"]:
                incoming = self.knowledge_graph.get_incoming_relationships(entity.id)
                if relationship_type:
                    incoming = [r for r in incoming if r.type == relationship_type]
                relationships.extend(
                    [
                        {
                            "direction": "incoming",
                            "type": r.type,
                            "source": (
                                self.knowledge_graph.entities[r.source].name
                                if r.source in self.knowledge_graph.entities
                                else r.source
                            ),
                            "source_id": r.source,
                            "confidence": r.confidence,
                            "properties": r.properties,
                        }
                        for r in incoming
                    ]
                )

            return ToolResult(
                success=True,
                data={
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "relationship_type_filter": relationship_type,
                    "direction": direction,
                    "relationships": relationships,
                    "count": len(relationships),
                },
                metadata={"entity_id": entity.id, "relationship_count": len(relationships)},
            )

        except Exception as e:
            logger.error(f"Get relationships failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class TimelineViewInput(ToolInput):
    query: Optional[str] = Field(None, description="Optional query to filter timeline")
    k: int = Field(10, description="Number of events to return", ge=1, le=50)


@register_tool
class TimelineViewTool(BaseTool):
    """Extract temporal information and create timeline."""

    name = "timeline_view"
    description = "Extract and organize temporal information into a timeline. Use for queries like 'show me the timeline of events' or 'when did things happen'"
    tier = 3
    input_schema = TimelineViewInput

    def execute_impl(self, query: Optional[str] = None, k: int = 10) -> ToolResult:
        """Create timeline from documents."""
        try:
            # Search for chunks with temporal information
            search_query = query if query else "date when time year month effective"
            query_embedding = self.embedder.embed_texts([search_query])

            results = self.vector_store.hierarchical_search(
                query_text=search_query,
                query_embedding=query_embedding,
                k_layer3=k * 3,
            )

            chunks = results.get("layer3", [])

            # Extract dates from chunks
            timeline_events = []

            # Date pattern matching (ISO dates, common formats)
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # ISO format
                r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
                r"\d{1,2}\.\d{1,2}\.\d{4}",  # DD.MM.YYYY
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            ]

            for chunk in chunks:
                content = chunk.get("raw_content", chunk.get("content", ""))

                # Try to find dates in content
                dates_found = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    dates_found.extend(matches)

                # Check metadata for dates
                chunk_date = chunk.get("date") or chunk.get("timestamp")
                if chunk_date:
                    dates_found.append(chunk_date)

                if dates_found or chunk_date:
                    timeline_events.append(
                        {
                            "dates": dates_found if dates_found else [chunk_date],
                            "content": format_chunk_result(chunk),
                            "doc_id": chunk.get("doc_id", "Unknown"),
                            "section_id": chunk.get("section_id", "N/A"),
                        }
                    )

            # Sort by first date found (simple heuristic)
            timeline_events = timeline_events[:k]

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "timeline_events": timeline_events,
                    "event_count": len(timeline_events),
                },
                citations=[event["doc_id"] for event in timeline_events],
                metadata={"query": query, "event_count": len(timeline_events)},
            )

        except Exception as e:
            logger.error(f"Timeline view failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class SummarizeSectionInput(ToolInput):
    doc_id: str = Field(..., description="Document ID")
    section_id: str = Field(..., description="Section ID to summarize")


@register_tool
class SummarizeSectionTool(BaseTool):
    """Summarize a specific document section."""

    name = "summarize_section"
    description = "Summarize a specific section of a document. Use for queries like 'summarize section 3 of GRI 306'"
    tier = 3
    input_schema = SummarizeSectionInput

    def execute_impl(self, doc_id: str, section_id: str) -> ToolResult:
        """Summarize a document section."""
        try:
            # Retrieve all chunks from the section
            results = self.vector_store.hierarchical_search(
                query_text=f"{doc_id} {section_id}",
                query_embedding=None,
                k_layer2=1,
                k_layer3=100,
            )

            # Try to find section-level summary (layer2)
            layer2_results = results.get("layer2", [])
            section_summary = None

            for section in layer2_results:
                if section.get("doc_id") == doc_id and section.get("section_id") == section_id:
                    section_summary = section
                    break

            # Get all chunks from this section
            layer3_results = results.get("layer3", [])
            section_chunks = [
                chunk
                for chunk in layer3_results
                if chunk.get("doc_id") == doc_id and chunk.get("section_id") == section_id
            ]

            if not section_summary and not section_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Section '{section_id}' not found in document '{doc_id}'",
                )

            # Prepare summary data
            summary_data = {
                "doc_id": doc_id,
                "section_id": section_id,
                "section_summary": (
                    format_chunk_result(section_summary) if section_summary else None
                ),
                "chunk_count": len(section_chunks),
                "chunks": [format_chunk_result(chunk) for chunk in section_chunks[:10]],
            }

            # Use context assembler to format nicely
            if self.context_assembler and section_chunks:
                assembled = self.context_assembler.assemble(chunks=section_chunks, max_chunks=10)
                summary_data["formatted_content"] = assembled.context

            return ToolResult(
                success=True,
                data=summary_data,
                citations=[doc_id],
                metadata={
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "chunk_count": len(section_chunks),
                },
            )

        except Exception as e:
            logger.error(f"Summarize section failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class GetStatisticsInput(ToolInput):
    stat_type: str = Field(
        "corpus",
        description="Statistics type: 'corpus' (overall), 'document' (per-doc), or 'entity' (KG stats)",
    )


@register_tool
class GetStatisticsTool(BaseTool):
    """Get corpus statistics and analytics."""

    name = "get_statistics"
    description = "Get statistics about the indexed corpus (document count, entity types, relationships). Use for queries like 'how many documents are indexed?' or 'show me corpus statistics'"
    tier = 3
    input_schema = GetStatisticsInput

    def execute_impl(self, stat_type: str = "corpus") -> ToolResult:
        """Get various statistics."""
        try:
            stats = {}

            if stat_type in ["corpus", "document"]:
                # Get vector store statistics
                vs_stats = self.vector_store.get_stats()
                stats.update(vs_stats)

                # Count unique documents
                # This is a simplified approach - in production you'd track this properly
                sample_results = self.vector_store.hierarchical_search(
                    query_text="",
                    query_embedding=None,
                    k_layer1=100,
                )

                unique_docs = set()
                for doc in sample_results.get("layer1", []):
                    unique_docs.add(doc.get("doc_id", "Unknown"))

                stats["unique_documents"] = len(unique_docs)
                stats["document_list"] = list(unique_docs)

            if stat_type in ["corpus", "entity"]:
                # Get knowledge graph statistics
                if self.knowledge_graph:
                    entity_types = Counter()
                    relationship_types = Counter()

                    for entity in self.knowledge_graph.entities.values():
                        entity_types[entity.type] += 1

                    for rel in self.knowledge_graph.relationships.values():
                        relationship_types[rel.type] += 1

                    stats["knowledge_graph"] = {
                        "total_entities": len(self.knowledge_graph.entities),
                        "total_relationships": len(self.knowledge_graph.relationships),
                        "entity_types": dict(entity_types),
                        "relationship_types": dict(relationship_types),
                        "top_entities": [
                            {"id": e.id, "name": e.name, "type": e.type}
                            for e in sorted(
                                self.knowledge_graph.entities.values(),
                                key=lambda x: x.confidence,
                                reverse=True,
                            )[:10]
                        ],
                    }
                else:
                    stats["knowledge_graph"] = None

            return ToolResult(
                success=True,
                data=stats,
                metadata={"stat_type": stat_type},
            )

        except Exception as e:
            logger.error(f"Get statistics failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class GetIndexStatisticsInput(ToolInput):
    include_cache_stats: bool = Field(
        False, description="Include embedding cache statistics"
    )


@register_tool
class GetIndexStatisticsTool(BaseTool):
    """Get comprehensive index statistics and metadata."""

    name = "get_index_statistics"
    description = "Get comprehensive statistics about the indexed corpus including document counts, embedding model info, cache statistics, and system configuration. Use for queries like 'what documents are indexed?' or 'show me system statistics'"
    tier = 3
    input_schema = GetIndexStatisticsInput

    def execute_impl(self, include_cache_stats: bool = False) -> ToolResult:
        """Get comprehensive index statistics."""
        try:
            stats = {}

            # Vector store statistics
            vs_stats = self.vector_store.get_stats()
            stats["vector_store"] = vs_stats

            # Detect if hybrid search is enabled
            stats["hybrid_search_enabled"] = vs_stats.get("hybrid_enabled", False)

            # Embedding model information
            if self.embedder:
                stats["embedding_model"] = {
                    "model_name": self.embedder.model_name,
                    "dimensions": self.embedder.dimensions,
                    "model_type": self.embedder.model_type,
                }

                # Cache statistics if requested
                if include_cache_stats and hasattr(self.embedder, "get_cache_stats"):
                    stats["embedding_cache"] = self.embedder.get_cache_stats()

            # Knowledge graph statistics
            if self.knowledge_graph:
                entity_types = Counter()
                relationship_types = Counter()

                for entity in self.knowledge_graph.entities.values():
                    entity_types[entity.type] += 1

                for rel in self.knowledge_graph.relationships.values():
                    relationship_types[rel.type] += 1

                stats["knowledge_graph"] = {
                    "total_entities": len(self.knowledge_graph.entities),
                    "total_relationships": len(self.knowledge_graph.relationships),
                    "entity_types": dict(entity_types),
                    "relationship_types": dict(relationship_types),
                }
            else:
                stats["knowledge_graph"] = None

            # Get document list
            # Sample layer 1 to get unique documents
            sample_results = self.vector_store.hierarchical_search(
                query_text="",
                query_embedding=None,
                k_layer1=100,
            )

            unique_docs = set()
            for doc in sample_results.get("layer1", []):
                unique_docs.add(doc.get("document_id", "Unknown"))

            stats["documents"] = {
                "count": len(unique_docs),
                "document_ids": sorted(list(unique_docs)),
            }

            # System configuration
            stats["configuration"] = {
                "reranking_enabled": hasattr(self, "reranker") and self.reranker is not None,
                "context_assembler_enabled": hasattr(self, "context_assembler") and self.context_assembler is not None,
            }

            return ToolResult(
                success=True,
                data=stats,
                metadata={"stat_categories": list(stats.keys())},
            )

        except Exception as e:
            logger.error(f"Get index statistics failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
