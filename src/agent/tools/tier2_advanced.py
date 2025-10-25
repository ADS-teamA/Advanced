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


class MultiHopSearchInput(ToolInput):
    query: str = Field(..., description="Multi-hop query requiring graph traversal")
    max_hops: int = Field(2, description="Maximum hops in graph", ge=1, le=3)
    k: int = Field(6, description="Number of results", ge=1, le=10)


@register_tool
class MultiHopSearchTool(BaseTool):
    """Multi-hop search using knowledge graph traversal."""

    name = "multi_hop_search"
    description = "Search requiring multi-hop reasoning across documents using knowledge graph. Use for queries like 'find all documents related to X through Y'"
    tier = 2
    input_schema = MultiHopSearchInput
    requires_kg = True

    def execute_impl(self, query: str, max_hops: int = 2, k: int = 6) -> ToolResult:
        """Execute multi-hop graph traversal search."""
        if not self.graph_retriever:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available. Use simple_search instead.",
            )

        try:
            # Use graph retriever for multi-hop reasoning
            query_embedding = self.embedder.embed_texts([query])

            results = self.graph_retriever.search_with_graph(
                query_text=query,
                query_embedding=query_embedding,
                layer="layer3",
                k=k * 2,  # Retrieve more for graph filtering
                enable_multi_hop=True,
                max_hops=max_hops,
            )

            if not results:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={"query": query, "hops": max_hops, "results_count": 0},
                )

            # Rerank if available
            if self.reranker:
                results = self.reranker.rerank(query, results, top_k=k)
            else:
                results = results[:k]

            formatted = [format_chunk_result(chunk) for chunk in results]
            citations = [
                f"{chunk.get('doc_id', 'Unknown')} (Section {chunk.get('section_id', 'N/A')})"
                for chunk in results
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "query": query,
                    "hops": max_hops,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Multi-hop search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class CompareDocumentsInput(ToolInput):
    doc_id_1: str = Field(..., description="First document ID")
    doc_id_2: str = Field(..., description="Second document ID")
    comparison_aspect: Optional[str] = Field(
        None, description="Optional: specific aspect to compare (e.g., 'requirements', 'dates')"
    )


@register_tool
class CompareDocumentsTool(BaseTool):
    """Compare two documents to find similarities and differences."""

    name = "compare_documents"
    description = "Compare two documents to find similarities, differences, or conflicts. Use for queries like 'compare contract X with regulation Y'"
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
    """
    Debug and explain why chunks were retrieved.

    Shows score breakdowns (BM25, Dense, RRF, Rerank) to understand
    why specific chunks appeared in search results.

    Uses: Score analysis from hybrid search
    Speed: <100ms (metadata lookup only)
    Use for: Understanding retrieval behavior and debugging
    """

    name = "explain_search_results"
    description = "Explain why specific chunks were retrieved by showing score breakdowns (BM25, Dense, RRF). Use for debugging retrieval or understanding why certain results appeared."
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
    """Input for unified filtered_search tool."""

    query: str = Field(..., description="Search query")
    filter_type: str = Field(
        ..., description="Type of filter to apply: 'document', 'section', 'metadata', 'temporal'"
    )
    filter_value: str = Field(
        ..., description="Filter value (document_id, section_title, or date range)"
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


@register_tool
class FilteredSearchTool(BaseTool):
    """
    Unified search with multiple filter types.

    Combines document_search, section_search, hybrid_search_with_filters, and temporal_search
    into a single tool with filter_type parameter.

    Uses: Hybrid search with specified filtering
    Speed: ~100-300ms depending on filter type
    Use for: Focused search with any type of filter (document, section, metadata, temporal)
    """

    name = "filtered_search"
    description = "Hybrid search with document, section, metadata, or temporal filters"
    tier = 2
    input_schema = FilteredSearchInput

    def execute_impl(
        self,
        query: str,
        filter_type: str,
        filter_value: str,
        document_type: Optional[str] = None,
        section_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = 6,
    ) -> ToolResult:
        from .utils import validate_k_parameter

        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        try:
            # Embed query
            query_embedding = self.embedder.embed_texts([query])

            if filter_type == "document":
                # Document search
                document_id = filter_value
                dense_results = self.vector_store.faiss_store.search_layer3(
                    query_embedding=query_embedding, k=50, document_filter=document_id
                )
                sparse_results = self.vector_store.bm25_store.search_layer3(
                    query=query, k=50, document_filter=document_id
                )
                chunks = self.vector_store._rrf_fusion(dense_results, sparse_results, k=k)

            elif filter_type == "section":
                # Section search
                section_title = filter_value
                results = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    k_layer3=k * 3,
                    use_doc_filtering=False,
                )
                section_lower = section_title.lower()
                chunks = [
                    c
                    for c in results["layer3"]
                    if section_lower in c.get("section_title", "").lower()
                ][:k]

            elif filter_type == "metadata":
                # Metadata filter search
                results = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    k_layer3=k * 4,
                )
                chunks = results.get("layer3", [])

                if document_type:
                    chunks = [
                        c for c in chunks if c.get("doc_type", "").lower() == document_type.lower()
                    ]
                if section_type:
                    chunks = [
                        c
                        for c in chunks
                        if c.get("section_type", "").lower() == section_type.lower()
                    ]

                # Rerank if available
                if self.reranker and len(chunks) > k:
                    chunks = self.reranker.rerank(query, chunks, top_k=k)
                else:
                    chunks = chunks[:k]

            elif filter_type == "temporal":
                # Temporal filter search
                from datetime import datetime

                start_dt = datetime.fromisoformat(start_date) if start_date else None
                end_dt = datetime.fromisoformat(end_date) if end_date else None

                results = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    k_layer3=k * 3,
                )
                chunks = results.get("layer3", [])

                # Filter by date
                if start_dt or end_dt:
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
                    chunks = filtered_chunks

                # Rerank if available
                if self.reranker and len(chunks) > k:
                    chunks = self.reranker.rerank(query, chunks, top_k=k)
                else:
                    chunks = chunks[:k]

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid filter_type: {filter_type}. Must be 'document', 'section', 'metadata', or 'temporal'",
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
                    "filter_type": filter_type,
                    "filter_value": filter_value,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Filtered search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


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
    """
    Unified similarity search tool.

    Combines find_related_chunks and chunk_similarity_search into a single tool.

    Uses: Chunk embedding for similarity search
    Speed: 500-1000ms
    Use for: Finding semantically related or similar chunks
    """

    name = "similarity_search"
    description = "Find semantically similar or related chunks"
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


class EntityToolInput(ToolInput):
    """Input for unified entity_tool."""

    entity_value: str = Field(..., description="Entity value, ID, or name")
    operation: str = Field(
        ...,
        description="Operation: 'search' (find mentions), 'explain' (detailed info), 'relationships' (get relationships)",
    )
    k: int = Field(6, description="Number of results (for search operation)", ge=1, le=10)
    relationship_type: Optional[str] = Field(
        None, description="Filter by relationship type (for relationships operation)"
    )
    direction: str = Field(
        "both", description="Direction for relationships: 'outgoing', 'incoming', 'both'"
    )


@register_tool
class EntityTool(BaseTool):
    """
    Unified entity operations tool.

    Combines entity_search, explain_entity, and get_entity_relationships.

    Uses: Vector search + Knowledge graph (for explain/relationships)
    Speed: 150ms-2s depending on operation
    Use for: All entity-related operations
    """

    name = "entity_tool"
    description = "Search, explain, or get relationships for entities"
    tier = 2
    input_schema = EntityToolInput

    def execute_impl(
        self,
        entity_value: str,
        operation: str,
        k: int = 6,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> ToolResult:
        from .utils import validate_k_parameter

        try:
            if operation == "search":
                # Entity search (doesn't require KG)
                k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")
                query = f'"{entity_value}"'
                query_embedding = self.embedder.embed_texts([query])

                results = self.vector_store.hierarchical_search(
                    query_text=query, query_embedding=query_embedding, k_layer3=k * 2
                )

                entity_lower = entity_value.lower()
                filtered = [
                    c for c in results["layer3"] if entity_lower in c.get("content", "").lower()
                ][:k]

                formatted = [format_chunk_result(c) for c in filtered]
                citations = [
                    f"[{i+1}] {c['document_id']}: {c['section_title']}"
                    for i, c in enumerate(formatted)
                ]

                return ToolResult(
                    success=True,
                    data=formatted,
                    citations=citations,
                    metadata={
                        "entity": entity_value,
                        "operation": operation,
                        "k": k,
                        "matches_found": len(filtered),
                    },
                )

            elif operation == "explain":
                # Explain entity (requires KG)
                if not self.knowledge_graph:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Knowledge graph not available. Use operation='search' instead.",
                    )

                # Find entity by ID or name
                entity = None
                if entity_value in self.knowledge_graph.entities:
                    entity = self.knowledge_graph.entities[entity_value]
                else:
                    entity_lower = entity_value.lower()
                    for ent_id, ent in self.knowledge_graph.entities.items():
                        if ent.name.lower() == entity_lower:
                            entity = ent
                            break

                if not entity:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Entity '{entity_value}' not found in knowledge graph",
                    )

                # Get relationships
                outgoing_rels = self.knowledge_graph.get_outgoing_relationships(entity.id)
                incoming_rels = self.knowledge_graph.get_incoming_relationships(entity.id)

                # Get related entities
                related_entities = []
                for rel in outgoing_rels + incoming_rels:
                    target_id = rel.target if rel.source == entity.id else rel.source
                    if target_id in self.knowledge_graph.entities:
                        related_entities.append(self.knowledge_graph.entities[target_id])

                # Get mentions
                query_embedding = self.embedder.embed_texts([entity.name])
                chunk_results = self.vector_store.hierarchical_search(
                    query_text=entity.name, query_embedding=query_embedding, k_layer3=10
                )
                chunks = chunk_results.get("layer3", [])

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
                    citations=[chunk.get("document_id", "Unknown") for chunk in chunks[:5]],
                    metadata={
                        "entity_id": entity.id,
                        "operation": operation,
                        "relationship_count": len(outgoing_rels) + len(incoming_rels),
                    },
                )

            elif operation == "relationships":
                # Get entity relationships (requires KG)
                if not self.knowledge_graph:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Knowledge graph not available.",
                    )

                # Find entity
                entity = None
                if entity_value in self.knowledge_graph.entities:
                    entity = self.knowledge_graph.entities[entity_value]
                else:
                    for ent_id, ent in self.knowledge_graph.entities.items():
                        if ent.name.lower() == entity_value.lower():
                            entity = ent
                            break

                if not entity:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Entity '{entity_value}' not found",
                    )

                # Get relationships with filters
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
                        "operation": operation,
                        "relationship_type_filter": relationship_type,
                        "direction": direction,
                        "relationships": relationships,
                        "count": len(relationships),
                    },
                    metadata={
                        "entity_id": entity.id,
                        "operation": operation,
                        "relationship_count": len(relationships),
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid operation: {operation}. Must be 'search', 'explain', or 'relationships'",
                )

        except Exception as e:
            logger.error(f"Entity tool failed: {e}", exc_info=True)
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
    """
    Unified context expansion tool.

    Combines get_chunk_context and expand_search_context into a single tool.

    Uses: Metadata + embeddings (for similarity mode)
    Speed: 100ms-1s depending on mode
    Use for: Expanding chunks with additional context
    """

    name = "expand_context"
    description = "Expand chunks with adjacent, section, similarity, or hybrid context"
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
