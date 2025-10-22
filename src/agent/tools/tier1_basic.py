"""
TIER 1: Basic Retrieval Tools

Fast tools (100-300ms) for common retrieval tasks.
These should handle 80% of user queries.
"""

import logging
from typing import List, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result, validate_k_parameter

logger = logging.getLogger(__name__)


# === Tool 1: Simple Search (Hybrid + Reranking) ===


class SimpleSearchInput(ToolInput):
    """Input for simple_search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(6, description="Number of results to return", ge=1, le=20)


@register_tool
class SimpleSearchTool(BaseTool):
    """
    Fast hybrid search with reranking.

    Uses: BM25 + Dense + RRF fusion → Cross-encoder reranking
    Speed: ~200-300ms
    Use for: Most queries (best quality/speed tradeoff)
    """

    name = "simple_search"
    description = "Search documents using hybrid retrieval (BM25 + dense embeddings) with reranking for best quality"
    tier = 1
    input_schema = SimpleSearchInput
    requires_reranker = True

    def execute_impl(self, query: str, k: int = 6) -> ToolResult:
        k = validate_k_parameter(k)

        # Embed query
        query_embedding = self.embedder.embed_texts([query])

        # Hybrid search (retrieve more candidates for reranking)
        candidates_k = k * 3 if self.reranker else k

        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=candidates_k,
        )

        chunks = results["layer3"]

        # Rerank if available
        if self.reranker and len(chunks) > k:
            logger.debug(f"Reranking {len(chunks)} candidates to top {k}")
            chunks = self.reranker.rerank(query=query, candidates=chunks, top_k=k)
        else:
            chunks = chunks[:k]

        # Format results
        formatted = [format_chunk_result(c) for c in chunks]

        # Generate citations
        citations = [
            f"[{i+1}] {c['document_id']}: {c['section_title']}" for i, c in enumerate(formatted)
        ]

        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata={
                "query": query,
                "k": k,
                "method": "hybrid+rerank" if self.reranker else "hybrid",
                "candidates_retrieved": len(results["layer3"]),
                "final_count": len(formatted),
            },
        )


# === Tool 2: Entity Search ===


class EntitySearchInput(ToolInput):
    """Input for entity_search tool."""

    entity_value: str = Field(
        ..., description="Entity value to search for (e.g., 'GRI 306', 'GSSB')"
    )
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class EntitySearchTool(BaseTool):
    """
    Find chunks mentioning specific entities.

    Uses: Semantic search for entity mentions
    Speed: ~150-200ms
    Use for: Finding where entities (standards, organizations, dates) are mentioned
    """

    name = "entity_search"
    description = "Find chunks mentioning specific entities (standards, organizations, dates, etc.)"
    tier = 1
    input_schema = EntitySearchInput

    def execute_impl(self, entity_value: str, k: int = 6) -> ToolResult:
        k = validate_k_parameter(k)

        # Create search query focused on entity
        query = f'"{entity_value}"'  # Quote for exact match emphasis

        # Embed and search
        query_embedding = self.embedder.embed_texts([query])

        results = self.vector_store.hierarchical_search(
            query_text=query, query_embedding=query_embedding, k_layer3=k * 2
        )

        # Filter chunks that actually mention the entity (case-insensitive)
        entity_lower = entity_value.lower()
        filtered = [c for c in results["layer3"] if entity_lower in c.get("content", "").lower()][
            :k
        ]

        formatted = [format_chunk_result(c) for c in filtered]

        citations = [
            f"[{i+1}] {c['document_id']}: {c['section_title']}" for i, c in enumerate(formatted)
        ]

        return ToolResult(
            success=True if filtered else False,
            data=formatted,
            error=None if filtered else f"No chunks found mentioning '{entity_value}'",
            citations=citations,
            metadata={
                "entity": entity_value,
                "k": k,
                "matches_found": len(filtered),
            },
        )


# === Tool 3: Document Search ===


class DocumentSearchInput(ToolInput):
    """Input for document_search tool."""

    query: str = Field(..., description="Search query")
    document_id: str = Field(..., description="Document ID to search within")
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class DocumentSearchTool(BaseTool):
    """
    Search within a specific document.

    Uses: Hybrid search with document filtering
    Speed: ~100-150ms
    Use for: Focused search in known document
    """

    name = "document_search"
    description = "Search within a specific document by document ID"
    tier = 1
    input_schema = DocumentSearchInput

    def execute_impl(self, query: str, document_id: str, k: int = 6) -> ToolResult:
        k = validate_k_parameter(k)

        # Embed query
        query_embedding = self.embedder.embed_texts([query])

        # Hybrid search with document filter
        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=k,
            use_doc_filtering=True,
        )

        # Filter to specific document (hierarchical_search may return top doc)
        chunks = [c for c in results["layer3"] if c.get("document_id") == document_id]

        if not chunks:
            return ToolResult(
                success=False,
                data=[],
                error=f"No results found in document '{document_id}' for query '{query}'",
                metadata={"query": query, "document_id": document_id},
            )

        formatted = [format_chunk_result(c) for c in chunks[:k]]

        citations = [f"[{i+1}] {c['section_title']}" for i, c in enumerate(formatted)]

        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata={
                "query": query,
                "document_id": document_id,
                "results_count": len(formatted),
            },
        )


# === Tool 4: Section Search ===


class SectionSearchInput(ToolInput):
    """Input for section_search tool."""

    query: str = Field(..., description="Search query")
    section_title: str = Field(..., description="Section title to search within")
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class SectionSearchTool(BaseTool):
    """
    Search within document sections.

    Uses: Layer 2 search (section-level embeddings)
    Speed: ~100ms
    Use for: Finding specific sections matching query
    """

    name = "section_search"
    description = "Search within document sections by section title"
    tier = 1
    input_schema = SectionSearchInput

    def execute_impl(self, query: str, section_title: str, k: int = 6) -> ToolResult:
        k = validate_k_parameter(k)

        # Embed query
        query_embedding = self.embedder.embed_texts([query])

        # Search at layer 2 (sections)
        # Note: HybridVectorStore doesn't expose layer 2 directly, so use layer 3 + filter
        results = self.vector_store.hierarchical_search(
            query_text=query, query_embedding=query_embedding, k_layer3=k * 3
        )

        # Filter by section title (case-insensitive partial match)
        section_lower = section_title.lower()
        chunks = [
            c for c in results["layer3"] if section_lower in c.get("section_title", "").lower()
        ][:k]

        if not chunks:
            return ToolResult(
                success=False,
                data=[],
                error=f"No results found in sections matching '{section_title}'",
                metadata={"query": query, "section_title": section_title},
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
                "section_title": section_title,
                "matches_found": len(formatted),
            },
        )


# === Tool 5: Keyword Search (Pure BM25) ===


class KeywordSearchInput(ToolInput):
    """Input for keyword_search tool."""

    keywords: str = Field(..., description="Keywords to search for (space-separated)")
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class KeywordSearchTool(BaseTool):
    """
    Pure BM25 keyword search (no embeddings).

    Uses: BM25 only (sparse retrieval)
    Speed: ~50-100ms (fastest)
    Use for: Exact keyword/phrase matching
    """

    name = "keyword_search"
    description = (
        "Fast keyword search using BM25 (no semantic embeddings) - best for exact terms/phrases"
    )
    tier = 1
    input_schema = KeywordSearchInput

    def execute_impl(self, keywords: str, k: int = 6) -> ToolResult:
        k = validate_k_parameter(k)

        # Use BM25 search directly (if available)
        if hasattr(self.vector_store, "bm25_store"):
            results = self.vector_store.bm25_store.search_layer3(query=keywords, k=k)
        else:
            # Fallback to hybrid with dummy embedding
            import numpy as np

            dummy_embedding = np.zeros((1, self.embedder.dimensions))
            results_dict = self.vector_store.hierarchical_search(
                query_text=keywords, query_embedding=dummy_embedding, k_layer3=k
            )
            results = results_dict["layer3"]

        if not results:
            return ToolResult(
                success=False,
                data=[],
                error=f"No results found for keywords '{keywords}'",
                metadata={"keywords": keywords},
            )

        formatted = [format_chunk_result(c) for c in results]

        citations = [
            f"[{i+1}] {c['document_id']}: {c['section_title']}" for i, c in enumerate(formatted)
        ]

        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata={
                "keywords": keywords,
                "method": "bm25",
                "results_count": len(formatted),
            },
        )


# === Tool 6: Get Document List ===


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool."""

    pass  # No parameters needed


@register_tool
class GetDocumentListTool(BaseTool):
    """
    List all indexed documents.

    Uses: Vector store metadata
    Speed: <10ms
    Use for: Discovering available documents
    """

    name = "get_document_list"
    description = "Get a list of all indexed documents in the corpus"
    tier = 1
    input_schema = GetDocumentListInput

    def execute_impl(self) -> ToolResult:
        # Get documents from vector store stats
        stats = self.vector_store.get_stats()

        # Extract document IDs from metadata
        documents = set()

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer1"):
            for meta in self.vector_store.metadata_layer1:
                doc_id = meta.get("document_id")
                if doc_id:
                    documents.add(doc_id)
        elif hasattr(self.vector_store, "faiss_store"):
            for meta in self.vector_store.faiss_store.metadata_layer1:
                doc_id = meta.get("document_id")
                if doc_id:
                    documents.add(doc_id)

        document_list = sorted(list(documents))

        return ToolResult(
            success=True,
            data={"documents": document_list, "count": len(document_list)},
            metadata={"total_documents": len(document_list)},
        )
