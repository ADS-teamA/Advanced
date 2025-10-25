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
    k: int = Field(6, description="Number of results to return", ge=1, le=10)


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
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

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
    k: int = Field(6, description="Number of results", ge=1, le=10)


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
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

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
            success=True,  # Search succeeded, just no results if filtered is empty
            data=formatted,
            citations=citations,
            metadata={
                "entity": entity_value,
                "k": k,
                "matches_found": len(filtered),
                "no_results": len(filtered) == 0,
            },
        )


# === Tool 3: Document Search ===


class DocumentSearchInput(ToolInput):
    """Input for document_search tool."""

    query: str = Field(..., description="Search query")
    document_id: str = Field(..., description="Document ID to search within")
    k: int = Field(6, description="Number of results", ge=1, le=10)


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
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Embed query
        query_embedding = self.embedder.embed_texts([query])

        # Perform manual hybrid search with explicit document filter
        # NOTE: We can't use hierarchical_search(use_doc_filtering=True) because it
        # auto-selects document from Layer 1, ignoring our document_id parameter.
        # Instead, we search Layer 3 directly with explicit document_filter.
        dense_results = self.vector_store.faiss_store.search_layer3(
            query_embedding=query_embedding,
            k=50,
            document_filter=document_id
        )
        sparse_results = self.vector_store.bm25_store.search_layer3(
            query=query,
            k=50,
            document_filter=document_id
        )

        # Apply RRF fusion to combine dense + sparse results
        chunks = self.vector_store._rrf_fusion(dense_results, sparse_results, k=k)

        if not chunks:
            logger.info(f"No results found in document '{document_id}' for query '{query}'")
            return ToolResult(
                success=True,  # Search succeeded, just no results
                data=[],
                metadata={"query": query, "document_id": document_id, "no_results": True},
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
    k: int = Field(6, description="Number of results", ge=1, le=10)


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
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Embed query
        query_embedding = self.embedder.embed_texts([query])

        # Search Layer 3 across all documents (no document filtering)
        # NOTE: We set use_doc_filtering=False to search across all documents,
        # then filter by section_title afterwards
        results = self.vector_store.hierarchical_search(
            query_text=query,
            query_embedding=query_embedding,
            k_layer3=k * 3,  # Get more candidates for section filtering
            use_doc_filtering=False  # Search all documents
        )

        # Filter by section title (case-insensitive partial match)
        section_lower = section_title.lower()
        chunks = [
            c for c in results["layer3"] if section_lower in c.get("section_title", "").lower()
        ][:k]

        if not chunks:
            logger.info(f"No results found in sections matching '{section_title}'")
            return ToolResult(
                success=True,  # Search succeeded, just no results
                data=[],
                metadata={"query": query, "section_title": section_title, "no_results": True},
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
    k: int = Field(6, description="Number of results", ge=1, le=10)


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
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

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
            logger.info(f"No results found for keywords '{keywords}'")
            return ToolResult(
                success=True,  # Search succeeded, just no matches
                data=[],
                metadata={"keywords": keywords, "no_results": True},
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
    List all indexed documents with their summaries.

    Uses: Vector store metadata (Layer 1 - document level)
    Speed: <10ms
    Use for: Discovering available documents and getting quick overviews
    """

    name = "get_document_list"
    description = "Get a list of all indexed documents with their summaries"
    tier = 1
    input_schema = GetDocumentListInput

    def execute_impl(self) -> ToolResult:
        # Extract document IDs and summaries from Layer 1 metadata
        documents_map = {}  # {doc_id: summary}

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer1"):
            for meta in self.vector_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")  # Layer 1 content is the document summary
                if doc_id and doc_id not in documents_map:
                    # Only store first occurrence (all Layer 1 entries for same doc have same summary)
                    documents_map[doc_id] = summary
        elif hasattr(self.vector_store, "faiss_store"):
            for meta in self.vector_store.faiss_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")
                if doc_id and doc_id not in documents_map:
                    documents_map[doc_id] = summary

        # Build list of document objects with id and summary
        document_list = [
            {"id": doc_id, "summary": summary}
            for doc_id, summary in sorted(documents_map.items())
        ]

        return ToolResult(
            success=True,
            data={"documents": document_list, "count": len(document_list)},
            metadata={"total_documents": len(document_list)},
        )


# === Tool 7: Get Document Summary ===


class GetDocumentSummaryInput(ToolInput):
    """Input for get_document_summary tool."""

    document_id: str = Field(..., description="Document ID to get summary for")


@register_tool
class GetDocumentSummaryTool(BaseTool):
    """
    Get document-level summary (Layer 1).

    Uses: Layer 1 metadata (generic summary from PHASE 2)
    Speed: <10ms
    Use for: Quick overview of what a document is about
    """

    name = "get_document_summary"
    description = "Get a high-level summary of a document by document ID"
    tier = 1
    input_schema = GetDocumentSummaryInput

    def execute_impl(self, document_id: str) -> ToolResult:
        # Get Layer 1 (document-level) metadata
        layer1_chunks = []

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer1"):
            layer1_chunks = self.vector_store.metadata_layer1
        elif hasattr(self.vector_store, "faiss_store"):
            layer1_chunks = self.vector_store.faiss_store.metadata_layer1

        # Find document summary
        doc_summary = None
        for meta in layer1_chunks:
            if meta.get("document_id") == document_id:
                doc_summary = meta.get("content")
                break

        if not doc_summary:
            logger.info(f"Document '{document_id}' not found or has no summary")
            return ToolResult(
                success=True,  # Lookup succeeded, document just not found
                data=None,
                metadata={"document_id": document_id, "found": False},
            )

        return ToolResult(
            success=True,
            data={"document_id": document_id, "summary": doc_summary},
            metadata={"document_id": document_id, "summary_length": len(doc_summary)},
        )


# === Tool 8: Get Document Sections ===


class GetDocumentSectionsInput(ToolInput):
    """Input for get_document_sections tool."""

    document_id: str = Field(..., description="Document ID to get sections for")


@register_tool
class GetDocumentSectionsTool(BaseTool):
    """
    List all sections in a document.

    Uses: Layer 2 metadata (section-level)
    Speed: <20ms
    Use for: Discovering document structure and navigation
    """

    name = "get_document_sections"
    description = "Get a list of all sections in a document by document ID. Dynamically adjusts section count based on token budget (typically 10-100 sections)."
    tier = 1
    input_schema = GetDocumentSectionsInput

    def execute_impl(self, document_id: str) -> ToolResult:
        # Get Layer 2 (section-level) metadata
        layer2_chunks = []

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer2"):
            layer2_chunks = self.vector_store.metadata_layer2
        elif hasattr(self.vector_store, "faiss_store"):
            layer2_chunks = self.vector_store.faiss_store.metadata_layer2

        # Find sections for this document
        sections = []
        for meta in layer2_chunks:
            if meta.get("document_id") == document_id:
                section_info = {
                    "section_id": meta.get("section_id"),
                    "section_title": meta.get("section_title"),
                    # Removed section_path and page_number to save tokens
                }
                sections.append(section_info)

        if not sections:
            logger.info(f"Document '{document_id}' not found or has no sections")
            return ToolResult(
                success=True,  # Lookup succeeded, document just not found
                data=None,
                metadata={"document_id": document_id, "found": False},
            )

        # Sort sections by section_id (preserves document order)
        sections.sort(key=lambda x: x.get("section_id", ""))

        # NEW: Use adaptive formatter for dynamic section limits
        try:
            from .token_manager import get_adaptive_formatter

            formatter = get_adaptive_formatter()
            formatted_sections, format_metadata = formatter.format_sections_with_budget(
                sections, include_summary=False
            )

            logger.info(
                f"Document '{document_id}': {format_metadata['total_sections']} sections total, "
                f"returning {format_metadata['returned_sections']} "
                f"(max_allowed={format_metadata['max_sections_allowed']}, "
                f"truncated={format_metadata['truncated']})"
            )

            return ToolResult(
                success=True,
                data={
                    "document_id": document_id,
                    "sections": formatted_sections,
                    "count": format_metadata["returned_sections"],
                    "total_sections": format_metadata["total_sections"],
                    "truncated": format_metadata["truncated"],
                    "max_sections_allowed": format_metadata["max_sections_allowed"],
                },
                metadata={
                    "document_id": document_id,
                    "section_count": format_metadata["returned_sections"],
                    "total_sections": format_metadata["total_sections"],
                    "truncated": format_metadata["truncated"],
                    "token_budget_used": format_metadata["total_tokens"],
                },
            )

        except ImportError:
            # Fallback to legacy hardcoded limit
            total_count = len(sections)
            max_sections = 50
            truncated = total_count > max_sections
            sections = sections[:max_sections]

            logger.warning(
                "token_manager not available, using legacy 50-section limit. "
                f"Document '{document_id}': {total_count} sections total, returning {len(sections)}"
            )

            return ToolResult(
                success=True,
                data={
                    "document_id": document_id,
                    "sections": sections,
                    "count": len(sections),
                    "total_sections": total_count,
                    "truncated": truncated,
                },
                metadata={
                    "document_id": document_id,
                    "section_count": len(sections),
                    "total_sections": total_count,
                    "truncated": truncated,
                },
            )


# === Tool 9: Get Section Details ===


class GetSectionDetailsInput(ToolInput):
    """Input for get_section_details tool."""

    document_id: str = Field(..., description="Document ID")
    section_id: str = Field(..., description="Section ID within the document")


@register_tool
class GetSectionDetailsTool(BaseTool):
    """
    Get detailed information about a specific section.

    Uses: Layer 2 metadata (section summary from PHASE 2)
    Speed: <20ms
    Use for: Quick section overview before deeper search
    """

    name = "get_section_details"
    description = "Get detailed information about a specific section including summary and metadata"
    tier = 1
    input_schema = GetSectionDetailsInput

    def execute_impl(self, document_id: str, section_id: str) -> ToolResult:
        # Get Layer 2 (section-level) metadata
        layer2_chunks = []

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer2"):
            layer2_chunks = self.vector_store.metadata_layer2
        elif hasattr(self.vector_store, "faiss_store"):
            layer2_chunks = self.vector_store.faiss_store.metadata_layer2

        # Find specific section
        section_data = None
        for meta in layer2_chunks:
            if meta.get("document_id") == document_id and meta.get("section_id") == section_id:
                section_data = {
                    "document_id": document_id,
                    "section_id": section_id,
                    "section_title": meta.get("section_title"),
                    "section_path": meta.get("section_path"),
                    "summary": meta.get("content"),  # Section summary from PHASE 2
                    "page_number": meta.get("page_number"),
                }
                break

        if not section_data:
            logger.info(f"Section '{section_id}' not found in document '{document_id}'")
            return ToolResult(
                success=True,  # Lookup succeeded, section just not found
                data=None,
                metadata={"document_id": document_id, "section_id": section_id, "found": False},
            )

        # Get chunk count for this section (Layer 3)
        layer3_chunks = []
        if hasattr(self.vector_store, "metadata_layer3"):
            layer3_chunks = self.vector_store.metadata_layer3
        elif hasattr(self.vector_store, "faiss_store"):
            layer3_chunks = self.vector_store.faiss_store.metadata_layer3

        chunk_count = sum(
            1
            for meta in layer3_chunks
            if meta.get("document_id") == document_id and meta.get("section_id") == section_id
        )

        section_data["chunk_count"] = chunk_count

        return ToolResult(
            success=True,
            data=section_data,
            metadata={
                "document_id": document_id,
                "section_id": section_id,
                "chunk_count": chunk_count,
            },
        )


# === Tool 10: Get Document Metadata ===


class GetDocumentMetadataInput(ToolInput):
    """Input for get_document_metadata tool."""

    document_id: str = Field(..., description="Document ID to get metadata for")


@register_tool
class GetDocumentMetadataTool(BaseTool):
    """
    Get comprehensive document metadata.

    Uses: Layer 1 + Layer 2 + Layer 3 metadata aggregation
    Speed: <50ms
    Use for: Discovering document properties, statistics, and structure
    """

    name = "get_document_metadata"
    description = "Get comprehensive metadata about a document including stats, structure, and properties"
    tier = 1
    input_schema = GetDocumentMetadataInput

    def execute_impl(self, document_id: str) -> ToolResult:
        # Collect metadata from all layers
        metadata = {"document_id": document_id}

        # Layer 1: Document summary
        layer1_chunks = []
        if hasattr(self.vector_store, "metadata_layer1"):
            layer1_chunks = self.vector_store.metadata_layer1
        elif hasattr(self.vector_store, "faiss_store"):
            layer1_chunks = self.vector_store.faiss_store.metadata_layer1

        for meta in layer1_chunks:
            if meta.get("document_id") == document_id:
                metadata["summary"] = meta.get("content")
                break

        # Layer 2: Section count and list
        layer2_chunks = []
        if hasattr(self.vector_store, "metadata_layer2"):
            layer2_chunks = self.vector_store.metadata_layer2
        elif hasattr(self.vector_store, "faiss_store"):
            layer2_chunks = self.vector_store.faiss_store.metadata_layer2

        sections = [
            meta.get("section_title")
            for meta in layer2_chunks
            if meta.get("document_id") == document_id
        ]
        metadata["section_count"] = len(sections)
        metadata["sections"] = sections

        # Layer 3: Chunk count
        layer3_chunks = []
        if hasattr(self.vector_store, "metadata_layer3"):
            layer3_chunks = self.vector_store.metadata_layer3
        elif hasattr(self.vector_store, "faiss_store"):
            layer3_chunks = self.vector_store.faiss_store.metadata_layer3

        chunk_count = sum(1 for meta in layer3_chunks if meta.get("document_id") == document_id)
        metadata["chunk_count"] = chunk_count

        # Estimate document length (from chunks)
        total_chars = 0
        for meta in layer3_chunks:
            if meta.get("document_id") == document_id:
                content = meta.get("content", "")
                total_chars += len(content)

        metadata["estimated_chars"] = total_chars
        metadata["estimated_words"] = total_chars // 5  # Rough estimate: 5 chars per word

        # Check if document exists
        if not metadata.get("summary") and metadata["section_count"] == 0:
            logger.info(f"Document '{document_id}' not found")
            return ToolResult(
                success=True,  # Lookup succeeded, document just not found
                data=None,
                metadata={"document_id": document_id, "found": False},
            )

        return ToolResult(
            success=True,
            data=metadata,
            metadata={"document_id": document_id, "total_sections": len(sections)},
        )


# === Tool 11: Get Chunk Context ===


class GetChunkContextInput(ToolInput):
    """Input for get_chunk_context tool."""

    chunk_id: str = Field(..., description="Chunk ID to get context for")


@register_tool
class GetChunkContextTool(BaseTool):
    """
    Get a chunk with surrounding chunks for context.

    Returns the target chunk plus context_window chunks before and after.
    Useful for understanding chunk content in broader narrative flow.

    Uses: Layer 3 metadata with adjacency detection
    Speed: <100ms
    Use for: Expanding a single chunk result with surrounding context
    """

    name = "get_chunk_context"
    description = "Get a chunk with surrounding chunks (before/after) for better context. Use when you need to understand a chunk in its broader narrative context."
    tier = 1
    input_schema = GetChunkContextInput

    def execute_impl(self, chunk_id: str) -> ToolResult:
        # Get Layer 3 (chunk-level) metadata
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

        # Extract document_id and section_id from target chunk
        document_id = target_chunk.get("document_id")
        section_id = target_chunk.get("section_id")

        # Build adjacency map: Find all chunks from same document and section
        same_section_chunks = []
        for i, meta in enumerate(layer3_chunks):
            if (
                meta.get("document_id") == document_id
                and meta.get("section_id") == section_id
            ):
                same_section_chunks.append((i, meta))

        # Sort by chunk_id (assumes lexicographic order preserves sequential ordering)
        same_section_chunks.sort(key=lambda x: x[1].get("chunk_id", ""))

        # Find target chunk index in sorted list
        target_position = None
        for pos, (_, meta) in enumerate(same_section_chunks):
            if meta.get("chunk_id") == chunk_id:
                target_position = pos
                break

        if target_position is None:
            # Fallback: Return just the target chunk
            logger.warning(
                f"Could not determine chunk position for {chunk_id}, returning without context"
            )
            return ToolResult(
                success=True,
                data={
                    "target_chunk": format_chunk_result(target_chunk),
                    "context_before": [],
                    "context_after": [],
                    "context_window": 0,
                },
                metadata={"chunk_id": chunk_id, "has_context": False},
            )

        # Get context_window from config (default: 2)
        context_window = self.config.context_window if self.config else 2

        # Extract context chunks
        start_pos = max(0, target_position - context_window)
        end_pos = min(len(same_section_chunks), target_position + context_window + 1)

        context_before = [
            format_chunk_result(same_section_chunks[i][1])
            for i in range(start_pos, target_position)
        ]
        context_after = [
            format_chunk_result(same_section_chunks[i][1])
            for i in range(target_position + 1, end_pos)
        ]

        return ToolResult(
            success=True,
            data={
                "target_chunk": format_chunk_result(target_chunk),
                "context_before": context_before,
                "context_after": context_after,
                "context_window": context_window,
                "document_id": document_id,
                "section_id": section_id,
            },
            metadata={
                "chunk_id": chunk_id,
                "has_context": True,
                "context_count": len(context_before) + len(context_after),
            },
            citations=[document_id],
        )


# === Tool 12: List Available Tools ===


class ListAvailableToolsInput(ToolInput):
    """Input for list_available_tools tool."""

    pass  # No parameters needed


@register_tool
class ListAvailableToolsTool(BaseTool):
    """
    List all available tools with descriptions and usage guidance.

    Returns complete tool catalog with:
    - Tool name
    - Description
    - Input parameters/schema
    - When to use (best practices)

    Uses: Tool registry metadata
    Speed: <10ms
    Use for: Understanding available capabilities and tool selection
    """

    name = "list_available_tools"
    description = "Get a complete list of all available tools with their descriptions, parameters, and usage guidelines. Use when you need to understand what tools are available or select the right tool for a task."
    tier = 1
    input_schema = ListAvailableToolsInput

    def execute_impl(self) -> ToolResult:
        """Get list of all available tools with metadata."""
        from .registry import get_registry

        registry = get_registry()
        all_tools = registry.get_all_tools()

        # Build tool list with metadata
        tools_list = []
        for tool in all_tools:
            # Extract input parameters from schema
            schema = tool.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # Build parameters info
            parameters = []
            for param_name, param_info in properties.items():
                param_desc = param_info.get("description", "No description")
                param_type = param_info.get("type", "unknown")
                is_required = param_name in required

                parameters.append({
                    "name": param_name,
                    "type": param_type,
                    "description": param_desc,
                    "required": is_required
                })

            # Extract "when to use" from description if present
            # Some tools have "Use for:" or "Use when:" in their docstring
            when_to_use = tool.description
            if hasattr(tool.__class__, "__doc__") and tool.__class__.__doc__:
                doc = tool.__class__.__doc__.strip()
                # Look for "Use for:" or "Use when:" lines
                for line in doc.split("\n"):
                    line = line.strip()
                    if line.startswith("Use for:") or line.startswith("Use when:"):
                        when_to_use = line
                        break

            # Add tier info for context (even though not grouping by tier)
            tier_label = {1: "Basic (fast)", 2: "Advanced (quality)", 3: "Analysis (deep)"}

            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "when_to_use": when_to_use,
                "tier": f"Tier {tool.tier} - {tier_label.get(tool.tier, 'Unknown')}",
            })

        # Sort by name for consistent ordering
        tools_list.sort(key=lambda t: t["name"])

        return ToolResult(
            success=True,
            data={
                "tools": tools_list,
                "total_count": len(tools_list),
                "best_practices": {
                    "general": [
                        "Start with Tier 1 (fast) tools before escalating to Tier 2/3",
                        "Use simple_search for most queries (hybrid + rerank = best quality)",
                        "For complex queries, decompose into sub-tasks and use multiple tools",
                        "Try multiple retrieval strategies before giving up"
                    ],
                    "selection_strategy": {
                        "most_queries": "simple_search",
                        "entity_focused": "entity_search",
                        "specific_document": "document_search",
                        "multi_hop_reasoning": "multi_hop_search (if KG available)",
                        "comparison": "compare_documents",
                        "temporal_info": "temporal_search or timeline_view"
                    }
                }
            },
            metadata={
                "total_tools": len(tools_list),
                "tier1_count": len([t for t in all_tools if t.tier == 1]),
                "tier2_count": len([t for t in all_tools if t.tier == 2]),
                "tier3_count": len([t for t in all_tools if t.tier == 3]),
            },
        )
