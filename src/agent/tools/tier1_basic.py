"""
TIER 1: Basic Retrieval Tools

Fast tools (100-300ms) for common retrieval tasks.
These should handle 80% of user queries.
"""

import logging
from typing import List, Literal, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result, validate_k_parameter

logger = logging.getLogger(__name__)


# === Tool 1: Simple Search (Hybrid + Reranking) ===


class SimpleSearchInput(ToolInput):
    """Input for simple_search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(5, description="Number of results to return (3-5 recommended)", ge=1, le=10)


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

    def execute_impl(self, query: str, k: int = 5) -> ToolResult:
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


# === Tool 2: Get Document List ===


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


# === Tool 3: List Available Tools ===


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
                        "entity_focused": "Use simple_search with entity names, or multi_hop_search if KG available",
                        "specific_document": "Use exact_match_search or filtered_search with document_id filter",
                        "multi_hop_reasoning": "multi_hop_search (requires KG)",
                        "comparison": "compare_documents",
                        "temporal_info": "filtered_search with filter_type='temporal' or timeline_view"
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


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================
#
# These tools combine multiple legacy tools for better UX and reduced tool count:
#
# get_document_info:
#   - Replaces: get_document_summary, get_document_metadata, get_document_sections, get_section_details
#   - Benefit: Single tool with info_type parameter instead of 4 separate tools
#
# exact_match_search:
#   - Replaces: keyword_search, cross_reference_search, entity_search
#   - Benefit: Unified interface with search_type parameter + ROI filtering


class GetDocumentInfoInput(ToolInput):
    """Input for unified get_document_info tool."""

    document_id: str = Field(..., description="Document ID")
    info_type: str = Field(
        ...,
        description="Type of information: 'summary' (high-level overview), 'metadata' (comprehensive stats), 'sections' (list all sections), 'section_details' (specific section info)"
    )
    section_id: Optional[str] = Field(None, description="Section ID (required for info_type='section_details')")


@register_tool
class GetDocumentInfoTool(BaseTool):
    """
    Unified document information tool.

    Combines get_document_summary, get_document_metadata, get_document_sections, and get_section_details
    into a single tool with info_type parameter.

    Uses: Vector store metadata (Layer 1/2/3)
    Speed: <50ms
    Use for: All document information queries
    """

    name = "get_document_info"
    description = "Get document information (summary, metadata, sections, or section details)"
    tier = 1
    input_schema = GetDocumentInfoInput

    def execute_impl(
        self, document_id: str, info_type: str, section_id: Optional[str] = None
    ) -> ToolResult:
        try:
            # Get layer metadata
            layer1_chunks = []
            layer2_chunks = []
            layer3_chunks = []

            if hasattr(self.vector_store, "metadata_layer1"):
                layer1_chunks = self.vector_store.metadata_layer1
                layer2_chunks = self.vector_store.metadata_layer2
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer1_chunks = self.vector_store.faiss_store.metadata_layer1
                layer2_chunks = self.vector_store.faiss_store.metadata_layer2
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            if info_type == "summary":
                # Get document summary (Layer 1)
                doc_summary = None
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        doc_summary = meta.get("content")
                        break

                if not doc_summary:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data={"document_id": document_id, "summary": doc_summary},
                    metadata={"document_id": document_id, "summary_length": len(doc_summary)},
                )

            elif info_type == "metadata":
                # Get comprehensive metadata (all layers)
                metadata = {"document_id": document_id}

                # Layer 1: Summary
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        metadata["summary"] = meta.get("content")
                        break

                # Layer 2: Sections
                sections = [
                    meta.get("section_title")
                    for meta in layer2_chunks
                    if meta.get("document_id") == document_id
                ]
                metadata["section_count"] = len(sections)
                metadata["sections"] = sections

                # Layer 3: Chunks
                chunk_count = sum(1 for meta in layer3_chunks if meta.get("document_id") == document_id)
                metadata["chunk_count"] = chunk_count

                # Estimate document length
                total_chars = sum(
                    len(meta.get("content", ""))
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                )
                metadata["estimated_chars"] = total_chars
                metadata["estimated_words"] = total_chars // 5

                if not metadata.get("summary") and metadata["section_count"] == 0:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data=metadata,
                    metadata={"document_id": document_id, "total_sections": len(sections)},
                )

            elif info_type == "sections":
                # Get list of sections (Layer 2)
                sections = []
                for meta in layer2_chunks:
                    if meta.get("document_id") == document_id:
                        section_info = {
                            "section_id": meta.get("section_id"),
                            "section_title": meta.get("section_title"),
                        }
                        sections.append(section_info)

                if not sections:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                # Sort sections by section_id
                sections.sort(key=lambda x: x.get("section_id", ""))

                # Use adaptive formatter for dynamic section limits
                try:
                    from .token_manager import get_adaptive_formatter

                    formatter = get_adaptive_formatter()
                    formatted_sections, format_metadata = formatter.format_sections_with_budget(
                        sections, include_summary=False
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
                        },
                    )

                except ImportError:
                    # Fallback
                    total_count = len(sections)
                    max_sections = 50
                    truncated = total_count > max_sections
                    sections = sections[:max_sections]

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

            elif info_type == "section_details":
                # Get specific section details (Layer 2 + Layer 3)
                if not section_id:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="section_id is required for info_type='section_details'",
                    )

                # Find section in Layer 2
                section_data = None
                for meta in layer2_chunks:
                    if meta.get("document_id") == document_id and meta.get("section_id") == section_id:
                        section_data = {
                            "document_id": document_id,
                            "section_id": section_id,
                            "section_title": meta.get("section_title"),
                            "section_path": meta.get("section_path"),
                            "summary": meta.get("content"),
                            "page_number": meta.get("page_number"),
                        }
                        break

                if not section_data:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "section_id": section_id, "found": False},
                    )

                # Get chunk count (Layer 3)
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

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid info_type: {info_type}. Must be 'summary', 'metadata', 'sections', or 'section_details'",
                )

        except Exception as e:
            logger.error(f"Get document info failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class ExactMatchSearchInput(ToolInput):
    """Input for unified exact_match_search tool."""

    query: str = Field(..., description="Search query (keywords or reference text)")
    search_type: Literal["keywords", "cross_references"] = Field(
        ...,
        description="Search type: 'keywords' (general keyword search), 'cross_references' (find references to specific clauses/articles)"
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)
    document_id: Optional[str] = Field(
        None,
        description="Optional: Filter search to specific document (index-level filtering for better performance)"
    )
    section_id: Optional[str] = Field(
        None,
        description="Optional: Filter results to specific section (requires document_id, uses post-retrieval filtering)"
    )


@register_tool
class ExactMatchSearchTool(BaseTool):
    """
    Unified exact match search tool.

    Combines keyword_search and cross_reference_search into a single tool.

    Uses: BM25 sparse retrieval
    Speed: ~50-100ms (fastest)
    Use for: Exact keyword/phrase matching and cross-reference finding
    """

    name = "exact_match_search"
    description = "Fast BM25 keyword search or cross-reference lookup with optional scope limiting (document/section ROI)"
    tier = 1
    input_schema = ExactMatchSearchInput

    def execute_impl(
        self,
        query: str,
        search_type: str,
        k: int = 6,
        document_id: Optional[str] = None,
        section_id: Optional[str] = None
    ) -> ToolResult:
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # Validate section_id requires document_id
        if section_id and not document_id:
            return ToolResult(
                success=False,
                data=None,
                error="section_id requires document_id to be specified",
            )

        try:
            # Retrieve 3x candidates when section_id filtering is needed, since we'll
            # be post-filtering results by section_id (BM25 doesn't support section-level
            # index filtering, only document-level). This ensures we get ~k results after filtering.
            # Note: If section has < k chunks, we may return fewer than k results.
            retrieval_k = k * 3 if section_id else k

            if search_type == "keywords":
                # Pure BM25 keyword search with optional document filter
                if hasattr(self.vector_store, "bm25_store"):
                    results = self.vector_store.bm25_store.search_layer3(
                        query=query,
                        k=retrieval_k,
                        document_filter=document_id  # BM25 supports document filtering
                    )
                else:
                    # Fallback
                    import numpy as np
                    dummy_embedding = np.zeros((1, self.embedder.dimensions))
                    results_dict = self.vector_store.hierarchical_search(
                        query_text=query,
                        query_embedding=dummy_embedding,
                        k_layer3=retrieval_k,
                        document_filter=document_id
                    )
                    results = results_dict["layer3"]

            elif search_type == "cross_references":
                # Cross-reference search: Find exact mentions of clauses/articles/sections
                # Uses BM25 for initial retrieval, then strict substring matching to ensure
                # the reference actually appears in the chunk (BM25 may match partial words).
                # Retrieves 2x candidates to account for filtering by substring match.
                results_dict = self.vector_store.hierarchical_search(
                    query_text=query,
                    query_embedding=None,
                    k_layer3=retrieval_k * 2,
                    document_filter=document_id  # Apply document filter
                )
                chunks = results_dict.get("layer3", [])

                # Filter chunks that actually contain the reference
                reference_lower = query.lower()
                results = [
                    chunk
                    for chunk in chunks
                    if reference_lower in chunk.get("raw_content", chunk.get("content", "")).lower()
                ][:retrieval_k]

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid search_type: {search_type}. Must be 'keywords' or 'cross_references'",
                )

            # Apply section_id filter if specified
            if section_id:
                results = [
                    chunk for chunk in results
                    if chunk.get("section_id") == section_id
                ][:k]

            # Determine search scope for metadata
            if section_id:
                search_scope = f"section (doc={document_id}, section={section_id})"
            elif document_id:
                search_scope = f"document (doc={document_id})"
            else:
                search_scope = "entire database"

            if not results:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "query": query,
                        "search_type": search_type,
                        "search_scope": search_scope,
                        "document_id": document_id,
                        "section_id": section_id,
                        "no_results": True,
                    },
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
                    "query": query,
                    "search_type": search_type,
                    "method": "bm25",
                    "search_scope": search_scope,
                    "document_id": document_id,
                    "section_id": section_id,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Exact match search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
