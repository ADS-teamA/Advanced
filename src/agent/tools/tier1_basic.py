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


# === Tool 0: Get Tool Help (Meta Tool) ===


class GetToolHelpInput(ToolInput):
    """Input for get_tool_help tool."""

    tool_name: str = Field(
        ...,
        description="Name of tool to get help for (e.g., 'search', 'compare_documents')"
    )


@register_tool
class GetToolHelpTool(BaseTool):
    """Get detailed documentation for a specific tool."""

    name = "get_tool_help"
    description = "Get detailed help for any tool"
    detailed_help = """
    Returns comprehensive documentation for a specific tool including:
    - Full description and use cases
    - All parameters with types and defaults
    - Examples of when to use this tool
    - Performance characteristics (tier, speed)

    Use this whenever you need to understand a tool's capabilities or parameters
    before using it for the first time.
    """
    tier = 1
    input_schema = GetToolHelpInput

    def execute_impl(self, tool_name: str) -> ToolResult:
        """Get detailed help for a tool."""
        from .registry import get_registry

        registry = get_registry()

        # Check if tool exists
        if tool_name not in registry._tool_classes:
            available_tools = sorted(registry._tool_classes.keys())
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools[:10])}...",
                metadata={"requested_tool": tool_name, "available_count": len(available_tools)},
            )

        # Get tool class
        tool_class = registry._tool_classes[tool_name]

        # Build detailed help
        help_text = f"# {tool_class.name}\n\n"
        help_text += f"**Tier:** {tool_class.tier} "
        help_text += f"({'Basic/Fast' if tool_class.tier == 1 else 'Advanced' if tool_class.tier == 2 else 'Analysis/Slow'})\n\n"

        # Description
        help_text += f"**Description:** {tool_class.description}\n\n"

        # Detailed help if available
        if tool_class.detailed_help:
            help_text += f"**Details:**\n{tool_class.detailed_help.strip()}\n\n"

        # Parameters from Pydantic schema
        if tool_class.input_schema and tool_class.input_schema != ToolInput:
            help_text += "**Parameters:**\n"
            schema = tool_class.input_schema.model_json_schema()

            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "No description")
                is_required = "✓ Required" if param_name in required else "Optional"
                default = param_info.get("default", "N/A")

                help_text += f"- `{param_name}` ({param_type}) - {is_required}\n"
                help_text += f"  {param_desc}\n"
                if default != "N/A":
                    help_text += f"  Default: {default}\n"
                help_text += "\n"

        # Requirements
        requirements = []
        if tool_class.requires_kg:
            requirements.append("Knowledge Graph")
        if tool_class.requires_reranker:
            requirements.append("Reranker")

        if requirements:
            help_text += f"**Requires:** {', '.join(requirements)}\n\n"

        return ToolResult(
            success=True,
            data={
                "tool_name": tool_name,
                "tier": tool_class.tier,
                "help_text": help_text,
                "short_description": tool_class.description,
                "requires_kg": tool_class.requires_kg,
                "requires_reranker": tool_class.requires_reranker,
            },
            metadata={"tool": tool_name},
        )


# === Tool 1: Unified Search (with Query Expansion) ===


class SearchInput(ToolInput):
    """Input for unified search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(5, description="Number of results to return (3-5 recommended)", ge=1, le=10)
    num_expands: int = Field(
        1,
        description="Number of query expansions: 1=original query only (fast), 3=multi-query search (better recall), 5=maximum expansion (best recall but slower). Warning: num_expands > 5 may impact performance",
        ge=1,
        le=10,
    )


@register_tool
class SearchTool(BaseTool):
    """Unified search with optional query expansion."""

    name = "search"
    description = "Unified hybrid search with optional query expansion for better recall"
    detailed_help = """
    Unified search tool combining hybrid retrieval (BM25 + Dense + RRF) with optional
    query expansion for improved recall. Agent can control both result count (k) and
    expansion level (num_expands).

    **Query Expansion:**
    - num_expands=1: Use original query only (fast, ~200-300ms)
    - num_expands=3: Multi-query search (+15-25% recall, ~500-800ms)
    - num_expands=5: Maximum expansion (+20-30% recall, ~1-1.5s)

    **When to use:**
    - Most queries (replaces simple_search)
    - When recall is important: Use num_expands=3 or higher
    - When speed is critical: Use num_expands=1
    - For ambiguous queries: Use num_expands=3 to capture different interpretations

    **Method:**
    1. Query expansion (if num_expands > 1): Generate N variations using LLM
    2. Hybrid search for each query: BM25 + Dense + RRF fusion
    3. RRF fusion across queries (if multiple queries)
    4. Cross-encoder reranking (final quality boost)

    **Performance:**
    - num_expands=1: ~200-300ms (same as simple_search)
    - num_expands=3: ~500-800ms (recommended)
    - num_expands=5: ~1-1.5s (maximum quality)

    **Best practices:**
    - Start with num_expands=1 for most queries
    - Use num_expands=3 when user query is ambiguous or recall is critical
    - Use num_expands=5 only when comprehensive recall is essential
    - Avoid num_expands > 5 (diminishing returns, performance impact)
    """
    tier = 1
    input_schema = SearchInput
    requires_reranker = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_expander = None  # Lazy initialization

    def _get_query_expander(self):
        """Lazy initialization of QueryExpander."""
        if self._query_expander is None:
            import os
            from ..query_expander import QueryExpander

            # Get provider and model from config (self.config is ToolConfig)
            provider = self.config.query_expansion_provider
            model = self.config.query_expansion_model

            # Get API keys from environment variables
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")

            try:
                self._query_expander = QueryExpander(
                    provider=provider,
                    model=model,
                    anthropic_api_key=anthropic_key,
                    openai_api_key=openai_key,
                )
                logger.info(f"QueryExpander initialized: provider={provider}, model={model}")
            except Exception as e:
                logger.warning(f"QueryExpander initialization failed: {e}. Query expansion will be disabled.")
                self._query_expander = None

        return self._query_expander

    def execute_impl(self, query: str, k: int = 5, num_expands: int = 1) -> ToolResult:
        k, _ = validate_k_parameter(k, adaptive=True, detail_level="medium")

        # === STEP 1: Query Expansion ===
        queries = [query]  # Default: original query only
        expansion_metadata = {"num_expands": num_expands, "expansion_method": "none"}

        if num_expands > 1:
            # Attempt query expansion
            expander = self._get_query_expander()

            if expander:
                try:
                    expansion_result = expander.expand(query, num_expansions=num_expands - 1)
                    queries = expansion_result.expanded_queries
                    expansion_metadata.update({
                        "expansion_method": expansion_result.expansion_method,
                        "model_used": expansion_result.model_used,
                        "queries_generated": len(queries),
                    })
                    logger.debug(f"Query expanded to {len(queries)} queries: {queries}")
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}. Using original query only.")
                    # Fallback to original query
                    queries = [query]
                    expansion_metadata["expansion_method"] = "failed"
            else:
                logger.warning("QueryExpander not available. Using original query only.")
                expansion_metadata["expansion_method"] = "unavailable"

        # === STEP 2: Hybrid Search for Each Query ===
        # Retrieve more candidates for reranking
        candidates_k = k * 3 if self.reranker else k

        all_chunks = []  # All chunks from all queries (with their original scores)

        for q in queries:
            # Embed query
            query_embedding = self.embedder.embed_texts([q])

            # Hybrid search
            results = self.vector_store.hierarchical_search(
                query_text=q,
                query_embedding=query_embedding,
                k_layer3=candidates_k,
            )

            chunks = results["layer3"]
            # Tag chunks with their source query for tracking
            for chunk in chunks:
                chunk["_source_query"] = q

            all_chunks.extend(chunks)

        # === STEP 3: RRF Fusion (if multiple queries) ===
        if len(queries) > 1:
            # Use RRF to combine results from multiple queries
            chunks = self._rrf_fusion(all_chunks, k=candidates_k)
            fusion_method = "rrf"
        else:
            # Single query: use chunks as-is
            chunks = all_chunks[:candidates_k]
            fusion_method = "none"

        # === STEP 4: Reranking ===
        if self.reranker and len(chunks) > k:
            logger.debug(f"Reranking {len(chunks)} candidates to top {k}")
            chunks = self.reranker.rerank(query=query, candidates=chunks, top_k=k)
            reranking_applied = True
        else:
            chunks = chunks[:k]
            reranking_applied = False

        # === STEP 5: Format Results ===
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
                "num_expands": num_expands,
                "expansion_metadata": expansion_metadata,
                "fusion_method": fusion_method,
                "reranking_applied": reranking_applied,
                "method": "hybrid+expansion+rerank" if num_expands > 1 and self.reranker else "hybrid+rerank" if self.reranker else "hybrid",
                "candidates_retrieved": len(all_chunks),
                "final_count": len(formatted),
            },
        )

    def _rrf_fusion(self, chunks: List[dict], k: int = 60) -> List[dict]:
        """
        RRF (Reciprocal Rank Fusion) for combining results from multiple queries.

        RRF formula: score(chunk) = sum(1 / (k + rank)) for all queries where chunk appears

        Args:
            chunks: List of chunks from all queries (may contain duplicates)
            k: RRF parameter (default: 60, optimal from research)

        Returns:
            Deduplicated and reranked chunks sorted by RRF score
        """
        # Group chunks by chunk_id
        chunk_ranks = {}  # {chunk_id: [(rank, source_query), ...]}

        current_rank = {}  # {source_query: current_rank}

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            source_query = chunk.get("_source_query", "")

            if source_query not in current_rank:
                current_rank[source_query] = 0

            rank = current_rank[source_query]
            current_rank[source_query] += 1

            if chunk_id not in chunk_ranks:
                chunk_ranks[chunk_id] = {
                    "chunk": chunk,
                    "ranks": [],
                }

            chunk_ranks[chunk_id]["ranks"].append((rank, source_query))

        # Calculate RRF scores
        rrf_scores = []
        for chunk_id, data in chunk_ranks.items():
            # RRF score: sum of 1/(k + rank) for all occurrences
            rrf_score = sum(1.0 / (k + rank) for rank, _ in data["ranks"])

            rrf_scores.append({
                "chunk": data["chunk"],
                "rrf_score": rrf_score,
                "appearances": len(data["ranks"]),
            })

        # Sort by RRF score (descending)
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Return top chunks
        return [item["chunk"] for item in rrf_scores]


# === Tool 3: Get Document List ===


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool."""

    pass  # No parameters needed


@register_tool
class GetDocumentListTool(BaseTool):
    """List all indexed documents."""

    name = "get_document_list"
    description = "List all indexed documents"
    detailed_help = """
    Returns a list of all indexed documents with their summaries.

    **When to use:**
    - User asks "what documents are available?"
    - Need to discover corpus contents
    - Before document-specific queries

    **Data source:** Vector store metadata (Layer 1 - document level)
    **Speed:** <10ms (metadata lookup only)
    """
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
            {"id": doc_id, "summary": summary} for doc_id, summary in sorted(documents_map.items())
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
    """List all available tools."""

    name = "list_available_tools"
    description = "List all available tools"
    detailed_help = """
    Returns a complete list of all available tools with short descriptions.
    For detailed help on a specific tool, use get_tool_help instead.

    **When to use:**
    - Need to see all available tools
    - Understand available capabilities
    - Select the right tool for a task

    **Best practice:** Use get_tool_help for detailed docs on specific tools.

    **Speed:** <10ms (metadata lookup)
    """
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

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc,
                        "required": is_required,
                    }
                )

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

            tools_list.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "when_to_use": when_to_use,
                    "tier": f"Tier {tool.tier} - {tier_label.get(tool.tier, 'Unknown')}",
                }
            )

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
                        "Use 'search' for most queries (hybrid + optional expansion + rerank = best quality)",
                        "Start with num_expands=1 for speed, increase to 3-5 for better recall when needed",
                        "For complex queries, decompose into sub-tasks and use multiple tools",
                        "Try multiple retrieval strategies before giving up",
                    ],
                    "selection_strategy": {
                        "most_queries": "search (with num_expands=1 for speed, 3-5 for recall)",
                        "entity_focused": "Use 'search' with entity names, or multi_hop_search if KG available",
                        "specific_document": "Use exact_match_search or filtered_search with document_id filter",
                        "multi_hop_reasoning": "multi_hop_search (requires KG)",
                        "comparison": "compare_documents",
                        "temporal_info": "filtered_search with filter_type='temporal' or timeline_view",
                    },
                },
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
        description="Type of information: 'summary' (high-level overview), 'metadata' (comprehensive stats), 'sections' (list all sections), 'section_details' (specific section info)",
    )
    section_id: Optional[str] = Field(
        None, description="Section ID (required for info_type='section_details')"
    )


@register_tool
class GetDocumentInfoTool(BaseTool):
    """Get document information."""

    name = "get_document_info"
    description = "Get document info/metadata"
    detailed_help = """
    Unified tool for retrieving document information with multiple info types:
    - 'summary': High-level document overview
    - 'metadata': Comprehensive stats (sections, chunks, source info)
    - 'sections': List all sections with titles and hierarchy
    - 'section_details': Detailed info about a specific section

    **When to use:**
    - Need document overview before detailed search
    - Want to understand document structure
    - Looking for specific section to search within

    **Best practices:**
    - Use 'summary' for quick overview
    - Use 'sections' to understand structure
    - Use 'metadata' for comprehensive stats
    - Combine with filtered_search to search within sections

    **Data source:** Vector store metadata (Layer 1/2/3)
    **Speed:** <50ms
    """
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
                chunk_count = sum(
                    1 for meta in layer3_chunks if meta.get("document_id") == document_id
                )
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
                    if (
                        meta.get("document_id") == document_id
                        and meta.get("section_id") == section_id
                    ):
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
                        metadata={
                            "document_id": document_id,
                            "section_id": section_id,
                            "found": False,
                        },
                    )

                # Get chunk count (Layer 3)
                chunk_count = sum(
                    1
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                    and meta.get("section_id") == section_id
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
        description="Search type: 'keywords' (general keyword search), 'cross_references' (find references to specific clauses/articles)",
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)
    document_id: Optional[str] = Field(
        None,
        description="Optional: Filter search to specific document (index-level filtering for better performance)",
    )
    section_id: Optional[str] = Field(
        None,
        description="Optional: Filter results to specific section (requires document_id, uses post-retrieval filtering)",
    )


@register_tool
class ExactMatchSearchTool(BaseTool):
    """Fast BM25 keyword/exact match search."""

    name = "exact_match_search"
    description = "Fast BM25 keyword search"
    detailed_help = """
    Fast BM25-based search optimized for exact keyword matches and cross-references.
    Supports optional scope limiting to specific documents or sections.

    **Search types:**
    - 'keywords': General keyword search
    - 'cross_references': Find references to specific clauses/articles

    **When to use:**
    - Exact keyword or phrase matching
    - Finding cross-references (e.g., "článek 5", "section 3.2")
    - When speed is critical over semantic understanding

    **Best practices for queries:**
    - Keep queries SHORT and SPECIFIC (2-5 words ideal)
    - Use exact terms from documents
    - For cross-refs: Use standard format (e.g., "článek 5", "§ 15")
    - Avoid long sentences (BM25 works better with keywords)
    - Use document_id filter when possible for faster results

    **Method:** BM25 sparse retrieval only (no embeddings)
    **Speed:** ~50-100ms (fastest tool)
    """
    tier = 1
    input_schema = ExactMatchSearchInput

    def execute_impl(
        self,
        query: str,
        search_type: str,
        k: int = 6,
        document_id: Optional[str] = None,
        section_id: Optional[str] = None,
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
                        document_filter=document_id,  # BM25 supports document filtering
                    )
                else:
                    # Fallback
                    import numpy as np

                    dummy_embedding = np.zeros((1, self.embedder.dimensions))
                    results_dict = self.vector_store.hierarchical_search(
                        query_text=query,
                        query_embedding=dummy_embedding,
                        k_layer3=retrieval_k,
                        document_filter=document_id,
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
                    document_filter=document_id,  # Apply document filter
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
                results = [chunk for chunk in results if chunk.get("section_id") == section_id][:k]

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
