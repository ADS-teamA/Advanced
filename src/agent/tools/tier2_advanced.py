"""
TIER 2: Advanced Retrieval Tools

Quality tools (500-1000ms) for complex retrieval tasks.
Use when Tier 1 tools are insufficient.
"""

from pydantic import Field
from typing import Optional, List
from datetime import datetime
from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 2 TOOLS: Advanced Retrieval
# ============================================================================


class MultiHopSearchInput(ToolInput):
    query: str = Field(..., description="Multi-hop query requiring graph traversal")
    max_hops: int = Field(2, description="Maximum hops in graph", ge=1, le=3)
    k: int = Field(6, description="Number of results", ge=1, le=20)


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
                    "doc1_relevant_chunks": [
                        format_chunk_result(c) for c in doc1_relevant
                    ],
                    "doc2_relevant_chunks": [
                        format_chunk_result(c) for c in doc2_relevant
                    ],
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


class FindRelatedChunksInput(ToolInput):
    chunk_id: str = Field(..., description="Chunk ID to find related chunks for")
    k: int = Field(6, description="Number of related chunks to return", ge=1, le=20)
    same_document_only: bool = Field(
        False, description="Only find related chunks from same document"
    )


@register_tool
class FindRelatedChunksTool(BaseTool):
    """Find chunks semantically related to a given chunk."""

    name = "find_related_chunks"
    description = "Find chunks semantically related to a specific chunk. Use for queries like 'find sections related to this clause'"
    tier = 2
    input_schema = FindRelatedChunksInput

    def execute_impl(
        self, chunk_id: str, k: int = 6, same_document_only: bool = False
    ) -> ToolResult:
        """Find related chunks using embedding similarity."""
        try:
            # Get the source chunk
            # Note: chunk_id format is "doc_id:section_id:chunk_index"
            source_chunk = self._get_chunk_by_id(chunk_id)

            if not source_chunk:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Chunk '{chunk_id}' not found",
                )

            # Use chunk content as query
            chunk_content = source_chunk.get("raw_content", source_chunk.get("content", ""))
            query_embedding = self.embedder.embed_texts([chunk_content])

            # Search for similar chunks
            document_filter = None
            if same_document_only:
                document_filter = source_chunk.get("doc_id")

            results = self.vector_store.hierarchical_search(
                query_text=chunk_content,
                query_embedding=query_embedding,
                k_layer3=k * 2,
                document_filter=document_filter,
            )

            related_chunks = results.get("layer3", [])

            # Filter out the source chunk itself
            related_chunks = [
                c for c in related_chunks if c.get("chunk_id") != chunk_id
            ]

            # Rerank if available
            if self.reranker and len(related_chunks) > k:
                related_chunks = self.reranker.rerank(chunk_content, related_chunks, top_k=k)
            else:
                related_chunks = related_chunks[:k]

            formatted = [format_chunk_result(chunk) for chunk in related_chunks]
            citations = [
                f"{chunk.get('doc_id', 'Unknown')} (Section {chunk.get('section_id', 'N/A')})"
                for chunk in related_chunks
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "source_chunk_id": chunk_id,
                    "same_document_only": same_document_only,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Find related chunks failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _get_chunk_by_id(self, chunk_id: str):
        """Get chunk by ID from vector store."""
        # This is a simplified implementation
        # In production, you'd need a proper chunk ID lookup
        try:
            # Try searching for the chunk ID as text
            results = self.vector_store.hierarchical_search(
                query_text=chunk_id,
                query_embedding=None,
                k_layer3=50,
            )

            chunks = results.get("layer3", [])
            for chunk in chunks:
                if chunk.get("chunk_id") == chunk_id:
                    return chunk

            return None
        except Exception:
            return None


class TemporalSearchInput(ToolInput):
    query: str = Field(..., description="Search query")
    start_date: Optional[str] = Field(None, description="Start date (ISO format: YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (ISO format: YYYY-MM-DD)")
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class TemporalSearchTool(BaseTool):
    """Search with date/time filtering."""

    name = "temporal_search"
    description = "Search documents with date/time filters. Use for queries like 'find regulations from 2023' or 'documents between Jan-Mar 2024'"
    tier = 2
    input_schema = TemporalSearchInput

    def execute_impl(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        k: int = 6,
    ) -> ToolResult:
        """Search with temporal filtering."""
        try:
            # Parse dates if provided
            start_dt = datetime.fromisoformat(start_date) if start_date else None
            end_dt = datetime.fromisoformat(end_date) if end_date else None

            # Perform hybrid search
            query_embedding = self.embedder.embed_texts([query])

            results = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=query_embedding,
                k_layer3=k * 3,
            )

            chunks = results.get("layer3", [])

            # Filter by date if metadata available
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
                        # Skip chunks with invalid dates
                        continue

                chunks = filtered_chunks

            if not chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "query": query,
                        "start_date": start_date,
                        "end_date": end_date,
                        "results_count": 0,
                    },
                )

            # Rerank if available
            if self.reranker:
                chunks = self.reranker.rerank(query, chunks, top_k=k)
            else:
                chunks = chunks[:k]

            formatted = [format_chunk_result(chunk) for chunk in chunks]
            citations = [
                f"{chunk.get('doc_id', 'Unknown')} ({chunk.get('date', 'No date')})"
                for chunk in chunks
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "query": query,
                    "start_date": start_date,
                    "end_date": end_date,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Temporal search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class HybridSearchWithFiltersInput(ToolInput):
    query: str = Field(..., description="Search query")
    document_type: Optional[str] = Field(None, description="Filter by document type")
    section_type: Optional[str] = Field(None, description="Filter by section type")
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class HybridSearchWithFiltersTool(BaseTool):
    """Hybrid search with metadata filters."""

    name = "hybrid_search_with_filters"
    description = "Hybrid search with metadata filters (document type, section type). Use when you need to narrow results by document characteristics"
    tier = 2
    input_schema = HybridSearchWithFiltersInput

    def execute_impl(
        self,
        query: str,
        document_type: Optional[str] = None,
        section_type: Optional[str] = None,
        k: int = 6,
    ) -> ToolResult:
        """Search with metadata filtering."""
        try:
            query_embedding = self.embedder.embed_texts([query])

            # Retrieve more candidates for filtering
            results = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=query_embedding,
                k_layer3=k * 4,
            )

            chunks = results.get("layer3", [])

            # Apply filters
            if document_type:
                chunks = [
                    c
                    for c in chunks
                    if c.get("doc_type", "").lower() == document_type.lower()
                ]

            if section_type:
                chunks = [
                    c
                    for c in chunks
                    if c.get("section_type", "").lower() == section_type.lower()
                ]

            if not chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "query": query,
                        "document_type": document_type,
                        "section_type": section_type,
                        "results_count": 0,
                    },
                )

            # Rerank if available
            if self.reranker:
                chunks = self.reranker.rerank(query, chunks, top_k=k)
            else:
                chunks = chunks[:k]

            formatted = [format_chunk_result(chunk) for chunk in chunks]
            citations = [
                f"{chunk.get('doc_id', 'Unknown')} (Section {chunk.get('section_id', 'N/A')})"
                for chunk in chunks
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "query": query,
                    "document_type": document_type,
                    "section_type": section_type,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Filtered search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class CrossReferenceSearchInput(ToolInput):
    reference_text: str = Field(
        ..., description="Reference text to search for (e.g., 'Article 5.2', 'Section 3')"
    )
    k: int = Field(6, description="Number of results", ge=1, le=20)


@register_tool
class CrossReferenceSearchTool(BaseTool):
    """Find cross-references and citations."""

    name = "cross_reference_search"
    description = "Find cross-references to specific clauses, articles, or sections. Use for queries like 'find all references to Article 5'"
    tier = 2
    input_schema = CrossReferenceSearchInput

    def execute_impl(self, reference_text: str, k: int = 6) -> ToolResult:
        """Search for cross-references."""
        try:
            # Use keyword search (BM25) for exact reference matching
            results = self.vector_store.hierarchical_search(
                query_text=reference_text,
                query_embedding=None,  # BM25 only
                k_layer3=k * 2,
            )

            chunks = results.get("layer3", [])

            # Filter chunks that actually contain the reference
            reference_lower = reference_text.lower()
            filtered_chunks = []

            for chunk in chunks:
                content = chunk.get("raw_content", chunk.get("content", "")).lower()
                if reference_lower in content:
                    filtered_chunks.append(chunk)

            if not filtered_chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "reference_text": reference_text,
                        "results_count": 0,
                    },
                )

            # Limit to k results
            filtered_chunks = filtered_chunks[:k]

            formatted = [format_chunk_result(chunk) for chunk in filtered_chunks]
            citations = [
                f"{chunk.get('doc_id', 'Unknown')} (Section {chunk.get('section_id', 'N/A')})"
                for chunk in filtered_chunks
            ]

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata={
                    "reference_text": reference_text,
                    "results_count": len(formatted),
                },
            )

        except Exception as e:
            logger.error(f"Cross-reference search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
