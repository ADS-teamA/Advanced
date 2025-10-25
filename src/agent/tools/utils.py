"""
Shared Tool Utilities

Helper functions used across multiple tools.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def format_chunk_result(
    chunk: Dict[str, Any], include_score: bool = True, max_content_length: int = 400
) -> Dict[str, Any]:
    """
    Format a chunk result for tool output.

    Args:
        chunk: Chunk dict from retrieval
        include_score: Whether to include score
        max_content_length: Maximum content length (chars). Default 400 to prevent token overflow.
                          Set to None for no truncation.

    Returns:
        Formatted dict
    """
    # Get content and truncate if needed
    content = chunk.get("content", chunk.get("raw_content", ""))
    if max_content_length and len(content) > max_content_length:
        content = content[:max_content_length] + "... [truncated]"

    result = {
        "content": content,
        "document_id": chunk.get("document_id", "unknown"),
        "section_title": chunk.get("section_title", ""),
        "chunk_id": chunk.get("chunk_id", ""),
    }

    if include_score:
        # Prefer rerank_score, fall back to boosted_score, then rrf_score, then score
        score = (
            chunk.get("rerank_score")
            or chunk.get("boosted_score")
            or chunk.get("rrf_score")
            or chunk.get("score")
            or 0.0
        )
        result["score"] = round(float(score), 4)

    # Add page number if available
    if "page_number" in chunk:
        result["page"] = chunk["page_number"]

    return result


def generate_citation(chunk: Dict[str, Any], chunk_number: int, format: str = "inline") -> str:
    """
    Generate citation string for a chunk.

    Args:
        chunk: Chunk dict
        chunk_number: Citation number (1-indexed)
        format: "inline", "detailed", or "footnote"

    Returns:
        Citation string
    """
    doc_name = chunk.get("document_name") or chunk.get("document_id", "unknown")
    section = chunk.get("section_title", "")
    page = chunk.get("page_number")

    if format == "inline":
        return f"[Chunk {chunk_number}]"

    elif format == "detailed":
        parts = [f"Doc: {doc_name}"]
        if section:
            parts.append(f"Section: {section}")
        if page:
            parts.append(f"Page: {page}")
        return f"[{', '.join(parts)}]"

    elif format == "footnote":
        return f"[{chunk_number}] {doc_name}" + (f", {section}" if section else "")

    else:
        return f"[{chunk_number}]"


def create_error_result(error_message: str, tool_name: str = None) -> Dict[str, Any]:
    """
    Create standardized error result.

    Args:
        error_message: Error description
        tool_name: Name of tool that failed

    Returns:
        Error dict
    """
    return {
        "error": error_message,
        "tool": tool_name,
        "success": False,
    }


def validate_k_parameter(k: int, max_k: int = 10) -> int:
    """
    Validate and clamp k parameter.

    Args:
        k: Number of results requested
        max_k: Maximum allowed value (default: 10 to prevent token overflow)

    Returns:
        Validated k value
    """
    if k < 1:
        logger.warning(f"k={k} is too small, using k=1")
        return 1

    if k > max_k:
        logger.warning(f"k={k} exceeds maximum {max_k}, clamping to {max_k}")
        return max_k

    return k


def deduplicate_chunks(chunks: List[Dict], key: str = "chunk_id") -> List[Dict]:
    """
    Remove duplicate chunks based on key.

    Args:
        chunks: List of chunk dicts
        key: Key to use for deduplication

    Returns:
        Deduplicated list (preserves order)
    """
    seen = set()
    result = []

    for chunk in chunks:
        chunk_key = chunk.get(key)
        if chunk_key and chunk_key not in seen:
            seen.add(chunk_key)
            result.append(chunk)

    if len(result) < len(chunks):
        logger.debug(f"Deduplicated {len(chunks)} chunks to {len(result)}")

    return result


def merge_chunk_lists(
    *chunk_lists: List[Dict], max_total: int = None, sort_by: str = "score"
) -> List[Dict]:
    """
    Merge multiple chunk lists, deduplicate, and optionally sort.

    Args:
        *chunk_lists: Variable number of chunk lists
        max_total: Maximum total chunks to return
        sort_by: Field to sort by (usually "score")

    Returns:
        Merged and sorted chunk list
    """
    # Flatten all lists
    all_chunks = []
    for chunk_list in chunk_lists:
        all_chunks.extend(chunk_list)

    # Deduplicate
    deduplicated = deduplicate_chunks(all_chunks)

    # Sort by score (descending)
    if sort_by in deduplicated[0] if deduplicated else False:
        deduplicated.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

    # Limit to max_total
    if max_total and len(deduplicated) > max_total:
        deduplicated = deduplicated[:max_total]

    return deduplicated
