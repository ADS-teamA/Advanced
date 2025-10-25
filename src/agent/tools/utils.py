"""
Shared Tool Utilities

Helper functions used across multiple tools.

IMPORTANT: This module now uses smart token management (token_manager.py) instead of
hardcoded character limits. Old functions preserved for backward compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import smart token management (lazy import to avoid circular deps)
_token_manager_imported = False
_DetailLevel = None
_get_adaptive_formatter = None
_get_token_counter = None
_SmartTruncator = None


def _ensure_token_manager_imported():
    """Lazy import token management to avoid circular dependencies."""
    global _token_manager_imported, _DetailLevel, _get_adaptive_formatter
    global _get_token_counter, _SmartTruncator

    if not _token_manager_imported:
        try:
            from .token_manager import (
                DetailLevel,
                get_adaptive_formatter,
                get_token_counter,
                SmartTruncator,
            )

            _DetailLevel = DetailLevel
            _get_adaptive_formatter = get_adaptive_formatter
            _get_token_counter = get_token_counter
            _SmartTruncator = SmartTruncator
            _token_manager_imported = True
        except ImportError as e:
            logger.warning(f"Could not import token_manager, using legacy limits: {e}")
            _token_manager_imported = False


def format_chunk_result(
    chunk: Dict[str, Any],
    include_score: bool = True,
    max_content_length: int = None,
    smart_truncate: bool = True,
    detail_level: str = "medium",
) -> Dict[str, Any]:
    """
    Format a chunk result for tool output.

    NEW: Now uses smart token-aware truncation by default (truncates at sentence boundaries).
    Falls back to character-based truncation if token_manager unavailable.

    Args:
        chunk: Chunk dict from retrieval
        include_score: Whether to include score
        max_content_length: Maximum content length (DEPRECATED: use detail_level instead).
                          If None, uses smart token-based limits.
                          If specified, overrides smart truncation.
        smart_truncate: Use token-aware sentence-boundary truncation (default: True)
        detail_level: Detail level ("summary", "medium", "full"). Ignored if max_content_length set.

    Returns:
        Formatted dict with 'content', 'document_id', 'section_title', 'chunk_id',
        optionally 'score', 'page', 'truncated'
    """
    content = chunk.get("content", chunk.get("raw_content", ""))

    # Determine truncation strategy
    if max_content_length is not None:
        # Legacy mode: character-based truncation (backward compatibility)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
            was_truncated = True
        else:
            was_truncated = False

    elif smart_truncate:
        # NEW: Smart token-aware truncation
        _ensure_token_manager_imported()

        if _token_manager_imported:
            # Map detail_level string to enum
            detail_map = {
                "summary": _DetailLevel.SUMMARY,  # ~100 tokens
                "medium": _DetailLevel.MEDIUM,  # ~300 tokens
                "full": _DetailLevel.FULL,  # ~600 tokens
            }
            level = detail_map.get(detail_level.lower(), _DetailLevel.MEDIUM)

            formatter = _get_adaptive_formatter()
            token_counter = _get_token_counter()
            max_tokens = formatter.budget.tokens_per_item(level)

            content, was_truncated = _SmartTruncator.truncate_at_sentence(
                content, max_tokens, token_counter
            )
        else:
            # Fallback: character-based with estimate
            char_limits = {"summary": 400, "medium": 1200, "full": 2400}
            char_limit = char_limits.get(detail_level.lower(), 1200)

            if len(content) > char_limit:
                content = content[:char_limit] + "... [truncated]"
                was_truncated = True
            else:
                was_truncated = False
    else:
        # No truncation
        was_truncated = False

    # Build result
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

    # Add truncation flag
    if was_truncated:
        result["truncated"] = True

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


def validate_k_parameter(
    k: int, max_k: int = None, adaptive: bool = True, detail_level: str = "medium"
) -> Tuple[int, Optional[str]]:
    """
    Validate and adaptively adjust k parameter based on token budget.

    NEW: Now adaptively calculates max_k based on available token budget.
    Falls back to hardcoded limits if token_manager unavailable.

    Args:
        k: Number of results requested
        max_k: Maximum allowed value (optional). If None, calculates from token budget.
        adaptive: Use adaptive k based on token budget (default: True)
        detail_level: Detail level for results ("summary", "medium", "full")

    Returns:
        Tuple of (validated_k, adjustment_reason)
        - validated_k: Adjusted k value
        - adjustment_reason: Why k was adjusted (None if not adjusted)
    """
    if k < 1:
        logger.warning(f"k={k} is too small, using k=1")
        return 1, "below_minimum"

    if adaptive and max_k is None:
        # NEW: Calculate max_k from token budget
        _ensure_token_manager_imported()

        if _token_manager_imported:
            formatter = _get_adaptive_formatter()
            detail_map = {
                "summary": _DetailLevel.SUMMARY,
                "medium": _DetailLevel.MEDIUM,
                "full": _DetailLevel.FULL,
            }
            level = detail_map.get(detail_level.lower(), _DetailLevel.MEDIUM)
            tokens_per_item = formatter.budget.tokens_per_item(level)

            calculated_k, reason = formatter.adaptive_k(
                requested_k=k, tokens_per_item=tokens_per_item, min_k=3, max_k=50  # Safety limit
            )

            if calculated_k != k:
                logger.info(
                    f"Adaptive k: {k} → {calculated_k} "
                    f"(detail_level={detail_level}, reason={reason})"
                )
                return calculated_k, reason
            else:
                return k, None
        else:
            # Fallback: use hardcoded max_k
            max_k = 10

    # Use provided or fallback max_k
    if max_k is None:
        max_k = 10  # Legacy default

    if k > max_k:
        logger.warning(f"k={k} exceeds maximum {max_k}, clamping to {max_k}")
        return max_k, "exceeded_maximum"

    return k, None


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
