"""
Claude Agent SDK RAG CLI

A pragmatic, production-ready agent for document retrieval and analysis.

Architecture:
- 27 RAG tools organized in 3 tiers (basic, advanced, analysis)
- Hybrid search with BM25 + FAISS + RRF fusion
- Knowledge graph integration
- Smart token management for adaptive tool output sizing
- Streaming responses via Claude SDK with prompt caching

Usage:
    python -m src.agent.cli --vector-store vector_db
"""

from .agent_core import AgentCore
from .config import AgentConfig

__version__ = "1.0.0"
__all__ = ["AgentConfig", "AgentCore"]
