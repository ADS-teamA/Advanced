"""
Claude Agent SDK RAG CLI

A pragmatic, production-ready agent for document retrieval and analysis.

Architecture:
- 17 RAG tools organized in 3 tiers (basic, advanced, analysis)
- Hybrid search with BM25 + FAISS + RRF fusion
- Knowledge graph integration
- HyDE and query decomposition
- Streaming responses via Claude SDK

Usage:
    python run_agent.py --store output/hybrid_store
"""

from .config import AgentConfig
from .agent_core import AgentCore

__version__ = "1.0.0"
__all__ = ["AgentConfig", "AgentCore"]
