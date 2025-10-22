"""
Query optimization modules.

Includes HyDE (Hypothetical Document Embeddings) and query decomposition.
"""

from .hyde import HyDEGenerator
from .decomposition import QueryDecomposer
from .optimizer import QueryOptimizer

__all__ = ["HyDEGenerator", "QueryDecomposer", "QueryOptimizer"]
