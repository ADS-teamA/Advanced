"""
Query optimization modules.

Includes HyDE (Hypothetical Document Embeddings) and query decomposition.
"""

from .decomposition import QueryDecomposer
from .hyde import HyDEGenerator
from .optimizer import QueryOptimizer

__all__ = ["HyDEGenerator", "QueryDecomposer", "QueryOptimizer"]
