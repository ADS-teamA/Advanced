"""
Chunking strategy interfaces.

Provides pluggable text chunking strategies (RCTS, fixed-size, etc.).
"""

from abc import ABC, abstractmethod
from typing import List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        pass

    @abstractmethod
    def get_chunk_size(self) -> int:
        """
        Get configured chunk size.

        Returns:
            Chunk size in characters
        """
        pass
