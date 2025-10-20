"""
Recursive Character Text Splitter (RCTS) implementation.

Based on LegalBench-RAG research:
- RCTS > Fixed-size chunking (Precision@1: 6.41% vs 2.40%)
- 500 characters optimal chunk size
- 0 overlap (RCTS handles boundaries naturally)
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.chunking_strategy import ChunkingStrategy
from src.core.config import ChunkingConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RCTSStrategy(ChunkingStrategy):
    """
    Recursive Character Text Splitter strategy.

    Evidence from LegalBench-RAG (Pipitone & Alami, 2024):
    - RCTS achieves 6.41% Precision@1 vs 2.40% for naive chunking (+167%)
    - RCTS achieves 62.22% Recall@64 vs ~35% for naive chunking (+78%)
    - RCTS respects natural text boundaries (paragraphs, sentences, clauses)
    - 500 characters is optimal chunk size (Reuter et al., 2024)
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize RCTS strategy.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # Create LangChain RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators,
            is_separator_regex=False
        )

        logger.info(
            f"Initialized RCTS with chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}"
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text using RCTS algorithm.

        The algorithm tries separators in order:
        1. Paragraph breaks (\\n\\n)
        2. Line breaks (\\n)
        3. Sentence ends (. )
        4. Clause separators (; )
        5. Sub-clause separators (, )
        6. Word boundaries ( )
        7. Character-level (fallback)

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text.strip():
            logger.warning("Empty text provided to split_text")
            return []

        chunks = self.splitter.split_text(text)

        logger.debug(
            f"Split {len(text)} chars into {len(chunks)} chunks "
            f"(avg: {len(text) // len(chunks) if chunks else 0} chars/chunk)"
        )

        return chunks

    def get_chunk_size(self) -> int:
        """Get configured chunk size."""
        return self.config.chunk_size
