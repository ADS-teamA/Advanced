"""
PHASE 4: Embedding Generation Module

Based on research:
- LegalBench-RAG: text-embedding-3-large baseline (3072 dims)
- MLEB 2025: Kanon 2 #1 (86% NDCG@10), Voyage 3 Large #2 (85.7%)
- Multilingual: BGE-M3 (100+ languages, 8192 tokens)

Supports:
- kanon-2 (Voyage AI, 1024 dims, MLEB #1, default)
- voyage-3-large (Voyage AI, 1024 dims, MLEB #2)
- voyage-law-2 (Voyage AI, 1024 dims, legal-optimized)
- text-embedding-3-large (OpenAI, 3072 dims)
- BAAI/bge-m3 (HuggingFace, 1024 dims, Czech support)

Implementation:
- Batch embedding for efficiency
- Layer-specific embedding (use 'content' field with SAC for Layer 3)
- Normalization for cosine similarity (FAISS IndexFlatIP)
"""

import logging
import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "bge-m3"  # Default: BGE-M3-v2 (multilingual, local, M1 optimized)
    batch_size: int = 32  # Optimized for local inference
    normalize: bool = True  # For cosine similarity (FAISS IndexFlatIP)
    dimensions: Optional[int] = None  # Auto-detected from model


class EmbeddingGenerator:
    """
    Generate embeddings for multi-layer chunks.

    Supports:
    - Voyage AI: kanon-2 (default, #1 MLEB), voyage-3-large, voyage-law-2
    - OpenAI: text-embedding-3-large (3072 dims)
    - HuggingFace: BAAI/bge-m3 (1024 dims, multilingual)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - MLEB 2025 benchmark
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator.

        Args:
            config: EmbeddingConfig instance (defaults to kanon-2)
        """
        self.config = config or EmbeddingConfig()
        self.model_name = self.config.model
        self.batch_size = self.config.batch_size

        logger.info(f"Initializing EmbeddingGenerator with model: {self.model_name}")

        # Initialize model based on type
        if "voyage" in self.model_name.lower() or "kanon" in self.model_name.lower():
            self._init_voyage_model()
        elif self.model_name.startswith("text-embedding"):
            self._init_openai_model()
        elif "bge" in self.model_name.lower():
            self._init_bge_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        logger.info(f"EmbeddingGenerator initialized: {self.dimensions} dimensions")

    def _init_voyage_model(self):
        """Initialize Voyage AI embedding model (Kanon 2, Voyage 3, etc.)."""
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage models. "
                "Install with: uv pip install voyageai"
            )

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable required for Voyage models. "
                "Get your key at: https://www.voyageai.com/"
            )

        self.client = voyageai.Client(api_key=api_key)
        self.model_type = "voyage"

        # All Voyage models use 1024 dimensions
        self.dimensions = 1024

        logger.info(f"Voyage AI model initialized: {self.model_name} ({self.dimensions}D)")

    def _init_openai_model(self):
        """Initialize OpenAI embedding model."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for text-embedding models. "
                "Install with: uv pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = OpenAI(api_key=api_key)
        self.model_type = "openai"

        # Dimensions for OpenAI models
        if self.model_name == "text-embedding-3-large":
            self.dimensions = 3072
        elif self.model_name == "text-embedding-3-small":
            self.dimensions = 1536
        else:
            self.dimensions = 1536  # Default

        logger.info(f"OpenAI model initialized: {self.model_name} ({self.dimensions}D)")

    def _init_bge_model(self):
        """Initialize BGE embedding model (HuggingFace)."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers required for BGE models. "
                "Install with: uv pip install sentence-transformers torch"
            )

        logger.info(f"Loading BGE model: {self.model_name}")

        # Map model names
        model_map = {
            "bge-m3": "BAAI/bge-m3",
            "bge-m3": "BAAI/bge-m3",
            "bge-large": "BAAI/bge-large-en-v1.5"
        }

        hf_model_name = model_map.get(self.model_name, self.model_name)

        # Detect device (MPS for M1/M2 Mac, CUDA for GPU, CPU fallback)
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA GPU acceleration")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU acceleration)")

        self.model = SentenceTransformer(hf_model_name, device=device)
        self.model_type = "sentence_transformer"
        self.dimensions = self.model.get_sentence_embedding_dimension()

        logger.info(f"BGE model loaded: {hf_model_name} ({self.dimensions}D) on {device}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings, shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts...")

        if self.model_type == "voyage":
            embeddings = self._embed_voyage(texts)
        elif self.model_type == "openai":
            embeddings = self._embed_openai(texts)
        elif self.model_type == "sentence_transformer":
            embeddings = self._embed_sentence_transformer(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Normalize for cosine similarity if configured
        if self.config.normalize:
            embeddings = self._normalize_embeddings(embeddings)

        logger.info(f"Embeddings generated: shape {embeddings.shape}")
        return embeddings

    def _embed_voyage(self, texts: List[str]) -> np.ndarray:
        """Embed texts using Voyage AI API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")

            result = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type="document"  # For indexing/storage
            )

            batch_embeddings = result.embeddings
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float"
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We handle normalization separately
        )

        return embeddings.astype(np.float32)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def embed_chunks(self, chunks: List, layer: int) -> np.ndarray:
        """
        Embed chunks from a specific layer.

        CRITICAL: For Layer 3, uses 'content' field (with SAC).
                  For Layers 1-2, also uses 'content' field.

        Args:
            chunks: List of Chunk objects
            layer: Layer number (1, 2, or 3)

        Returns:
            numpy array of embeddings
        """
        if not chunks:
            return np.array([])

        logger.info(f"Embedding Layer {layer}: {len(chunks)} chunks")

        # Extract texts - always use 'content' field
        # For Layer 3, this includes SAC summary (58% DRM reduction!)
        texts = [chunk.content for chunk in chunks]

        # Filter empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            logger.warning(f"Layer {layer}: No valid texts to embed")
            return np.array([])

        # Generate embeddings
        embeddings = self.embed_texts(valid_texts)

        logger.info(
            f"Layer {layer} embeddings generated: "
            f"{len(embeddings)} vectors, {self.dimensions}D"
        )

        return embeddings

    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Get statistics about embeddings."""
        if embeddings.size == 0:
            return {}

        return {
            "count": len(embeddings),
            "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else 0,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "memory_mb": float(embeddings.nbytes / (1024 * 1024))
        }


# Example usage
if __name__ == "__main__":
    from config import ExtractionConfig
    from multi_layer_chunker import MultiLayerChunker
    from docling_extractor_v2 import DoclingExtractorV2

    # Extract and chunk document
    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=True
    )

    extractor = DoclingExtractorV2(config)
    result = extractor.extract("document.pdf")

    chunker = MultiLayerChunker(chunk_size=500, enable_sac=True)
    chunks = chunker.chunk_document(result)

    # Generate embeddings
    embedding_config = EmbeddingConfig(
        model="text-embedding-3-large",
        batch_size=100,
        normalize=True
    )

    embedder = EmbeddingGenerator(embedding_config)

    # Embed all layers
    layer1_embeddings = embedder.embed_chunks(chunks["layer1"], layer=1)
    layer2_embeddings = embedder.embed_chunks(chunks["layer2"], layer=2)
    layer3_embeddings = embedder.embed_chunks(chunks["layer3"], layer=3)

    print(f"Layer 1: {layer1_embeddings.shape}")
    print(f"Layer 2: {layer2_embeddings.shape}")
    print(f"Layer 3: {layer3_embeddings.shape}")
