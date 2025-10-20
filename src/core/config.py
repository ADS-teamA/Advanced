"""
Configuration management for the RAG pipeline.

All configuration values are evidence-based from PIPELINE.md and
the 4 research papers (LegalBench-RAG, SAC, Multi-Layer Embeddings, NLI).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PreprocessingConfig:
    """
    Phase 1: Document preprocessing configuration.
    """
    pdf_processor_type: str = "pypdf2"  # 'pypdf2' or 'ocr'
    enable_ocr: bool = False
    normalize_whitespace: bool = True
    extract_metadata: bool = True


@dataclass
class SummarizationConfig:
    """
    Phase 2: Summarization configuration.

    Evidence-based settings from Reuter et al., 2024:
    - Generic summaries > expert-guided summaries (counterintuitive!)
    - 150 chars optimal summary length
    - ±20 tolerance acceptable
    """
    provider: str = "claude"  # 'claude' or 'openai'
    model: str = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku by default
    max_chars: int = 150  # Optimal from research (Reuter 2024, Table 1)
    tolerance: int = 20   # ±20 chars acceptable
    style: str = "generic"  # NOT expert-guided (generic performs better!)
    temperature: float = 0.3
    max_tokens: int = 50
    retry_on_exceed: bool = True
    max_retries: int = 3


@dataclass
class ChunkingConfig:
    """
    Phase 3: Chunking configuration.

    Evidence-based settings from research papers:
    - RCTS > fixed-size chunking (LegalBench-RAG: 6.41% vs 2.40% Prec@1)
    - 500 chars optimal chunk size (Reuter 2024)
    - 0 overlap (RCTS handles boundaries naturally)
    - SAC reduces DRM by 58% (Reuter 2024)
    - Multi-layer improves essential chunks by 2.3x (Lima 2024)
    """
    method: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 500  # Characters (optimal from Reuter 2024)
    chunk_overlap: int = 0  # RCTS handles boundaries naturally
    enable_sac: bool = True  # Critical: 58% DRM reduction
    enable_multi_layer: bool = True  # 2.3x essential chunks improvement

    # RCTS separators optimized for legal text
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence ends
        "; ",    # Clause separators
        ", ",    # Sub-clause separators
        " ",     # Word boundaries
        ""       # Character fallback
    ])


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Phase configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # API Keys - Both can be loaded simultaneously
    # Claude is used for LLM operations (Phase 2: Summarization)
    # OpenAI will be used for embeddings (Phase 4: Embedding, future)
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "logs/pipeline.log"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create nested config objects
        preprocessing = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        summarization = SummarizationConfig(**config_dict.get('summarization', {}))
        chunking = ChunkingConfig(**config_dict.get('chunking', {}))

        # Create main config
        main_config = config_dict.get('pipeline', {})
        return cls(
            preprocessing=preprocessing,
            summarization=summarization,
            chunking=chunking,
            **main_config
        )

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = {
            'preprocessing': {
                'pdf_processor_type': self.preprocessing.pdf_processor_type,
                'enable_ocr': self.preprocessing.enable_ocr,
                'normalize_whitespace': self.preprocessing.normalize_whitespace,
                'extract_metadata': self.preprocessing.extract_metadata
            },
            'summarization': {
                'provider': self.summarization.provider,
                'model': self.summarization.model,
                'max_chars': self.summarization.max_chars,
                'tolerance': self.summarization.tolerance,
                'style': self.summarization.style,
                'temperature': self.summarization.temperature,
                'max_tokens': self.summarization.max_tokens,
                'retry_on_exceed': self.summarization.retry_on_exceed,
                'max_retries': self.summarization.max_retries
            },
            'chunking': {
                'method': self.chunking.method,
                'chunk_size': self.chunking.chunk_size,
                'chunk_overlap': self.chunking.chunk_overlap,
                'enable_sac': self.chunking.enable_sac,
                'enable_multi_layer': self.chunking.enable_multi_layer,
                'separators': self.chunking.separators
            },
            'pipeline': {
                'log_level': self.log_level,
                'log_format': self.log_format,
                'log_file': self.log_file
            }
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """
        Validate configuration values.

        Note: Both ANTHROPIC_API_KEY and OPENAI_API_KEY can be loaded simultaneously.
        Only the API key for the configured provider (summarization.provider) is validated.
        This allows using Claude for LLM operations and OpenAI for embeddings (Phase 4).

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate provider type
        if self.summarization.provider not in ['claude', 'openai']:
            raise ValueError(
                f"Unknown LLM provider: {self.summarization.provider}. "
                "Supported providers: 'claude', 'openai'"
            )

        # Validate API key for the configured provider
        # Note: Both keys can coexist - only the active provider's key is validated
        if self.summarization.provider == 'claude':
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required for Claude models. "
                    "Please set it in your .env file or environment."
                )
        elif self.summarization.provider == 'openai':
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OpenAI models. "
                    "Please set it in your .env file or environment."
                )

        # Validate chunk size
        if self.chunking.chunk_size < 100:
            raise ValueError(f"Chunk size {self.chunking.chunk_size} is too small (min: 100)")

        if self.chunking.chunk_size > 2000:
            raise ValueError(f"Chunk size {self.chunking.chunk_size} is too large (max: 2000)")

        # Validate summary length
        if self.summarization.max_chars < 50:
            raise ValueError(f"Summary max_chars {self.summarization.max_chars} is too small (min: 50)")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory if logging to file
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
