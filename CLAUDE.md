# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MY_SUJBOT is a research-based RAG system optimized for legal/technical documents. All 7 phases complete (structure extraction → hybrid search → KG → agent with 27 tools).

**Key Principles:**
- **RCTS chunking (500 chars)** - DO NOT change without research justification
- **Generic summaries (150 chars)** - NOT expert summaries (counterintuitive but proven)
- **Multi-layer indexing** - 3 separate FAISS indexes (document/section/chunk)
- **No Cohere reranking** - Hurts performance on legal docs (use cross-encoder)

## Architecture

**Pipeline Flow (PHASE 1-7):**
```
Document → [1] Hierarchy → [2] Summaries → [3] Chunking+SAC →
[4] Embedding+FAISS → [5] Hybrid/KG/Rerank → [6] Assembly → [7] Agent
```

**Key Modules:**
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/agent/` - RAG Agent CLI (27 tools, Claude SDK)
- `src/graph/` - Knowledge graph (entities, relationships)
- `src/hybrid_search.py` - BM25 + Dense + RRF fusion
- `src/config.py` - Central config (load from .env via `from_env()`)

## Installation & Setup

**Prerequisites:** Python 3.10+, `uv` package manager

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Install dependencies
uv sync  # macOS/Linux
# Windows: See INSTALL.md (PyTorch must be installed first)

# Configure
cp .env.example .env
# Add API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY (see .env.example)
```

**CRITICAL - Platform-Specific:**
- **Windows:** Install PyTorch FIRST before `uv sync` (see INSTALL.md)
- **Embedding models:** Windows→cloud (text-embedding-3-large), macOS→local (bge-m3)

## OCR Configuration

**Tesseract OCR** (Czech support, 90%+ accuracy) - Automatic via Docling
- Default: `ocr_language=["ces", "eng"]`
- Auto-fixes Czech diacritics in malformed PDFs (see `src/docling_extractor_v2.py:normalize_unicode()`)
- Common codes: `ces` (Czech), `eng`, `deu`, `slk`, `pol`, `auto`

## Common Commands

### Run Indexing Pipeline

```bash
# Single document
uv run python run_pipeline.py data/document.pdf

# Batch processing
uv run python run_pipeline.py data/regulace/GRI

# Background processing (long-running tasks)
uv run python run_pipeline.py data/ &  # Run in background
```

**Speed Modes:**
- `speed_mode="fast"` (default): 2-3 min, full price (ThreadPoolExecutor)
- `speed_mode="eco"`: 15-30 min, 50% cheaper (OpenAI Batch API)
  - Use for overnight bulk indexing: `IndexingConfig(speed_mode="eco")`

### Central Vector Database (Recommended)

```bash
# Add document to central DB (creates if not exists)
uv run python manage_vector_db.py add data/document.pdf

# Migrate existing vector store
uv run python manage_vector_db.py migrate output/old_store/phase4_vector_store

# Show stats
uv run python manage_vector_db.py stats
```

**Benefits:** All docs in `vector_db/`, incremental indexing, agent has full access

### Run RAG Agent CLI

```bash
# Launch agent (27 tools, Claude SDK)
uv run python -m src.agent.cli

# With central database (recommended)
uv run python -m src.agent.cli --vector-store vector_db

# Debug mode
uv run python -m src.agent.cli --debug

# Run in background (for long sessions)
nohup uv run python -m src.agent.cli > agent.log 2>&1 &
```

**Tools:** 12 basic (fast) + 9 advanced (quality) + 6 analysis (deep)

### Testing

```bash
# All tests
uv run pytest tests/ -v

# Specific phase
uv run pytest tests/test_phase4_indexing.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Single test
uv run pytest tests/agent/test_validation.py::test_api_key_validation -v
```

### Development

```bash
# Format (Black + isort)
uv run black src/ tests/ --line-length 100
uv run isort src/ tests/ --profile black

# Type check
uv run mypy src/
```

## Configuration

**Load from .env (recommended):**
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig

# Load all settings from .env
config = IndexingConfig.from_env()
pipeline = IndexingPipeline(config)

# Override specific settings
config = IndexingConfig.from_env(
    enable_knowledge_graph=True,
    enable_hybrid_search=True,
    speed_mode="eco"  # 50% cheaper
)
```

**Key .env variables:**
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (required for summaries)
- `LLM_MODEL=gpt-5-nano` (summaries & agent)
- `EMBEDDING_MODEL=text-embedding-3-large` (Windows) or `bge-m3` (macOS)
- `SPEED_MODE=fast` (fast or eco)
- `ENABLE_HYBRID_SEARCH=true`, `ENABLE_KNOWLEDGE_GRAPH=true`

**Config hierarchy:** `IndexingConfig` → `ExtractionConfig`, `SummarizationConfig`, `ChunkingConfig`, `EmbeddingConfig`, `KnowledgeGraphConfig`

## Best Practices

**Background Processing:**
```bash
# Run pipeline in background (long-running)
nohup uv run python run_pipeline.py data/ > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log

# Agent in background
nohup uv run python -m src.agent.cli > agent.log 2>&1 &
```

**Configuration:**
- Always use `from_env()` to load config from `.env`
- Override only what you need: `IndexingConfig.from_env(speed_mode="eco")`
- Check `.env.example` for all available options

**Performance:**
- Use `speed_mode="eco"` for bulk indexing (50% savings)
- Enable embedding cache: `EMBEDDING_CACHE_ENABLED=true`
- Use central vector DB for incremental indexing

## Critical Implementation Rules

**DO NOT CHANGE (research-backed):**
- Chunk size: 500 chars (RCTS optimal, +167% Precision@1)
- Summary length: 150 chars, generic style (NOT expert)
- No Cohere reranking (hurts legal doc performance)

**Cross-Platform Requirements:**
- Use `pathlib.Path` for all paths (not string concatenation)
- Windows: PyTorch must be installed BEFORE `uv sync` (see INSTALL.md)
- Test on multiple platforms or document platform-specific code
- Gracefully handle GPU detection (CPU/CUDA/MPS)

**API Keys:**
- PHASE 2 summaries: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` required
- Embeddings: Cloud models need API key, local (bge-m3) doesn't
- Always validate API keys before pipeline execution

**Embedding Model Selection:**
- **Windows:** `text-embedding-3-large` (cloud) - avoids PyTorch DLL issues
- **macOS M1/M2/M3:** `bge-m3` (local, FREE, GPU-accelerated)
- **Linux GPU:** `bge-m3` (local), **Linux CPU:** `text-embedding-3-large` (cloud)

**Common Pitfalls:**
- Windows DLL errors → See INSTALL.md for PyTorch pre-installation
- Missing API keys → Validate at startup, fail early with clear message
- FAISS dimensions must match embeddings → Use `embedder.dimensions`
- SAC context: Prepended during embedding, stripped during retrieval

**Performance:**
- Docling extraction is CPU-intensive
- FAISS indexes are in-memory (large datasets need disk-based)
- KG extraction parallelized (5 workers)
- BGE-M3: Fast on GPU, slow on CPU (use cloud embeddings for CPU)
