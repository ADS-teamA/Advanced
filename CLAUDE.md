# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MY_SUJBOT is a research-based RAG (Retrieval-Augmented Generation) pipeline optimized for legal and technical documents. The system implements state-of-the-art techniques from multiple research papers to achieve superior retrieval quality through hierarchical structure extraction, contextual chunking, multi-layer embeddings, knowledge graph construction, and context assembly.

**Current Status:** PHASE 1-6 Complete (Full SOTA 2025 Retrieval + Context Assembly)
**Next Steps:** PHASE 7 (Answer Generation with LLM)

## Core Architecture

The pipeline follows a multi-phase architecture where each phase builds on the previous:

### Phase Flow
1. **PHASE 1:** Font-size based hierarchical structure extraction (Docling)
2. **PHASE 2:** Generic summary generation (gpt-4o-mini, 150 chars)
3. **PHASE 3:** Multi-layer chunking with Summary-Augmented Chunks (RCTS 500 chars)
4. **PHASE 4:** Embedding generation and FAISS indexing (text-embedding-3-large, 3072D)
5. **PHASE 5A:** Knowledge Graph construction (entities and relationships)
6. **PHASE 5B:** Hybrid Search (BM25 + Dense + RRF fusion)
7. **PHASE 5C:** Cross-Encoder Reranking (two-stage retrieval, +25% accuracy)
8. **PHASE 5D:** Graph-Vector Integration (triple-modal fusion, +60% multi-hop)
9. **PHASE 6:** Context Assembly (SAC stripping, citations, provenance tracking)

### Key Design Principles
- **Contextual Retrieval:** Chunks are augmented with LLM-generated context before embedding (-49% retrieval errors)
- **RCTS over Fixed Chunking:** Recursive Character Text Splitting at 500 chars (+167% Precision@1)
- **Generic Summaries:** Counter-intuitively better than expert summaries for semantic alignment
- **Multi-Layer Indexing:** Three separate FAISS indexes (document, section, chunk) for granular retrieval
- **No Cohere Reranking:** Research shows it hurts performance on legal documents (use cross-encoder instead)

## Project Structure

```
src/
├── config.py                   # Central configuration classes
├── docling_extractor_v2.py     # PHASE 1+2: Extraction with Docling
├── summary_generator.py        # PHASE 2: Generic summary generation
├── multi_layer_chunker.py      # PHASE 3: Multi-layer chunking + SAC
├── contextual_retrieval.py     # PHASE 3: Context generation for chunks
├── embedding_generator.py      # PHASE 4: Embedding with OpenAI
├── faiss_vector_store.py       # PHASE 4: FAISS vector storage
├── hybrid_search.py            # PHASE 5B: BM25 + RRF fusion
├── reranker.py                 # PHASE 5C: Cross-encoder reranking
├── graph_retrieval.py          # PHASE 5D: Graph-vector integration
├── context_assembly.py         # PHASE 6: Context assembly for LLM
├── indexing_pipeline.py        # Main orchestrator for PHASE 1-6
└── graph/                      # PHASE 5A: Knowledge Graph
    ├── models.py               # Entity, Relationship, KnowledgeGraph
    ├── config.py               # KG configuration
    ├── entity_extractor.py     # LLM-based entity extraction
    ├── relationship_extractor.py # LLM-based relationship extraction
    ├── graph_builder.py        # Graph storage backends
    └── kg_pipeline.py          # KG orchestrator

tests/
├── test_pipeline.py              # Integration tests
├── test_complete_pipeline.py     # PHASE 1-3 tests
├── test_phase4_indexing.py       # PHASE 4 tests
├── test_phase5c_reranking.py     # PHASE 5C tests
├── test_phase5d_graph_retrieval.py # PHASE 5D tests
├── test_phase6_context_assembly.py # PHASE 6 tests
└── graph/                        # PHASE 5A tests
```

## Environment Setup

### Prerequisites
```bash
# Python 3.10+ required
python --version

# Install uv package manager (recommended)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Platform-Specific Installation

**CRITICAL:** PyTorch installation differs by platform. Windows users must follow specific steps to avoid DLL errors.

**See `INSTALL.md` for detailed platform-specific instructions:**
- Windows: Requires PyTorch pre-installation before other dependencies
- macOS: Works with standard `uv sync`
- Linux: Choose CPU or CUDA version based on hardware

### Quick Installation

**Windows:**
```bash
# 1. Install PyTorch FIRST (avoids DLL errors)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install application
uv sync
```

**macOS/Linux:**
```bash
uv sync
```

### API Keys
API keys are **required** for the pipeline to function:

1. Copy `.env.example` to `.env`
2. Add your API keys:
   - `ANTHROPIC_API_KEY`: For summary generation (PHASE 2) and optional KG extraction
   - `OPENAI_API_KEY`: For OpenAI embeddings (PHASE 4) or KG extraction (PHASE 5A)
   - `VOYAGE_API_KEY`: For Voyage AI embeddings (optional, SOTA quality)

3. Choose embedding model:
   - **Windows:** Use `text-embedding-3-large` (cloud, avoids PyTorch issues)
   - **macOS (M1/M2/M3):** Use `bge-m3` (local, FREE, GPU-accelerated)
   - **Linux with GPU:** Use `bge-m3` (local, FREE, GPU-accelerated)
   - **Any platform:** Use `voyage-3-large` (cloud, best quality)

**IMPORTANT:** Without API keys, the pipeline will fail at PHASE 2 (summaries). Embedding models can be either cloud-based (API key required) or local (no API key).

## Common Commands

### Running the Pipeline

```bash
# Single document (all formats supported: PDF, DOCX, PPTX, XLSX, HTML)
python run_pipeline.py data/document.pdf

# Batch processing (entire directory)
python run_pipeline.py data/regulace/GRI

# Direct Python usage
python -c "
from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from pathlib import Path

config = IndexingConfig(
    enable_knowledge_graph=True,  # Optional: enable KG
    kg_llm_model='gpt-4o-mini',
    kg_backend='simple'
)
pipeline = IndexingPipeline(config)
result = pipeline.index_document(Path('data/doc.pdf'))
result['vector_store'].save('output/vector_store')
"
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_complete_pipeline.py -v      # PHASE 1-3
pytest tests/test_phase4_indexing.py -v        # PHASE 4
pytest tests/graph/ -v                         # PHASE 5A (KG)
pytest tests/test_phase5c_reranking.py -v      # PHASE 5C
pytest tests/test_phase5d_graph_retrieval.py -v # PHASE 5D
pytest tests/test_phase6_context_assembly.py -v # PHASE 6

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Development Tools

```bash
# Format code (Black)
black src/ tests/ --line-length 100

# Sort imports (isort)
isort src/ tests/ --profile black

# Type checking (mypy)
mypy src/ --config-file pyproject.toml
```

## Configuration System

The pipeline uses a centralized configuration system with sensible defaults based on research:

### IndexingConfig (Main Pipeline)
Located in `src/indexing_pipeline.py`, controls all phases:
- `enable_smart_hierarchy=True`: Font-size based hierarchy (PHASE 1)
- `generate_summaries=True`: Generic summaries (PHASE 2)
- `chunk_size=500`: RCTS optimal chunk size (PHASE 3)
- `enable_sac=True`: Summary-Augmented Chunking (PHASE 3)
- `embedding_model="text-embedding-3-large"`: 3072D embeddings (PHASE 4)
- `enable_knowledge_graph=False`: Knowledge graph construction (PHASE 5A)

### Knowledge Graph Configuration
Enable KG by setting `enable_knowledge_graph=True` in IndexingConfig:
- `kg_llm_provider`: "openai" or "anthropic"
- `kg_llm_model`: Model for extraction (default: "gpt-4o-mini")
- `kg_backend`: "simple", "neo4j", or "networkx"
- `kg_min_entity_confidence=0.6`: Minimum confidence for entities
- `kg_min_relationship_confidence=0.5`: Minimum confidence for relationships

## Knowledge Graph (PHASE 5A)

The Knowledge Graph module extracts structured information from documents:

### Entity Types (9)
STANDARD, ORGANIZATION, DATE, CLAUSE, TOPIC, REGULATION, CONTRACT, PERSON, LOCATION

### Relationship Types (18)
- Document: SUPERSEDED_BY, SUPERSEDES, REFERENCES
- Organizational: ISSUED_BY, DEVELOPED_BY, PUBLISHED_BY
- Temporal: EFFECTIVE_DATE, EXPIRY_DATE
- Content: COVERS_TOPIC, CONTAINS_CLAUSE
- Structural: PART_OF, CONTAINS

### Usage Example
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig

config = IndexingConfig(enable_knowledge_graph=True)
pipeline = IndexingPipeline(config)
result = pipeline.index_document("doc.pdf")

kg = result["knowledge_graph"]
print(f"Entities: {len(kg.entities)}")
print(f"Relationships: {len(kg.relationships)}")

# Query entities
standards = [e for e in kg.entities if e.type == "STANDARD"]
for std in standards:
    rels = kg.get_outgoing_relationships(std.id)
```

## Hybrid Search (PHASE 5B)

PHASE 5B implements hybrid retrieval combining dense (FAISS) and sparse (BM25) search with Reciprocal Rank Fusion (RRF).

### Key Features
- **BM25 Sparse Retrieval**: Keyword/exact match via term frequency
- **Dense Retrieval**: Semantic similarity via embeddings (existing FAISS)
- **RRF Fusion**: Combines both rankings using formula: `score = 1/(k + rank)`, k=60
- **Multi-Layer Support**: All 3 layers (document, section, chunk)
- **Contextual Indexing**: BM25 indexes same text as FAISS (context + raw_content)

### Expected Impact
Based on research: **+23% precision improvement** over dense-only retrieval for legal documents.

### Usage Example
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig

# Enable hybrid search
config = IndexingConfig(
    enable_hybrid_search=True,  # Enable PHASE 5B
    hybrid_fusion_k=60,  # RRF parameter (research-optimal)
    enable_knowledge_graph=False
)

pipeline = IndexingPipeline(config)
result = pipeline.index_document("doc.pdf")

# result["vector_store"] is now HybridVectorStore
hybrid_store = result["vector_store"]

# Search requires both text and embedding
from src.embedding_generator import EmbeddingGenerator

embedder = EmbeddingGenerator()
query_text = "waste disposal requirements"
query_embedding = embedder.embed_texts([query_text])

# Hybrid search with RRF fusion
results = hybrid_store.hierarchical_search(
    query_text=query_text,  # For BM25
    query_embedding=query_embedding,  # For FAISS
    k_layer3=6
)

# Results contain RRF-fused chunks
for chunk in results["layer3"]:
    print(f"RRF Score: {chunk['rrf_score']:.4f}")
    print(f"Content: {chunk['content'][:100]}...")
```

### Architecture
```
src/hybrid_search.py:
├── BM25Index: Single-layer BM25 index
├── BM25Store: Multi-layer wrapper (3 BM25 indexes)
└── HybridVectorStore: FAISS + BM25 + RRF fusion
```

### Save/Load
```python
# Save hybrid store (saves both FAISS and BM25)
hybrid_store.save(Path("output/hybrid_store"))

# Load
from src.hybrid_search import HybridVectorStore
loaded = HybridVectorStore.load(Path("output/hybrid_store"))
```

### Backward Compatibility
Hybrid search is **optional** via config flag. When disabled, pipeline behaves exactly as before (dense-only FAISS retrieval).

## Indexing Pipeline Integration

The `IndexingPipeline` class in `src/indexing_pipeline.py` is the **main entry point** for the entire system. It orchestrates all phases automatically:

```python
# Knowledge Graph is automatically integrated
result = pipeline.index_document("doc.pdf")

# Returns dict with:
# - vector_store: FAISSVectorStore (PHASE 4)
# - knowledge_graph: KnowledgeGraph (PHASE 5A, if enabled)
# - stats: Pipeline statistics
```

### Batch Processing
```python
result = pipeline.index_batch(
    document_paths=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    output_dir=Path("output/batch"),
    save_per_document=True
)
# Creates combined vector store and merged knowledge graph
```

## Research Foundation

The implementation is based on four key research papers:

1. **LegalBench-RAG** (Pipitone & Alami, 2024): RCTS chunking, text-embedding-3-large
2. **Summary-Augmented Chunking** (Reuter et al., 2024): SAC reduces DRM by 58%
3. **Multi-Layer Embeddings** (Lima, 2024): 3 separate indexes, 2.3x essential chunks
4. **Contextual Retrieval** (Anthropic, 2024): Context augmentation reduces errors by 49%

### Key Research Findings
- RCTS (500 chars) > Fixed chunking: +167% Precision@1
- Generic summaries > Expert summaries (counterintuitive!)
- Dense retrieval > Hybrid (for legal docs, may differ for general docs)
- NO Cohere reranking (worse than baseline on legal docs)
- Multi-layer indexing: 2.3x more essential chunks retrieved

## Context Assembly (PHASE 6)

PHASE 6 prepares retrieved chunks for LLM consumption by:
1. **Stripping SAC summaries** - Removes LLM-generated contexts used during embedding
2. **Formatting with citations** - Adds provenance tracking and source attribution
3. **Managing context length** - Respects token limits for LLM context windows

### Usage Example

```python
from src.context_assembly import ContextAssembler, CitationFormat

# Initialize assembler with desired citation format
assembler = ContextAssembler(
    citation_format=CitationFormat.INLINE,  # or SIMPLE, DETAILED, FOOTNOTE
    include_metadata=True,
    max_chunk_length=1000  # Optional chunk truncation
)

# Assemble retrieved chunks
result = assembler.assemble(
    chunks=retrieved_chunks,  # From reranker or graph retrieval
    max_chunks=6,
    max_tokens=4000  # ~16K characters
)

# Use assembled context for LLM prompt
prompt = f"""Context:
{result.context}

Question: {user_question}

Answer (with citations):"""
```

### Citation Formats

- **INLINE**: `[Chunk 1]` - Simple inline citations
- **SIMPLE**: `[1]` - Numbered references
- **DETAILED**: `[Doc: GRI 306, Section: 3.2, Page: 15]` - Full metadata
- **FOOTNOTE**: Numbered with sources section at end

### Key Features

- **SAC Stripping**: During embedding, chunks use `context + raw_content`. During assembly, only `raw_content` is used
- **Provenance Tracking**: Each chunk maintains document, section, page metadata
- **Token Management**: Respects max_tokens limit (~4 chars = 1 token)
- **Flexible Formatting**: Customizable separators, headers, citation styles

See `src/context_assembly.py` for full implementation details.

---

## Supported Document Formats

The pipeline supports multiple document formats through Docling:
- **PDF:** Primary format, full hierarchy extraction
- **DOCX:** Word documents, structure preservation
- **PPTX:** PowerPoint presentations
- **XLSX:** Excel spreadsheets
- **HTML/HTM:** Web pages

All formats are processed through the same pipeline with consistent structure extraction.

## Output Structure

For each document processed, the pipeline saves:
```
output/<document_name>/
├── phase1_extraction.json    # Document structure & hierarchy
├── phase2_summaries.json      # Generic summaries
├── phase3_chunks.json         # Multi-layer chunks with SAC
├── phase4_vector_store/       # FAISS indexes (3 files)
│   ├── index_layer1.faiss
│   ├── index_layer2.faiss
│   └── index_layer3.faiss
└── <document_id>_kg.json      # Knowledge graph (if enabled)
```

## Development Workflow

### Adding New Features

1. **Identify the phase:** Determine which phase (1-5A or future 5B-7) the feature belongs to
2. **Check research:** Ensure the feature aligns with research findings in `PIPELINE.md`
3. **Update config:** Add configuration options to appropriate config class
4. **Implement:** Add code to relevant module
5. **Test:** Write tests in `tests/` following existing patterns
6. **Document:** Update `PIPELINE.md` with implementation details

### Testing Strategy

- **Unit tests:** Test individual components in isolation
- **Integration tests:** Test phase interactions (e.g., `test_complete_pipeline.py`)
- **Research validation:** Compare results against research benchmarks

### Code Style

- **Line length:** 100 characters (Black/isort configured)
- **Type hints:** Use where beneficial, not required everywhere (mypy config: `disallow_untyped_defs=false`)
- **Docstrings:** Required for public classes and functions
- **Logging:** Use `logging` module, not print statements

## Future Development (PHASE 7)

The roadmap for SOTA 2025 upgrade is documented in `PIPELINE.md`:

### ✅ PHASE 5B: Hybrid Search - COMPLETE
- BM25 sparse retrieval with contextual indexing
- Reciprocal Rank Fusion (RRF) for combining dense + sparse
- Achieved: +23% precision improvement

### ✅ PHASE 5C: Reranking - COMPLETE
- Cross-encoder reranking (ms-marco-MiniLM)
- Two-stage retrieval (fast → precise)
- Achieved: +25% accuracy improvement

### ✅ PHASE 5D: Graph-Vector Integration - COMPLETE
- Triple-modal fusion: Dense + Sparse + Graph
- Entity-aware search and graph boosting
- Achieved: +60% improvement on multi-hop queries

### ✅ PHASE 6: Context Assembly - COMPLETE
- Strip SAC summaries from retrieved chunks
- Concatenate chunks with proper citations
- Add provenance tracking with multiple citation formats
- Module: `src/context_assembly.py`

### ⏳ PHASE 7: Answer Generation - PENDING
- GPT-4 or Mixtral 8x7B for answer generation
- Mandatory citations from retrieved chunks
- Temperature: 0.1-0.3 for consistency
- System prompt engineering for legal domain
- Citation validation and enforcement

## Important Notes for Claude Code

### Cross-Platform Compatibility

**CRITICAL:** This codebase must work on Windows, macOS, and Linux.

**Platform-Specific Issues to Avoid:**
- **PyTorch Installation:** Windows requires specific pre-installation steps (see `INSTALL.md`)
- **macOS-Specific Code:** No `ocrmac` or other macOS-only dependencies
- **GPU Detection:** Code must gracefully handle CPU-only, CUDA, and MPS (Apple Silicon)
- **Path Separators:** Use `pathlib.Path` instead of string concatenation
- **Line Endings:** Git handles this, but be aware

**When Adding Dependencies:**
1. Check if it works on all platforms
2. If platform-specific, make it optional
3. Document platform requirements in `INSTALL.md`
4. Update `.env.example` with platform recommendations

### API Key Management
- Always check for API keys before running pipeline components
- PHASE 2 (summaries) requires ANTHROPIC_API_KEY or OPENAI_API_KEY
- PHASE 4 (embeddings):
  - Cloud models (text-embedding-3-large, voyage-*): Require API key
  - Local models (bge-m3): No API key needed
- PHASE 5A (KG) requires either key depending on `kg_llm_provider`

### Embedding Model Selection

**Windows Users:**
- Recommend `text-embedding-3-large` (cloud) to avoid PyTorch DLL issues
- If BGE-M3 needed, ensure PyTorch is installed first (see INSTALL.md)

**Apple Silicon Users:**
- Recommend `bge-m3` (local) for FREE GPU-accelerated embeddings
- Gracefully detect MPS availability: `torch.backends.mps.is_available()`

**Linux Users:**
- With NVIDIA GPU: Recommend `bge-m3` (local)
- CPU only: Recommend `text-embedding-3-large` (cloud)

### Research-Backed Decisions
Do NOT change these without strong justification:
- Chunk size: 500 chars (RCTS optimal)
- Summary length: 150 chars (research-validated)
- Summary style: "generic" not "expert" (counterintuitive but proven)
- Embedding models: text-embedding-3-large, voyage-3-large, or bge-m3 (research-validated)
- No Cohere reranking (hurts performance on legal docs)

### Performance Considerations
- Docling extraction is CPU-intensive (use for structure, not speed)
- Docling requires PyTorch for layout detection (unavoidable but CPU-only is fine)
- FAISS indexes are in-memory (large datasets may need disk-based indexes)
- Knowledge Graph extraction is parallelized (5 workers default)
- Batch processing merges indexes (careful with large batches)
- BGE-M3 local inference:
  - Fast on Apple Silicon (MPS) or NVIDIA GPU
  - Slow on CPU (recommend cloud embeddings instead)

### Common Pitfalls
- **Windows DLL errors:** PyTorch not installed correctly → See INSTALL.md for fix
- Missing API keys causes silent failures → Always validate env vars first
- Knowledge Graph requires graph module imports → Check KG_AVAILABLE flag
- FAISS dimensions must match embeddings → Use `embedder.dimensions`
- SAC context is prepended during embedding, stripped during retrieval
- **Platform assumptions:** Always test cross-platform or document platform-specific code

### Troubleshooting Windows Issues

**Error:** `OSError: [WinError 1114] DLL load failed`

**Root Cause:** PyTorch DLL dependencies not correctly installed

**Solutions (in order of preference):**
1. **Use cloud embeddings:** Set `EMBEDDING_MODEL=text-embedding-3-large` in `.env`
2. **Install Visual C++ Redistributables:** Download from Microsoft
3. **Reinstall PyTorch:** Use platform-specific wheel from pytorch.org
4. See `INSTALL.md` for complete troubleshooting steps

**Error:** `ImportError: No module named 'sentence_transformers'`

**Solution:**
- For BGE-M3: `uv pip install sentence-transformers`
- Or use cloud embeddings (no installation needed)
