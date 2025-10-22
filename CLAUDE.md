# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MY_SUJBOT is a research-based RAG (Retrieval-Augmented Generation) pipeline optimized for legal and technical documents. The system implements state-of-the-art techniques from multiple research papers to achieve superior retrieval quality through hierarchical structure extraction, contextual chunking, multi-layer embeddings, and knowledge graph construction.

**Current Status:** PHASE 1-5A Complete (Extraction → Knowledge Graph)
**Next Steps:** PHASE 5B-7 (Hybrid Search, Reranking, Answer Generation)

## Core Architecture

The pipeline follows a multi-phase architecture where each phase builds on the previous:

### Phase Flow
1. **PHASE 1:** Font-size based hierarchical structure extraction (Docling)
2. **PHASE 2:** Generic summary generation (gpt-4o-mini, 150 chars)
3. **PHASE 3:** Multi-layer chunking with Summary-Augmented Chunks (RCTS 500 chars)
4. **PHASE 4:** Embedding generation and FAISS indexing (text-embedding-3-large, 3072D)
5. **PHASE 5A:** Knowledge Graph construction (entities and relationships)

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
├── indexing_pipeline.py        # Main orchestrator for PHASE 1-5A
└── graph/                      # PHASE 5A: Knowledge Graph
    ├── models.py               # Entity, Relationship, KnowledgeGraph
    ├── config.py               # KG configuration
    ├── entity_extractor.py     # LLM-based entity extraction
    ├── relationship_extractor.py # LLM-based relationship extraction
    ├── graph_builder.py        # Graph storage backends
    └── kg_pipeline.py          # KG orchestrator

tests/
├── test_pipeline.py            # Integration tests
├── test_complete_pipeline.py   # PHASE 1-3 tests
├── test_phase4_indexing.py     # PHASE 4 tests
└── graph/                      # PHASE 5A tests
```

## Environment Setup

### Prerequisites
```bash
# Python 3.10+ required
python --version

# Install dependencies
uv sync
# OR
pip install -e .
```

### API Keys
API keys are **required** for the pipeline to function:

1. Copy `.env.example` to `.env`
2. Add your API keys:
   - `ANTHROPIC_API_KEY`: For summary generation (PHASE 2) and optional KG extraction
   - `OPENAI_API_KEY`: For embeddings (PHASE 4) and KG extraction (PHASE 5A)

**IMPORTANT:** Without API keys, the pipeline will fail at PHASE 2 (summaries) and PHASE 4 (embeddings).

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

## Future Development (PHASE 5B-7)

The roadmap for SOTA 2025 upgrade is documented in `PIPELINE.md`:

### PHASE 5B: Hybrid Search
- BM25 sparse retrieval with contextual indexing
- Reciprocal Rank Fusion (RRF) for combining dense + sparse
- Expected: +23% precision improvement

### PHASE 5C: Reranking
- Cross-encoder reranking (ms-marco-MiniLM)
- Two-stage retrieval (fast → precise)
- Expected: +25% accuracy improvement
- **Critical:** Test on legal documents first (Cohere failed in research)

### PHASE 6: Context Assembly
- Strip SAC summaries from retrieved chunks
- Concatenate chunks with proper citations
- Add provenance tracking

### PHASE 7: Answer Generation
- GPT-4 or Mixtral 8x7B
- Mandatory citations from retrieved chunks
- Temperature: 0.1-0.3 for consistency

## Important Notes for Claude Code

### API Key Management
- Always check for API keys before running pipeline components
- PHASE 2 (summaries) requires ANTHROPIC_API_KEY or OPENAI_API_KEY
- PHASE 4 (embeddings) requires OPENAI_API_KEY
- PHASE 5A (KG) requires either key depending on `kg_llm_provider`

### Research-Backed Decisions
Do NOT change these without strong justification:
- Chunk size: 500 chars (RCTS optimal)
- Summary length: 150 chars (research-validated)
- Summary style: "generic" not "expert" (counterintuitive but proven)
- Embedding model: text-embedding-3-large (best for legal)
- No Cohere reranking (hurts performance on legal docs)

### Performance Considerations
- Docling extraction is CPU-intensive (use for structure, not speed)
- FAISS indexes are in-memory (large datasets may need disk-based indexes)
- Knowledge Graph extraction is parallelized (5 workers default)
- Batch processing merges indexes (careful with large batches)

### Common Pitfalls
- Missing API keys causes silent failures → Always validate env vars first
- Knowledge Graph requires graph module imports → Check KG_AVAILABLE flag
- FAISS dimensions must match embeddings → Use `embedder.dimensions`
- SAC context is prepended during embedding, stripped during retrieval
