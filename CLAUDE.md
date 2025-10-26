# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**MY_SUJBOT** is a production-ready RAG (Retrieval-Augmented Generation) system optimized for legal and technical documents. It implements state-of-the-art techniques from 4 research papers (2024-2025) and features a 7-phase pipeline with an interactive AI agent.

**Status:** PHASE 1-7 COMPLETE ✅ (Full SOTA 2025 RAG System + 27-Tool Agent)

**Core Technologies:**
- Document processing: IBM Docling (hierarchical structure extraction)
- Embeddings: OpenAI text-embedding-3-large or local BGE-M3
- Vector store: FAISS (3-layer indexing)
- Retrieval: Hybrid (BM25 + Dense + RRF fusion) with cross-encoder reranking
- Knowledge Graph: Entity/relationship extraction with NetworkX/Neo4j
- Agent: Claude SDK with 27 specialized tools (Anthropic Sonnet/Haiku)

---

## Development Setup

### Installation & Environment

**Package Manager:** Uses `uv` (required)
```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY

# Platform-specific notes:
# Windows: Install PyTorch FIRST before uv sync
# macOS M1/M2/M3: Use EMBEDDING_MODEL=bge-m3 (local, free, GPU-accelerated)
# Windows: Use EMBEDDING_MODEL=text-embedding-3-large (cloud)
```

**Required API Keys:**
- `ANTHROPIC_API_KEY` - Required (for agent and optional summaries)
- `OPENAI_API_KEY` - Optional (for embeddings, summaries, knowledge graph)

### Common Commands

**Run Pipeline (Index Documents):**
```bash
# Single document
uv run python run_pipeline.py data/document.pdf

# Batch processing
uv run python run_pipeline.py data/regulace/

# Fast mode (default): 2-3 min, full price
uv run python run_pipeline.py data/document.pdf

# Eco mode: 15-30 min, 50% cheaper (set SPEED_MODE=eco in .env)
```

**Run Agent (Interactive CLI):**
```bash
# Launch agent
uv run python -m src.agent.cli

# With specific vector store
uv run python -m src.agent.cli --vector-store output/my_doc/phase4_vector_store

# Debug mode
uv run python -m src.agent.cli --debug
```

**Run Tests:**
```bash
# All tests
uv run pytest tests/ -v

# Specific phase/component
uv run pytest tests/test_phase4_indexing.py -v
uv run pytest tests/agent/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Single test
uv run pytest tests/agent/test_validation.py::test_api_key_validation -v
```

**Code Quality:**
```bash
# Format code
uv run black src/ tests/ --line-length 100
uv run isort src/ tests/ --profile black

# Type checking
uv run mypy src/
```

---

## Architecture Overview

### 7-Phase Pipeline

The system processes documents through 7 distinct phases:

**PHASE 1: Hierarchy Extraction**
- Tool: IBM Docling
- Purpose: Extract document structure using font-size classification
- Output: Hierarchical sections (depth=4), metadata
- File: `src/docling_extractor_v2.py`

**PHASE 2: Summary Generation**
- Model: gpt-4o-mini (or gpt-5-nano)
- Purpose: Generate generic summaries (150 chars) for documents and sections
- Critical: Use GENERIC summaries (NOT expert) - counterintuitive but proven better
- File: `src/summary_generator.py`

**PHASE 3: Multi-Layer Chunking + SAC**
- Method: RCTS (500 chars, no overlap)
- Layers: Document (L1), Section (L2), Chunk (L3 - PRIMARY)
- SAC: Summary-Augmented Chunking - prepends context to each chunk (-58% DRM)
- File: `src/multi_layer_chunker.py`

**PHASE 4: Embedding + FAISS Indexing**
- Embeddings: text-embedding-3-large (3072D) or bge-m3 (1024D)
- Storage: 3 separate FAISS indexes (IndexFlatIP, cosine similarity)
- Context: Embeds `context + raw_content`, stores only `raw_content`
- File: `src/embedding_generator.py`, `src/faiss_vector_store.py`

**PHASE 5: Advanced Retrieval** (3 sub-phases)
- **5A: Knowledge Graph** - Entity/relationship extraction (18 types)
  - Files: `src/graph/`, `src/graph_retrieval.py`
- **5B: Hybrid Search** - BM25 + Dense + RRF fusion (+23% precision)
  - File: `src/hybrid_search.py`
- **5C: Reranking** - Cross-encoder 2-stage retrieval (+25% accuracy)
  - File: `src/reranker.py`

**PHASE 6: Context Assembly**
- Purpose: Prepare chunks for LLM
- Tasks: Strip SAC summaries, add citations, manage token limits
- File: `src/context_assembly.py`

**PHASE 7: RAG Agent**
- Framework: Claude SDK (official Anthropic SDK)
- Tools: 27 specialized tools (3 tiers: basic, advanced, analysis)
- Features: Streaming, prompt caching (90% savings), cost tracking, **query expansion**
- Query Expansion: Multi-query generation (research suggests +15-25% recall) with `num_expands` parameter
- Files: `src/agent/`, `src/agent/query_expander.py`

### Configuration Architecture

**Centralized Config System:**
- Main: `src/config.py` - Shared configs (Extraction, Summarization, Chunking, Embedding)
- Pipeline: `src/indexing_pipeline.py` - `IndexingConfig` (orchestrates all phases)
- Agent: `src/agent/config.py` - `AgentConfig` (agent-specific settings)
- Knowledge Graph: `src/graph/config.py` - `KnowledgeGraphConfig`

**Loading Pattern:**
```python
# Load from environment (.env)
from src.indexing_pipeline import IndexingConfig
config = IndexingConfig.from_env()

# Override specific settings
config = IndexingConfig.from_env(
    enable_knowledge_graph=True,
    speed_mode="eco"
)
```

### Cost Tracking System

**Global Tracker Pattern:**
```python
from src.cost_tracker import get_global_tracker, reset_global_tracker

# Reset at start of operation
reset_global_tracker()

# Track usage (automatic in pipeline components)
tracker = get_global_tracker()
tracker.track_llm("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=500)

# Get summary
print(tracker.get_summary())
```

**Supported Providers:**
- Anthropic: Claude Haiku, Sonnet, Opus
- OpenAI: GPT-4o, GPT-5, o-series, embeddings
- Voyage AI: Embeddings
- Local models: Free (bge-m3, etc.)

---

## Critical Implementation Rules

### Research-Based Constraints (DO NOT CHANGE)

These decisions are backed by research papers and extensive testing:

1. **RCTS > Fixed-size chunking** (LegalBench-RAG)
   - Chunk size: **500 chars** (optimal for legal/technical docs)
   - Overlap: 0 (RCTS handles naturally via hierarchy)

2. **Generic > Expert summaries** (Reuter et al., counterintuitive!)
   - Summary length: **150 chars**
   - Style: **Generic** (NOT expert terminology)

3. **SAC reduces DRM by 58%** (Reuter et al.)
   - Prepend document summary to each chunk during embedding
   - Strip summaries during retrieval (context assembly)

4. **Multi-layer embeddings** (Lima 2024)
   - 3 separate FAISS indexes (not merged)
   - 2.3x essential chunks compared to single-layer

5. **No Cohere reranking** (LegalBench-RAG)
   - Cohere reranker performs WORSE than baseline on legal documents
   - Use cross-encoder models instead (ms-marco, bge-reranker)

6. **Hybrid > Pure Dense** (Industry best practice 2025)
   - BM25 + Dense + RRF fusion outperforms dense-only by +23% precision
   - RRF k=60 is optimal parameter

### Agent Tool Guidelines

**Tool Tiers (Speed/Quality Tradeoff):**
- **TIER 1** (11 tools): Fast (<100ms), basic retrieval - Use first
  - Key tool: **`search`** (unified hybrid search with optional query expansion) ✅ **NEW: Query Expansion**
    - `num_expands=0`: Fast mode (~200ms) - original query only, 1 query total (default)
    - `num_expands=1`: Balanced mode (~500ms) - original + 1 paraphrase, 2 queries total (+15-25% recall est.)
    - `num_expands=2`: Better recall (~800ms) - original + 2 paraphrases, 3 queries total (+20-30% recall est.)
    - `num_expands=3-5`: Best recall (~1.2-2s) - original + 3-5 paraphrases, 4-6 queries total (max quality)
    - Uses LLM-based paraphrasing (GPT-5 nano or Claude Haiku) to find docs with different terminology
    - Implementation: `src/agent/query_expander.py` + `src/agent/tools/tier1_basic.py`
- **TIER 2** (9 tools): Quality (500-1000ms), advanced retrieval - Use for complex queries
- **TIER 3** (6 tools): Deep (1-3s), analysis and insights - Use sparingly

**Tool Design Principles:**
- All tools inherit from `BaseTool` in `src/agent/tools/base.py`
- Register with `@register_tool` decorator
- Validation via Pydantic `ToolInput` schemas
- Return `ToolResult` with success/data/citations/metadata
- Tools must handle graceful degradation (e.g., reranker unavailable)

**Adding New Tools:**
1. Create tool class in appropriate tier file (`tier1_basic.py`, `tier2_advanced.py`, `tier3_analysis.py`)
2. Define input schema with Pydantic
3. Implement `execute_impl()` method
4. Register with `@register_tool`
5. Add tests in `tests/agent/tools/`

### Prompt Caching Strategy (Anthropic Only)

**Automatic Caching Points:**
- System prompt (agent instructions)
- Tool definitions (27 tools)
- Initial messages (document list)
- Long tool results (>1024 tokens)

**Cache Management:**
- Enabled via `ENABLE_PROMPT_CACHING=true` in `.env`
- Saves 90% on cached tokens
- Context pruning at 50K tokens to prevent quadratic growth

**Implementation:**
See `src/agent/agent_core.py:_create_messages()` for cache control block formatting.

---

## Key File Locations

### Pipeline Core
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/config.py` - Shared configuration classes
- `src/cost_tracker.py` - API cost tracking
- `run_pipeline.py` - CLI entry point

### Phase Implementations
- `src/docling_extractor_v2.py` - PHASE 1 (hierarchy)
- `src/summary_generator.py` - PHASE 2 (summaries)
- `src/multi_layer_chunker.py` - PHASE 3 (chunking + SAC)
- `src/embedding_generator.py` - PHASE 4 (embeddings)
- `src/faiss_vector_store.py` - PHASE 4 (FAISS)
- `src/hybrid_search.py` - PHASE 5B (BM25 + RRF)
- `src/graph/` - PHASE 5A (knowledge graph)
- `src/reranker.py` - PHASE 5C (cross-encoder)
- `src/graph_retrieval.py` - PHASE 5D (graph-vector fusion)
- `src/context_assembly.py` - PHASE 6 (context prep)

### Agent (PHASE 7)
- `src/agent/cli.py` - Interactive CLI and REPL
- `src/agent/agent_core.py` - Core agent with streaming
- `src/agent/config.py` - Agent configuration
- `src/agent/validation.py` - Comprehensive validation
- `src/agent/tools/` - 27 specialized tools
  - `base.py` - Base classes
  - `registry.py` - Tool registry
  - `tier1_basic.py` - 11 fast tools
  - `tier2_advanced.py` - 9 quality tools
  - `tier3_analysis.py` - 6 analysis tools
  - `token_manager.py` - Token estimation
  - `utils.py` - Shared utilities

### Tests
- `tests/agent/` - Agent tests (49 tests)
- `tests/graph/` - Knowledge graph tests
- `tests/test_phase*.py` - Pipeline phase tests
- `tests/test_complete_pipeline.py` - End-to-end integration

---

## Common Development Tasks

### Adding a New Document Format

1. Update `DoclingExtractorV2` in `src/docling_extractor_v2.py`
2. Add format to `supported_formats` list in `IndexingPipeline.index_document()`
3. Add integration test in `tests/test_complete_pipeline.py`

### Modifying Chunk Size

**WARNING:** Changing chunk size invalidates all existing vector stores.

```python
# In .env or IndexingConfig
CHUNK_SIZE=500  # Research-optimal, don't change without testing

# If you must change:
# 1. Re-index ALL documents
# 2. Update tests to expect new chunk counts
# 3. Benchmark retrieval quality (Precision@k, Recall@k)
```

### Adding a New Embedding Model

1. Add pricing to `src/cost_tracker.py:PRICING`
2. Update `EmbeddingGenerator._get_provider()` in `src/embedding_generator.py`
3. Add platform-specific guidance to `.env.example`
4. Test on your platform (especially Windows DLL issues)

### Extending the Agent with New Tools

See "Agent Tool Guidelines" above. Example:

```python
# In src/agent/tools/tier1_basic.py

class MyToolInput(ToolInput):
    """Input schema with validation."""
    query: str = Field(..., description="User query")
    limit: int = Field(10, ge=1, le=100)

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    tier = 1
    input_schema = MyToolInput

    def execute_impl(self, query: str, limit: int = 10) -> ToolResult:
        # Implementation
        results = self.vector_store.search(query, k=limit)

        return ToolResult(
            success=True,
            data=results,
            citations=[...],
            metadata={"query": query, "count": len(results)}
        )
```

### Debugging Retrieval Issues

1. **Enable debug mode:**
   ```bash
   uv run python -m src.agent.cli --debug
   ```

2. **Check vector store stats:**
   ```python
   from src.hybrid_search import HybridVectorStore
   store = HybridVectorStore.load("vector_db")
   print(store.get_stats())
   ```

3. **Inspect search results:**
   ```python
   # In agent, use explain_search_results tool
   # Shows BM25, Dense, RRF score breakdown
   ```

4. **Verify embeddings:**
   ```python
   from src.embedding_generator import EmbeddingGenerator
   embedder = EmbeddingGenerator()

   # Check embedding cache stats
   stats = embedder.get_cache_stats()
   print(f"Hit rate: {stats['hit_rate']:.1%}")
   ```

### Testing Strategy

**Unit Tests:**
- Test individual components (extractors, chunkers, embedders)
- Mock external API calls
- Focus on edge cases and error handling

**Integration Tests:**
- Test phase interactions (extraction → chunking → embedding)
- Use small test documents (<10 pages)
- Verify intermediate outputs

**Agent Tests:**
- Test tool execution and validation
- Test streaming and error recovery
- Test prompt caching behavior

**Run tests before PR:**
```bash
# Full test suite
uv run pytest tests/ -v

# Fast feedback loop (unit tests only)
uv run pytest tests/agent/ tests/graph/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Debug & Optimization System (/debug-optimize)

MY_SUJBOT includes a sophisticated multi-agent debugging system accessible via the `/debug-optimize` slash command. This system uses 5 specialized agents running in parallel to find bugs, optimize costs, and automatically apply fixes.

**When to Use:**
- Agent errors or tool failures
- High API costs or cache misses
- Pipeline failures or phase transition errors
- Validation errors or configuration issues
- Performance problems or slow retrieval

**Usage:**
```bash
/debug-optimize
[Paste your error description and conversation context here]

Example:
Error: Cache not working - showing 0 cached tokens
Conversation:
> What is RAG?
💰 This message: $0.0025
  Input (cached): 0 tokens
...
```

**System Architecture:**

The command orchestrates 5 specialized agents:

1. **cost-optimizer** (Haiku) - Cost optimization expert
   - Finds redundant context copying
   - Identifies cache misses and inefficient API usage
   - Optimizes token consumption
   - Tools: Read, Grep, Glob, Bash

2. **rag-debugger** (Haiku + WebSearch) - RAG pipeline expert
   - Debugs retrieval, chunking, embedding issues
   - Fixes FAISS index problems
   - Validates research constraints
   - Tools: Read, Grep, Glob, Bash, WebSearch

3. **validation-expert** (Haiku) - Configuration & validation expert
   - Fixes API key errors
   - Resolves schema mismatches
   - Debugs type checking issues
   - Tools: Read, Grep, Glob, Bash

4. **pipeline-expert** (Haiku) - Pipeline orchestration expert
   - Fixes phase transition errors
   - Debugs speed mode issues
   - Validates data flow between phases
   - Tools: Read, Grep, Glob, Bash

5. **agent-expert** (Haiku + WebSearch) - Agent framework expert
   - Debugs tool execution failures
   - Fixes streaming errors
   - Resolves Claude SDK integration bugs
   - Tools: Read, Grep, Glob, Bash, WebSearch

**Agent Routing:**
- Cost issues → cost-optimizer
- Retrieval/indexing bugs → rag-debugger
- Config/validation → validation-expert
- Pipeline failures → pipeline-expert
- Agent/tool errors → agent-expert

**Auto-Fix Process:**
1. Parses error and conversation context
2. Routes to relevant agents (parallel execution)
3. Agents analyze and generate fix plans
4. Detects and resolves conflicts
5. Auto-applies fixes (up to 20 per run)
6. Runs validation tests
7. Git commits if all tests pass
8. Generates comprehensive report

**Safety Features:**
- Respects research constraints (won't auto-change chunk size, summary style, etc.)
- Validates fixes before applying
- Automatic rollback on test failures
- Saves fix plans to `output/debug-fixes/` if validation fails
- Max 20 fixes auto-applied (prevents runaway changes)

**Templates & Documentation:**
- `.claude/templates/debug-context.md` - Debug context extraction format
- `.claude/templates/fix-plan.md` - Fix plan structure
- `.claude/agents/*.md` - 5 specialized agent definitions
- `.claude/commands/debug-optimize.md` - Main orchestration logic

**Example Workflow:**
```
User: /debug-optimize
Error: Tool validation failing - k > 100

System:
1. Parses error → type: agent, component: tools
2. Routes to agent-expert + validation-expert (parallel)
3. Agents analyze:
   - agent-expert: "Schema too strict for use case"
   - validation-expert: "Same issue - k max should be 200"
4. Merges identical fixes (high confidence)
5. Applies: k: int = Field(10, ge=1, le=200)
6. Runs: pytest tests/agent/tools/test_tier1_basic.py
7. Commits: "fix: Increase k parameter max to 200"
8. Reports: ✅ 1 fix applied, tests passed
```

**Cost Optimization Features:**
- Identifies duplicate embeddings
- Finds missing prompt cache control
- Detects redundant tool results
- Optimizes context assembly patterns
- Estimates savings per fix

---

## Troubleshooting

### Common Issues

**1. Vector store not found**
```bash
# Run indexing first
uv run python run_pipeline.py data/your_documents/
```

**2. PyTorch DLL errors (Windows)**
```bash
# Install PyTorch FIRST before uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. API key errors**
```bash
# Check .env file
cat .env | grep API_KEY

# Set for current session
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

**4. Out of memory (embedding generation)**
```python
# Reduce batch size in .env or config
EMBEDDING_BATCH_SIZE=32  # Default is 100
```

**5. Knowledge graph construction fails**
```bash
# Check API key for KG LLM provider
echo $OPENAI_API_KEY  # or $ANTHROPIC_API_KEY

# Disable KG if not needed
ENABLE_KNOWLEDGE_GRAPH=false
```

**6. Agent tools failing**
- Check validation output on startup (shows which tools are available)
- Missing reranker → Tier 2 tools degraded
- Missing knowledge graph → Tier 3 graph tools unavailable

**7. GPT-5 API Compatibility Issues**

GPT-5 models (gpt-5-nano, gpt-5-*, o1-*, o3-*) have **breaking changes** compared to GPT-4:

**Error: "Unsupported parameter: 'max_tokens'"**
```python
# WRONG (GPT-4 style):
response = client.chat.completions.create(
    model="gpt-5-nano",
    max_tokens=300  # ❌ Not supported
)

# CORRECT (GPT-5 style):
response = client.chat.completions.create(
    model="gpt-5-nano",
    max_completion_tokens=300  # ✅ Use this instead
)
```

**Error: "Unsupported value: 'temperature' does not support 0.7"**
```python
# WRONG:
response = client.chat.completions.create(
    model="gpt-5-nano",
    temperature=0.7  # ❌ Not supported
)

# CORRECT:
response = client.chat.completions.create(
    model="gpt-5-nano"
    # ✅ Use default temperature (1.0) - don't set parameter
)
```

**Implementation Pattern:**
```python
# Auto-detect model and use correct parameters
params = {"model": model, "messages": messages}

if model.startswith(("gpt-5", "o1-", "o3-")):
    # GPT-5/o-series models
    params["max_completion_tokens"] = 300
    # Don't set temperature (only default 1.0 supported)
else:
    # GPT-4 and earlier
    params["max_tokens"] = 300
    params["temperature"] = 0.7

response = client.chat.completions.create(**params)
```

**Where to check:**
- Query expansion: `src/agent/query_expander.py` ✅ Already fixed
- Summary generation: `src/summary_generator.py` (if using GPT-5)
- Knowledge graph: `src/graph/` (if using GPT-5)
- Any custom LLM calls using OpenAI API

### Performance Optimization

**Indexing Speed:**
- Fast mode (default): ThreadPoolExecutor, 2-3 min per doc
- Eco mode: OpenAI Batch API, 15-30 min per doc, 50% cheaper
- Use eco mode for overnight bulk indexing

**Agent Response Speed:**
- Enable prompt caching: 90% cost reduction on repeated queries
- Use TIER 1 tools first (fast, <100ms)
- Lazy load reranker to speed up startup

**Memory Usage:**
- Local embeddings (bge-m3): ~2GB RAM
- Cloud embeddings: Minimal memory (<500MB)
- FAISS indexes: ~10MB per 1000 documents

---

## Resources

### Documentation
- **README.md** - User guide and quick start
- **PIPELINE.md** - Complete pipeline specification with research
- **INSTALL.md** - Platform-specific installation
- **docs/agent/README.md** - Agent CLI documentation
- **docs/cost-tracking.md** - Cost optimization guide

### Research Papers
1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS chunking, reranking
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending

### API Documentation
- Anthropic Claude: https://docs.anthropic.com/
- OpenAI: https://platform.openai.com/docs
- IBM Docling: https://ds4sd.github.io/docling/

---

## Code Style & Conventions

**Formatting:**
- Black (line length: 100)
- isort (profile: black)
- Type hints encouraged (mypy compatible)

**Naming:**
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private: _leading_underscore

**Documentation:**
- Docstrings: Google style
- Type hints: Required for public APIs
- Comments: Explain WHY, not WHAT

**Logging:**
```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed diagnostic info")
logger.info("High-level progress")
logger.warning("Degraded mode or non-critical issues")
logger.error("Errors that don't crash the program")
```

---

## Version & Status

**Last Updated:** 2025-10-26
**Status:** PHASE 1-7 COMPLETE ✅ + Query Expansion ✅
**Agent Tools:** 27 (11 basic + 9 advanced + 6 analysis)
**Pipeline:** Full SOTA 2025 RAG (Hybrid + Reranking + Graph + Query Expansion + Context Assembly)
**Recent Updates:**
- Query Expansion (2025-10-26): Multi-query generation with `num_expands` parameter (research-based +15-25% recall improvement)
