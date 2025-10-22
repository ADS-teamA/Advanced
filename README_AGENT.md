# RAG Agent CLI - Document Assistant

Interactive CLI agent powered by Claude SDK for legal and technical document retrieval and analysis.

## ✨ Features

- **17 Specialized RAG Tools** organized in 3 tiers (basic, advanced, analysis)
- **Hybrid Search**: BM25 + Dense embeddings + RRF fusion + Cross-encoder reranking
- **Knowledge Graph Integration**: Entity-aware search and relationship queries
- **Query Optimization**: HyDE (Hypothetical Document Embeddings) and query decomposition
- **Autonomous Tool Orchestration**: Claude decides which tools to use automatically
- **Streaming Responses**: Real-time output as Claude generates answers
- **Config-Driven**: All settings configurable via CLI arguments or environment variables
- **Production-Ready**: Startup validation, error handling, execution statistics

## 🚀 Quick Start

### Prerequisites

1. **API Keys** (required):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."  # For embeddings
   ```

2. **Indexed Documents** (run indexing pipeline first):
   ```bash
   python run_pipeline.py data/your_documents/
   ```

   This creates `output/hybrid_store/` with vector embeddings.

### Basic Usage

```bash
# Start agent with hybrid vector store
python run_agent.py --store output/hybrid_store

# With knowledge graph
python run_agent.py --store output/hybrid_store --kg output/knowledge_graph.json

# Custom model
python run_agent.py --store output/hybrid_store --model claude-sonnet-4-5-20250929

# Enable HyDE for better retrieval
python run_agent.py --store output/hybrid_store --enable-hyde

# Enable query decomposition
python run_agent.py --store output/hybrid_store --enable-decomposition

# Full-featured mode
python run_agent.py \
  --store output/hybrid_store \
  --kg output/knowledge_graph.json \
  --enable-hyde \
  --enable-decomposition \
  --model claude-sonnet-4-5-20250929
```

### Interactive Session

```
╭─────────────────────────────────────────────────────────────╮
│              RAG Agent - Document Assistant                 │
│  Type your question or use /help for commands              │
╰─────────────────────────────────────────────────────────────╯

> What are the waste disposal requirements in GRI 306?

[Using simple_search...]
Based on GRI 306 waste disposal requirements, organizations must:
- Track waste generation by type and disposal method
- Report waste diverted from disposal and directed to disposal
- Document waste management practices and improvement initiatives

Sources: GRI 306 (Section 3.1), GRI 306 (Section 3.2)

> /stats

📊 Tool Execution Statistics:
Total tools: 17
Total calls: 1
Success rate: 100%
Average time per call: 234ms

🔝 Most Used Tools:
  simple_search          1 calls,  234ms avg

> /exit

👋 Goodbye!
```

## 📚 Tool Architecture

The agent has access to 17 specialized tools organized in 3 performance tiers:

### Tier 1: Basic Tools (6 tools, ~100-300ms)

Fast, frequently-used tools for common retrieval tasks:

- **simple_search** - Hybrid retrieval (BM25 + dense + reranking)
- **entity_search** - Find chunks mentioning specific entities
- **document_search** - Search within a specific document
- **section_search** - Search within document sections
- **keyword_search** - Pure BM25 keyword/phrase search
- **get_document_list** - List all indexed documents

**Use when:** Standard retrieval, keyword matching, document browsing

### Tier 2: Advanced Tools (6 tools, ~500-1000ms)

Quality tools for complex retrieval scenarios:

- **multi_hop_search** - Multi-hop graph traversal (requires KG)
- **compare_documents** - Compare two documents for similarities/differences
- **find_related_chunks** - Find semantically related chunks
- **temporal_search** - Search with date/time filters
- **hybrid_search_with_filters** - Hybrid search + metadata filtering
- **cross_reference_search** - Find cross-references to clauses/articles

**Use when:** Complex queries, document comparison, temporal analysis, cross-referencing

### Tier 3: Analysis Tools (5 tools, ~1-3s)

Deep analysis tools for specialized insights:

- **explain_entity** - Comprehensive entity information + relationships (requires KG)
- **get_entity_relationships** - Filtered relationship queries (requires KG)
- **timeline_view** - Extract and organize temporal information
- **summarize_section** - Detailed section summarization
- **get_statistics** - Corpus statistics and analytics

**Use when:** Entity analysis, timeline construction, summarization, corpus analytics

## 🎮 REPL Commands

Interactive commands available in the agent CLI:

```bash
/help, /h        # Show help message
/stats, /s       # Show tool execution statistics
/config, /c      # Show current configuration
/clear, /reset   # Clear conversation history
/exit, /quit, /q # Exit the agent
```

## ⚙️ Configuration

### CLI Arguments

```bash
python run_agent.py [OPTIONS]

Required:
  --store PATH              Path to hybrid vector store directory

Optional:
  --kg, --knowledge-graph PATH   Path to knowledge graph JSON
  --model TEXT              Claude model (default: claude-sonnet-4-5-20250929)
  --max-tokens INT          Max response tokens (default: 4096)
  --temperature FLOAT       Model temperature (default: 0.3)

Features:
  --enable-hyde             Enable HyDE (Hypothetical Document Embeddings)
  --enable-decomposition    Enable query decomposition
  --no-reranking            Disable cross-encoder reranking
  --no-stream               Disable streaming responses

Display:
  --citation-format [inline|detailed|footnote]
                            Citation format (default: inline)
  --hide-tool-calls         Don't show tool execution messages
  -v, --verbose             Enable verbose logging
```

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."  # For embeddings

# Optional defaults
export AGENT_MODEL="claude-sonnet-4-5-20250929"
export VECTOR_STORE_PATH="output/hybrid_store"
```

### Configuration File

Edit `src/agent/config.py` to customize defaults:

```python
@dataclass
class AgentConfig:
    # Core settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.3
    
    # Tool settings
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    
    # Query optimization
    enable_hyde: bool = False
    enable_query_decomposition: bool = False
```

## 🔬 Query Optimization

### HyDE (Hypothetical Document Embeddings)

**What it does:** Generates hypothetical answers to your query, then searches using those instead of the original query.

**When to enable:** Improves retrieval quality when:
- Questions are abstract or conceptual
- Queries don't use document-specific terminology
- You want to find semantically similar content

**Example:**
```bash
python run_agent.py --store output/hybrid_store --enable-hyde

> What are best practices for waste management?

# Agent generates hypothetical document like:
# "Organizations should implement waste segregation systems,
#  track waste generation metrics, and document disposal methods..."
# Then searches using this hypothetical document
```

**Research:** Based on Gao et al. (2022) "Precise Zero-Shot Dense Retrieval"

### Query Decomposition

**What it does:** Breaks complex multi-part queries into simpler sub-queries that are answered independently.

**When to enable:** Useful for:
- Complex queries with multiple parts ("Find X and check if Y")
- Comparison tasks ("Compare A and B")
- Multi-step reasoning ("Find X, then analyze Y")

**Example:**
```bash
python run_agent.py --store output/hybrid_store --enable-decomposition

> Find waste requirements in GRI 306 and check if our contract complies

# Agent decomposes into:
# 1. What are the waste disposal requirements in GRI 306?
# 2. What waste disposal provisions are in our contract?
# 3. Do the contract provisions comply with GRI 306 requirements?
```

**Research:** Based on "Least-to-Most Prompting" (Zhou et al., 2022)

### Combining Both

```bash
python run_agent.py --store output/hybrid_store --enable-hyde --enable-decomposition

# Agent will:
# 1. Decompose complex queries into sub-queries
# 2. Apply HyDE to each sub-query
# 3. Search with optimized queries
# 4. Combine results into comprehensive answer
```

## 💡 Example Queries

### Basic Retrieval

```
> What is GRI 306?
> Find information about waste disposal
> Search for "hazardous waste" in GRI 306
> List all indexed documents
```

### Entity-Focused

```
> Find all mentions of GRI 306
> What documents reference ISO 14001?
> Show me everything about GDPR compliance
```

### Document Comparison

```
> Compare GRI 305 and GRI 306
> What are the differences between our 2023 and 2024 contracts?
> Find conflicts between contract X and regulation Y
```

### Cross-Referencing

```
> Find all references to Article 5.2
> What clauses reference Section 3?
> Show cross-references to GDPR Article 6
```

### Temporal Analysis

```
> Find regulations from 2023
> Show documents between January and March 2024
> What changed after 2022?
```

### Advanced Analysis

```
> Explain entity GRI 306 (requires KG)
> Show timeline of environmental regulations
> Get statistics about the corpus
> Summarize section 3 of ISO 14001
```

### Complex Multi-Part Queries

```
> Find waste disposal requirements in GRI 306 and check if our contract complies
> Compare environmental reporting in GRI 305 and GRI 306, then summarize key differences
> Search for hazardous waste regulations, find related clauses, and explain compliance requirements
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Agent CLI (run_agent.py)             │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface (cli.py)                                     │
│  ├─ Startup validation                                      │
│  ├─ Component initialization                                │
│  └─ REPL loop                                               │
├─────────────────────────────────────────────────────────────┤
│  Agent Core (agent_core.py)                                 │
│  ├─ Claude SDK orchestration                                │
│  ├─ Tool execution loop                                     │
│  ├─ Streaming support                                       │
│  └─ Conversation management                                 │
├─────────────────────────────────────────────────────────────┤
│  Query Optimization (query/)                                │
│  ├─ HyDE generator                                          │
│  ├─ Query decomposer                                        │
│  └─ Query optimizer                                         │
├─────────────────────────────────────────────────────────────┤
│  Tool System (tools/)                                       │
│  ├─ Base tool abstraction                                   │
│  ├─ Tool registry (17 tools)                                │
│  │   ├─ Tier 1: Basic (6 tools)                            │
│  │   ├─ Tier 2: Advanced (6 tools)                         │
│  │   └─ Tier 3: Analysis (5 tools)                         │
│  └─ Utility functions                                       │
├─────────────────────────────────────────────────────────────┤
│  RAG Pipeline Components                                    │
│  ├─ HybridVectorStore (hybrid_search.py)                   │
│  ├─ EmbeddingGenerator (embedding_generator.py)            │
│  ├─ CrossEncoderReranker (reranker.py)                     │
│  ├─ GraphEnhancedRetriever (graph_retrieval.py)            │
│  ├─ KnowledgeGraph (graph/models.py)                       │
│  └─ ContextAssembler (context_assembly.py)                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Tool Registry Pattern**: Auto-discovery and registration of tools via decorator
2. **Dependency Injection**: Components injected into tools at initialization
3. **Pydantic Validation**: Type-safe input validation for all tools
4. **Config-Driven**: No hardcoded values, everything configurable
5. **Lazy Loading**: Expensive resources (reranker, KG) loaded on-demand
6. **Statistics Tracking**: All tool executions tracked for analytics

## 🔧 Troubleshooting

### "Vector store not found"

**Problem:** Agent can't find indexed documents

**Solution:**
```bash
# Run indexing pipeline first
python run_pipeline.py data/your_documents/

# Then specify correct path
python run_agent.py --store output/hybrid_store
```

### "Knowledge graph not available"

**Problem:** Tier 3 KG tools failing

**Solutions:**
1. **Run indexing with KG enabled:**
   ```bash
   python run_pipeline.py data/docs/ --enable-kg
   ```

2. **Specify KG path:**
   ```bash
   python run_agent.py --store output/hybrid_store --kg output/knowledge_graph.json
   ```

3. **Use non-KG alternatives:**
   - Instead of `explain_entity` → use `entity_search`
   - Instead of `multi_hop_search` → use `simple_search`

### "Reranker model download slow"

**Problem:** First run downloads cross-encoder model (~500MB)

**Solutions:**
1. **Disable reranking:**
   ```bash
   python run_agent.py --store output/hybrid_store --no-reranking
   ```

2. **Use lazy loading (default):**
   - Reranker only loads when first needed
   - Configure in `AgentConfig.tool_config.lazy_load_reranker = True`

### "API rate limits"

**Problem:** Too many API calls with HyDE/decomposition

**Solutions:**
1. **Use simpler queries** - Avoid complex multi-part questions
2. **Disable optimization temporarily:**
   ```bash
   python run_agent.py --store output/hybrid_store  # No --enable-hyde/decomposition
   ```
3. **Use faster model:**
   ```bash
   python run_agent.py --store output/hybrid_store --model claude-haiku-4-5
   ```

### "Streaming not working"

**Problem:** Responses appear all at once

**Check:**
1. Streaming enabled? (default: yes)
2. Try forcing streaming mode:
   ```bash
   python run_agent.py --store output/hybrid_store  # Don't use --no-stream
   ```

### "Tool execution errors"

**Problem:** Tools failing with errors

**Debug steps:**
1. **Enable verbose logging:**
   ```bash
   python run_agent.py --store output/hybrid_store -v
   ```

2. **Check logs:**
   ```bash
   tail -f agent.log
   ```

3. **Validate vector store:**
   ```python
   from src.hybrid_search import HybridVectorStore
   store = HybridVectorStore.load("output/hybrid_store")
   print(store.get_stats())
   ```

## 📊 Performance Tips

### Speed Optimization

1. **Disable reranking** for faster responses:
   ```bash
   python run_agent.py --store output/hybrid_store --no-reranking
   ```

2. **Use Haiku model** (5x faster):
   ```bash
   python run_agent.py --store output/hybrid_store --model claude-haiku-4-5
   ```

3. **Disable query optimization**:
   ```bash
   # Don't use --enable-hyde or --enable-decomposition
   python run_agent.py --store output/hybrid_store
   ```

### Quality Optimization

1. **Enable all features**:
   ```bash
   python run_agent.py --store output/hybrid_store \
     --kg output/knowledge_graph.json \
     --enable-hyde \
     --enable-decomposition
   ```

2. **Use Sonnet model** (better reasoning):
   ```bash
   python run_agent.py --store output/hybrid_store --model claude-sonnet-4-5-20250929
   ```

3. **Keep reranking enabled** (default)

### Balanced Configuration

```bash
# Good balance of speed and quality
python run_agent.py --store output/hybrid_store \
  --model claude-sonnet-4-5-20250929 \
  --enable-hyde
  # Reranking: enabled (default)
  # Decomposition: disabled (slower)
  # KG: optional
```

## 🧪 Development

### Adding New Tools

1. **Create tool class** in appropriate tier file:

```python
from pydantic import Field
from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool

class MyToolInput(ToolInput):
    query: str = Field(..., description="Search query")

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    tier = 1  # 1, 2, or 3
    input_schema = MyToolInput
    
    def execute_impl(self, query: str) -> ToolResult:
        # Implementation here
        return ToolResult(success=True, data=result)
```

2. **Tool auto-registers** via `@register_tool` decorator

3. **Test tool:**
```python
pytest tests/test_agent_tools.py::test_my_tool -v
```

### Testing

```bash
# Test with example indexed data
python run_agent.py --store tests/fixtures/test_store

# Unit tests
pytest tests/test_agent_tools.py -v

# Integration tests
pytest tests/test_agent_integration.py -v
```

## 📖 Further Reading

- **Claude SDK**: https://docs.anthropic.com/claude/docs/claude-sdk
- **HyDE Paper**: https://arxiv.org/abs/2212.10496
- **Query Decomposition**: "Least-to-Most Prompting" (Zhou et al., 2022)
- **Hybrid Search**: See `PIPELINE.md` PHASE 5B documentation
- **Knowledge Graphs**: See `CLAUDE.md` PHASE 5A documentation

## 🤝 Contributing

To extend the agent:

1. **Add tools** to `src/agent/tools/` (see Development section)
2. **Update config** in `src/agent/config.py`
3. **Add tests** in `tests/`
4. **Update README** with new features

## 📝 License

See main project LICENSE file.

---

**Questions or issues?** See the main project README and CLAUDE.md for detailed documentation.

## 🐛 Debug Mode

The agent includes comprehensive debugging capabilities to help diagnose issues.

### Enabling Debug Mode

```bash
# Enable debug mode with --debug flag
python run_agent.py --store output/hybrid_store --debug

# Debug mode features:
# - Detailed logging to agent.log
# - Console output of all operations
# - Component initialization tracking
# - Tool execution details
# - API call logging
```

### What Debug Mode Does

1. **Comprehensive Logging**: All operations logged with timestamps, module names, and function names
2. **Startup Validation**: Detailed validation of all components with diagnostics
3. **Component Tracking**: Logs initialization of every component
4. **Tool Execution**: Detailed logging of tool calls and results
5. **Error Tracing**: Full stack traces for all errors

### Log Format

Debug mode creates `agent.log` with detailed format:

```
2025-01-15 10:23:45 | src.agent.agent_core         | DEBUG    | __init__              | Initializing AgentCore...
2025-01-15 10:23:45 | src.agent.agent_core         | DEBUG    | __init__              | Model: claude-sonnet-4-5-20250929
2025-01-15 10:23:46 | src.agent.tools.registry     | DEBUG    | initialize_tools      | Initializing tools with dependencies
2025-01-15 10:23:46 | src.agent.tools.tier1_basic  | DEBUG    | execute               | Executing simple_search with query='test'
```

### Validation Checks

Debug mode runs comprehensive validation on startup:

```bash
python run_agent.py --store output/hybrid_store --debug

# Checks performed:
# ✅ Python version compatibility (3.10+)
# ✅ Required dependencies (anthropic, pydantic, faiss, numpy)
# ✅ Optional dependencies (sentence_transformers, torch)
# ✅ API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
# ✅ Vector store integrity (files, load test, vector count)
# ✅ Knowledge graph (if enabled)
# ✅ Embedding compatibility
# ✅ Claude model access
# ✅ Tool registry (all 17 tools registered)
```

### Integration Test

Run standalone integration test to verify setup:

```bash
python test_agent_integration.py

# Tests:
# - Python version
# - All dependencies
# - Agent module imports
# - Pipeline imports
# - Tool registry
# - Config creation
# - API keys
# - Vector store existence
```

### Debugging Common Issues

#### Issue: "Vector store not found"

**Debug:**
```bash
python run_agent.py --store output/hybrid_store --debug

# Check log for:
# - Path resolution
# - File existence checks
# - Load attempt details
```

**Solution:** See `agent.log` for exact path being checked and why it failed.

#### Issue: "Tool execution failing"

**Debug:**
```bash
python run_agent.py --store output/hybrid_store --debug

# Watch console for:
# - Tool call details
# - Input parameters
# - Execution results
# - Error stack traces
```

**Solution:** Log shows exact tool inputs and where failure occurred.

#### Issue: "API errors"

**Debug:**
```bash
python run_agent.py --store output/hybrid_store --debug

# Check for:
# - API key validation
# - Model access checks
# - API call parameters
# - Response errors
```

**Solution:** Detailed API interaction logging helps identify rate limits, invalid keys, or model issues.

### Log Levels

Different verbosity levels:

```bash
# Normal mode: No console logging, warnings to agent.log
python run_agent.py --store output/hybrid_store

# Verbose mode: INFO level to console and file
python run_agent.py --store output/hybrid_store -v

# Debug mode: DEBUG level to console and file
python run_agent.py --store output/hybrid_store --debug
```

### Performance Impact

Debug mode has minimal performance impact:
- **Logging overhead**: ~5-10ms per operation
- **Validation checks**: +2-3 seconds on startup
- **File I/O**: Asynchronous, non-blocking

**Recommendation:** Always enable debug mode when troubleshooting, disable for production use.

### Example Debug Session

```bash
$ python run_agent.py --store output/hybrid_store --debug

2025-01-15 10:23:45 | root                         | INFO     | main                  | ================================================================================
2025-01-15 10:23:45 | root                         | INFO     | main                  | RAG AGENT DEBUG MODE ENABLED
2025-01-15 10:23:45 | root                         | INFO     | main                  | ================================================================================
2025-01-15 10:23:45 | root                         | DEBUG    | main                  | Python version: 3.10.12
2025-01-15 10:23:45 | root                         | DEBUG    | main                  | Working directory: /Users/user/MY_SUJBOT
2025-01-15 10:23:45 | root                         | DEBUG    | main                  | Starting RAG Agent in DEBUG mode
2025-01-15 10:23:45 | root                         | DEBUG    | main                  | Command line arguments: {'store': 'output/hybrid_store', 'debug': True, ...}

🔍 Validating environment...

================================================================================
STARTING COMPREHENSIVE VALIDATION
================================================================================

✅ Python Version: Python 3.10.12 (compatible)
✅ Dependency: anthropic - Claude SDK - installed
✅ Dependency: pydantic - Input validation - installed
✅ Dependency: faiss - Vector search - installed
✅ API Key: ANTHROPIC - Anthropic API key present (format valid)
✅ Vector Store - Vector store loaded successfully (12,453 vectors)
✅ Tool Registry - All 17 tools registered

================================================================================
VALIDATION SUMMARY: 15/15 checks passed
================================================================================

✅ All validation checks passed - agent ready to start

🚀 Initializing agent components...
Loading vector store...
Initializing embedder...
Loading reranker...
✅ 17 tools initialized

✅ Agent ready!

╭─────────────────────────────────────────────────────────────╮
│              RAG Agent - Document Assistant                 │
│  Type your question or use /help for commands              │
╰─────────────────────────────────────────────────────────────╯

>
```

### Troubleshooting Tips

1. **Always check `agent.log` first** - Most errors have detailed traces there
2. **Run integration test** - `python test_agent_integration.py` to verify setup
3. **Enable debug mode** - Get full visibility into what's happening
4. **Check validation report** - Shows exactly what failed and why
5. **Compare with working config** - Use `--debug` to see differences

