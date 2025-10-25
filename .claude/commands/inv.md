---
description: Investigate pipeline phases for improvement opportunities using parallel research agents
argument-hint: "[phase-numbers or 'all']"
allowed-tools: Task, TodoWrite
model: sonnet
---

# Pipeline Investigation Command

Launch parallel research agents to investigate MY_SUJBOT pipeline phases and identify state-of-the-art improvements.

## Instructions

You will launch **pipeline-investigator** subagents in parallel to research improvements for the RAG pipeline. Each subagent uses the Haiku model to minimize costs while conducting thorough research.

### Available Phases

1. **PHASE 1** - Extraction (Docling, structure parsing)
2. **PHASE 2** - Summarization (generic summaries, optimal length)
3. **PHASE 3** - Chunking (RCTS, SAC, multi-layer)
4. **PHASE 4** - Embedding (vector generation, FAISS)
5. **PHASE 5A** - Knowledge Graph (entity/relationship extraction)
6. **PHASE 5B** - Hybrid Search (BM25 + Dense + RRF)
7. **PHASE 5C** - Reranking (cross-encoder, two-stage)
8. **PHASE 5D** - Graph-Vector Integration (triple-modal fusion)
9. **PHASE 6** - Context Assembly (SAC stripping, citations)
10. **PHASE 7** - Agent (27 tools, Claude SDK)

### Phase Selection

**Arguments**: `$ARGUMENTS`

- **No arguments or "all"**: Investigate all 10 phases
- **Specific phases**: E.g., "1,3,5" investigates phases 1, 3, and 5
- **Phase ranges**: E.g., "5-7" investigates phases 5, 6, and 7

### Your Task

1. **Parse Arguments**
   - Determine which phases to investigate
   - Default to "all" if no arguments provided

2. **Create Task List**
   - Use TodoWrite to create a task for each phase investigation
   - Mark the first task as in_progress

3. **Launch Parallel Agents**
   - Launch all pipeline-investigator agents **in parallel** (single message with multiple Task calls)
   - Each agent investigates one phase
   - Each agent prompt should specify:
     - Phase number and name
     - Relevant source files (see Phase-to-File mapping below)
     - Focus areas for research

4. **Synthesize Findings**
   - After all agents complete, analyze their reports
   - Identify cross-phase improvements
   - Prioritize by impact and feasibility
   - Present a consolidated summary with:
     - Top 10 improvement opportunities across all phases
     - Quick wins (low complexity, high impact)
     - Long-term opportunities (high complexity, transformative)
     - Research citations

### Phase-to-File Mapping

| Phase | Primary Files |
|-------|---------------|
| 1 | `src/docling_extractor_v2.py`, `src/structure_extractor.py` |
| 2 | `src/summarizer.py` |
| 3 | `src/chunker.py` |
| 4 | `src/embedding_generator.py`, `src/faiss_vector_store.py` |
| 5A | `src/graph/kg_extractor.py`, `src/graph/knowledge_graph.py` |
| 5B | `src/hybrid_search.py` |
| 5C | `src/reranker.py` |
| 5D | `src/graph_vector_fusion.py` |
| 6 | `src/context_assembly.py` |
| 7 | `src/agent/agent_core.py`, `src/agent/tools/` |

### Example Agent Prompt Template

```
Investigate PHASE X: [Phase Name]

**Focus**: [Brief description of what this phase does]

**Current Implementation**:
- Primary files: [file1.py, file2.py]
- Key techniques: [list from CLAUDE.md]

**Your Mission**:
1. Analyze current implementation in the specified files
2. Search for recent research (2024-2025) on [phase topic]
3. Identify 3-5 concrete improvement opportunities
4. Consider trade-offs and feasibility

**Research Focus Areas**:
- [Specific area 1 relevant to this phase]
- [Specific area 2]
- [Specific area 3]

Provide a structured report following the format in your system prompt.
```

### Critical Reminders

- **Launch agents in PARALLEL**: Use a single message with multiple Task tool calls
- **Use Haiku model**: Agents are configured to use Haiku (cost-effective)
- **Be specific**: Each agent prompt should include exact file paths and focus areas
- **Respect constraints**: Check CLAUDE.md for "DO NOT CHANGE" rules
- **Track progress**: Use TodoWrite to show progress through phases

### Expected Output

After all agents complete, provide:

1. **Executive Summary** (3-5 sentences on overall findings)
2. **Top 10 Improvements** (ranked by impact × feasibility)
3. **Quick Wins** (3-5 low-hanging fruit)
4. **Long-Term Opportunities** (2-3 transformative but complex)
5. **Research Citations** (all papers/blogs referenced)

Begin by parsing the arguments and determining which phases to investigate.
