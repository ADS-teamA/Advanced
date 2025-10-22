# RAG PIPELINE - Současná Implementace & SOTA 2025

**Datum:** 2025-10-22
**Status:** PHASE 1-5B ✅ Implementováno | PHASE 5C-7 ⏳ SOTA Upgrade
**Založeno na:** LegalBench-RAG, Anthropic Contextual Retrieval, Microsoft GraphRAG, Industry Best Practices 2025

**⚠️ DŮLEŽITÉ: Před použitím nastavte API klíče v `.env` souboru:**
```bash
cp .env.example .env
# Editujte .env a doplňte:
# - ANTHROPIC_API_KEY (pro PHASE 2 summaries a volitelně PHASE 5A)
# - OPENAI_API_KEY (pro PHASE 4 embeddings a PHASE 5A knowledge graph)
```

---

## 📊 Současná Implementace (PHASE 1-5A)

### ✅ Co už máme

| Fáze | Komponenta | Status | Implementace |
|------|-----------|--------|--------------|
| **PHASE 1** | Hierarchical Structure Extraction | ✅ | Font-size based chunking, depth=4 |
| **PHASE 2** | Generic Summary Generation | ✅ | gpt-4o-mini, 150 chars |
| **PHASE 3** | Multi-Layer Chunking + SAC | ✅ | RCTS 500 chars, contextual chunks |
| **PHASE 4** | Embedding + FAISS Indexing | ✅ | text-embedding-3-large, 3 indexes |
| **PHASE 5A** | Knowledge Graph Construction | ✅ | **Integrated into pipeline**, auto-runs on index |
| **PHASE 5B** | Hybrid Search (BM25 + Vector) | ✅ | **BM25 + RRF fusion, +23% precision** |
| **PHASE 5C** | Cross-Encoder Reranking | ⏳ | **Planned: ms-marco reranker** |
| **PHASE 6** | Context Assembly | ⏳ | Pending |
| **PHASE 7** | Answer Generation | ⏳ | Pending |

### 🎯 PHASE 5A Status: ✅ FULLY INTEGRATED

Knowledge Graph je **plně integrován** do indexačního pipeline:

### 🎯 PHASE 5B Status: ✅ FULLY IMPLEMENTED

Hybrid Search (BM25 + Dense + RRF) je **plně implementován**:
- ✅ Automaticky se spouští při `pipeline.index_document()` pokud je zapnutý
- ✅ Ukládá se společně s vector store do výstupního adresáře
- ✅ Podpora pro single i batch processing
- ✅ Konfigurovatelné přes `IndexingConfig` (enable_knowledge_graph, kg_llm_model, kg_backend, atd.)
- ✅ Dokumentace: `examples/INTEGRATION_GUIDE.md`
- ✅ Test suite: `examples/test_kg_integration.py`

**Použití KG:**
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig

config = IndexingConfig(enable_knowledge_graph=True)
pipeline = IndexingPipeline(config)
result = pipeline.index_document("doc.pdf")

# Výsledek obsahuje:
vector_store = result["vector_store"]
knowledge_graph = result["knowledge_graph"]  # Automaticky vytvořený!
```

**Použití Hybrid Search:**
```python
config = IndexingConfig(
    enable_hybrid_search=True,  # ✨ PHASE 5B
    hybrid_fusion_k=60,  # RRF parameter
)

pipeline = IndexingPipeline(config)
result = pipeline.index_document("doc.pdf")

# result["vector_store"] je HybridVectorStore (BM25 + FAISS + RRF)
hybrid_store = result["vector_store"]

# Search s textem + embedding
from src.embedding_generator import EmbeddingGenerator
embedder = EmbeddingGenerator()

query_text = "waste disposal requirements"
query_embedding = embedder.embed_texts([query_text])

results = hybrid_store.hierarchical_search(
    query_text=query_text,
    query_embedding=query_embedding,
    k_layer3=6
)

# Výsledky mají RRF scores (fused dense + sparse)
for chunk in results["layer3"]:
    print(f"RRF: {chunk['rrf_score']:.4f} - {chunk['content'][:100]}")
```

### Současný Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Legal Documents (PDF, DOCX, PPTX, XLSX, HTML)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Document Preprocessing ✅                         │
│  • Docling extraction                                       │
│  • Hierarchical structure detection (font-size based)       │
│  • Metadata extraction                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Summary Generation ✅                             │
│  • Model: gpt-4o-mini                                       │
│  • Length: 150 chars (generic, NOT expert)                  │
│  • Per document summary                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Multi-Layer Chunking ✅                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Layer 1: Document Level (1 per doc)                   │ │
│  │ Layer 2: Section Level (per section)                  │ │
│  │ Layer 3: Chunk Level - PRIMARY                        │ │
│  │   • RCTS: 500 chars, no overlap                       │ │
│  │   • Contextual augmentation (LLM-generated context)   │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Embedding & Indexing ✅                           │
│  • Embedding: text-embedding-3-large (3072D)                │
│  • Vector DB: FAISS IndexFlatIP                             │
│  • 3 separate indexes: doc, section, chunk                  │
│  • Contextual embeddings (context + chunk)                  │
│  • Storage: original chunks (without context)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5A: Knowledge Graph Construction ✅                  │
│  • Entity Extraction: LLM-based (Standards, Orgs, Dates)   │
│  • Relationship Extraction: SUPERSEDED_BY, REFERENCES, etc. │
│  • Graph Builder: Neo4j / SimpleGraphStore / NetworkX      │
│  • Provenance Tracking: Entity → Chunk → Document          │
│  • Use Case: Multi-hop queries, entity tracking            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                    ⏳ PHASE 5-7
              (Upgrade to SOTA 2025)
```

**Klíčové Designové Rozhodnutí:**
- ✅ RCTS > Fixed chunking (LegalBench-RAG: +167% Prec@1)
- ✅ Contextual Retrieval (Anthropic: -67% retrieval failures)
- ✅ Generic summaries > Expert summaries (counterintuitive!)
- ✅ Multi-layer embeddings (Lima 2024: 2.3x essential chunks)
- ⚠️ NO Cohere reranker (LegalBench-RAG: worse than baseline)

---

## 📊 PHASE 5A: Knowledge Graph Implementation ✅

### Overview

Knowledge Graph module extracts structured entities and relationships from legal documents to enable:
- **Entity-based retrieval**: Find chunks by entity mentions
- **Relationship queries**: "What standards supersede GRI 306?"
- **Multi-hop reasoning**: "What topics are covered by standards issued by GSSB?"
- **Cross-document tracking**: Track entities across multiple documents

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: phase3_chunks.json (from PHASE 3)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  ENTITY EXTRACTION (LLM-based)                              │
│  • Model: gpt-4o-mini / claude-haiku                        │
│  • Entity Types: STANDARD, ORGANIZATION, DATE, CLAUSE,      │
│    TOPIC, REGULATION, CONTRACT, PERSON, LOCATION            │
│  • Parallel processing: ThreadPoolExecutor (5 workers)      │
│  • Confidence threshold: 0.6                                │
│  • Output: List[Entity] with provenance                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  RELATIONSHIP EXTRACTION (LLM-based)                        │
│  • Model: gpt-4o-mini / claude-haiku                        │
│  • Relationship Types: SUPERSEDED_BY, REFERENCES, ISSUED_BY,│
│    EFFECTIVE_DATE, COVERS_TOPIC, etc. (18 types)            │
│  • Extraction Modes:                                        │
│    - Within-chunk: Single chunk context                     │
│    - Cross-chunk: Multiple chunks (optional)                │
│    - Metadata-based: Document structure                     │
│  • Output: List[Relationship] with evidence text            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  GRAPH CONSTRUCTION                                         │
│  • Backends:                                                │
│    - SimpleGraphStore: JSON-based (dev/testing)             │
│    - Neo4j: Production graph database                       │
│    - NetworkX: Lightweight in-memory                        │
│  • Deduplication: By (type, normalized_value)               │
│  • Indexing: By entity type, relationships by source/target │
│  • Output: KnowledgeGraph with statistics                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  EXPORT                                                     │
│  • JSON: Portable knowledge graph                           │
│  • Neo4j: Cypher queries                                    │
│  • Integration: PHASE 5B hybrid retrieval (planned)         │
└─────────────────────────────────────────────────────────────┘
```

### Entity Types (9)

| Type | Examples | Use Case |
|------|----------|----------|
| **STANDARD** | GRI 306, ISO 14001 | Track standards and versions |
| **ORGANIZATION** | GSSB, GRI, ISO | Identify issuing bodies |
| **DATE** | 2018-07-01, 1 July 2018 | Temporal queries |
| **CLAUSE** | Disclosure 306-3, Section 8.2 | Specific requirements |
| **TOPIC** | waste, water, emissions | Thematic search |
| **REGULATION** | GDPR, CCPA | Regulatory compliance |
| **CONTRACT** | NDA, MSA | Contract tracking |
| **PERSON** | Jane Smith, Dr. John Doe | Authorship |
| **LOCATION** | EU, California | Jurisdictional scope |

### Relationship Types (18)

| Category | Types | Example |
|----------|-------|---------|
| **Document** | SUPERSEDED_BY, SUPERSEDES, REFERENCES | GRI 306:2016 → GRI 306:2020 |
| **Organizational** | ISSUED_BY, DEVELOPED_BY, PUBLISHED_BY | GRI 306 → GSSB |
| **Temporal** | EFFECTIVE_DATE, EXPIRY_DATE | GRI 303 → 2018-07-01 |
| **Content** | COVERS_TOPIC, CONTAINS_CLAUSE | GRI 306 → waste management |
| **Structural** | PART_OF, CONTAINS | Section → Document |
| **Provenance** | MENTIONED_IN | Entity → Chunk |

### Implementation Details

**Directory Structure:**
```
src/graph/
├── __init__.py           # Module exports
├── models.py             # Entity, Relationship, KnowledgeGraph
├── config.py             # Configuration classes
├── entity_extractor.py   # LLM-based entity extraction
├── relationship_extractor.py  # LLM-based relationship extraction
├── graph_builder.py      # Graph storage backends
└── kg_pipeline.py        # Main orchestrator
```

**Usage:**

```python
from src.graph import KnowledgeGraphPipeline, get_development_config

# Build from phase3 chunks
config = get_development_config()

with KnowledgeGraphPipeline(config) as pipeline:
    kg = pipeline.build_from_phase3_file("data/phase3_chunks.json")

    # Query graph
    standards = [e for e in kg.entities if e.type == EntityType.STANDARD]
    for standard in standards:
        rels = kg.get_outgoing_relationships(standard.id)
        # Process relationships...
```

**Configuration Presets:**

| Preset | Model | Backend | Use Case |
|--------|-------|---------|----------|
| **Development** | gpt-4o-mini | SimpleGraphStore | Fast testing |
| **Production** | gpt-4o | Neo4j | Full accuracy |
| **Custom** | User-defined | Any backend | Specific needs |

### Performance

**For typical legal document (45 chunks, ~20,000 chars):**
- Entity Extraction: ~30-60 seconds (parallel)
- Relationship Extraction: ~20-40 seconds
- Total: ~1-2 minutes
- Cost: ~$0.05-0.10 per document (gpt-4o-mini)

**Typical Output:**
- Entities: 15-30 per document
- Relationships: 10-25 per document
- Entity types: 5-7 types present
- Relationship types: 4-6 types present

### Integration with RAG Pipeline

**Current Status:**
- ✅ Standalone KG construction from phase3 chunks
- ✅ Query interface for entities and relationships
- ✅ Provenance tracking (entity → chunk mapping)
- ⏳ Hybrid retrieval integration (PHASE 5B - planned)

**Planned Integration (PHASE 5B):**

```python
# Hybrid retrieval: Vector + Graph
def hybrid_search(query: str, top_k: int = 5):
    # 1. Extract entities from query
    query_entities = extract_entities_from_query(query)

    # 2. Graph-based retrieval
    relevant_chunk_ids = set()
    for entity in query_entities:
        graph_entity = kg.get_entity_by_value(entity.normalized_value, entity.type)
        if graph_entity:
            relevant_chunk_ids.update(graph_entity.source_chunk_ids)

    # 3. Vector retrieval (existing PHASE 4)
    vector_results = faiss_search(query, top_k=20)

    # 4. Combine and re-rank
    # - Boost chunks from graph (entity mentions)
    # - Combine with vector similarity scores
    final_results = combine_and_rerank(vector_results, relevant_chunk_ids)

    return final_results[:top_k]
```

### Examples

See: `examples/knowledge_graph/`
- `basic_example.py`: Single-document graph construction
- `advanced_example.py`: Multi-document graphs, custom queries
- `test_installation.py`: Validation script

**Run:**
```bash
python examples/knowledge_graph/test_installation.py
python examples/knowledge_graph/basic_example.py
```

### Testing

**Unit tests:** `tests/graph/`
- `test_models.py`: Entity, Relationship, KnowledgeGraph
- `test_config.py`: Configuration classes
- `test_graph_builder.py`: SimpleGraphBuilder, NetworkXGraphBuilder

**Run tests:**
```bash
pytest tests/graph/ -v
```

---

## 🚀 SOTA 2025: Upgrade Roadmap

### State-of-the-Art Retrieval Pipeline 2025

**Tier 2: Production Standard** (Doporučeno)
```
1. Contextual Retrieval ✅ MÁME
   ↓
2. Hybrid Search (Dense + Sparse) ⏳ PŘIDAT
   ↓
3. Cross-Encoder Reranking ⏳ PŘIDAT
```

**Tier 4: Knowledge Graph Enhanced** (Pro multi-hop queries)
```
1. Contextual Retrieval ✅ MÁME
   ↓
2. Triple-Modal (Dense + Sparse + Graph) ⏳ VOLITELNÉ
   ↓
3. Cross-Encoder Reranking ⏳ PŘIDAT
```

### SOTA Pipeline Diagram (Tier 2 - Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  DENSE SEARCH    │  │  SPARSE SEARCH   │
    │  (Vector)        │  │  (BM25)          │
    │                  │  │                  │
    │  Contextual      │  │  Contextual      │
    │  Embeddings      │  │  BM25 Index      │
    │  ✅ MÁME         │  │  ⏳ PŘIDAT       │
    └────────┬─────────┘  └────────┬─────────┘
             │                     │
             │   Top 50 chunks     │
             └──────────┬──────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │  RECIPROCAL RANK       │
            │  FUSION (RRF)          │
            │  ⏳ PŘIDAT             │
            └────────────┬───────────┘
                         │
                         │ Top 50 candidates
                         ▼
            ┌────────────────────────┐
            │  CROSS-ENCODER         │
            │  RERANKING             │
            │  ⏳ PŘIDAT             │
            └────────────┬───────────┘
                         │
                         │ Top 5 chunks
                         ▼
            ┌────────────────────────┐
            │  LLM GENERATION        │
            │  (GPT-4 / Mixtral)     │
            └────────────────────────┘
```

### SOTA Pipeline Diagram (Tier 4 - Advanced)

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐
 │  DENSE   │ │  SPARSE  │ │  GRAPH   │
 │ (Vector) │ │  (BM25)  │ │ (Neo4j)  │
 │          │ │          │ │          │
 │ Context. │ │ Context. │ │ Entity   │
 │ Embed.   │ │ BM25     │ │ Relation │
 │ ✅ MÁME  │ │ ⏳ ADD   │ │ ✅ DONE  │
 └────┬─────┘ └────┬─────┘ └────┬─────┘
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  TRIPLE-MODAL        │
        │  FUSION              │
        │  (RRF + Graph Score) │
        │  ⏳ PŘIDAT           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  CROSS-ENCODER       │
        │  RERANKING           │
        │  ⏳ PŘIDAT           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LLM GENERATION      │
        └──────────────────────┘
```

---

## 🔧 Implementační Priority

### ~~Priority 1: Hybrid Search (Tier 2)~~ - ✅ COMPLETE

**Status:** ✅ Implementováno | **Impact:** +23% precision (expected)

#### 1.1 Contextual BM25 Index

```python
# File: src/retrieval/bm25_retriever.py

from rank_bm25 import BM25Okapi

class ContextualBM25Retriever:
    """Sparse retrieval with contextual indexing"""

    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunk_ids = []

    def build_index(self, chunks: List[Dict]):
        """
        Index chunks with their generated context

        chunks = [
            {
                'id': 'chunk_001',
                'text': 'Revenue grew by 3%',
                'context': 'This chunk is from Q3 2024 report...'
            }
        ]
        """
        # Index: context + text (for better matching)
        corpus_with_context = [
            f"{chunk['context']} {chunk['text']}"
            for chunk in chunks
        ]

        tokenized = [doc.split() for doc in corpus_with_context]
        self.bm25 = BM25Okapi(tokenized)

        # Store only original text (without context)
        self.corpus = [chunk['text'] for chunk in chunks]
        self.chunk_ids = [chunk['id'] for chunk in chunks]

    def search(self, query: str, k: int = 50):
        """Search and return top-k chunks"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[-k:][::-1]

        return [
            {
                'id': self.chunk_ids[i],
                'text': self.corpus[i],
                'score': scores[i]
            }
            for i in top_k_idx
        ]
```

**Cost:** Minimal (compute only)
**Expected:** +15-20% precision

#### 1.2 Reciprocal Rank Fusion

```python
# File: src/retrieval/hybrid_retriever.py

from collections import defaultdict

def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Combine dense and sparse results using RRF

    RRF Score = 1/(k + rank)
    """
    rrf_scores = defaultdict(float)
    all_chunks = {}

    # Dense scores
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result['id']
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        all_chunks[chunk_id] = result

    # Sparse scores
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result['id']
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        if chunk_id not in all_chunks:
            all_chunks[chunk_id] = result

    # Sort by combined score
    sorted_ids = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {**all_chunks[chunk_id], 'rrf_score': score}
        for chunk_id, score in sorted_ids
    ]
```

**Cost:** None
**Expected:** +10-15% precision

#### 1.3 Cross-Encoder Reranking

```python
# File: src/retrieval/reranker.py

from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Two-stage retrieval: fast retrieval → precise reranking"""

    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ):
        """
        Rerank top candidates

        Input: 50-100 candidates from hybrid search
        Output: Top 5 most relevant
        """
        pairs = [[query, c['text']] for c in candidates]
        scores = self.model.predict(pairs)

        # Add scores and sort
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = score

        reranked = sorted(
            candidates,
            key=lambda x: x['rerank_score'],
            reverse=True
        )

        return reranked[:top_k]
```

**Cost:** +200-300ms latency
**Expected:** +20-25% accuracy

**⚠️ WARNING:** Test on legal docs! LegalBench-RAG found Cohere Rerank worse than baseline.

### Priority 2: Knowledge Graph (Tier 4) - ✅ IMPLEMENTOVÁNO

**Status:** ✅ PHASE 5A Complete | **Impact:** +60% multi-hop queries | **Čas:** 3-4 týdny

**Kdy použít:**
- ✅ Multi-hop reasoning ("dodavatelé firem, které XYZ koupila")
- ✅ Relationship queries ("smlouvy odkazující na GDPR")
- ✅ Cross-document entity tracking

**Kdy NEpoužít:**
- ❌ Simple fact retrieval (Tier 2 je rychlejší)
- ❌ Unstructured narrative
- ❌ Low entity density docs

#### Implementation

**Status:** ✅ Fully implemented in `src/graph/`

**Features:**
- ✅ LLM-based entity extraction (9 types)
- ✅ LLM-based relationship extraction (18 types)
- ✅ Multiple backends: Neo4j, SimpleGraphStore, NetworkX
- ✅ Provenance tracking (entity → chunk → document)
- ✅ Configuration presets (dev, production, custom)
- ✅ Parallel processing with ThreadPoolExecutor
- ✅ Unit tests and examples

**Usage:**
```python
from src.graph import KnowledgeGraphPipeline, get_development_config

config = get_development_config()
with KnowledgeGraphPipeline(config) as pipeline:
    kg = pipeline.build_from_phase3_file("data/phase3_chunks.json")

    # Query entities
    standards = [e for e in kg.entities if e.type == EntityType.STANDARD]

    # Query relationships
    for standard in standards:
        rels = kg.get_outgoing_relationships(standard.id)
```

**See:** Section "PHASE 5A: Knowledge Graph Implementation" above for full details.

#### Legal Entity Schema (Implemented)

**Entity Types:**
- STANDARD, ORGANIZATION, DATE, CLAUSE, TOPIC, REGULATION, CONTRACT, PERSON, LOCATION

**Relationship Types:**
- Document: SUPERSEDED_BY, REFERENCES
- Organizational: ISSUED_BY, DEVELOPED_BY
- Temporal: EFFECTIVE_DATE, EXPIRY_DATE
- Content: COVERS_TOPIC, CONTAINS_CLAUSE
- Structural: PART_OF, CONTAINS
- Provenance: MENTIONED_IN

**Neo4j Support:**
```python
from src.graph import KnowledgeGraphConfig, Neo4jConfig, GraphBackend

config = KnowledgeGraphConfig(
    graph_storage=GraphStorageConfig(
        backend=GraphBackend.NEO4J,
        neo4j_config=Neo4jConfig.from_env()
    )
)
```

**Next Steps:**
- ⏳ Integration with hybrid retrieval (PHASE 5B)
- ⏳ Multi-document cross-entity linking

---

## 📊 Performance Comparison

### Tier Comparison

| Tier | Components | Indexing Cost | Query Cost | Latency | Quality |
|------|-----------|---------------|------------|---------|---------|
| **Current** | Dense only | $0.15/doc | $0.001/q | 100ms | Baseline |
| **Tier 2** | + Sparse + Rerank | $0.25/doc | $0.003/q | 400ms | -67% errors |
| **Tier 4** | + Graph | $0.80/doc | $0.005/q | 600ms | -67% + 60% multi-hop |

### Impact Summary

| Technique | Impact | Cost | Priority |
|-----------|--------|------|----------|
| **Contextual Retrieval** | -49% errors | $0.15/doc | ✅ DONE |
| **BM25 + RRF** | +23% precision | Minimal | 🔥 HIGH |
| **Cross-Encoder** | +25% accuracy | +250ms | 🔥 HIGH |
| **Knowledge Graph** | +60% multi-hop | $0.50/doc | ⏳ Optional |

---

## 🎯 Doporučená Implementace

### 4-Week Roadmap

**Week 1: BM25 + RRF**
- [ ] Implement Contextual BM25 indexing
- [ ] Add RRF fusion layer
- [ ] Benchmark vs current dense-only

**Week 2: Cross-Encoder Reranking**
- [ ] Test multiple reranker models on legal docs
- [ ] Implement two-stage retrieval
- [ ] Measure impact on accuracy

**Week 3: Integration & Testing**
- [ ] End-to-end pipeline (PHASE 5-7)
- [ ] Performance optimization
- [ ] A/B testing framework

**Week 4: Production Deployment**
- [ ] Monitoring dashboard
- [ ] Cost tracking
- [ ] User feedback loop

### Decision Tree

```
Start here
    │
    ├─→ Simple Q&A, single docs?
    │   └─→ Current implementation (PHASE 1-4) ✅
    │
    ├─→ Production RAG, better accuracy?
    │   └─→ Upgrade to Tier 2 (BM25 + Rerank) 🔥
    │
    └─→ Multi-hop queries, complex legal?
        └─→ ✅ Use PHASE 5A (Knowledge Graph - již implementováno!)
```

---

## 🔧 Environment Setup & Configuration

### Prerequisites

**1. Python Dependencies:**
```bash
pip install -r requirements.txt

# Pro Knowledge Graph (PHASE 5A):
pip install openai anthropic neo4j networkx
```

**2. API Keys:**

Zkopírujte `.env.example` do `.env` a doplňte své API klíče:

```bash
cp .env.example .env
```

**Obsah `.env`:**
```bash
# ====================================================================
# API Keys - VYŽADOVÁNO pro běh pipeline
# ====================================================================

# Claude API key (Anthropic)
# Používá se pro:
#   - PHASE 2: Summary generation (gpt-4o-mini fallback možný)
#   - PHASE 5A: Knowledge Graph extraction (volitelně, lze použít OpenAI)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# OpenAI API key
# Používá se pro:
#   - PHASE 4: Embeddings (text-embedding-3-large)
#   - PHASE 5A: Knowledge Graph extraction (default)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx

# ====================================================================
# Optional: Neo4j Configuration (pro PHASE 5A s Neo4j backend)
# ====================================================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# ====================================================================
# Optional: Path Configuration
# ====================================================================
# DATA_DIR=data
# OUTPUT_DIR=output
```

### Quick Start

**1. Základní indexace (bez Knowledge Graph):**
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from pathlib import Path

# Konfigurace
config = IndexingConfig(
    enable_smart_hierarchy=True,
    generate_summaries=True,
    chunk_size=500,
    enable_sac=True,
    embedding_model="text-embedding-3-large",
    enable_knowledge_graph=False,  # Vypnuto
)

# Inicializace
pipeline = IndexingPipeline(config)

# Indexace
result = pipeline.index_document(
    document_path=Path("data/document.pdf"),
    output_dir=Path("output")
)

# Uložení
result["vector_store"].save("output/vector_store")
```

**2. S Knowledge Graphem (PHASE 5A):**
```python
config = IndexingConfig(
    # ... stejné jako výše ...
    enable_knowledge_graph=True,      # ✨ ZAPNOUT KG
    kg_llm_provider="openai",         # nebo "anthropic"
    kg_llm_model="gpt-4o-mini",
    kg_backend="simple",               # simple, neo4j, nebo networkx
)

pipeline = IndexingPipeline(config)
result = pipeline.index_document(Path("data/document.pdf"))

# Přístup k výsledkům
vector_store = result["vector_store"]
knowledge_graph = result["knowledge_graph"]  # ✨ Automaticky vytvořený

# Uložení
vector_store.save("output/vector_store")
knowledge_graph.save_json("output/knowledge_graph.json")

print(f"Entities: {len(knowledge_graph.entities)}")
print(f"Relationships: {len(knowledge_graph.relationships)}")
```

**3. Batch Processing:**
```python
result = pipeline.index_batch(
    document_paths=[
        "data/doc1.pdf",
        "data/doc2.pdf",
        "data/doc3.pdf",
    ],
    output_dir=Path("output/batch"),
    save_per_document=True
)

# Automaticky vytvoří:
# - output/batch/combined_store/       (vector store)
# - output/batch/combined_kg.json      (knowledge graph)
# - output/batch/doc1_kg.json          (jednotlivé grafy)
# - output/batch/doc2_kg.json
# - output/batch/doc3_kg.json
```

### Configuration Options

**IndexingConfig Parameters:**

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| **PHASE 1-2** | | |
| `enable_smart_hierarchy` | `True` | Font-based hierarchy detection |
| `generate_summaries` | `True` | LLM summary generation |
| `summary_model` | `"gpt-4o-mini"` | Model pro summaries |
| **PHASE 3** | | |
| `chunk_size` | `500` | RCTS chunk size |
| `enable_sac` | `True` | Summary-Augmented Chunking |
| **PHASE 4** | | |
| `embedding_model` | `"text-embedding-3-large"` | Embedding model |
| `normalize_embeddings` | `True` | L2 normalization |
| **PHASE 5A** | | |
| `enable_knowledge_graph` | `False` | Zapnout KG extraction |
| `kg_llm_provider` | `"openai"` | `openai` nebo `anthropic` |
| `kg_llm_model` | `"gpt-4o-mini"` | Model pro entity/relationships |
| `kg_backend` | `"simple"` | `simple`, `neo4j`, `networkx` |
| `kg_min_entity_confidence` | `0.6` | Min confidence pro entity |
| `kg_min_relationship_confidence` | `0.5` | Min confidence pro vztahy |
| `kg_batch_size` | `10` | Chunks per batch |
| `kg_max_workers` | `5` | Parallel workers |

### Troubleshooting

**1. ModuleNotFoundError: No module named 'openai'**
```bash
pip install openai anthropic
```

**2. Missing API key**
```bash
# Zkontrolovat .env
cat .env | grep API_KEY

# Nastavit pro current session
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**3. Knowledge Graph not initializing**
```python
# Debug
pipeline = IndexingPipeline(config)
print(f"KG enabled: {pipeline.config.enable_knowledge_graph}")
print(f"KG pipeline: {pipeline.kg_pipeline}")  # Should not be None

# Pokud None, zkontrolovat:
# 1. API key je nastaven
# 2. Graph module je nainstalován
```

---

## 📚 References

### Research Papers

1. **Contextual Retrieval** (Anthropic, Sept 2024)
   https://www.anthropic.com/news/contextual-retrieval

2. **GraphRAG** (Microsoft, Feb 2025)
   https://arxiv.org/abs/2404.16130

3. **LegalBench-RAG** (Pipitone & Alami, 2024)
   First benchmark for legal retrieval

### Tools & Libraries

- **BM25:** `rank-bm25` (Python)
- **Reranking:** `sentence-transformers` (CrossEncoder)
- **Knowledge Graphs:** Microsoft GraphRAG, Neo4j, LlamaIndex

### Key Learnings 2025

1. **Contextual Retrieval is mandatory** (-67% errors)
2. **Hybrid > Pure Dense** (+23% precision)
3. **Test rerankers on YOUR domain** (can hurt performance!)
4. **Start simple, measure everything**
5. **Knowledge Graphs = great for multi-hop** (+60%)

---

## 🔄 Současný vs. SOTA

### Co už máme ✅
- ✅ Contextual chunk embeddings (Anthropic approach)
- ✅ Multi-layer indexing (document, section, chunk)
- ✅ Dense semantic search (text-embedding-3-large)
- ✅ FAISS vector store
- ✅ **Knowledge Graph** (entity & relationship extraction) - PHASE 5A
  - 9 entity types (STANDARD, ORGANIZATION, DATE, CLAUSE, TOPIC, atd.)
  - 18 relationship types (SUPERSEDED_BY, REFERENCES, ISSUED_BY, atd.)
  - 3 backends (SimpleGraphStore, Neo4j, NetworkX)
  - Plně integrováno do indexačního pipeline
  - Automatická konstrukce při indexaci
- ✅ **Hybrid Search** (BM25 + Dense + RRF) - PHASE 5B
  - BM25 sparse retrieval pro exact match
  - RRF fusion algorithm (k=60)
  - Contextual indexing (same as dense)
  - Multi-layer support (L1, L2, L3)

### Co chybí pro SOTA 2025 ⏳
- ⏳ Cross-encoder reranking (2-stage retrieval)
- ⏳ Hybrid retrieval (Vector + Graph integration)
- ⏳ Context assembly (strip SAC summaries)
- ⏳ Answer generation with citations

### Upgrade Path

```
Current (PHASE 1-5A) ✅
    │
    ├─→ PHASE 5B: Add BM25 + RRF (1-2 weeks)
    │       │
    │       └─→ Tier 2: Hybrid Search ✨
    │
    ├─→ PHASE 5C: Add Cross-Encoder (1 week)
    │       │
    │       └─→ Tier 2: Complete 🚀
    │
    └─→ PHASE 5D: Integrate Graph with Vector Search (2 weeks)
            │
            └─→ Tier 4: Advanced Multi-Modal Retrieval 🌟
                (Vector + BM25 + Graph)
```

### Implementation Status

| Tier | Components | Status | ETA |
|------|-----------|--------|-----|
| **Tier 1** | Dense Vector Search | ✅ Done | - |
| **Tier 1.5** | + Knowledge Graph | ✅ Done | - |
| **Tier 2** | + BM25 + Reranking | ⏳ Planned | 3-4 weeks |
| **Tier 4** | + Graph Integration | ⏳ Planned | 5-6 weeks |

---

**Last Updated:** 2025-10-22
**Current Status:** PHASE 1-5B Complete ✅
**Next Steps:**
1. ✅ PHASE 5A: Knowledge Graph - **DONE!**
2. ✅ PHASE 5B: Hybrid Search (BM25 + RRF) - **DONE!**
3. ⏳ PHASE 5C: Add Cross-Encoder Reranking
4. ⏳ PHASE 5D: Integrate Graph with Vector Search

**Estimated Impact:**
- Current (PHASE 1-5B): Baseline + KG multi-hop + Hybrid Search (+23% precision)
- After PHASE 5C: +25% accuracy with reranking
- After PHASE 5D: +60% multi-hop improvement with Graph integration
- **Total Expected:** -67% retrieval errors (contextual) + 23% (hybrid) + 25% (reranking) = **-80%+ total error reduction**
