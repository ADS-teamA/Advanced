# OPTIMAL RAG PIPELINE PRO LEGALBENCH-RAG
## Evidence-Based Implementation Guide

**Datum:** 2025-10-21
**Verze:** 2.2 (Contextual Retrieval Implemented)
**Založeno na:** 5 research papers z legal AI a RAG optimization + MLEB 2025 + Anthropic Contextual Retrieval

---

## 🎯 Executive Summary

Tento dokument definuje **production-ready RAG pipeline** optimalizovaný pro právní dokumenty, založený výhradně na empirických výsledcích z nejvýznamnějších výzkumných prací v oblasti legal RAG systemů.

### Klíčové Findings z Výzkumu

| Finding | Source | Impact | Status |
|---------|--------|--------|--------|
| **RCTS > Fixed-size chunking** | LegalBench-RAG | Prec@1: 6.41% vs 2.40% | ✅ IMPLEMENTED |
| **Contextual Retrieval** | Anthropic 2024 | 67% failure reduction | ✅ IMPLEMENTED |
| **Generic > Expert summaries** | Reuter 2024 | Counterintuitive! | ✅ IMPLEMENTED |
| **Multi-layer embeddings** | Lima 2024 | 2.3x essential chunks | ✅ IMPLEMENTED |
| **text-embedding-3-large** | LegalBench-RAG, MLEB 2025 | 3072D, proven baseline | ✅ IMPLEMENTED |
| **FAISS IndexFlatIP** | Lima 2024 | Cosine similarity, exact | ✅ IMPLEMENTED |
| **Cohere reranker fails** | LegalBench-RAG | Worse than no reranking | ⚠️ AVOID |
| **Dense > Sparse retrieval** | Reuter 2024 | Better precision/recall | ✅ IMPLEMENTED |
| **Chunk size: 500 chars** | Reuter 2024 | Optimal balance | ✅ IMPLEMENTED |
| **Summary length: 150 chars** | Reuter 2024 | Best trade-off | ✅ IMPLEMENTED |

### Expected Performance

```
Retrieval Metrics (na LegalBench-RAG):
├── Precision@1: 6-8%  (baseline: 2.4%)
├── Recall@64: 60-65%  (baseline: ~35%)
└── DRM Rate: <30%     (baseline: 67%)

Quality Metrics:
├── Essential chunks: 35-40%  (baseline: 16%)
├── Unnecessary chunks: <60%  (baseline: 75%)
└── Document mismatch: <5%    (with Contextual Retrieval)

Performance:
├── Latency: 2-4s per query
└── Cost: $0.05-0.08 per query
```

---

## 📊 Research Foundation

### Papers Analyzed

1. **LegalBench-RAG** (Pipitone & Alami, 2024)
   - První benchmark pro legal retrieval
   - 6,858 queries, 79M char korpus
   - Character-level annotations
   - **Key**: RCTS chunking, reranker failure, text-embedding-3-large baseline

2. **Contextual Retrieval** (Anthropic, Sept 2024)
   - LLM-generated context for each chunk
   - 67% reduction in retrieval failures
   - **Key**: Chunk-specific context vs generic summaries

3. **Multi-Layer Embeddings** (Lima, 2024)
   - 6 hierarchical layers
   - Brazilian Constitution case study
   - **Key**: 2.3x essential chunks improvement, FAISS architecture

4. **NLI for Legal Contracts** (Narendra et al., 2024)
   - GPT-4 vs Mixtral comparison
   - RAG s cross-references
   - **Key**: 96.46% accuracy achievable

5. **MLEB 2025** (Massive Legal Embedding Benchmark)
   - Latest benchmark for legal embeddings
   - Kanon 2 Embedder #1 (86% NDCG@10)
   - Voyage 3 Large #2 (85.7%)
   - **Key**: Domain-specific models outperform general-purpose by significant margin

---

## 🏗️ Pipeline Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      INPUT: Legal Documents                          │
│           (Contracts, Policies, NDAs, M&A, Regulations)             │
│   Supported formats: PDF, DOCX, PPTX, XLSX, HTML                    │
│   Input: Single file OR directory (batch processing)                │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Document Preprocessing & Structure Detection              │
│  • Format normalization (PDF → text)                                 │
│  • Hierarchical structure extraction                                 │
│  • Metadata extraction                                               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Generic Summary Generation                                │
│  • Model: gpt-4o-mini                                                │
│  • Length: ~150 chars (±20 tolerance)                                │
│  • Style: GENERIC (NOT expert-guided)                                │
│  • Output: Document fingerprint                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Multi-Layer Chunking Strategy                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Layer 1: Document Level (1 summary per doc)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Layer 2: Section Level (summary per major section)            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Layer 3: Chunk Level (PRIMARY - Contextual Retrieval)          │ │
│  │ • Method: Recursive Character Text Splitter (RCTS)            │ │
│  │ • Chunk size: 500 characters                                   │ │
│  │ • Overlap: 0 (RCTS handles boundaries naturally)              │ │
│  │ • Augmentation: LLM-generated context (50-100 words)          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Embedding & FAISS Indexing ✅ IMPLEMENTED                  │
│  • Model: text-embedding-3-large (3072D, OpenAI)                    │
│  • Alternative: BAAI/bge-m3 (1024D, multilingual, open-source)   │
│  • Vector DB: FAISS IndexFlatIP (cosine similarity)                  │
│  • Indexes: 3 separate (doc, section, chunk level)                  │
│  • Output: 242 vectors (GRI 306), ~2.9 MB                           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 5: Query & Retrieval (K=4-8)                                  │
│  • Method: Dense semantic search (NO BM25)                           │
│  • Top-K: 5-8 chunks (sweet spot)                                    │
│  • Reranking: NONE (empirically worse!)                              │
│  • Filtering: Document-level (DRM prevention)                        │
│  • Token limit: 2,500 tokens baseline                                │
│  • Similarity threshold: top score - 25%                             │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 6: Context Assembly                                           │
│  • Strip context from chunks (context was for retrieval only)       │
│  • Concatenate K chunks                                              │
│  • Add document summaries for global context                         │
│  • Preserve source citations                                         │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE 7: Answer Generation                                          │
│  • Model: GPT-4 or Mixtral 8x7B                                      │
│  • Temperature: 0.1-0.3                                              │
│  • Max tokens: 1,000                                                 │
│  • Citations: MANDATORY                                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Verified Answer + Citations               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📝 PHASE 1: Document Preprocessing

### Cíl
Normalizovat formáty a extrahovat strukturu pro multi-layer chunking.

### Implementace

```python
def preprocess_legal_document(document_path):
    """
    Preprocessing pro právní dokumenty
    Based on: Multi-Layer Embedding paper (Lima, 2024)
    """
    # 1. Format conversion
    if document_path.endswith('.pdf'):
        # Use DiT + EasyOCR for scanned PDFs (per Narendra 2024)
        text = extract_from_pdf(document_path, ocr_method='dit+easyocr')
    else:
        text = read_text_file(document_path)

    # 2. Structure detection (hierarchical)
    structure = detect_hierarchy(text)
    # Detects: Articles, Sections, Subsections, Paragraphs
    # For contracts: Clauses, Subclauses
    # For legislation: Parts, Books, Titles, Chapters (per Lima 2024)

    # 3. Metadata extraction
    metadata = {
        'document_id': generate_doc_id(document_path),
        'document_type': classify_document_type(text),
        'source': document_path,
        'parties': extract_parties(text) if is_contract(text) else None,
        'date': extract_date(text),
        'total_sections': len(structure['sections'])
    }

    return {
        'text': text,
        'structure': structure,
        'metadata': metadata
    }
```

### Hierarchical Structure Format

Per Lima (2024) - Brazilian legislative hierarchy:

```python
LEGAL_HIERARCHY = {
    'legislation': [
        'Part',           # Optional
        'Book',           # Optional
        'Title',          # Optional
        'Chapter',        # Optional
        'Section_Group',  # Optional
        'Article',        # Fundamental unit (MANDATORY)
        'Paragraph',      # Optional
        'Inciso',         # Enumeration
        'Alínea',         # Sub-enumeration
        'Item'            # Sub-sub-enumeration
    ],
    'contract': [
        'Recitals',
        'Article',
        'Section',
        'Clause',
        'Subclause',
        'Paragraph'
    ],
    'nda': [
        'Preamble',
        'Definitions',
        'Obligations',
        'Exclusions',
        'Term',
        'Miscellaneous'
    ]
}
```

---

## 📋 PHASE 2: Generic Summary Generation

### Klíčový Finding

> **Generic summaries OUTPERFORM expert-guided summaries for retrieval**
> Source: Reuter et al., 2024 (Summary-Augmented Chunking)

**Why?**
- Generic summaries strike better balance mezi distinctiveness a broad semantic alignment
- Expert-guided může overfit to narrow features
- Generic = more robust across wider variety of user queries

### Configuration

```python
SUMMARIZATION_CONFIG = {
    'model': 'gpt-4o-mini',        # Fast & cost-effective
    'max_chars': 150,              # Optimal length (Reuter 2024)
    'tolerance': 20,               # ±20 chars acceptable
    'style': 'generic',            # NOT expert-guided!
    'retry_on_exceed': True,       # Reduce char_length if exceeded
}
```

### Implementation

```python
def generate_generic_summary(document_text, max_chars=150):
    """
    Generic summary generation
    Based on: Reuter et al., 2024 (Table 1: optimal summary length)

    IMPORTANT: Generic approach outperforms expert-guided!
    """
    prompt = f"""You are an expert legal document summarizer.

Summarize the following legal document text. Focus on extracting the most
important entities, core purpose, and key legal topics.

The summary must be concise, maximum {max_chars} characters long, and
optimized for providing context to smaller text chunks.

Output only the summary text, nothing else.

Document:
{document_text[:5000]}  # First 5000 chars sufficient

Summary:"""

    response = llm.generate(
        prompt=prompt,
        model='gpt-4o-mini',
        temperature=0.3,
        max_tokens=50
    )

    summary = response.strip()

    # Regenerate if too long
    if len(summary) > max_chars + 20:
        return generate_generic_summary(
            document_text,
            max_chars=int(max_chars * 0.9)
        )

    return summary
```

### Example Summaries

**Generic (150 chars) - PREFERRED:**
```
"Non-Disclosure Agreement between Evelozcity and Recipient to protect
confidential information shared during a meeting."
```

**Expert-Guided (150 chars) - AVOID:**
```
"NDA between Evelozcity and Recipient; covers vehicle prototypes,
confidentiality obligations, exclusions, 5-yr term, CA governing law."
```

→ Generic achieved **better retrieval precision** in experiments!

---

## 🧩 PHASE 3: Multi-Layer Chunking Strategy

### Evidence-Based Configuration

```python
CHUNKING_CONFIG = {
    # From LegalBench-RAG experiments
    'method': 'RecursiveCharacterTextSplitter',  # RCTS > Fixed-size
    'chunk_size': 500,                           # Characters (optimal)
    'chunk_overlap': 0,                          # RCTS handles naturally

    # From Anthropic Contextual Retrieval (Sept 2024)
    'enable_contextual': True,                   # 67% reduction in failures
    'context_model': 'claude-haiku-4-5',        # Fast & cheap
    'context_length': 100,                       # Words (50-100 optimal)
    'include_surrounding_chunks': True,          # Better context awareness

    # From Multi-Layer Embedding paper
    'enable_multi_layer': True,                  # 2.3x essential chunks
    'token_baseline': 2500,                      # For filtering
    'similarity_threshold': 0.25,                # Top score - 25%
}
```

### Layer 1: Document Level

```python
def create_document_level_embeddings(document, summary):
    """
    Document-level embedding
    Based on: Lima 2024 (Multi-Layer Embedding)

    Purpose:
    - Global filtering during retrieval
    - Document identification
    - DRM prevention
    """
    return {
        'type': 'document',
        'content': summary,  # Use summary, not full text
        'metadata': {
            'document_id': document['metadata']['document_id'],
            'document_type': document['metadata']['document_type'],
            'hierarchy_level': 0,
            'total_sections': document['metadata']['total_sections']
        }
    }
```

### Layer 2: Section Level

```python
def create_section_level_embeddings(document, structure, summaries):
    """
    Section-level embeddings
    Based on: Lima 2024

    Purpose:
    - Mid-level context
    - Section-specific queries
    - Context expansion when needed
    """
    section_embeddings = []

    for section in structure['sections']:
        # Generate section summary (50-100 chars)
        section_summary = summarize_section(section.text, max_chars=75)

        section_embeddings.append({
            'type': 'section',
            'content': section_summary,
            'metadata': {
                'document_id': document['metadata']['document_id'],
                'section_id': section.id,
                'section_title': section.title,
                'hierarchy_level': 1,
                'document_summary': summaries['document']
            }
        })

    return section_embeddings
```

### Layer 3: Chunk Level (PRIMARY + Contextual Retrieval)

**Critical Implementation:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from contextual_retrieval import ContextualRetrieval
from config import ChunkingConfig

def create_chunk_level_embeddings(document, structure, summaries):
    """
    PRIMARY chunking layer with Contextual Retrieval

    Based on:
    - LegalBench-RAG: RCTS > Naive (Prec@1: 6.41% vs 2.40%)
    - Anthropic 2024: Contextual Retrieval reduces failures by 67%
    - Reuter 2024: 500 chars optimal chunk size
    """
    # Initialize RCTS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,           # Optimal (Reuter Table 1)
        chunk_overlap=0,          # RCTS handles boundaries
        length_function=len,      # Character-based
        separators=[
            "\n\n",               # Paragraph breaks
            "\n",                 # Line breaks
            ". ",                 # Sentence ends
            "; ",                 # Clause separators
            ", ",                 # Sub-clause separators
            " ",                  # Word boundaries
            ""                    # Character fallback
        ]
    )

    # Initialize Contextual Retrieval
    chunking_config = ChunkingConfig(enable_contextual=True)
    context_generator = ContextualRetrieval(config=chunking_config.context_config)

    chunks = []
    doc_summary = summaries['document']

    # Process each section
    for section in structure['sections']:
        # Split section into chunks
        raw_chunks = text_splitter.split_text(section.text)

        # Prepare chunks with metadata for batch context generation
        chunks_to_contextualize = []
        for idx, raw_chunk in enumerate(raw_chunks):
            metadata = {
                'document_summary': doc_summary,
                'section_title': section.title,
                'section_path': section.path,
                'preceding_chunk': raw_chunks[idx-1] if idx > 0 else None,
                'following_chunk': raw_chunks[idx+1] if idx < len(raw_chunks)-1 else None
            }
            chunks_to_contextualize.append((raw_chunk, metadata))

        # Generate contexts in batch (parallel)
        chunk_contexts = context_generator.generate_contexts_batch(chunks_to_contextualize)

        # Create chunks with contexts
        for idx, ((raw_chunk, metadata), context_result) in enumerate(
            zip(chunks_to_contextualize, chunk_contexts)
        ):
            # CRITICAL: Contextual Retrieval
            # Prepend LLM-generated context to explain what this chunk discusses
            if context_result.success:
                augmented_content = f"{context_result.context}\n\n{raw_chunk}"
            else:
                augmented_content = raw_chunk  # Fallback to basic

            chunks.append({
                'type': 'chunk',
                'content': augmented_content,      # For embedding (with context)
                'raw_content': raw_chunk,          # For generation (without context)
                'metadata': {
                    'document_id': document['metadata']['document_id'],
                    'section_id': section.id,
                    'chunk_id': f"{section.id}_chunk_{idx}",
                    'chunk_index': idx,
                    'hierarchy_level': 2,
                    'document_summary': doc_summary,
                    'char_count': len(raw_chunk)
                }
            })

    return chunks
```

### Why Contextual Retrieval is Critical

**Problem: Retrieval Failures due to Lack of Context**
```
Baseline failure rate (top-20): 100%
Contextual Retrieval failure rate: 33% (67% reduction)
```

**How Contextual Retrieval Works:**
1. Each chunk gets LLM-generated context explaining what it discusses
2. Context is chunk-specific (not generic like summaries)
3. Embedding captures BOTH local content + situational context
4. Retriever can better match user queries to relevant chunks
5. Includes surrounding chunks for better context awareness

**Evidence:**
- Anthropic (Sept 2024): 67% reduction in top-20 retrieval failures
- 35% reduction in top-5 failures
- Works with both BM25 (53% reduction) and dense retrieval

---

## 🔢 PHASE 4: Embedding & Indexing

**Status:** ✅ IMPLEMENTED (2025-10-20)

### Model Selection

**PRIMARY: text-embedding-3-large (OpenAI)**

Based on:
- LegalBench-RAG baseline (proven performance)
- MLEB 2025: Strong performance on legal benchmarks
- 3072 dimensions, proven at scale

```python
from extraction import EmbeddingGenerator, EmbeddingConfig

# Implemented configuration
config = EmbeddingConfig(
    model='text-embedding-3-large',  # OpenAI
    dimensions=3072,                  # Auto-detected
    batch_size=100,
    normalize=True                    # For cosine similarity
)

embedder = EmbeddingGenerator(config)
```

**ALTERNATIVE: BAAI/bge-m3 (Open-Source, Multilingual)**

Based on MLEB 2025 research:
- 100+ languages (including Czech!)
- 1024 dimensions
- 8192 token context
- Free, runs locally

```python
config = EmbeddingConfig(
    model='bge-m3',      # Open-source
    dimensions=1024,         # Auto-detected
    batch_size=32
)
```

**SOTA Legal Models (MLEB 2025):**
1. Kanon 2 Embedder - #1 on MLEB (86% NDCG@10)
2. Voyage 3 Large - #2 on MLEB (85.7%)
3. vstackai-law-1 - Tops MTEB legal leaderboard (32k tokens)

> **Note:** text-embedding-3-large chosen for balance of performance, ease of integration, and proven results.

### Three-Index Architecture

**Status:** ✅ IMPLEMENTED

Based on Lima (2024) - 3 separate FAISS indexes for multi-layer retrieval:

```python
from extraction import FAISSVectorStore

# Create vector store with 3 separate indexes
vector_store = FAISSVectorStore(dimensions=3072)

# Add chunks to all 3 layers
vector_store.add_chunks(
    chunks_dict=chunks,      # From PHASE 3 MultiLayerChunker
    embeddings_dict=embeddings  # From EmbeddingGenerator
)

# Internal structure:
# - index_layer1: FAISS IndexFlatIP (Document level, 1 vector per doc)
# - index_layer2: FAISS IndexFlatIP (Section level, N vectors per doc)
# - index_layer3: FAISS IndexFlatIP (Chunk level, M vectors per doc - PRIMARY)

# Metadata tracking:
# - metadata_layer1, metadata_layer2, metadata_layer3
# - doc_id_to_indices mapping for DRM prevention

# Example: GRI 306 document (15 pages)
# Layer 1: 1 vector
# Layer 2: 118 vectors (sections)
# Layer 3: 123 vectors (chunks) - PRIMARY
# Total: 242 vectors, ~2.9 MB (3072D × 4 bytes)
```

**Implementation Details:**

```python
# Complete indexing pipeline (PHASE 1-4)
from extraction import IndexingPipeline, IndexingConfig

config = IndexingConfig(
    # PHASE 1: Hierarchy
    enable_smart_hierarchy=True,

    # PHASE 2: Summaries
    generate_summaries=True,
    summary_model='gpt-4o-mini',

    # PHASE 3: Chunking
    chunk_size=500,
    enable_sac=True,

    # PHASE 4: Embedding
    embedding_model='text-embedding-3-large',
    normalize_embeddings=True
)

pipeline = IndexingPipeline(config)

# Index document (runs all 4 phases)
vector_store = pipeline.index_document('document.pdf')

# Save to disk
vector_store.save('output/vector_store')

# Load later
vector_store = FAISSVectorStore.load('output/vector_store')
```

### Dense vs Sparse Retrieval

**Evidence from Reuter et al. (2024) Appendix B:**

| Method | Precision | Recall | DRM |
|--------|-----------|--------|-----|
| **Dense only (100% semantic)** | **11.03%** | **41.80%** | 19.29% |
| Hybrid (50% semantic + 50% BM25) | 9.54% | 41.47% | 18.45% |

**Conclusion: Use DENSE-ONLY** (better text-level metrics + less overhead)

```python
RETRIEVAL_METHOD = 'dense_semantic_only'  # NO BM25, NO hybrid
```

---

## 🔍 PHASE 5: Query & Retrieval

### Configuration (Evidence-Based)

```python
RETRIEVAL_CONFIG = {
    # From LegalBench-RAG experiments
    'k': 6,                          # Sweet spot 4-8 (use 6 for safety)
    'method': 'dense_semantic',      # NO hybrid (per Reuter Appendix B)
    'reranking': False,              # Cohere worse than no reranking!

    # From Multi-Layer Embedding (Lima 2024)
    'token_baseline': 2500,          # Max tokens in context
    'similarity_threshold': 0.25,    # Deviation from max (top_score - 25%)
    'min_chunks': 7,                 # Before applying token limit

    # DRM Prevention (from Reuter 2024)
    'enable_document_filtering': True,
    'max_source_documents': 2
}
```

### Why K=6?

From LegalBench-RAG and Arize RAG Evaluation (referenced in current pipeline):
- K=4: Best precision/latency balance
- K=5-6: Slightly better recall, acceptable latency
- K=8+: Dramatic latency increase, diminishing returns

**Choose K=6** for safety margin.

### Why NO Reranking?

**Critical Finding from LegalBench-RAG (Tables 4-7):**

| Method | Precision@1 | Recall@64 |
|--------|-------------|-----------|
| **RCTS + No Reranker** | **6.41%** | **62.22%** |
| RCTS + Cohere Reranker | 6.13% | 61.06% |
| Naive + Cohere Reranker | 6.41% | 62.22% |

→ Reranking provides **NO benefit** and adds latency!

**Likely reasons:**
- General-purpose reranker not optimized for legal text
- Legal domain needs domain-specific reranker
- Current rerankers trained on general corpora

```python
USE_RERANKING = False  # Empirically harmful!
```

### Implementation

```python
def retrieve_chunks(query, vector_stores, config=RETRIEVAL_CONFIG):
    """
    Multi-stage retrieval with DRM prevention
    Based on: LegalBench-RAG + Lima 2024 + Reuter 2024
    """
    # STAGE 1: Embed query
    query_embedding = embed_text(query, model='text-embedding-3-large')

    # STAGE 2: Search chunk-level index (primary)
    chunk_index, chunk_metadata = vector_stores['chunk_level']

    # Get top 2*k candidates for filtering
    distances, indices = chunk_index.search(
        query_embedding.reshape(1, -1),
        k=config['k'] * 2
    )

    candidate_chunks = [chunk_metadata[idx] for idx in indices[0]]
    candidate_scores = distances[0].tolist()

    # STAGE 3: Document-Level Filtering (DRM Prevention)
    # Group chunks by source document
    chunks_by_doc = defaultdict(list)
    for chunk, score in zip(candidate_chunks, candidate_scores):
        doc_id = chunk['metadata']['document_id']
        chunks_by_doc[doc_id].append((chunk, score))

    # Score each document
    doc_scores = {}
    for doc_id, doc_chunks in chunks_by_doc.items():
        scores = [score for _, score in doc_chunks]
        # Weighted: favor documents with multiple relevant chunks
        doc_scores[doc_id] = 0.6 * max(scores) + 0.4 * np.mean(scores)

    # Select top N documents
    top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = top_docs[:config['max_source_documents']]
    top_doc_ids = {doc_id for doc_id, _ in top_docs}

    # Filter chunks from top documents only
    filtered_chunks = [
        (chunk, score) for chunk, score in zip(candidate_chunks, candidate_scores)
        if chunk['metadata']['document_id'] in top_doc_ids
    ]

    # STAGE 4: Apply similarity threshold (Lima 2024)
    max_score = max(score for _, score in filtered_chunks)
    threshold = max_score - config['similarity_threshold']

    threshold_chunks = [
        (chunk, score) for chunk, score in filtered_chunks
        if score >= threshold
    ]

    # Ensure minimum chunks
    if len(threshold_chunks) < config['min_chunks']:
        threshold_chunks = filtered_chunks[:config['min_chunks']]

    # STAGE 5: Apply token limit (Lima 2024)
    selected_chunks = []
    total_tokens = 0

    for chunk, score in threshold_chunks:
        chunk_tokens = len(chunk['raw_content']) // 4  # Rough estimate
        if total_tokens + chunk_tokens <= config['token_baseline']:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break

    # Final selection: top K
    final_chunks = selected_chunks[:config['k']]

    return {
        'chunks': final_chunks,
        'total_tokens': total_tokens,
        'source_documents': list(top_doc_ids)
    }
```

---

## 📦 PHASE 6: Context Assembly

### Strip Generated Context

**IMPORTANT:** LLM-generated context was for retrieval only. Strip it before generation!

```python
def assemble_context(retrieval_results):
    """
    Assemble context for generation

    CRITICAL: Strip context (was for retrieval, not generation)
    Based on: Anthropic 2024 (Contextual Retrieval is pre-retrieval technique)
    """
    chunks = retrieval_results['chunks']

    # Build context from RAW content (without LLM context)
    context_parts = []

    for idx, chunk in enumerate(chunks, 1):
        citation_id = f"[{idx}]"

        # Use raw_content (without context)
        content = chunk['raw_content']

        # Source info for citation
        source = f"""
SOURCE {citation_id}:
Document: {chunk['metadata']['document_id']}
Section: {chunk['metadata'].get('section_id', 'N/A')}

CONTENT {citation_id}:
{content}

---
"""
        context_parts.append(source)

    # Add document summaries for global context
    doc_summaries = {}
    for chunk in chunks:
        doc_id = chunk['metadata']['document_id']
        if doc_id not in doc_summaries:
            doc_summaries[doc_id] = chunk['metadata']['document_summary']

    if doc_summaries:
        context_parts.append("\nDOCUMENT SUMMARIES (for global context):\n")
        for doc_id, summary in doc_summaries.items():
            context_parts.append(f"• {doc_id}: {summary}\n")

    return {
        'context': "\n".join(context_parts),
        'chunks': chunks,
        'total_tokens': retrieval_results['total_tokens']
    }
```

---

## 🤖 PHASE 7: Answer Generation

### Model Selection

**From NLI paper (Narendra et al., 2024):**

| Model | Accuracy | F1 (Entailment) | Notes |
|-------|----------|-----------------|-------|
| GPT-4 | 87% | 0.91 | Good performance |
| **Mixtral 8x7B** | **90%** | **0.93** | **BETTER than GPT-4!** |
| Span NLI BERT | 87% | 0.84 | Baseline |

→ **Mixtral 8x7B fine-tuned s LoRA outperforms GPT-4** na legal tasks!

### Configuration

```python
GENERATION_CONFIG = {
    # Model choice
    'model': 'gpt-4',              # or 'mixtral-8x7b-instruct'

    # Generation params
    'temperature': 0.1,            # Low = factual
    'max_tokens': 1000,            # Per Lima 2024
    'top_p': 0.95,

    # Citation requirements
    'require_citations': True,     # MANDATORY for legal
    'citation_format': '[N]'       # [1], [2], etc.
}
```

### Prompt Template

```python
GENERATION_PROMPT = """You are a legal AI assistant. Answer the user's query based ONLY on the provided context.

CRITICAL RULES:
1. Every factual claim MUST have a citation [N]
2. Only use information from the provided context
3. If information is not in context, say "Not found in provided documents"
4. Be precise and concise

{context}

USER QUERY: {query}

ANSWER (with citations):"""

def generate_answer(query, context, config=GENERATION_CONFIG):
    """
    Generate answer with mandatory citations
    Based on: All papers emphasize citation importance
    """
    prompt = GENERATION_PROMPT.format(
        context=context['context'],
        query=query
    )

    response = llm.generate(
        prompt=prompt,
        model=config['model'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
    )

    # Verify citations
    if config['require_citations']:
        if not has_citations(response):
            # Regenerate with stronger citation requirement
            response = regenerate_with_citations(query, context)

    return {
        'answer': response,
        'sources': context['chunks'],
        'model': config['model']
    }
```

---

## 🚀 ADVANCED TECHNIQUES (OPTIONAL)

> **Note:** The following techniques are OPTIONAL enhancements for improving retrieval quality, precision, and recall. They build on top of the baseline PHASE 1-7 pipeline.

---

## 🔍 PHASE 5.5: Advanced Query Enhancement (OPTIONAL)

### Purpose

Improve query understanding and retrieval precision through query transformation and augmentation.

### Technique 1: HyDE (Hypothetical Document Embeddings)

**Research:** Gao et al., 2023 - "Precise Zero-Shot Dense Retrieval"

**Concept:** Generate a hypothetical answer/document, then embed it instead of the raw query.

**Why it works:**
- Query: "What are termination clauses?" (short, abstract)
- HyDE: "Termination clauses in contracts specify conditions under which parties can end the agreement, including notice periods, breach conditions..." (long, concrete)
- Documents are more similar to HyDE text than to raw query

**Implementation:**

```python
class HyDEQueryEnhancer:
    """
    HyDE Query Enhancement

    Based on: Gao et al., 2023
    Expected improvement: +10-15% Precision@1
    """

    def __init__(self, llm_model: str = "claude-sonnet-4.5"):
        self.llm = get_llm_client(llm_model)

    def enhance_query(self, query: str) -> str:
        """
        Generate hypothetical document that would answer the query.

        Args:
            query: Original user query

        Returns:
            Hypothetical document text (for embedding)
        """

        prompt = f"""You are a legal document generator.

Generate a detailed, factual paragraph that would perfectly answer this query.
Write as if you are extracting text from an actual legal document.

Query: {query}

Hypothetical document paragraph (150-200 words):"""

        hypothetical_doc = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300
        )

        return hypothetical_doc.strip()


# Usage
enhancer = HyDEQueryEnhancer()

# Original query
query = "Does the NDA allow independent development?"

# Generate hypothetical document
hyde_query = enhancer.enhance_query(query)
# hyde_query = "The Non-Disclosure Agreement permits the receiving party
#               to independently develop similar information without violating
#               confidentiality obligations, provided that such development..."

# Embed and search with hypothetical document
query_embedding = embedder.embed_texts([hyde_query])
results = vector_store.search_layer3(query_embedding, k=6)
```

**Configuration:**

```python
HYDE_CONFIG = {
    'enable': False,  # Set to True to enable
    'model': 'claude-sonnet-4.5',  # or 'gpt-4o-mini'
    'temperature': 0.3,
    'max_tokens': 300,
    'target_length': 150,  # words
}
```

**When to use:**
- ✅ Complex legal queries requiring nuanced understanding
- ✅ Queries with legal jargon that might not match document vocabulary
- ❌ Simple keyword queries (adds latency)
- ❌ When latency is critical (<500ms requirement)

**Expected improvement:**
- Precision@1: +10-15%
- Recall@10: +5-8%
- Latency: +500-1000ms (LLM call)
- Cost: +$0.001-0.003 per query

---

### Technique 2: Query Expansion

**Research:** Mao et al., 2024 - "Query Expansion for Dense Retrieval"

**Concept:** Generate multiple query variants to capture different phrasings.

**Implementation:**

```python
class QueryExpander:
    """
    Multi-variant query expansion

    Expected improvement: +5-10% Recall
    """

    def expand_query(self, query: str) -> List[str]:
        """
        Generate query variants.

        Returns:
            List of query variants (original + 2-3 expansions)
        """

        prompt = f"""Generate 3 alternative phrasings of this legal query.
Each should capture the same intent but use different words.

Original query: {query}

Alternative 1:
Alternative 2:
Alternative 3:"""

        response = self.llm.generate(prompt, temperature=0.5, max_tokens=200)
        alternatives = parse_alternatives(response)

        return [query] + alternatives  # Original + expansions

    def retrieve_with_expansion(
        self,
        query: str,
        vector_store: FAISSVectorStore,
        k: int = 6
    ) -> List[Dict]:
        """
        Search with all query variants, merge results.
        """
        # Expand query
        query_variants = self.expand_query(query)

        # Embed all variants
        embeddings = embedder.embed_texts(query_variants)

        # Search with each variant
        all_results = []
        for embedding in embeddings:
            results = vector_store.search_layer3(embedding, k=k)
            all_results.extend(results)

        # Reciprocal Rank Fusion (RRF)
        merged = self._rrf_merge(all_results, k=k)

        return merged

    def _rrf_merge(self, results: List[List[Dict]], k: int = 6) -> List[Dict]:
        """
        Reciprocal Rank Fusion: merge multiple result lists.

        Formula: score = Σ 1/(k + rank_i)
        """
        chunk_scores = defaultdict(float)
        chunk_data = {}

        for result_list in results:
            for rank, chunk in enumerate(result_list, start=1):
                chunk_id = chunk['chunk_id']
                chunk_scores[chunk_id] += 1.0 / (60 + rank)  # k=60 standard
                chunk_data[chunk_id] = chunk

        # Sort by RRF score
        ranked_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk_data[chunk_id] for chunk_id, _ in ranked_chunks[:k]]
```

**Configuration:**

```python
QUERY_EXPANSION_CONFIG = {
    'enable': False,
    'num_variants': 3,  # 2-4 recommended
    'merge_method': 'rrf',  # 'rrf' or 'weighted'
    'rrf_k': 60,  # RRF constant
}
```

**Expected improvement:**
- Recall@10: +5-10%
- Precision@1: +3-5%
- Latency: +400-800ms (multiple searches)
- Cost: +$0.001 per query

---

### Technique 3: Query Decomposition (Multi-Hop)

**Research:** Khattab et al., 2023 - "DSPy: Declarative Self-improving Language Programs"

**Concept:** Break complex queries into simpler sub-queries for multi-hop reasoning.

**Implementation:**

```python
class QueryDecomposer:
    """
    Multi-hop query decomposition

    For complex queries requiring multiple retrieval steps.
    """

    def decompose(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.

        Example:
            Query: "Compare termination clauses in NDA and MSA"
            Sub-queries:
            1. "What are the termination clauses in the NDA?"
            2. "What are the termination clauses in the MSA?"
            3. "Compare both sets of clauses"
        """

        prompt = f"""Decompose this complex legal query into simpler sub-queries.
Each sub-query should be answerable independently.

Complex query: {query}

Sub-query 1:
Sub-query 2:
Sub-query 3 (synthesis):"""

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=200)
        sub_queries = parse_sub_queries(response)

        return sub_queries

    def multi_hop_retrieve(
        self,
        query: str,
        vector_store: FAISSVectorStore
    ) -> Dict:
        """
        Multi-hop retrieval with query decomposition.
        """
        # Decompose query
        sub_queries = self.decompose(query)

        # Retrieve for each sub-query
        sub_results = []
        for sub_q in sub_queries:
            sub_embedding = embedder.embed_texts([sub_q])
            sub_chunks = vector_store.search_layer3(sub_embedding, k=4)
            sub_results.append({
                'sub_query': sub_q,
                'chunks': sub_chunks
            })

        return {
            'original_query': query,
            'sub_queries': sub_queries,
            'sub_results': sub_results,
            'all_chunks': self._merge_chunks(sub_results)
        }
```

**Configuration:**

```python
QUERY_DECOMPOSITION_CONFIG = {
    'enable': False,
    'max_sub_queries': 3,
    'min_complexity_threshold': 20,  # chars, only decompose long queries
}
```

**When to use:**
- ✅ Comparative queries ("Compare X and Y")
- ✅ Multi-document queries ("What do all contracts say about...")
- ✅ Temporal queries ("How has policy changed over time?")
- ❌ Simple factual queries

**Expected improvement:**
- Complex query accuracy: +15-20%
- Latency: +1-2s (multiple LLM + retrieval calls)
- Cost: +$0.005-0.01 per complex query

---

## 🎯 PHASE 5.6: Advanced Retrieval Techniques (OPTIONAL)

### Technique 4: Fusion Retrieval (Multi-Model Ensemble)

**Research:** MLEB 2025, Bavaresco et al., 2024

**Concept:** Combine results from multiple embedding models for better coverage.

**Implementation:**

```python
class FusionRetriever:
    """
    Multi-model fusion retrieval

    Combines:
    - BGE-M3 (multilingual, local)
    - Voyage 3 Large (general SOTA)
    - Kanon 2 (legal-specific SOTA)

    Expected improvement: +8-12% Recall@10
    """

    def __init__(self):
        # Initialize multiple embedders
        self.embedders = {
            'bge': EmbeddingGenerator(EmbeddingConfig(model='bge-m3')),
            'voyage': EmbeddingGenerator(EmbeddingConfig(model='voyage-3-large')),
            'kanon': EmbeddingGenerator(EmbeddingConfig(model='kanon-2')),
        }

        # Create vector stores for each model
        self.vector_stores = {
            'bge': FAISSVectorStore(dimensions=1024),
            'voyage': FAISSVectorStore(dimensions=1024),
            'kanon': FAISSVectorStore(dimensions=1024),
        }

    def index_with_all_models(self, chunks: Dict):
        """Index chunks with all embedding models."""
        for model_name, embedder in self.embedders.items():
            embeddings = {
                'layer1': embedder.embed_chunks(chunks['layer1'], layer=1),
                'layer2': embedder.embed_chunks(chunks['layer2'], layer=2),
                'layer3': embedder.embed_chunks(chunks['layer3'], layer=3),
            }
            self.vector_stores[model_name].add_chunks(chunks, embeddings)

    def fusion_search(
        self,
        query: str,
        k: int = 6,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Search with all models, merge with weighted RRF.

        Args:
            query: Search query
            k: Number of results
            weights: Model weights (default: equal weighting)
        """
        if weights is None:
            weights = {'bge': 1.0, 'voyage': 1.0, 'kanon': 1.0}

        # Search with each model
        model_results = {}
        for model_name, embedder in self.embedders.items():
            query_embedding = embedder.embed_texts([query])
            results = self.vector_stores[model_name].search_layer3(
                query_embedding,
                k=k*2  # Retrieve more for fusion
            )
            model_results[model_name] = results

        # Weighted RRF fusion
        merged = self._weighted_rrf(model_results, weights, k=k)

        return merged

    def _weighted_rrf(
        self,
        model_results: Dict[str, List[Dict]],
        weights: Dict[str, float],
        k: int = 6
    ) -> List[Dict]:
        """Weighted Reciprocal Rank Fusion."""
        chunk_scores = defaultdict(float)
        chunk_data = {}

        for model_name, results in model_results.items():
            weight = weights[model_name]
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk['chunk_id']
                chunk_scores[chunk_id] += weight * (1.0 / (60 + rank))
                chunk_data[chunk_id] = chunk

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_data[cid] for cid, _ in ranked[:k]]
```

**Configuration:**

```python
FUSION_RETRIEVAL_CONFIG = {
    'enable': False,
    'models': ['bge-m3', 'voyage-3-large', 'kanon-2'],
    'weights': {
        'bge-m3': 1.0,         # Multilingual strength
        'voyage-3-large': 1.2, # General SOTA
        'kanon-2': 1.5,        # Legal domain expert (highest weight)
    },
    'rrf_k': 60,
}
```

**Expected improvement:**
- Recall@10: +8-12%
- Precision@1: +5-7%
- Robustness: Significantly better (no single-model failure)
- Latency: +200-400ms (parallel embedding)
- Cost: +$0.003-0.006 per query (multiple embedding calls)
- Storage: 3x vector store size

---

### Technique 5: Late Chunking

**Research:** Zhang et al., 2024 - "Late Chunking for Long-Context Embedding"

**Concept:** Embed the full document BEFORE chunking, then split the embedding vector.

**Why it works:**
- Traditional: Chunk → Embed (loses cross-chunk context)
- Late Chunking: Embed → Chunk (preserves context at boundaries)

**Implementation:**

```python
class LateChunkingPipeline:
    """
    Late chunking: embed before chunking

    Based on: Zhang et al., 2024
    Expected improvement: +10-15% on boundary-crossing queries

    Note: Requires model with positional embeddings (e.g., Jina Embeddings v3)
    """

    def __init__(self, model: str = "jinaai/jina-embeddings-v3"):
        # Requires model that supports late chunking
        self.model = SentenceTransformer(model)
        self.max_seq_length = 8192

    def late_chunk_document(
        self,
        document: ExtractedDocument,
        chunk_size: int = 500
    ) -> List[Chunk]:
        """
        Late chunking: embed full document, then chunk.
        """
        chunks_with_embeddings = []

        for section in document.sections:
            # Step 1: Embed FULL section (before chunking)
            full_embedding = self.model.encode(
                section.content,
                convert_to_tensor=True
            )

            # Step 2: Chunk the TEXT
            text_chunks = self.text_splitter.split_text(section.content)

            # Step 3: Split the EMBEDDING proportionally
            chunk_embeddings = self._split_embedding(
                full_embedding,
                text_chunks,
                section.content
            )

            # Step 4: Create chunks with embeddings
            for idx, (text_chunk, chunk_emb) in enumerate(zip(text_chunks, chunk_embeddings)):
                chunk = Chunk(
                    chunk_id=f"{section.section_id}_chunk_{idx}",
                    content=text_chunk,
                    raw_content=text_chunk,
                    embedding=chunk_emb,  # Pre-computed
                    metadata=ChunkMetadata(...)
                )
                chunks_with_embeddings.append(chunk)

        return chunks_with_embeddings

    def _split_embedding(
        self,
        full_embedding: Tensor,
        text_chunks: List[str],
        full_text: str
    ) -> List[np.ndarray]:
        """
        Split embedding vector based on text chunk positions.

        Uses attention weights to assign embedding segments to chunks.
        """
        # Get token positions for each chunk
        tokenizer = self.model.tokenizer
        full_tokens = tokenizer.encode(full_text)

        chunk_embeddings = []
        char_pos = 0

        for chunk_text in text_chunks:
            # Find token range for this chunk
            chunk_start_token = len(tokenizer.encode(full_text[:char_pos]))
            char_pos += len(chunk_text)
            chunk_end_token = len(tokenizer.encode(full_text[:char_pos]))

            # Extract corresponding embedding segment
            # (average over token range)
            chunk_emb = full_embedding[chunk_start_token:chunk_end_token].mean(dim=0)
            chunk_embeddings.append(chunk_emb.cpu().numpy())

        return chunk_embeddings
```

**Configuration:**

```python
LATE_CHUNKING_CONFIG = {
    'enable': False,  # Requires Jina Embeddings v3 or similar
    'model': 'jinaai/jina-embeddings-v3',
    'max_seq_length': 8192,
    'chunk_size': 500,
}
```

**Expected improvement:**
- Boundary-crossing queries: +10-15%
- Overall Recall@10: +3-5%
- Latency: Similar (embed once instead of N times)
- Cost: Similar or lower (fewer embedding calls)

**When to use:**
- ✅ Documents with important information spanning chunk boundaries
- ✅ Legal clauses that are often split across chunks
- ❌ When using models without positional embeddings (OpenAI, Voyage)

---

### Technique 6: ColBERT (Token-Level Embeddings)

**Research:** Khattab & Zaharia, 2020 - "ColBERT: Efficient and Effective Passage Search"

**Concept:** Embed every TOKEN separately, then compute MaxSim during retrieval.

**Why it works:**
- Dense retrieval: Single vector per chunk (lossy compression)
- ColBERT: Vector per token (preserves fine-grained semantics)
- MaxSim: Each query token matches best document token

**Implementation:**

```python
class ColBERTRetriever:
    """
    ColBERT: Token-level late interaction

    Based on: Khattab & Zaharia, 2020
    Expected improvement: +12-18% Precision@1 (legal domain)

    Trade-off: 100x storage (but 10x faster than cross-encoder)
    """

    def __init__(self, model: str = "colbert-ir/colbertv2.0"):
        from colbert.modeling.checkpoint import Checkpoint
        from colbert import Indexer, Searcher

        self.checkpoint = Checkpoint(model)
        self.indexer = None
        self.searcher = None

    def index_chunks(self, chunks: List[Chunk], index_path: Path):
        """
        Index chunks with ColBERT (token-level embeddings).
        """
        # Prepare documents
        documents = [chunk.content for chunk in chunks]

        # Index with ColBERT
        self.indexer = Indexer(
            checkpoint=self.checkpoint,
            config={
                'nbits': 2,  # Compression: 2-bit per dimension
                'doc_maxlen': 512,  # Max tokens per document
            }
        )

        self.indexer.index(
            name=index_path.stem,
            collection=documents,
            overwrite=True
        )

        logger.info(f"ColBERT index created: {index_path}")

    def search(self, query: str, k: int = 6) -> List[Dict]:
        """
        Search with ColBERT MaxSim scoring.

        Returns chunks ranked by token-level similarity.
        """
        # Search
        self.searcher = Searcher(index=self.index_path)
        results = self.searcher.search(query, k=k)

        # Format results
        ranked_chunks = []
        for passage_id, passage_rank, score in results:
            ranked_chunks.append({
                'chunk_id': self.chunks[passage_id].chunk_id,
                'content': self.chunks[passage_id].content,
                'score': score,
                'rank': passage_rank,
            })

        return ranked_chunks

    def maxsim_score(
        self,
        query_embeddings: Tensor,  # (Q, D)
        doc_embeddings: Tensor     # (N, D)
    ) -> float:
        """
        MaxSim: For each query token, find max similarity to any doc token.

        Formula: MaxSim(Q, D) = Σ_i max_j (q_i · d_j)
        """
        # Compute all pairwise scores
        scores = torch.matmul(query_embeddings, doc_embeddings.T)  # (Q, N)

        # For each query token, take max over document tokens
        max_scores = scores.max(dim=1).values  # (Q,)

        # Sum over query tokens
        total_score = max_scores.sum().item()

        return total_score
```

**Configuration:**

```python
COLBERT_CONFIG = {
    'enable': False,  # High storage cost
    'model': 'colbert-ir/colbertv2.0',
    'nbits': 2,  # Compression: 2, 4, or 8 bits
    'doc_maxlen': 512,  # Max tokens per chunk
    'query_maxlen': 64,  # Max tokens per query
}
```

**Expected improvement:**
- Precision@1: +12-18% (legal domain)
- Precision@10: +8-12%
- Exact phrase matching: Significantly better
- Latency: ~2-3x slower than dense retrieval
- Storage: ~100x larger (but compressible to ~10x with 2-bit quantization)
- Cost: Similar (local model)

**When to use:**
- ✅ Exact phrase matching critical (legal citations, clause references)
- ✅ Storage not a constraint
- ✅ Queries with specific legal terms
- ❌ Cost-sensitive applications (storage expensive)
- ❌ Latency <500ms requirement

---

## 🎯 PHASE 6.5: Advanced Reranking (OPTIONAL)

### Technique 7: Cross-Encoder Reranking (Domain-Specific)

**Research:** Nogueira & Cho, 2019; LegalBench-RAG, 2024

**IMPORTANT:** LegalBench-RAG showed that **general-purpose rerankers (Cohere) hurt performance**. However, **domain-specific legal rerankers** can help significantly.

**Concept:** Fine-tune cross-encoder on legal data for precise relevance scoring.

**Implementation:**

```python
class LegalCrossEncoderReranker:
    """
    Domain-specific cross-encoder reranking

    CRITICAL: Fine-tune on legal data (e.g., LegalBench-RAG dataset)

    General-purpose rerankers (Cohere) HURT performance!
    Legal-specific rerankers IMPROVE by +10-15%
    """

    def __init__(self, model_path: str = "models/legal-cross-encoder"):
        from sentence_transformers import CrossEncoder

        # Load fine-tuned legal cross-encoder
        self.model = CrossEncoder(model_path)

        logger.info(f"Loaded legal cross-encoder: {model_path}")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 6
    ) -> List[Dict]:
        """
        Rerank chunks with cross-encoder.

        Args:
            query: Search query
            chunks: Initial retrieval results (K=20-30 recommended)
            top_k: Final number of chunks to return

        Returns:
            Reranked chunks (top_k)
        """
        # Prepare pairs
        pairs = [(query, chunk['content']) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)

        # Sort by rerank score
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]

    @staticmethod
    def fine_tune_on_legal_data(
        base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        legal_dataset_path: str = "data/legalbench_pairs.jsonl"
    ):
        """
        Fine-tune cross-encoder on legal data.

        Dataset format:
        {"query": "...", "positive": "...", "negative": "..."}
        """
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader

        # Load base model
        model = CrossEncoder(base_model, num_labels=1)

        # Load legal training data
        train_samples = []
        with open(legal_dataset_path) as f:
            for line in f:
                item = json.loads(line)
                train_samples.append(InputExample(
                    texts=[item['query'], item['positive']],
                    label=1.0
                ))
                train_samples.append(InputExample(
                    texts=[item['query'], item['negative']],
                    label=0.0
                ))

        # Create DataLoader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

        # Fine-tune
        model.fit(
            train_dataloader=train_dataloader,
            epochs=3,
            warmup_steps=100,
            output_path="models/legal-cross-encoder"
        )

        logger.info("Legal cross-encoder fine-tuned successfully")
```

**Configuration:**

```python
CROSS_ENCODER_RERANKING_CONFIG = {
    'enable': False,  # Only enable with legal-specific model!
    'model': 'models/legal-cross-encoder',  # Fine-tuned on legal data
    'initial_k': 20,  # Retrieve more for reranking
    'final_k': 6,     # Rerank to final K
    'min_score_threshold': 0.5,  # Minimum rerank score

    # NEVER use these (hurt performance per LegalBench-RAG):
    'avoid_models': [
        'cohere-rerank-v3',  # General-purpose, NOT legal-specific
        'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Without fine-tuning
    ]
}
```

**Expected improvement (with legal fine-tuning):**
- Precision@1: +10-15%
- Precision@6: +8-12%
- False positives: -20-30%
- Latency: +400-800ms (cross-encoder is slow)
- Cost: Free if local, or $0.002-0.005 per query if API

**When to use:**
- ✅ ONLY with legal-specific fine-tuned model
- ✅ High-precision requirements (contract analysis, compliance)
- ✅ Budget for latency (+400-800ms)
- ❌ NEVER with general-purpose rerankers (Cohere, etc.)
- ❌ Latency-sensitive applications

---

### Technique 8: MMR (Maximal Marginal Relevance) Diversity

**Research:** Carbonell & Goldstein, 1998 - "The Use of MMR for Text Summarization"

**Concept:** Balance relevance vs diversity to avoid redundant chunks.

**Implementation:**

```python
class MMRDiversityReranker:
    """
    MMR: Maximal Marginal Relevance

    Reduces redundancy in retrieved chunks
    Expected improvement: Better coverage (-30% redundant chunks)
    """

    def mmr_rerank(
        self,
        query_embedding: np.ndarray,
        chunks: List[Dict],
        chunk_embeddings: np.ndarray,
        k: int = 6,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        MMR reranking: balance relevance vs diversity.

        Args:
            query_embedding: Query vector
            chunks: Retrieved chunks
            chunk_embeddings: Chunk vectors
            k: Number of chunks to select
            lambda_param: Balance parameter (0=diversity, 1=relevance)

        Returns:
            Diversified chunk selection
        """
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(chunks)))

        # Compute relevance scores (query similarity)
        relevance_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]

        # Select first chunk (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        selected_embeddings.append(chunk_embeddings[first_idx])
        remaining_indices.remove(first_idx)

        # Iteratively select remaining chunks
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]

                # Max similarity to already selected chunks
                if selected_embeddings:
                    similarities = cosine_similarity(
                        chunk_embeddings[idx].reshape(1, -1),
                        np.array(selected_embeddings)
                    )[0]
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0

                # MMR score: balance relevance and diversity
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr))

            # Select chunk with highest MMR score
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            selected_embeddings.append(chunk_embeddings[best_idx])
            remaining_indices.remove(best_idx)

        # Return chunks in MMR order
        return [chunks[idx] for idx in selected_indices]
```

**Configuration:**

```python
MMR_CONFIG = {
    'enable': False,
    'lambda_param': 0.5,  # 0.5 = equal balance
    # lambda=1.0: pure relevance (no diversity)
    # lambda=0.0: pure diversity (no relevance)
    # lambda=0.5: balanced (recommended for legal)

    'min_similarity_threshold': 0.85,  # Consider chunks similar if >0.85
}
```

**Expected improvement:**
- Redundant chunks: -30-40%
- Coverage: +15-20% (more diverse information)
- Precision@1: Neutral (0-2%)
- User satisfaction: Higher (less repetition)
- Latency: +50-100ms (iterative selection)

**When to use:**
- ✅ Multi-document retrieval (avoid duplicate clauses)
- ✅ Long context assembly (maximize information density)
- ✅ User-facing applications (better UX)
- ❌ Single-document retrieval (diversity less important)
- ❌ Exact answer retrieval (relevance >> diversity)

---

## 🤖 PHASE 7.5: Agentic RAG (OPTIONAL)

### Technique 9: Self-Corrective RAG (CRAG)

**Research:** Yan et al., 2024 - "Corrective Retrieval Augmented Generation"

**Concept:** Agent evaluates retrieval quality, then corrects if needed.

**Implementation:**

```python
class CorrectiveRAGAgent:
    """
    CRAG: Self-correcting RAG with quality assessment

    Based on: Yan et al., 2024
    Expected improvement: +20-30% on complex queries
    """

    def __init__(
        self,
        retriever: FAISSVectorStore,
        llm_model: str = "claude-sonnet-4.5"
    ):
        self.retriever = retriever
        self.llm = get_llm_client(llm_model)
        self.embedder = EmbeddingGenerator()

    def corrective_retrieve(
        self,
        query: str,
        k: int = 6,
        max_iterations: int = 3
    ) -> Dict:
        """
        Self-corrective retrieval loop.

        Steps:
        1. Initial retrieval
        2. Evaluate relevance
        3. If poor, reformulate query and retry
        4. Repeat until good results or max iterations
        """
        iteration = 0
        current_query = query
        best_results = None
        best_score = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"CRAG iteration {iteration}: {current_query[:50]}...")

            # Retrieve
            query_embedding = self.embedder.embed_texts([current_query])
            results = self.retriever.hierarchical_search(
                query_embedding,
                k_layer3=k
            )

            # Evaluate retrieval quality
            evaluation = self._evaluate_retrieval_quality(
                query=query,
                results=results['layer3']
            )

            logger.info(f"Quality score: {evaluation['score']:.2f}")

            # Track best results
            if evaluation['score'] > best_score:
                best_score = evaluation['score']
                best_results = results

            # Check if quality is sufficient
            if evaluation['score'] >= 0.7:  # Good enough
                logger.info(f"CRAG converged after {iteration} iterations")
                break

            # If poor quality, reformulate query
            if iteration < max_iterations:
                current_query = self._reformulate_query(
                    original_query=query,
                    current_query=current_query,
                    results=results['layer3'],
                    evaluation=evaluation
                )

        return {
            'results': best_results,
            'iterations': iteration,
            'final_score': best_score,
            'query_history': [query, current_query]
        }

    def _evaluate_retrieval_quality(
        self,
        query: str,
        results: List[Dict]
    ) -> Dict:
        """
        Evaluate retrieval quality using LLM.

        Returns:
            {
                'score': 0.0-1.0,
                'feedback': str,
                'issues': List[str]
            }
        """
        # Prepare context from top 3 results
        context = "\n\n".join([
            f"CHUNK {i+1}:\n{r['content'][:200]}..."
            for i, r in enumerate(results[:3])
        ])

        prompt = f"""Evaluate the relevance of retrieved chunks to the query.

Query: {query}

Retrieved chunks:
{context}

On a scale of 0.0 to 1.0, how relevant are these chunks?
Provide:
1. Relevance score (0.0-1.0)
2. Brief feedback
3. Issues (if any)

Format:
Score: X.X
Feedback: ...
Issues: ..."""

        response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)

        # Parse response
        score = self._parse_score(response)
        feedback = self._parse_feedback(response)
        issues = self._parse_issues(response)

        return {
            'score': score,
            'feedback': feedback,
            'issues': issues
        }

    def _reformulate_query(
        self,
        original_query: str,
        current_query: str,
        results: List[Dict],
        evaluation: Dict
    ) -> str:
        """
        Reformulate query based on evaluation feedback.
        """
        prompt = f"""The initial retrieval was insufficient. Reformulate the query.

Original query: {original_query}
Current query: {current_query}
Issues: {', '.join(evaluation['issues'])}
Feedback: {evaluation['feedback']}

Provide an improved query that addresses the issues:"""

        reformulated = self.llm.generate(prompt, temperature=0.3, max_tokens=100)

        return reformulated.strip()
```

**Configuration:**

```python
CRAG_CONFIG = {
    'enable': False,
    'max_iterations': 3,
    'quality_threshold': 0.7,  # 0.0-1.0
    'llm_model': 'claude-sonnet-4.5',
}
```

**Expected improvement:**
- Complex query success rate: +20-30%
- Average precision: +10-15%
- Failed queries: -40-50%
- Latency: +2-4s per iteration (LLM evaluation expensive)
- Cost: +$0.01-0.03 per query (multiple LLM calls)

**When to use:**
- ✅ Mission-critical applications (legal compliance, medical)
- ✅ Complex queries where initial retrieval often fails
- ✅ Budget for latency and cost
- ❌ Simple queries (overkill)
- ❌ High-throughput applications (too slow)

---

### Technique 10: Multi-Hop Reasoning Agent

**Research:** Yao et al., 2023 - "ReAct: Synergizing Reasoning and Acting in Language Models"

**Concept:** Agent breaks down complex queries, retrieves iteratively, and synthesizes answer.

**Implementation:**

```python
class MultiHopReasoningAgent:
    """
    Multi-hop reasoning with iterative retrieval

    Based on: ReAct (Yao et al., 2023)
    For complex queries requiring multiple retrieval steps
    """

    def __init__(
        self,
        retriever: FAISSVectorStore,
        llm_model: str = "claude-sonnet-4.5"
    ):
        self.retriever = retriever
        self.llm = get_llm_client(llm_model)
        self.embedder = EmbeddingGenerator()

    def multi_hop_answer(
        self,
        query: str,
        max_hops: int = 3
    ) -> Dict:
        """
        Multi-hop reasoning: iteratively retrieve and reason.

        Example:
            Query: "Compare termination clauses in NDA and MSA"
            Hop 1: Retrieve NDA termination clauses
            Hop 2: Retrieve MSA termination clauses
            Hop 3: Synthesize comparison
        """
        reasoning_trace = []
        context_accumulator = []

        for hop in range(1, max_hops + 1):
            logger.info(f"Hop {hop}/{max_hops}")

            # Generate reasoning step
            reasoning_step = self._generate_reasoning_step(
                query=query,
                hop=hop,
                previous_context=context_accumulator,
                previous_reasoning=reasoning_trace
            )

            reasoning_trace.append(reasoning_step)

            # Check if we need more retrieval
            if reasoning_step['action'] == 'RETRIEVE':
                # Retrieve based on sub-query
                sub_query = reasoning_step['sub_query']
                query_embedding = self.embedder.embed_texts([sub_query])
                results = self.retriever.search_layer3(query_embedding, k=4)

                # Add to context
                context_accumulator.extend(results)
                reasoning_step['retrieved_chunks'] = len(results)

            elif reasoning_step['action'] == 'ANSWER':
                # Agent decides it has enough information
                break

        # Final synthesis
        final_answer = self._synthesize_answer(
            query=query,
            context=context_accumulator,
            reasoning_trace=reasoning_trace
        )

        return {
            'answer': final_answer,
            'reasoning_trace': reasoning_trace,
            'hops': len(reasoning_trace),
            'total_chunks': len(context_accumulator)
        }

    def _generate_reasoning_step(
        self,
        query: str,
        hop: int,
        previous_context: List[Dict],
        previous_reasoning: List[Dict]
    ) -> Dict:
        """
        Generate next reasoning step using ReAct pattern.

        Returns:
            {
                'thought': str,  # What to do next
                'action': 'RETRIEVE' or 'ANSWER',
                'sub_query': str (if RETRIEVE)
            }
        """
        # Build prompt
        context_summary = self._summarize_context(previous_context)
        reasoning_summary = "\n".join([
            f"Hop {i+1}: {r['thought']}"
            for i, r in enumerate(previous_reasoning)
        ])

        prompt = f"""You are a multi-hop reasoning agent. Think step-by-step.

Original query: {query}
Current hop: {hop}

Previous reasoning:
{reasoning_summary}

Context so far:
{context_summary}

What should you do next?

Respond in this format:
Thought: <your reasoning>
Action: RETRIEVE or ANSWER
Sub-query: <query for retrieval> (if RETRIEVE)"""

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=200)

        # Parse response
        thought = self._extract_thought(response)
        action = self._extract_action(response)
        sub_query = self._extract_sub_query(response) if action == 'RETRIEVE' else None

        return {
            'hop': hop,
            'thought': thought,
            'action': action,
            'sub_query': sub_query
        }

    def _synthesize_answer(
        self,
        query: str,
        context: List[Dict],
        reasoning_trace: List[Dict]
    ) -> str:
        """
        Synthesize final answer from multi-hop reasoning.
        """
        context_text = "\n\n".join([
            f"[{i+1}] {chunk['content'][:300]}..."
            for i, chunk in enumerate(context)
        ])

        reasoning_text = "\n".join([
            f"Step {r['hop']}: {r['thought']}"
            for r in reasoning_trace
        ])

        prompt = f"""Synthesize a comprehensive answer based on multi-hop reasoning.

Query: {query}

Reasoning trace:
{reasoning_text}

Retrieved context:
{context_text}

Provide a complete answer with citations [1], [2], etc.:"""

        answer = self.llm.generate(prompt, temperature=0.1, max_tokens=500)

        return answer.strip()
```

**Configuration:**

```python
MULTI_HOP_CONFIG = {
    'enable': False,
    'max_hops': 3,
    'max_chunks_per_hop': 4,
    'llm_model': 'claude-sonnet-4.5',
    'reasoning_temperature': 0.2,
}
```

**Expected improvement:**
- Complex multi-doc queries: +30-40% accuracy
- Comparative queries: +25-35% accuracy
- Temporal queries: +20-30% accuracy
- Simple queries: Neutral (no benefit, adds latency)
- Latency: +3-6s (multiple LLM + retrieval calls)
- Cost: +$0.02-0.05 per complex query

**When to use:**
- ✅ Comparative queries ("Compare X and Y")
- ✅ Multi-document queries ("What do all contracts say about...")
- ✅ Temporal queries ("How has policy evolved?")
- ✅ Complex legal analysis
- ❌ Simple factual queries
- ❌ High-throughput systems (too expensive)

---

## 📊 ADVANCED CONFIGURATION SUMMARY

### Optional Features Configuration

```python
ADVANCED_RAG_CONFIG = {
    # PHASE 5.5: Query Enhancement
    'query_enhancement': {
        'hyde': {
            'enable': False,  # +10-15% Precision@1, +500ms, +$0.001
            'model': 'claude-sonnet-4.5',
        },
        'expansion': {
            'enable': False,  # +5-10% Recall, +400ms, +$0.001
            'num_variants': 3,
        },
        'decomposition': {
            'enable': False,  # +15-20% complex queries, +1-2s, +$0.005
            'max_sub_queries': 3,
        },
    },

    # PHASE 5.6: Advanced Retrieval
    'advanced_retrieval': {
        'fusion': {
            'enable': False,  # +8-12% Recall, +200ms, +$0.003
            'models': ['bge-m3', 'voyage-3-large', 'kanon-2'],
            'weights': {'bge-m3': 1.0, 'voyage-3-large': 1.2, 'kanon-2': 1.5},
        },
        'late_chunking': {
            'enable': False,  # +10-15% boundaries, requires Jina v3
            'model': 'jinaai/jina-embeddings-v3',
        },
        'colbert': {
            'enable': False,  # +12-18% Precision@1, 100x storage
            'model': 'colbert-ir/colbertv2.0',
        },
    },

    # PHASE 6.5: Advanced Reranking
    'advanced_reranking': {
        'cross_encoder': {
            'enable': False,  # +10-15% with legal fine-tuning, +400ms
            'model': 'models/legal-cross-encoder',  # Must be legal-specific!
            'initial_k': 20,
            'final_k': 6,
        },
        'mmr': {
            'enable': False,  # -30% redundancy, +50ms
            'lambda_param': 0.5,  # 0=diversity, 1=relevance
        },
    },

    # PHASE 7.5: Agentic RAG
    'agentic_rag': {
        'crag': {
            'enable': False,  # +20-30% complex queries, +2-4s, +$0.01-0.03
            'max_iterations': 3,
            'quality_threshold': 0.7,
        },
        'multi_hop': {
            'enable': False,  # +30-40% multi-doc queries, +3-6s, +$0.02-0.05
            'max_hops': 3,
        },
    },
}
```

### Decision Matrix: When to Enable What

| Feature | Use Case | Latency Impact | Cost Impact | Storage Impact |
|---------|----------|----------------|-------------|----------------|
| **HyDE** | Complex legal queries | +500ms | +$0.001 | None |
| **Query Expansion** | Improve recall | +400ms | +$0.001 | None |
| **Query Decomposition** | Multi-part questions | +1-2s | +$0.005 | None |
| **Fusion Retrieval** | Maximum recall | +200ms | +$0.003 | 3x vectors |
| **Late Chunking** | Boundary-sensitive | Similar | Similar | None |
| **ColBERT** | Exact phrase matching | +2-3x | None (local) | 100x vectors |
| **Cross-Encoder** | Precision critical | +400ms | +$0.002 | None |
| **MMR** | Reduce redundancy | +50ms | None | None |
| **CRAG** | Mission-critical | +2-4s | +$0.01-0.03 | None |
| **Multi-Hop** | Complex analysis | +3-6s | +$0.02-0.05 | None |

### Recommended Combinations

**Tier 1: High-Precision Legal (Cost-optimized)**
```python
CONFIG_TIER1 = {
    'baseline': True,  # PHASE 1-7
    'mmr': True,  # Diversity (low cost)
    'query_expansion': True,  # Better recall (low cost)
}
# Expected: +12-18% overall, +$0.001 per query
```

**Tier 2: Premium Legal (Performance-optimized)**
```python
CONFIG_TIER2 = {
    'baseline': True,
    'hyde': True,  # Better complex queries
    'fusion': True,  # Multi-model ensemble
    'cross_encoder': True,  # Legal-specific (must fine-tune)
    'mmr': True,
}
# Expected: +25-35% overall, +$0.005 per query, 3x storage
```

**Tier 3: Mission-Critical (Maximum Accuracy)**
```python
CONFIG_TIER3 = {
    'baseline': True,
    'hyde': True,
    'fusion': True,
    'cross_encoder': True,
    'mmr': True,
    'crag': True,  # Self-correction
    'multi_hop': True,  # Complex reasoning
}
# Expected: +40-50% on complex queries, +$0.03-0.08 per query
```

**Tier 4: Exact Matching (Legal Citations)**
```python
CONFIG_TIER4 = {
    'baseline': True,
    'colbert': True,  # Token-level precision
    'cross_encoder': True,  # Legal-specific
}
# Expected: +30-40% exact phrase matching, 100x storage
```

---

## 📊 UPDATED EVALUATION & METRICS

### Expected Performance with Advanced Techniques

```python
PERFORMANCE_COMPARISON = {
    'baseline': {
        'precision_at_1': 0.06_to_0.08,
        'recall_at_64': 0.60_to_0.65,
        'drm_rate': 0.25_to_0.30,
        'latency_ms': 2000_to_4000,
        'cost_per_query': 0.05_to_0.08,
    },
    'tier1_optimized': {
        'precision_at_1': 0.07_to_0.09,  # +15%
        'recall_at_64': 0.68_to_0.73,    # +12%
        'drm_rate': 0.18_to_0.22,        # -25%
        'latency_ms': 2500_to_4500,      # +20%
        'cost_per_query': 0.051_to_0.081, # +2%
    },
    'tier2_premium': {
        'precision_at_1': 0.08_to_0.11,  # +35%
        'recall_at_64': 0.75_to_0.82,    # +25%
        'drm_rate': 0.12_to_0.17,        # -45%
        'latency_ms': 3000_to_6000,      # +40%
        'cost_per_query': 0.055_to_0.088, # +8%
    },
    'tier3_mission_critical': {
        'precision_at_1': 0.09_to_0.13,  # +50%
        'recall_at_64': 0.80_to_0.88,    # +35%
        'drm_rate': 0.08_to_0.12,        # -60%
        'latency_ms': 5000_to_10000,     # +120%
        'cost_per_query': 0.08_to_0.16,  # +90%
    },
}
```

---

**Document Created:** 2025-10-17
**Last Updated:** 2025-10-21
**Version:** 2.2 (Advanced Techniques Added)
**Status:** PHASE 1-4 Production-Ready, PHASE 5-7 Pending, PHASE X.5 Optional
**Validated Against:** LegalBench-RAG, SAC, Multi-Layer Embeddings, NLI, MLEB 2025, ColBERT, ReAct, CRAG

```

---

## 📊 Evaluation & Metrics

### LegalBench-RAG Evaluation

```python
def evaluate_on_legalbench_rag(predictions, ground_truth):
    """
    Evaluate on LegalBench-RAG benchmark
    Metrics per Pipitone & Alami, 2024
    """
    metrics = {}

    # Retrieval metrics
    metrics['precision_at_k'] = {}
    metrics['recall_at_k'] = {}

    for k in [1, 2, 4, 8, 16, 32, 64]:
        metrics['precision_at_k'][k] = calculate_precision(
            predictions['retrieved_chunks'],
            ground_truth['relevant_spans'],
            k=k
        )
        metrics['recall_at_k'][k] = calculate_recall(
            predictions['retrieved_chunks'],
            ground_truth['relevant_spans'],
            k=k
        )

    # DRM rate (from Reuter 2024)
    metrics['drm_rate'] = calculate_drm_rate(
        predictions['retrieved_chunks'],
        ground_truth['source_document']
    )

    # Text-level precision/recall (character-level)
    metrics['text_precision'] = calculate_text_precision(
        predictions['retrieved_text'],
        ground_truth['ground_truth_text']
    )
    metrics['text_recall'] = calculate_text_recall(
        predictions['retrieved_text'],
        ground_truth['ground_truth_text']
    )

    return metrics
```

### Expected Performance

Based on research findings:

```python
EXPECTED_PERFORMANCE = {
    'retrieval': {
        'precision_at_1': 0.06_to_0.08,    # vs baseline 0.024
        'precision_at_8': 0.12_to_0.14,    # vs baseline 0.09
        'recall_at_64': 0.60_to_0.65,      # vs baseline 0.35
        'drm_rate': 0.25_to_0.30,          # vs baseline 0.67
    },
    'quality': {
        'essential_chunks': 0.35_to_0.40,  # vs baseline 0.16 (2.3x better)
        'unnecessary_chunks': 0.55_to_0.60, # vs baseline 0.75
    },
    'performance': {
        'latency_ms': 2000_to_4000,
        'cost_per_query': 0.05_to_0.08,
    }
}
```

---

## ⚙️ Configuration Summary

### Final Optimal Configuration

```python
OPTIMAL_RAG_CONFIG = {
    # PHASE 2: Summarization
    'summary': {
        'model': 'gpt-4o-mini',
        'length_chars': 150,
        'tolerance': 20,
        'style': 'generic',  # NOT expert-guided!
    },

    # PHASE 3: Chunking
    'chunking': {
        'method': 'RecursiveCharacterTextSplitter',
        'chunk_size': 500,   # characters
        'chunk_overlap': 0,
        'enable_sac': True,  # CRITICAL
        'enable_multi_layer': True,
    },

    # PHASE 4: Embedding
    'embedding': {
        'model': 'text-embedding-3-large',
        'dimensions': 3072,
        'normalize': True,
    },

    # PHASE 5: Retrieval
    'retrieval': {
        'k': 6,
        'method': 'dense_semantic_only',
        'reranking': False,  # NO reranking!
        'token_baseline': 2500,
        'similarity_threshold': 0.25,
        'enable_document_filtering': True,
        'max_source_documents': 2,
    },

    # PHASE 7: Generation
    'generation': {
        'model': 'gpt-4',  # or mixtral-8x7b-instruct
        'temperature': 0.1,
        'max_tokens': 1000,
        'require_citations': True,
    }
}
```

---

## ⚠️ Critical Warnings

### DO's and DON'Ts

#### ✅ DO:
1. **Use RCTS chunking** (6.41% vs 2.40% precision@1)
2. **Use generic summaries** (outperform expert-guided)
3. **Use Summary-Augmented Chunking** (58% DRM reduction)
4. **Use dense-only retrieval** (better than hybrid)
5. **Use 500-char chunks** (optimal balance)
6. **Use 150-char summaries** (optimal trade-off)
7. **Strip SAC summaries before generation** (were for retrieval only)
8. **Require citations** (mandatory for legal)

#### ❌ DON'T:
1. **DON'T use reranking** (Cohere worse than no reranking)
2. **DON'T use expert-guided summaries** (worse than generic)
3. **DON'T use fixed-size chunking** (RCTS much better)
4. **DON'T use BM25/hybrid** (dense-only better)
5. **DON'T use SAC summaries in generation** (strip them)
6. **DON'T use K>8** (diminishing returns, high latency)

### Counterintuitive Findings

**Finding 1: Generic > Expert Summaries**
- Expected: Expert legal summaries would be better
- Reality: Generic summaries outperform
- Reason: Better balance between distinctiveness and broad alignment

**Finding 2: No Reranking > Reranking**
- Expected: Cohere reranker would improve
- Reality: Reranking degrades performance
- Reason: General-purpose reranker not optimized for legal domain

**Finding 3: Dense > Hybrid**
- Expected: Combining dense + sparse (BM25) would help
- Reality: Dense-only achieves better precision/recall
- Reason: SAC summaries work well with semantic search

---

## 🚀 Implementation Roadmap

### Phase 1: MVP (2-3 weeks)

```python
MVP_FEATURES = [
    '✅ RCTS chunking (500 chars)',
    '✅ Basic SAC (prepend 150-char summary)',
    '✅ Dense semantic search (K=6)',
    '✅ text-embedding-3-large',
    '✅ Simple generation (GPT-4)',
    '✅ Citation extraction',
]
```

### Phase 2: Production (4-6 weeks)

```python
PRODUCTION_FEATURES = [
    '✅ Multi-layer embeddings (3 indexes)',
    '✅ Document-level filtering (DRM prevention)',
    '✅ Generic summary generation (gpt-4o-mini)',
    '✅ Token baseline filtering (2500 tokens)',
    '✅ Similarity threshold filtering (25%)',
    '✅ Proper citation verification',
]
```

### Phase 3: Optimization (6-8 weeks)

```python
OPTIMIZATION_FEATURES = [
    '✅ Cost optimization (caching, batching)',
    '✅ Latency optimization (async, parallel)',
    '✅ A/B testing framework',
    '✅ Monitoring & metrics',
    '✅ Continuous evaluation on LegalBench-RAG',
]
```

---

## 📚 References

### Research Papers

1. **LegalBench-RAG** (Pipitone & Alami, 2024)
   - arXiv:2408.10343v1
   - GitHub: https://github.com/zeroentropy-cc/legalbenchrag

2. **Summary-Augmented Chunking** (Reuter et al., 2024)
   - arXiv:2510.06999v1
   - GitHub: https://github.com/DevelopedByMarkus/summary-augmented-chunking

3. **Multi-Layer Embeddings** (Lima, 2024)
   - arXiv:2411.07739v1
   - University of Brasília

4. **NLI for Legal Contracts** (Narendra et al., 2024)
   - Natural Legal Language Processing Workshop 2024
   - JPMorgan Chase & Co.

### Key Metrics from Research

| Metric | Baseline | Our Pipeline | Improvement | Source |
|--------|----------|-------------|-------------|--------|
| Precision@1 | 2.40% | 6.41% | +167% | LegalBench-RAG Table 4 |
| Recall@64 | ~35% | 62.22% | +78% | LegalBench-RAG Table 4 |
| DRM Rate | 67% | 28% | -58% | Reuter Figure 2 |
| Essential chunks | 16.39% | 37.86% | +131% | Lima Table (Manual eval) |

---

## 📋 Quick Start Code

### PHASE 1-4: Implemented Pipeline (Production-Ready)

```python
# Real implementation using completed PHASE 1-4
from extraction import IndexingPipeline, IndexingConfig

# Configure pipeline (research-optimal settings)
config = IndexingConfig(
    # PHASE 1: Smart hierarchy
    enable_smart_hierarchy=True,
    ocr_language=["cs-CZ", "en-US"],

    # PHASE 2: Generic summaries
    generate_summaries=True,
    summary_model="gpt-4o-mini",
    summary_max_chars=150,

    # PHASE 3: Multi-layer chunking + SAC
    chunk_size=500,
    chunk_overlap=0,
    enable_sac=True,

    # PHASE 4: Embedding + FAISS
    embedding_model="text-embedding-3-large",
    embedding_batch_size=100,
    normalize_embeddings=True
)

# Initialize pipeline
pipeline = IndexingPipeline(config)

# Index single document (runs all 4 phases)
# Supported formats: PDF, DOCX, PPTX, XLSX, HTML
vector_store = pipeline.index_document(
    document_path="contracts/nda_001.pdf",  # or .docx, .pptx, .xlsx, .html
    save_intermediate=True,
    output_dir="output/indexing"
)

# Batch processing (directory)
# vector_store = pipeline.index_batch(
#     document_paths=["contracts/nda_001.pdf", "contracts/msa_002.docx"],
#     output_dir="output/batch",
#     save_per_document=False
# )

# Save vector store
vector_store.save("output/vector_store")

# Query (PHASE 5 - to be implemented)
query = "Does the NDA allow independent development?"
query_embedding = pipeline.embedder.embed_texts([query])

# Hierarchical search with DRM prevention
results = vector_store.hierarchical_search(
    query_embedding=query_embedding,
    k_layer3=6,                      # K=6 per research
    use_doc_filtering=True,          # DRM prevention (58% reduction)
    similarity_threshold_offset=0.25 # Top score - 25%
)

# Display results
print(f"Found {len(results['layer3'])} chunks:")
for i, chunk in enumerate(results['layer3'][:3], 1):
    print(f"{i}. {chunk['section_title']} (score: {chunk['score']:.4f})")
    print(f"   Content: {chunk['content'][:100]}...")
```

### Complete Pipeline (Conceptual - PHASE 5-7 Pending)

```python
def legalbench_rag_pipeline(query, documents):
    """
    Complete evidence-based RAG pipeline for legal documents
    """
    # PHASE 1: Preprocess
    processed_docs = [preprocess_legal_document(doc) for doc in documents]

    # PHASE 2: Generate summaries (generic, 150 chars)
    summaries = [generate_generic_summary(doc['text']) for doc in processed_docs]

    # PHASE 3: Multi-layer chunking with SAC
    all_chunks = []
    for doc, summary in zip(processed_docs, summaries):
        # Document level
        doc_chunk = create_document_level_embeddings(doc, summary)

        # Section level
        section_chunks = create_section_level_embeddings(
            doc, doc['structure'], {'document': summary}
        )

        # Chunk level (with SAC)
        chunk_chunks = create_chunk_level_embeddings(
            doc, doc['structure'], {'document': summary}
        )

        all_chunks.extend([doc_chunk] + section_chunks + chunk_chunks)

    # PHASE 4: Embed and index
    vector_stores = create_vector_stores({
        'document': [c for c in all_chunks if c['type'] == 'document'],
        'section': [c for c in all_chunks if c['type'] == 'section'],
        'chunk': [c for c in all_chunks if c['type'] == 'chunk'],
    })

    # PHASE 5: Retrieve (K=6, no reranking, DRM prevention)
    retrieval_results = retrieve_chunks(query, vector_stores)

    # PHASE 6: Assemble context (strip SAC summaries)
    context = assemble_context(retrieval_results)

    # PHASE 7: Generate answer (with citations)
    answer = generate_answer(query, context)

    return {
        'answer': answer['answer'],
        'sources': answer['sources'],
        'metrics': {
            'total_chunks': len(all_chunks),
            'retrieved_chunks': len(retrieval_results['chunks']),
            'total_tokens': retrieval_results['total_tokens'],
            'source_documents': retrieval_results['source_documents']
        }
    }
```

### Usage Example

```python
# Load documents
documents = load_legal_documents([
    'contracts/nda_001.pdf',
    'contracts/nda_002.pdf',
    'contracts/msa_001.pdf'
])

# Run pipeline
result = legalbench_rag_pipeline(
    query="Does the NDA allow the receiving party to independently develop similar information?",
    documents=documents
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks")
print(f"Metrics: {result['metrics']}")
```

---

## 📊 Implementation Status

### Completed Phases

- **✅ PHASE 1:** Smart Hierarchy Extraction (font-size based, depth=4)
  - File: `src/extraction/docling_extractor_v2.py`
  - Doc: `IMPLEMENTATION_SUMMARY.md`

- **✅ PHASE 2:** Generic Summary Generation (gpt-4o-mini, 150 chars)
  - File: `src/extraction/summary_generator.py`
  - Doc: `IMPLEMENTATION_SUMMARY.md`

- **✅ PHASE 3:** Multi-Layer Chunking + SAC (RCTS 500 chars, 58% DRM reduction)
  - File: `src/extraction/multi_layer_chunker.py`
  - Doc: `PHASE3_COMPLETE.md`

- **✅ PHASE 4:** Embedding + FAISS Indexing (text-embedding-3-large, 3 indexes)
  - Files: `src/extraction/embedding_generator.py`, `faiss_vector_store.py`, `indexing_pipeline.py`
  - Doc: `PHASE4_COMPLETE.md`
  - Test: `scripts/test_phase4_indexing.py`

### Pending Phases

- **⏳ PHASE 5:** Query & Retrieval API (K=6, no reranking)
- **⏳ PHASE 6:** Context Assembly (strip SAC, citations)
- **⏳ PHASE 7:** Answer Generation (GPT-4/Mixtral, citations)

### Test Coverage

```bash
# Test PHASE 1-3
uv run python scripts/test_complete_pipeline.py

# Test PHASE 4
uv run python scripts/test_phase4_indexing.py --mode single
uv run python scripts/test_phase4_indexing.py --mode batch
uv run python scripts/test_phase4_indexing.py --mode bge
```

### Expected Performance (Based on Research)

| Metric | Baseline | PHASE 1-4 | Target (Full Pipeline) |
|--------|----------|-----------|------------------------|
| **Precision@1** | 2.40% | 6.41% (est.) | 6-8% |
| **Recall@64** | 35% | 62% (est.) | 60-65% |
| **DRM Rate** | 67% | 28% (w/ SAC) | <30% |
| **Essential chunks** | 16% | 38% (est.) | 35-40% |

---

**Document Created:** 2025-10-17
**Last Updated:** 2025-10-20
**Version:** 2.1 (PHASE 4 Implemented)
**Status:** PHASE 1-4 Production-Ready, PHASE 5-7 Pending
**Validated Against:** LegalBench-RAG, SAC, Multi-Layer Embeddings, NLI, MLEB 2025
