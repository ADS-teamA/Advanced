# LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain

**Autoři:** Nicholas Pipitone, Ghita Houir Alami
**Instituce:** ZeroEntropy, San Francisco, CA
**Rok:** 2024
**arXiv:** 2408.10343v1 [cs.AI]

---

## Shrnutí

Paper představuje **LegalBench-RAG**, první benchmark specificky navržený pro evaluaci retrieval komponenty RAG (Retrieval-Augmented Generation) systémů v legal doméně. Benchmark klade důraz na **přesný retrieval** minimálních, vysoce relevantních text snippets místo velkých chunků nebo document IDs. Dataset obsahuje 6,858 query-answer párů nad korpusem více než 79M znaků, kompletně human-annotovaný právními experty.

---

## Co bylo implementováno

### 1. LegalBench-RAG Dataset

#### Hlavní komponenty
- **Korpus**: 714 legal dokumentů v .txt formátu (79,704,214 znaků)
- **QA páry**: 6,858 dotazů s přesnými character-level annotations
- **4 source datasety**:
  - **ContractNLI**: 946 queries, 95 dokumentů (NDA related)
  - **CUAD**: 4,042 queries, 462 dokumentů (Private Contracts)
  - **MAUD**: 1,676 queries, 150 dokumentů (M&A documents)
  - **PrivacyQA**: 194 queries, 7 dokumentů (Privacy policies)

#### Struktura dat
```json
{
  "query": "Consider the Software License... Are the licenses non-transferable?",
  "snippets": [
    {
      "file_path": "CardlyticsInc_..._Agreement1.txt",
      "span": [44579, 45211],
      "answer": "Supplier hereby grants Bank of America..."
    }
  ]
}
```

### 2. LegalBench-RAG-mini

- **Lightweight verze** pro rychlou iteraci a experimentaci
- **776 queries** (194 z každého ze 4 datasetů)
- **72 dokumentů**, 8,682,104 znaků
- Zachovává reprezentativní vzorek plného benchmarku

### 3. Construction Process

#### Krok 1: Tracing Back to Original Sources
- Zpětné trasování každého text segmentu z LegalBench k původní lokaci
- Vytvoření mappingu mezi queries a relevant spans
- Každý span = range of characters v původním dokumentu

#### Krok 2: Query Generation Format
```
"Consider (document description); (interrogative)"
```

Komponenty:
- **Document description**: Automaticky vygenerováno pomocí GPT-4o-mini z:
  - Filename
  - První a poslední odstavce dokumentu
  - Validace pomocí regex
- **Interrogative**: Mapping z annotation kategorií na otázky

#### Krok 3: Quality Control (3 kritické kontrolní body)

1. **Mapping Annotation Categories to Interrogatives**
   - Manuální konstrukce mappingu
   - Vyloučení kategorií s nekonzistentní annotation precision

2. **Mapping Document IDs to Descriptions**
   - GPT-4o-mini pro auto-generování descriptions
   - Manuální inspekce každé description
   - Embedding similarity check pro detekci podobných descriptions
   - Vyloučení nedistinktivních párů

3. **Selection of Annotation Categories**
   - Manuální evaluace precision levels napříč kategoriemi
   - Vyloučení nekonzistentních kategorií

### 4. Benchmark Pipeline Implementation

#### RAG System Components Testované

**Pre-processing (Chunking Strategies):**
1. **Naive Fixed-Size Method**:
   - Chunk size: 500 characters
   - No overlap

2. **Recursive Character Text Splitter (RCTS)**:
   - Sekvenční splitting na predefined characters
   - Zachování paragraphs, sentences, words together

**Post-processing:**
1. **No Reranker**: Direct retrieval results
2. **Cohere Reranker**: "rerank-english-v3.0" model

**Technology Stack:**
- **Embedding Model**: OpenAI "text-embedding-3-large"
- **Vector Database**: SQLite Vec
- **Reranker**: Cohere "rerank-english-v3.0"
- **Evaluation**: Precision@k and Recall@k metrics

---

## Technická implementace

### Dataset Statistics

| Dataset | Documents | Corpus Characters | Avg Doc Length | Queries |
|---------|-----------|------------------|----------------|---------|
| ContractNLI | 95 | 1,013,969 | 10,673 | 977 |
| MAUD | 150 | 52,721,337 | 351,476 | 1,676 |
| CUAD | 462 | 25,792,044 | 55,827 | 4,042 |
| PrivacyQA | 7 | 176,864 | 25,266 | 194 |
| **Total** | **714** | **79,704,214** | **443,242** | **6,889** |

### LegalBench-RAG-mini Statistics

| Dataset | Documents | Corpus Characters | Queries |
|---------|-----------|------------------|---------|
| ContractNLI | 18 | 184,267 | 194 |
| MAUD | 18 | 7,109,200 | 194 |
| CUAD | 29 | 1,211,773 | 194 |
| PrivacyQA | 7 | 176,864 | 194 |
| **Total** | **72** | **8,682,104** | **776** |

### RAG Pipeline Architecture

```
Query q → Embedding Model → Vector Similarity Search
                                    ↓
                            Top-k' Retrieval
                                    ↓
                            (Optional) Reranker
                                    ↓
                            Top-k Chunks Rq
                                    ↓
                      LLM_P(q, Rq) → Answer
```

**Formální definice:**
```
Contextual Retriever(q, D) → Rq
LLM_P(q, Rq) → answer
```

Kde:
- `q` = query
- `D` = knowledge base (set of documents)
- `Rq` = retrieved chunks {r1, r2, ..., rK}
- `P` = system prompt

### Evaluation Metrics

**Precision@k**:
- Podíl relevantních snippets v top-k retrieved chunks
- Klesá s rostoucím k

**Recall@k**:
- Podíl nalezených ground truth snippets z celkového počtu
- Roste s rostoucím k

**Testované hodnoty k**: 1, 2, 4, 8, 16, 32, 64

---

## Výsledky

### Srovnání Chunking Strategies

#### Naive Method (Fixed-Size 500 chars, No Reranker)

| Dataset | Precision@1 | Precision@8 | Recall@1 | Recall@64 |
|---------|-------------|-------------|----------|-----------|
| PrivacyQA | 7.86 | 5.06 | 7.45 | 66.07 |
| ContractNLI | 16.45 | 9.73 | 11.32 | 86.57 |
| MAUD | 3.36 | 1.89 | 2.54 | 25.62 |
| CUAD | 9.27 | 4.33 | 12.60 | 75.71 |
| **ALL** | **2.40** | **4.33** | **3.37** | **76.39** |

#### RCTS Method (No Reranker) - **NEJLEPŠÍ VÝSLEDKY**

| Dataset | Precision@1 | Precision@8 | Recall@1 | Recall@64 |
|---------|-------------|-------------|----------|-----------|
| **PrivacyQA** | **14.38** | **9.03** | **8.85** | **84.19** |
| ContractNLI | 6.63 | 2.81 | 7.63 | 61.72 |
| MAUD | 2.65 | 1.40 | 1.65 | 28.28 |
| CUAD | 1.97 | 4.20 | 1.62 | 74.70 |
| **ALL** | **6.41** | **4.36** | **4.94** | **62.22** |

### Impact of Reranking

#### Naive + Cohere Reranker

| Dataset | Precision@1 | Recall@64 |
|---------|-------------|-----------|
| PrivacyQA | 14.38 | 84.19 |
| ContractNLI | 6.63 | 61.72 |
| MAUD | 2.64 | 28.28 |
| CUAD | 1.97 | 74.70 |
| **ALL** | **6.41** | **62.22** |

#### RCTS + Cohere Reranker

| Dataset | Precision@1 | Recall@64 |
|---------|-------------|-----------|
| PrivacyQA | 13.94 | 79.61 |
| ContractNLI | 5.08 | 62.97 |
| MAUD | 1.94 | 31.46 |
| CUAD | 3.53 | 70.19 |
| **ALL** | **6.13** | **61.06** |

### Klíčová zjištění

1. **Nejlepší konfigurace**: RCTS bez rerankeru
   - Nejvyšší Precision@k a Recall@k
   - Překvapivě lepší než s rerankerem

2. **Reranker Performance**:
   - Cohere Reranker měl **horší výkon** než no reranker
   - Možné důvody:
     - Obtížnost benchmarku
     - Focus na legal text
     - General-purpose model není dostatečně domain-specific

3. **Dataset Difficulty Ranking**:
   - **Nejlehčí**: PrivacyQA (Precision@1: 14.38%, Recall@64: 84.19%)
     - Jednodušší jazyk (non-lawyer questions)
     - Privacy policies consumer apps
   - **Nejtěžší**: MAUD (Precision@1: 2.65%, Recall@64: 28.28%)
     - Highly technical legal jargon
     - M&A dokumenty s komplexní terminologií
   - **Střední**: ContractNLI, CUAD

4. **Trendy v metrikách**:
   - **Recall** roste s increasing k (očekávané)
   - **Precision** klesá s increasing k (očekávané)
   - Low precision values kvůli highly targeted ground truth

### Performance Across All Methods (Aggregated)

**Recall@k Across Datasets:**
- PrivacyQA: Konzistentně nejvyšší napříč všemi metodami
- MAUD: Konzistentně nejnižší napříč všemi metodami
- ContractNLI & CUAD: Střední výkon

**Precision@k Across Datasets:**
- Podobné trendy jako u Recall
- Sharper decline s rostoucím k u všech datasetů

---

## Srovnání s related work

### Existing Benchmarks

| Benchmark | Focus | Domain | Evaluation |
|-----------|-------|--------|------------|
| **LegalBench** | Generation phase | Legal reasoning | LLM reasoning capabilities |
| **HotPotQA** | Q&A | General | Multi-hop reasoning |
| **RGB** | RAG systems | General | Overall RAG quality |
| **RECALL** | RAG robustness | General | Counterfactual knowledge |
| **MultiHop-RAG** | Multi-hop queries | General | Complex reasoning |
| **LegalBench-RAG** | **Retrieval phase** | **Legal-specific** | **Precise snippet retrieval** |

### Unique Contributions LegalBench-RAG

1. **First legal-specific retrieval benchmark**
2. **Character-level precision annotations**
3. **Emphasis on minimal, precise snippets** (not broad chunks)
4. **Human-in-the-loop verification support**
5. **Built from expert-annotated legal datasets**

---

## Inovace a přínosy

### Research Contributions

1. **První dedicated benchmark pro legal retrieval**
   - Zaměření na retrieval step of RAG pipeline
   - Legal-specific nuances and complex relationships

2. **Precise snippet retrieval**
   - Character-level annotations
   - Minimální, highly relevant text segments
   - Avoid context window limitations

3. **Comprehensive quality control**
   - Manual inspection všech datapoints
   - Multiple validation checkpoints
   - Domain expert annotations

4. **Practical applicability**
   - Publicly available dataset
   - Code for running benchmark
   - Mini version for rapid iteration

### Practical Benefits

1. **Reduced hallucination risk**
   - Precise retrieval minimizes irrelevant context
   - Less noise in LLM input

2. **Citation generation**
   - Precise snippets allow LLM to generate accurate citations
   - Human verification support

3. **Cost and latency optimization**
   - Shorter context windows = lower costs
   - Faster processing
   - Reduced token usage

4. **Standardized evaluation framework**
   - Compare competing retrieval algorithms
   - Iterate on RAG techniques
   - Commercial and academic use

### Dataset Value Estimation

**CUAD dataset creation cost estimate:**
- 1 year effort
- 40+ lawyers
- 13,000 annotations
- 9,283 pages reviewed 4+ times
- 5-10 minutes per page
- $500/hour legal expertise
- **Estimated cost: ~$2,000,000**

LegalBench-RAG leverages this existing investment to create first public retrieval-focused legal benchmark.

---

## Limitace a future work

### Identifikované limitace

1. **Document Type Coverage**
   - NDAs, M&A agreements, commercial contracts, privacy policies
   - **Chybí**:
     - Structured numerical data parsing (financial fraud)
     - Medical records analysis (personal injury)
     - Court decisions, statutes, regulations

2. **Single Document Queries**
   - Každá query answerable z exactly one document
   - **Nehodnotí**: Multi-document reasoning
   - Některé queries vyžadují multi-hop reasoning (identified manually)

3. **Annotation Quality**
   - Annotators: Legal documentation experience, but **not trained lawyers**
   - Potential impact na accuracy v complex cases

4. **Missing Evaluation**
   - Neutral class nebyla evaluována (focus on Entailment/Contradiction)

5. **General-purpose Tools Limitations**
   - Cohere Reranker not optimized for legal text
   - Need for domain-specific rerankers

### Future Work Directions

1. **Specialized Rerankers**
   - Fine-tuning on legal corpora
   - Legal-specific features
   - Training on larger, diverse legal datasets

2. **More Complex Queries**
   - Higher number of hops
   - Multi-document reasoning
   - Cross-reference resolution

3. **Expanded Document Types**
   - Court decisions
   - Statutes and regulations
   - Medical records
   - Financial documents

4. **Domain-Specific Embeddings**
   - Legal-specific embedding models
   - Better representation of legal terminology
   - Improved semantic understanding

---

## Závěr

LegalBench-RAG představuje **první benchmark specificky navržený pro evaluaci retrieval komponenty RAG systémů v legal doméně**. Klíčové přínosy:

### 1. Technical Achievement
- **6,858 query-answer párů** nad 79M character korpusem
- **Character-level precision annotations** z expert-annotated datasetů
- **Comprehensive quality control** s manuální inspekcí všech datapoints
- **Public availability** pro commercial i academic use

### 2. Key Findings
- **RCTS chunking strategy** výrazně lepší než naive fixed-size
- **General-purpose rerankers** (Cohere) horší než no reranking
  - Highlighting need for **domain-specific tools**
- **Dataset difficulty**: PrivacyQA (easiest) → ContractNLI/CUAD → MAUD (hardest)
- **Best configuration**: RCTS + No Reranker
  - Precision@1: 6.41%
  - Recall@64: 62.22%

### 3. Impact on Legal AI
- **Standardized evaluation** pro retrieval algorithms
- **Foundation** pro development of legal-specific RAG systems
- **Critical gap addressed** v legal AI ecosystem
- **Support** pro innovation a reliability of AI-driven legal tools

### 4. Future Directions
- Development of **legal-specific rerankers**
- **Multi-document reasoning** capabilities
- **Broader document type coverage**
- **More complex query types** s vyšším počtem hops

Tento benchmark poskytuje crucial tool pro companies a researchers zaměřené na enhancing accuracy a performance RAG systémů v legal domain, s přímou aplikovatelností v AI-powered legal applications.

**Dataset dostupný na**: https://github.com/zeroentropy-cc/legalbenchrag
