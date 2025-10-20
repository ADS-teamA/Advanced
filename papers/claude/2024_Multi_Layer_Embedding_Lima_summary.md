# Unlocking Legal Knowledge with Multi-Layered Embedding-Based Retrieval

**Autor:** João Alberto de Oliveira Lima
**Instituce:** University of Brasília; Federal Senate of Brazil
**Rok:** 2024
**Publikováno:** arXiv:2411.07739v1 [cs.AI] 12 Nov 2024

---

## Shrnutí

Paper navrhuje multi-layered embedding-based retrieval metodu pro právní a legislativní texty, která zachycuje komplexitu právních znalostí na různých úrovních granularity. Metoda vytváří embeddings nejen pro jednotlivé články, ale i pro jejich komponenty (paragrafy, klauzule) a strukturní seskupení (knihy, tituly, kapitoly), čímž umožňuje RAG systémům poskytovat přesné odpovědi přizpůsobené uživatelskému dotazu.

---

## Co bylo implementováno

### 1. Multi-Layered Embedding System

Systém definuje **6 vrstev** pro segmentaci právních dokumentů:

#### Document Level
- 1 embedding pro celý dokument
- Zachycuje overarching theme, purpose a scope
- Použitelné pro automatickou klasifikaci dokumentu

#### Document Component Level
- Embeddings pro komponenty dokumentu:
  - Main text (systematicky prezentovaný články)
  - Annexes (tabulky, nestrukturovaný text)
  - Justifications (u bills)
  - Schedules
- Každá komponenta má vlastní embedding

#### Basic Unit Hierarchy Level
- Embeddings pro strukturní seskupení:
  - Books (Livros)
  - Titles (Títulos)
  - Chapters (Capítulos)
  - Sections [group of Articles] (Seções)
- Zachycuje broader themes a vztahy mezi skupinami článků

#### Basic Unit Level
- Každý **článek** (Article) jako fundamentální jednotka má vlastní embedding
- Článek = základní jednotka v brazilském právu (odpovídá "Section" v US law)
- Zachycuje specifický právní issues a core provisions

#### Basic Unit Component Level
- Embeddings pro komponenty článků:
  - **Lead paragraph** (caput) - povinný
  - **Paragraphs** (Parágrafos) - volitelné, vysvětlují detaily/výjimky
- Detailní porozumění sémantice každého ustanovení

#### Enumeration Level
- Embeddings pro enumerativní elementy:
  - **Sections** (Incisos) - identifikované římskými číslicemi
  - **Items** (Alíneas) - identifikované malými písmeny
  - **Subitems** (Item) - nejspecifičtější detaily
- **Kontextové embedding**: Enumerativní elementy jsou embeddovány s kontextem superior elements až po Lead paragraph

**Příklad kontextového embeddingu:**
```
Místo izolovaného: "a dignidade da pessoa humana"
Embedded text: "[A República Federativa do Brasil, formada pela união
indissolúvel dos Estados e Municípios e do Distrito Federal, constitui-se
em Estado democrático de direito e tem como fundamentos:] dignidade da
pessoa humana"
```

### 2. Filtering Strategy (Retrieval Phase)

**Parametry:**
- **Baseline token count**: 2,500 tokens
- **Baseline similarity deviation**: 25%
- **Minimum chunks**: 7 (před aplikací token limit)

**Proces:**
1. Výpočet cosine similarity mezi query a všemi chunks
2. Seřazení podle similarity scores
3. Filtering duplicitního contentu:
   - Pokud textový range jednoho chunku je již pokryt jiným s vyšší similarity, není zahrnut
   - Příklad: Pokud je vybrán "Art. 5, § 1", není třeba vybírat "Art. 5, § 1, I"
4. Pokračování výběru až do:
   - Překročení baseline token count, NEBO
   - Similarity klesne pod 25% highest similarity
   - Whichever comes first

### 3. Komparativní Analýza

**Flat Chunking:**
- Tradiční přístup: 1 chunk = 1 článek
- Brazilská ústava: **276 chunks**

**Multi-Layer Chunking:**
- Všech 6 vrstev embeddings
- Brazilská ústava: **2,954 chunks**

**Příklad - Title I (4 články):**

| Layer | Flat | Multi-Layer |
|-------|------|-------------|
| Document Level | 0 | 1 |
| Text Component Level | 0 | 1 |
| Article Level | 4 | 4 |
| Article Component Level | 0 | 25 (4 caputs + 2 sole paragraphs + 19 incisos) |
| Grouping Level | 0 | 1 (1 title) |
| **TOTAL** | **4** | **32** |

### 4. Vizualizační Nástroje

**Dimensionality Reduction:**
- **PACMAP algorithm**: Reduction z 3,072 nebo 256 dimensions na 2D/3D
- Preserves global data topology
- Meaningful representation v reduced space

**Interactive Visualization:**
- **Plotly library**: Interaktivní 3D grafy
- Visualization semantic proximity
- Color-coding podle layer type
- Similarity matrices

**Příklady vizualizace:**
- Hohfeld's Legal Positions (8 concepts)
- Article 5 region (70+ fundamental rights jako cluster)
- Full Constitution embeddings

---

## Technická implementace

### Embedding Model

**OpenAI text-embedding-3-large:**
- Konfigurace: **256-dimensional vectors** (během proof-of-concept)
- Capability: až 3,072 dimensions
- Dense vector representations
- Semantic + syntactic features capture

**Token Limitation Handling:**
- Pokud text překročí max token limit:
  1. Segmentace na semantic chunks
  2. Embedding pro každý chunk
  3. Arithmetic mean embeddings = representation celého textu

### Generation Model

**GPT-4-turbo-preview:**
- Max output: 1,000 tokens
- Temperature: 0.3
- Kombinuje retrieved chunks + original query
- Contextually relevant responses

### RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ Legal Document → Parse Hierarchy → Multi-Layer Chunking     │
│                                   ↓                          │
│                           Create Embeddings                  │
│                                   ↓                          │
│                           Vector Database                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ User Query → Embedding → Cosine Similarity                  │
│                              ↓                               │
│                      Rank by Similarity                      │
│                              ↓                               │
│                      Apply Filters:                          │
│                      • Token baseline (2500)                 │
│                      • Similarity deviation (25%)            │
│                      • Duplicate content removal             │
│                      • Minimum 7 chunks                      │
│                              ↓                               │
│                      Selected Chunks                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   GENERATION PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ Concatenated Chunks + Query → GPT-4 → Response             │
└─────────────────────────────────────────────────────────────┘
```

### Brazilská legislativní hierarchie (Complementary Law No. 95/1998)

**Struktura (top-down):**
```
Legal Norm (Norma Jurídica)
├── Main Text (Texto Principal) [mandatory]
│   ├── Part (Parte) [optional, multiple možné]
│   │   └── Book (Livro) [optional, multiple možné]
│   │       └── Title (Título) [optional, multiple možné]
│   │           └── Chapter (Capítulo) [optional, multiple možné]
│   │               ├── Section [group] (Seção) [optional]
│   │               └── Subsection (Subseção) [optional]
│   │                   └── ARTICLE (Artigo) [FUNDAMENTAL UNIT]
│   │                       ├── Lead Paragraph (Caput) [mandatory]
│   │                       └── Paragraph (Parágrafo) [optional, multiple]
│   │                           └── Section [enum] (Inciso) [optional]
│   │                               └── Item (Alínea) [optional]
│   │                                   └── Subitem (Item) [optional]
└── Annex (Anexo) [optional, 0-N možné]
```

**Základní jednotka - ARTICLE:**
- **Caput**: Povinný, hlavní ustanovení
- **Sole Paragraph** (Parágrafo único): Pokud jen 1 paragraph
- **Numbered Paragraphs**: Pokud více paragraphs
- **Sections** (Incisos): Enumerace v rámci caput/paragraph (římské číslice)
- **Items** (Alíneas): Sub-enumeration (malá písmena)
- **Subitems**: Nejdetailnější úroveň

---

## Výsledky

### Testovací Dataset

**Brazilská Federální Ústava:**
- Kompletní text ústavy
- 8 test questions
- Různé typy dotazů:
  - Specific (např. "What are the attributes of the vote?")
  - Comprehensive (např. "What are the rights of children and teenagers?")

### 8 Test Questions

| # | Question | ML Chunks | Flat Chunks | ML Tokens | Flat Tokens |
|---|----------|-----------|-------------|-----------|-------------|
| Q1 | What are the foundations of the republic? | 19 | 7 | 2,502 | 3,110 |
| Q2 | Talk about the social function of property | 12 | 11 | 2,586 | 2,625 |
| Q3 | What are the attributes of the vote? | 7 | 7 | 2,966 | 4,549 |
| Q4 | How is tax revenue distributed? | 8 | 7 | 2,796 | 3,595 |
| Q5 | Rights of children and teenagers? | 7 | 7 | 4,163 | 3,054 |
| Q6 | What is assured to the jury? | 18 | 7 | 2,552 | 4,699 |
| Q7 | How is right to association given? | 20 | 7 | 2,230 | 2,093 |
| Q8 | Legal assistance for insufficient funds? | 12 | 8 | 5,689 | 1,523 |

### Chunk Relevance Classification

Každý retrieved chunk byl manuálně klasifikován:
- **E (Essential)**: Kritický pro odpověď
- **C (Complementary)**: Podporující informace
- **U (Unnecessary)**: Nerelevantní

**Celkové výsledky:**

| Classification | Multi-Layer | Flat |
|----------------|-------------|------|
| **Essential** | **37.86%** | **16.39%** |
| Complementary | 3.88% | 8.20% |
| Unnecessary | 58.25% | 75.41% |

**Key Findings:**
- Multi-layer má **2.3x více essential chunks** než flat
- Multi-layer má **nižší proportion unnecessary chunks**
- Flat approach struggle s "semantic overload" (např. Article 5)

### Similarity Metrics

**Box Plot Analysis (Max/Min Similarity):**

**Multi-Layer:**
- Narrower range at extremes
- More consistent similarity scores
- Better semantic alignment s query

**Flat:**
- Wider variability
- Vyšší proportion low-similarity chunks
- Méně konzistentní semantic matching

**Median Max Similarity:**
- Multi-Layer: ~0.52
- Flat: ~0.49

**Median Min Similarity:**
- Multi-Layer: ~0.40
- Flat: ~0.36

### Problematické případy

**Article 5 - Semantic Overload:**
- Obsahuje **70+ fundamental rights**
- Flat approach: často nevybere správnou sub-clause
- Multi-layer approach: 70+ samostatných embeddings pro každé právo

**Příklad - Question 6 ("What is assured to the jury?"):**
- **Multi-Layer**: Správně identifikoval Art. 5, Inciso XXXVIII (similarity 0.354712)
- **Flat**: Nezahrnul Art. 5 vůbec (semantic overload)

**Příklad - Question 7 ("How is right to association given?"):**
- **Multi-Layer**: Vybralo 8 sub-clauses z Art. 5 (Incisos XVII, XVIII, XIX, XX, XXI, LXX, LXX-b, XVI)
- **Flat**: Nezahrnul Art. 5, pouze Art. 8 (unions)

### Language Independence

**Flexibility of Query Embeddings:**
Stejný sémantický dotaz v různých jazycích/formulacích produkuje podobné výsledky:

- **Legal expert (PT)**: "De onde emana o poder?"
- **Legal expert (EN)**: "Where does power emanate from?"
- **Legal expert (IT)**: "Da dove viene il potere?"
- **Layperson (PT)**: "De onde vem a autoridade desse povo que manda?"

→ Všechny dotazy vedou k **Art. 1, Sole Paragraph** (source of power)

---

## Teoretické koncepty

### 1. Aboutness

**Gilbert Ryle (1933) - "About conversational":**
- Výraz zachycuje central theme konverzace, pokud:
  - Considerable number sentences přímo používá expression
  - Nebo používá synonym
  - Nebo dělá indirect reference na koncept
- Occurrence musí dominovat bez competing expression

**Aplikace na embeddings:**
- Embeddings = digital "aboutness" of content
- Reflection of thematic consistency across document
- Ne jen explicit text, ale semantic theme

### 2. Semantic Chunking

**Traditional Semantic Chunking (semchunk):**
- Recursive division technique
- Contextually meaningful separators
- Segments reach specific size
- **Limitation**: Treats all segments on same level, ignoruje hierarchical organization

**Multi-Layer Semantic Chunking:**
- Respects intrinsic hierarchy
- Multiple layers from hierarchical organization
- Segments na různých úrovních legal significance
- Captures interrelations between provisions

### 3. RAG (Retrieval Augmented Generation)

**3 hlavní fáze:**

#### Indexing Phase
**Traditional:**
- Text normalization (tokenization, stemming)
- Stopwords removal
- TF/IDF or BM25 indexing

**Advanced (Embeddings):**
- Pre-trained language models
- Semantic vectors (embeddings)
- Decision na indexing unit

#### Retrieval Phase
**Traditional:**
- Term frequency
- Presence of specific terms
- TF/IDF, BM25 ranking

**Advanced (Embeddings):**
- Semantic proximity (cosine similarity, Euclidean distance)
- Evaluates semantic nuances
- Language independence (synonyms, different languages)

#### Generation Phase
- Combines retrieved chunks + original query
- Large language model
- Contextually relevant response

---

## Srovnání přístupů

### Flat Chunking (Traditional)

**Charakteristiky:**
- 1 chunk = 1 článek
- Jednoduchá implementace
- Nižší počet chunks
- Uniform granularity

**Výhody:**
- Rychlejší indexing
- Menší storage requirements
- Jednodušší maintenance

**Nevýhody:**
- **Semantic overload**: Dlouhé články s mnoha koncepty
- **Granularity mismatch**: Nemůže odpovědět na very specific questions
- **Context loss**: Ztráta hierarchických vztahů
- **Lower precision**: Vyšší proportion unnecessary chunks (75.41%)

### Multi-Layer Chunking (Proposed)

**Charakteristiky:**
- Multiple chunks per článek
- 6 hierarchical layers
- Variable granularity
- Context-aware enumeration embedding

**Výhody:**
- **Higher precision**: 37.86% essential chunks vs 16.39%
- **Semantic clarity**: Handling semantic overload
- **Flexibility**: Odpovídá na specific i comprehensive queries
- **Context preservation**: Hierarchical relationships maintained
- **Better similarity consistency**: Narrower range, higher consistency

**Nevýhody:**
- Komplexnější implementace
- Vyšší storage requirements (10.7x více chunks)
- Delší indexing time
- Potential redundancy (overlapping content)

**Trade-offs:**

| Aspect | Flat | Multi-Layer | Winner |
|--------|------|-------------|--------|
| Precision | 16.39% essential | 37.86% essential | **ML** |
| Simplicity | Simple | Complex | Flat |
| Storage | 276 chunks | 2,954 chunks | Flat |
| Semantic Overload | Struggles | Handles well | **ML** |
| Query Flexibility | Limited | High | **ML** |
| Maintenance | Easy | Moderate | Flat |

---

## Koncepty z Related Work

### Embeddings vs Traditional IR

**TF/IDF (Term Frequency-Inverse Document Frequency):**
- Focus na term occurrence frequency
- Keyword-based
- Fail na semantic nuances

**Embeddings:**
- Dense vector representations
- Semantic content focus
- Captures meanings, themes, relationships
- Distance metrics (cosine similarity, Euclidean)
- **Syntactic features**: Slight variations → close vectors

**Příklad - Hohfeld's Legal Positions:**
- 8 legal concepts (Right, Duty, Power, Immunity, etc.)
- Embeddings capture correlatives and opposites
- 3D visualization shows semantic relationships
- Query "What is Legal Immunity?" → closest vector "Immunity"

---

## Praktické aplikace

### 1. Legislative Consultancy

**Use Case:**
- Drafting new legislation
- Analyzing existing laws
- Comparing similar provisions across documents

**Benefits:**
- Rychlejší access k relevant provisions
- Better understanding of hierarchical structure
- Informed decision-making

### 2. Legal Information Retrieval

**Use Case:**
- Lawyers searching case law
- Citizens understanding their rights
- Legal professionals researching precedents

**Benefits:**
- Natural language queries
- Language-independent search
- Precise results at appropriate granularity

### 3. Contract Analysis

**Use Case:**
- Comparing contracts to templates
- Identifying deviations
- Understanding obligations

**Benefits:**
- Multi-level comparison
- Hierarchical understanding
- Detailed or broad analysis as needed

### 4. Regulatory Compliance

**Use Case:**
- Checking compliance s regulations
- Understanding regulatory requirements
- Identifying gaps

**Benefits:**
- Comprehensive coverage check
- Detailed provision analysis
- Hierarchical compliance mapping

---

## Limitace a future work

### Identifikované limitace

1. **Annotator Expertise:**
   - Annotators zkušení s legal documentation
   - Ale ne trained lawyers
   - Může impact accuracy v complex cases

2. **Proof-of-Concept Dimensions:**
   - Použito 256 dimensions
   - Model podporuje až 3,072 dimensions
   - Nevyhodnoceno, zda vyšší dimensions = better performance

3. **Single Legal System:**
   - Testováno pouze na Brazilian Constitution
   - Civil law tradition
   - Aplikovatelnost na common law systems neověřena

4. **Redundancy:**
   - Some overlapping content v chunks
   - Např. Art. 5, § 1 obsahuje Art. 5, § 1, I
   - Trade-off mezi redundancy a precision

### Budoucí výzkum

**1. Inter-Article Relationships:**
- Embeddings pro pairs/groups of articles
- Cross-references a dependencies
- Network of relationships

**2. Temporal Dimension:**
- Embeddings pro different versions
- Evolution of legal texts over time
- Amendment tracking

**3. Vector Dimensions:**
- Testing 256 vs 3,072 dimensions
- Trade-off: computational cost vs accuracy
- Optimal dimension pro legal texts

**4. Cross-System Validation:**
- Testing on common law systems
- Different legal traditions
- International comparison

**5. Dynamic Updates:**
- Real-time embedding updates
- Amendment incorporation
- Version control

---

## Závěr

Paper demonstruje significantní advantage multi-layered embedding-based retrieval pro právní texty:

### Klíčové přínosy

1. **Higher Precision:**
   - 2.3x více essential chunks (37.86% vs 16.39%)
   - Nižší proportion unnecessary chunks (58.25% vs 75.41%)

2. **Semantic Overload Solution:**
   - Handling articles s mnoha concepts (např. Art. 5 s 70+ rights)
   - Separate embeddings pro každý concept
   - Precise retrieval specific provisions

3. **Query Flexibility:**
   - Odpovídá na specific i comprehensive queries
   - Natural language queries
   - Language-independent

4. **Hierarchical Understanding:**
   - Preserves legal document structure
   - Multi-level representation
   - Context-aware embeddings

5. **Improved Consistency:**
   - Narrower similarity score ranges
   - More consistent semantic matching
   - Better alignment s user queries

### Aplikovatelnost

**Beyond Legal Domain:**
Principles extend to any field s hierarchical text:
- Technical documentation
- Regulatory frameworks
- Medical guidelines
- Educational materials
- Corporate policies

**Legal Domain:**
- Civil law systems (verified)
- Common law systems (principle applicable)
- International treaties
- Legislative drafting
- Contract management

### Praktický impact

Pro **legislative consultants**:
- Faster access k relevant provisions
- Better informed decision-making
- Improved quality legislative outputs

Pro **legal professionals**:
- Enhanced research efficiency
- Comprehensive understanding
- Precise information retrieval

Pro **general public**:
- Accessible legal information
- Natural language queries
- Bridging gap mezi legal experts a laypersons

### Final Thoughts

Multi-layered embedding-based retrieval není jen technical improvement - je to **paradigm shift** v legal information retrieval. Umožňuje systémům:
- Understand legal knowledge na multiple levels
- Respond s appropriate granularity
- Maintain semantic coherence
- Bridge linguistic divides

Jak legal corpora rostou v complexity a volume, tento přístup nabízí scalable, linguistically diverse, a contextually rich solution pro navigaci legal knowledge.
