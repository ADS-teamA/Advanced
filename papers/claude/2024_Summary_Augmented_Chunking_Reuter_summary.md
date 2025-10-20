# Towards Reliable Retrieval in RAG Systems for Large Legal Datasets

**Autoři:** Markus Reuter, Tobias Lingenberg, Rūta Liepiņa, Francesca Lagioia, Marco Lippi, Giovanni Sartor, Andrea Passerini, Burcu Sayin

**Instituce:**
- Technical University of Darmstadt
- University of Florence
- University of Bologna (ALMA-AI, Faculty of Law)
- University of Trento (DISI)
- European University Institute

**Rok:** 2024
**Publikováno:** arXiv:2510.06999v1 [cs.CL] 8 Oct 2025

---

## Shrnutí

Paper identifikuje kritický problém RAG systémů v právní doméně - Document-Level Retrieval Mismatch (DRM), kde retriever vybírá informace z úplně nesprávných zdrojových dokumentů. Jako řešení navrhuje Summary-Augmented Chunking (SAC), jednoduchou a computationally efficient techniku, která obohacuje každý text chunk o document-level synthetic summary. Tím se injektuje globální kontext, který by jinak byl ztracen během standardního chunking procesu.

---

## Co bylo implementováno

### 1. Document-Level Retrieval Mismatch (DRM) Metrika

**Definice:**
Proportion top-k retrieved chunks, které **nepocházejí z dokumentu obsahujícího ground-truth text**.

**Vzorec:**
```
DRM = (Počet chunks z nesprávného dokumentu) / (Total top-k chunks)
```

**Proč je důležité:**
- V legal domain, kde dokumenty jsou strukturálně velmi podobné
- Answering question correctly je insufficient, pokud supporting text je z wrong document
- Undermines legal validity a erodes user trust
- Legal professionals vyžadují transparent a verifiable "reasoning trail"

**Baseline výsledky:**
- ContractNLI (362 NDAs): **DRM > 95%**
- Caused by: highly standardized, boilerplate nature of NDAs
- Linguistic homogeneity confuses retrieval models

### 2. Summary-Augmented Chunking (SAC)

**4-step proces:**

#### Step 1: Summarization
- Pro každý document v corpus → generování **single concise summary**
- Summary length: **~150 characters** (optimal)
- Acts as "document fingerprint"
- Jeden LLM call per document

#### Step 2: Chunking
- **Recursive character splitting** strategy
- Partition document do smaller, manageable chunks
- Established method, well-performing na legal texts

#### Step 3: Augmentation
- **Prepend** document-level summary k EACH chunk derived from that document
- Injects crucial global context
- Summary + Chunk = Augmented Chunk

#### Step 4: Indexing
- Summary-augmented chunks → embedded
- Indexed v vector database pro retrieval
- Standard embedding models (např. thenlper/gte-large)

**Klíčové vlastnosti:**
- **Minimal computational overhead**: pouze 1 additional LLM call per document
- **Modular**: smoothly integrable do existing RAG pipelines
- **Scalable**: practical for large, dynamically changing legal databases

### 3. Expert-Guided Summarization

Kromě generic SAC, vyvinuli s právními experty **meta-prompt** pro expert-guided summarization:

**Cílené document types:**
1. **Non-Disclosure Agreements (NDAs)**
2. **Privacy Policies**

**Expert prompt obsahuje templates specifying:**

**Pro NDAs:**
- Definition of Confidential Information
- Parties to the Agreement (Disclosing/Receiving)
- Obligations of Receiving Party
- Exclusions from Confidentiality
- Term and Duration
- Purpose of Disclosure
- Remedies for Breach
- Governing Law and Jurisdiction
- Miscellaneous Clauses

**Pro Privacy Policies:**
- Personal Data Collected
- Identity of Controller
- Purposes of Processing
- Legal Basis (GDPR)
- Recipients of Data
- International Transfers
- Data Retention
- Data Subject Rights
- Right to Lodge Complaint
- Automated Decision-Making

**Hypotéza:**
Tailoring summaries k nuances specific legal document types → further enhance retrieval

**Result (překvapivý):**
Generic summaries **outperformed** expert-guided ones (více v Results)

### 4. Evaluation Metriky

**Document-Level:**
- **DRM** (Document-Level Retrieval Mismatch)
- Lower = better
- Primary metric pro measuring retrieval reliability

**Character-Level (Text-Level):**
- **Precision**: Fraction retrieved text that is part of ground truth
  - High precision = concise, minimal noise

- **Recall**: Fraction ground truth text found by retrieval
  - High recall = system found all necessary information

**Computation:**
```
Precision = |Retrieved ∩ Ground Truth| / |Retrieved|
Recall = |Retrieved ∩ Ground Truth| / |Ground Truth|
```

---

## Technická implementace

### Dataset: LegalBench-RAG

**Proč LegalBench-RAG:**
- Specifically designed k **isolate retrieval component** z RAG
- Blends retrieval performance separate od LLM's internal knowledge
- Unlike LegalBench/LexGLUE (test intrinsic reasoning)

**4 Sub-datasets:**

| Dataset | Type | Description |
|---------|------|-------------|
| **CUAD** | Contract Understanding Atticus Dataset | General contracts |
| **MAUD** | Merger Agreement Understanding Dataset | Merger agreements |
| **ContractNLI** | | Non-disclosure agreements (362 docs) |
| **PrivacyQA** | | Privacy policies from mobile apps |

**Ground Truth obsahuje:**
- Source document ID
- Relevant text snippet(s)
- Character-level span positions [start, end]
- User query

### RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRE-RETRIEVAL (Indexing)             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Document → Summarize → Summary (~150 chars)            │
│      ↓                                                   │
│  Recursive Character Splitting → Chunks (500 chars)     │
│      ↓                                                   │
│  Prepend Summary to Each Chunk                          │
│      ↓                                                   │
│  Embed (thenlper/gte-large)                             │
│      ↓                                                   │
│  Index in FAISS Vector DB (cosine similarity)           │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  User Query → Embed                                      │
│      ↓                                                   │
│  Cosine Similarity Search                                │
│      ↓                                                   │
│  Retrieve Top-K Chunks                                   │
│      ↓                                                   │
│  Evaluate: DRM, Precision, Recall                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Hyperparameters

**Optimal Configuration (Table 1):**

| Parameter | Tested Values | Optimal | Why |
|-----------|---------------|---------|-----|
| **Chunk Size** | 200, 500, 800 chars | **500 chars** | Best precision/recall balance |
| **Summary Length** | 150, 300 chars | **150 chars** | Most balanced trade-off |
| **Tolerance** | ±20 chars | ±20 chars | LLMs deviate from length |

**Performance by Configuration:**
```
Chunk=500, Sum=150: Prec=11.03%, Rec=41.80%, DRM=19.29%  ← OPTIMAL
Chunk=500, Sum=300: Prec=8.45%,  Rec=37.77%, DRM=20.61%
Chunk=800, Sum=300: Prec=6.79%,  Rec=43.90%, DRM=29.68%
```

### Embedding Models

**Tested Models:**
1. **thenlper/gte-large** ← **SELECTED** (open-source, reproducible)
2. text-embedding-3-large (OpenAI) ← best performance, but API limits
3. BAAI/bge-base-en-v1.5
4. nlpaueb/legal-bert-base-uncased

**Final Choice:**
- **thenlper/gte-large**
- **Dimensions**: Not specified in final config (capable up to 3,072)
- **Reasons**: Open-source, reproducibility, no API rate limits

### Dense vs Sparse Retrieval (Appendix B)

**Experiment: Adding BM25 (sparse) to dense retrieval**

| Config | w_semantic | w_keyword (BM25) | Prec | Rec | DRM |
|--------|------------|------------------|------|-----|-----|
| **Dense only** | 100% | 0% | **11.03%** | **41.80%** | 19.29% |
| Hybrid 1 | 75% | 25% | 10.57% | 42.54% | **19.11%** |
| Hybrid 2 | 50% | 50% | 9.54% | 41.47% | **18.45%** |
| Hybrid 3 | 25% | 75% | 8.23% | 36.56% | **18.18%** |

**Findings:**
- BM25 **improved DRM** slightly (better document selection)
- But **lowered precision and recall** (worse snippet quality)
- Explanation:
  - Summaries: structured, rich in identifiers → sparse matching excels
  - Chunk bodies: natural language, few keywords → semantic similarity better
- **Decision**: Use **dense-only** due to better text-level metrics + less overhead

### Summarization Prompts

**Generic Prompt:**
```
System: You are an expert legal document summarizer.
User: Summarize the following legal document text.
Focus on extracting the most important entities, core purpose,
and key legal topics. The summary must be concise, maximum
{char_length} characters long, and optimized for providing
context to smaller text chunks. Output only the summary text.
Document: {document_content}
```

**Expert-Guided Prompt Structure:**
```
1. Identify document type (NDA, Privacy Policy, Other)
2. Apply type-specific template
3. Extract key differentiating legal variables
4. Prioritize critical identifiers (parties, dates, subject matter)
5. Max {char_length} characters
6. Ignore missing template fields
```

**Generation:**
- Model: **gpt-4o-mini** (Hurst et al., 2024)
- Length: ~150 chars with ±20 tolerance
- Regeneration if exceeds with reduced char_length

---

## Výsledky

### DRM Reduction (Primary Finding)

**Figure 2: DRM Comparison**

| Method | DRM @ k=1 | DRM @ k=8 | DRM @ k=64 | Average |
|--------|-----------|-----------|------------|---------|
| **Baseline** | ~65% | ~60% | ~80% | **~67%** |
| **SAC** | ~22% | ~25% | ~38% | **~28%** |

**Improvement: ~58% reduction in DRM** (halving mismatch rate)

**Per-Dataset DRM @ k=8 (SAC):**
- **cuad**: ~10% (general contracts)
- **maud**: ~15% (merger agreements)
- **contractnli**: ~42% (NDAs) ← still challenging
- **privacy_qa**: ~28% (privacy policies)

**Conclusion:**
SAC dramatically reduces document-level mismatch across wide range of hyperparameters.

### Text-Level Precision & Recall (Figure 3)

**Averaged across all datasets:**

| Top-K | Baseline Prec | SAC Prec | Baseline Rec | SAC Rec |
|-------|---------------|----------|--------------|---------|
| k=1 | 0.20 | **0.16** | 0.08 | **0.16** |
| k=4 | 0.12 | **0.14** | 0.22 | **0.34** |
| k=8 | 0.09 | **0.12** | 0.28 | **0.42** |
| k=16 | 0.07 | **0.10** | 0.32 | **0.48** |
| k=32 | 0.04 | **0.07** | 0.33 | **0.54** |
| k=64 | 0.02 | **0.04** | 0.34 | **0.58** |

**Key Findings:**
- SAC consistently outperforms baseline
- As top-k increases:
  - Precision decreases (more noise)
  - Recall increases (more coverage)
- SAC maintains better precision at all k values
- SAC achieves substantially higher recall

### Generic vs Expert-Guided SAC (Unexpected Result!)

**Figure 3 shows virtually identical performance:**

| Method | Precision @ k=8 | Recall @ k=8 |
|--------|-----------------|--------------|
| **Generic SAC** | **~0.12** | **~0.42** |
| Expert-Guided SAC | ~0.12 | ~0.38 |
| Baseline | ~0.09 | ~0.28 |

**Findings:**
- Generic summaries **outperform or match** expert-guided
- Expert-guided only slightly better v specific settings (larger chunks)
- Counter-intuitive given legal precision of expert templates

**Qualitative Example (Section 5.2) - NDA Query:**

Query: "Consider Evelozcity's NDA; Does document allow Receiving Party to independently develop similar information?"

**A. Baseline (No Summary):**
- ✗ Retrieved document: NDA-ROI-Corporation.txt (WRONG!)
- Retrieved snippet: "NON-DISCLOSURE AGREEMENT FOR PROSPECTIVE PURCHASERS"
- Comment: Complete failure, distracted by structural similarity

**B. Generic Summary (150 chars):**
- ✔ Retrieved document: NDA-Evelozcity.txt (CORRECT!)
- Summary: "Non-Disclosure Agreement between Evelozcity and Recipient to protect confidential information shared during a meeting."
- Retrieved snippet: "(d) is independently developed by or for the Recipient..."
- **Precision: 97%, Recall: 50%**
- Comment: Successful! Relevant clause retrieved

**C. Expert Summary (150 chars):**
- ✔ Retrieved document: NDA-Evelozcity.txt (CORRECT!)
- Summary: "NDA between Evelozcity and Recipient; covers vehicle prototypes, confidentiality obligations, exclusions, 5-yr term, CA governing law."
- Retrieved snippet: "NON-DISCLOSURE AGREEMENT This NON-DISCLOSURE..."
- Comment: Correct document, but **irrelevant boilerplate snippet**

**D. Expert Summary (300 chars):**
- ✔ Retrieved document: NDA-Evelozcity.txt (CORRECT!)
- Summary: "**Definition**: Non-public vehicle prototypes. **Parties**: Evelozcity, CA; Recipient: [Name Not Provided]. **Obligations**: Keep confidential, limit access. **Exclusions**: Public knowledge, independent development."
- Retrieved snippet: Same boilerplate as C
- Comment: Richer summary, but same irrelevant snippet

### Hypotézy vysvětlující Generic > Expert

**Hypothesis 1: Balance vs Specificity**
- Generic summaries strike better balance mezi:
  - Distinctiveness (odlišení dokumentů)
  - Broad semantic alignment (alignment s wide variety queries)
- Expert-guided cues may **overfit to narrow features**
- Improves retrieval only v very specific cases
- Reduces robustness across broader range user intents

**Hypothesis 2: Embedding Model Capacity**
- Informationally dense, structured language expert summaries
- May pose challenges for **smaller embedding models**
- Must compress both summary + chunk → single vector
- Needs stronger, more capacious embedding models
- Future experiments needed

**Hypothesis 3: Embedding Space Complexity**
- Strong global signal from summary může overshadow local relevance chunk
- Interaction mezi summaries + chunks v embedding space is complex
- Needs direct analysis: clustering, dimensionality reduction

**Legal Expert Perspective:**
- Expert summaries are legally **richer, more structured**
- Contain highly discriminative information
- **Superior for differentiating** between documents of same type
- But this doesn't translate to better text-level snippet retrieval

---

## Challenges of Legal Text for RAG (Section 2.2)

### 1. Lexical Redundancy
- **Highly standardized** language
- **Boilerplate clauses**, formally defined phrases
- Specialized terminology **repeated across thousands** documents
- Example: NDAs within database structurally **almost identical**
- Differ only v critical variables: party names, dates
- High similarity **confuses retrieval models**

### 2. Hierarchical Structure
- Complex layouts: nested sections, subsections, cross-references
- Standard chunking **ignores document hierarchy**
- Cuts off logical connections
- Retrieved chunks may appear relevant but **lose intended meaning**

### 3. Fragmented Information
- Answering legal question often requires **synthesizing info scattered** across:
  - Multiple sections
  - Different documents
- Example: Interpreting exception clause may depend on definitions introduced earlier
- Retrieval must **capture distributed factual dependencies**

### 4. Provenance and Traceability
- **Provenance of information = high importance**
- Answering correctly is insufficient if supporting text from **wrong document**
- Undermines legal validity, erodes user trust
- Legal professionals require **transparent, verifiable reasoning trail**
- Every information piece must be **validated against its source**
- Makes **document-faithful retrieval** fundamental reliability measure

---

## Srovnání s Related Work

### Context Enrichment Strategies

| Approach | Type | Complexity | Our Work |
|----------|------|------------|----------|
| **Small2Big** | Local | Low | Expand chunks s surrounding sentences |
| **Metadata** | Global | Low-Med | Standard (timestamps) or Artificial |
| **Contextual Retrieval** (Anthropic) | Global | Medium | Chunk-specific explanatory context |
| **Reverse HyDE** | Global | Medium | Synthetic questions chunk could answer |
| **QuIM-RAG** | Global | Medium | Question-based metadata |
| **SAC (ours)** | **Global** | **Low** | **Single document summary → all chunks** |
| **RAPTOR** | Hierarchical | High | Hierarchical indexing structure |
| **Knowledge Graphs** | Structural | High | Model legal relationships |
| **Late Chunking** | Embedding | High | Embed full document, then chunk |
| **Long-context Models** | Architecture | Very High | Process 100k+ tokens, no chunking |

**SAC Advantages:**
- **Lightweight**: 1 LLM call per document
- **Modular**: Integrable do existing pipelines
- **Scalable**: Minimal computational overhead
- **Resource-efficient**: Unlike complex architectural solutions

### Legal NLP Specific Work

**Italian Tax Law Decisions (Pisano et al., 2025):**
- Modular, expert-validated summarization
- Solid basis for downstream semantic search
- Two-step method combines separate summary parts
- Our meta-prompt: conditional logic integrates classification + summarization implicitly

**Legal Document Summarization (Akter et al., 2025):**
- Moving beyond generic summarization vital for complex legal texts
- Our findings align with importance expert knowledge
- But show generic can be sufficient for retrieval purposes

---

## Limitace

### 1. Dataset Scope
- **Restricted to particular categories** legal documents
- **Exclusively in English**
- Don't cover full spectrum: legislation, case law, other contracts
- **Jurisdiction-specific**: largely common-law contexts
- Legal meaning highly jurisdiction-specific

### 2. Isolated Intervention
- Focus on **isolated intervention** within standard RAG
- Clearly measure impact
- **Residual mismatch rates remain significant**
- SAC = valuable component, **not complete solution**

### 3. Retrieval-Only Evaluation
- Study focuses **exclusively on retrieval stage**
- No end-to-end generation evaluation
- Future work: impact on downstream generation

### 4. Generic vs Expert Mystery
- Counter-intuitive result not fully explained
- Needs deeper technical analysis
- Embedding space visualization needed

---

## Future Work

### Promising Next Steps

**1. Hierarchical Summarization:**
- Extend principle hierarchically
- Summaries at paragraph, section, document level
- Context at multiple granularities

**2. Query Optimization:**
- Query transformation, expansion, routing
- Bridge semantic gap mezi user questions a formal legal language

**3. Reranking:**
- More powerful model re-evaluates top-k chunks
- Improve final selection before generation

**4. Benchmark Against Other Methods:**
- Compare SAC vs Late Chunking (Günther et al., 2024)
- Compare SAC vs RAPTOR (Sarthi et al., 2024)
- Understand relative strengths

**5. Stronger Embedding Models:**
- Test with more capacious models
- Investigate expert summary performance
- Analyze if dense legal info processed better

**6. Embedding Space Analysis:**
- Clustering, dimensionality reduction
- Visualize summary + chunk concatenation behavior
- Understand interaction dynamics

**7. End-to-End Evaluation:**
- Adapt benchmark (e.g., Australian Legal QA)
- Measure impact on generation quality
- Evaluate hallucination rates

**8. Multi-Jurisdiction Testing:**
- Test beyond common law
- Different legal systems
- Multi-language support

---

## Závěr

Paper addresses **critical challenge retrieval reliability** v RAG systems pro large legal databases.

### Klíčové přínosy

**1. Identified & Quantified DRM:**
- Document-Level Retrieval Mismatch = dominant failure mode
- Retrievers confused by legal boilerplate
- Select text from entirely incorrect documents
- **DRM rates >95%** in some datasets (baseline)

**2. Proposed SAC Solution:**
- Summary-Augmented Chunking = simple, efficient technique
- Prepends document-level summaries k each chunk
- Injects global context
- **Drastically reduces DRM** (~58% reduction, halving mismatch)
- **Improves text-level precision & recall**

**3. Generic > Expert Finding:**
- Generic summaries outperform expert-guided
- For retrieval purposes, **broad semantic cues more robust**
- Dense, structured legal summaries less effective
- **Meaningful gains without heavy domain engineering**

### Praktický dopad

**Pro practitioners:**
- **Easily adoptable technique**
- **Tangible improvements** bez need domain-specific fine-tuning
- **Minimal infrastructure changes**
- **Scalable** even to large, dynamic legal databases

**Pro "LLMs as legal readers" vision:**
- If future legal documents become longer & more comprehensive
- Retrieval reliability **even more critical**
- SAC = practical step toward building **systems process such documents reliably**

**Not a complete solution alone:**
- But valuable component for reliable RAG
- Best path: **combining SAC with other modules**
- Query optimization, reranking, hierarchical summarization

### Final Thoughts

SAC demonstrates that **simple, modular interventions** v pre-retrieval stage can yield significant improvements. Unlike complex architectural solutions:
- **Inexpensive**: 1 summary per document
- **Integrable**: Seamlessly into existing pipelines
- **Practical**: Real-world deployable

Work brings us **closer to future kde AI serves as trusted partner** v legal profession, making AI more trustworthy tool for navigating complexity legal texts.

---

## Code Availability

**GitHub Repository:**
https://github.com/DevelopedByMarkus/summary-augmented-chunking.git

**Reproducibility:**
- Open-source embedding models
- Clear hyperparameters
- Public benchmark (LegalBench-RAG)
