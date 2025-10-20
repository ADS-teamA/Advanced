# Enhancing Contract Negotiations with LLM-Based Legal Document Comparison

**Autoři:** Savinay Narendra, Kaushal Shetty, Adwait Ratnaparkhi
**Instituce:** Machine Learning Center of Excellence, JPMorgan Chase & Co.
**Rok:** 2024
**Workshop:** Natural Legal Language Processing Workshop 2024

---

## Shrnutí

Paper představuje první LLM-based přístup pro srovnávání právních smluv s jejich šablonami pomocí Natural Language Inference (NLI). Systém identifikuje odchylky mezi šablonami a skutečnými smlouvami, což pomáhá právníkům při vyjednávání smluv a vylepšování šablon.

---

## Co bylo implementováno

### 1. Hlavní systém pro porovnání smluv
- **NLI-based approach**: Rozdělení šablonových klauzulí na klíčové koncepty (sub-clauses)
- **Bidirectional comparison**:
  - Template → Contract: Testování, zda koncepty ze šablony jsou entailed v kontraktu
  - Contract → Template: Identifikace dodatečných informací v kontraktu, které nejsou v šabloně
- **Clause library**: Organizovaný katalog non-entailed konceptů filtrovaný podle frekvence
- **Evidence extraction**: Identifikace konkrétních spanů v dokumentech, které podporují NLI rozhodnutí

### 2. RAG (Retrieval Augmented Generation) Pipeline
- Vector database pro ukládání embeddings
- Retrieval relevantních klauzulí z kontraktů
- Cross-reference resolution - připojení odkazů na jiné sekce dokumentu
- Generování odpovědí pomocí GPT-4 na specifické dotazy o konceptech

### 3. OCR Pipeline pro scannované PDFs
- **Document Image Transformer (DiT)**: Identifikace bounding boxes pro sekce dokumentu
- **EasyOCR**: Extrakce textu z identifikovaných bounding boxes
- Intelligent chunking dokumentů do subsections

### 4. Amendment Handling System
- Automatické začlenění amendments do master agreements
- 3-stage process:
  1. Summarizace amendments (extrakce sekcí a změn)
  2. Extrakce key data v JSON formátu
  3. Generování modifikované master smlouvy

### 5. Concept Extraction
- Automatické rozdělení template clauses na fundamentální koncepty
- Každá klauzule je rozdělena na multiple sub-clauses/concepts
- Zachování integrity a kontextu původní klauzule

---

## Technická implementace

### Modely

#### GPT-4
- Komerční API od OpenAI
- Multimodal model s human-level performance
- Použit pro inference a prompt-based tasks
- Žádný fine-tuning

#### Mixtral 8x7B
- Sparse Mixture of Experts (SMoE) model
- Fine-tuning pomocí **LoRA (Low-Rank Adaptation)**:
  - Zmrazení pre-trained weights
  - Injekce trainable rank decomposition matrices
  - Výrazně menší počet trainable parametrů
- Training parametry:
  - Batch size: 1 (per device)
  - Gradient accumulation steps: 1
  - Total training steps: 4000
  - Learning rate: 2.5e-5
  - Precision: bf16
  - Gradient checkpointing: enabled

### Datasety

#### ContractNLI (Public)
- 607 non-disclosure agreements (NDAs)
- 17 fixed hypotheses per agreement
- Labels: Entailment, Contradiction, NotMentioned
- Evidence spans jako lista indexů
- První dataset pro document-level NLI na kontraktech

#### Internal Dataset (JPMorgan)
- 25 master contracts
- 5 kontraktů s amendments
- Časové rozpětí: červen 2007 - srpen 2023 (15+ let)
- Typy kontraktů:
  1. Software and Maintenance Agreement
  2. Professional Services Agreement
  3. Software License Agreement
  4. Application Service Provider Agreement
  5. Hardware Agreement

### Prompting Strategy

#### Prompt 1 - NLI Classification
```
Given a document and a hypothesis, determine whether the document
entails or contradicts the hypothesis. Answer strictly as
"Entailment" or "Contradiction"
```

#### Prompt 2 - Evidence Extraction
```
Given a document and a hypothesis, if the label is 'Entailment'
extract evidence verbatim from the document that support the
hypothesis. If the label is 'Contradiction', extract evidence
verbatim from the document that contradicts the hypothesis
```

#### Další prompty (Appendix A)
- Prompt pro summarizaci amendments
- Prompt pro extrakci key data z amendments v JSON
- Prompt pro generování modified master contract
- Prompt pro concept extraction z template clauses
- Prompt pro RAG query (coverage check)
- Prompt pro reverse comparison (additional info detection)

### Pipeline Architecture

```
Amendments 1,2,...n → Summarize Amendments → Create Consolidated Contract
                                              (Master + Amendments)
                                                      ↓
                                              Embedding Model
                                                      ↓
Template Concepts ← Concept Extraction        Vector Database
       ↓                                              ↓
   Embedding ← ───────────────────────────→ Retrieved Chunks
   Model                                              ↓
       ↓                                          GPT-4 ← Prompts
   Vector DB                                          ↓
                                                  Response
```

### Analyzované klauzule
- Limitations of Liability
- Insurance
- Indemnity
- Representations and Warranties
- Red Flags
- System Modifications
- Assignment
- Source Code Escrow
- Audits

---

## Výsledky

### ContractNLI Dataset - NLI Task

| Model | F1 (Contradiction) | F1 (Entailment) | Accuracy |
|-------|-------------------|-----------------|----------|
| **GPT-4** | **0.70** | **0.91** | **0.87** |
| **Mixtral 8x7B** | **0.74** | **0.93** | **0.90** |
| Span NLI BERT | 0.389 | 0.839 | 0.87 |

**Klíčová zjištění:**
- LLM modely výrazně lepší na Contradiction label (GPT-4: +80%, Mixtral: +90% improvement)
- Mixtral dosáhl nejvyšší F1 score na obou labels
- Oba LLM modely překonaly Span NLI BERT

### ContractNLI Dataset - Evidence Identification

| Model | Mean Average Precision |
|-------|----------------------|
| **GPT-4** | **92.68%** |
| Mixtral | 79.8% |
| Span NLI BERT | 92.2% |

**Klíčová zjištění:**
- GPT-4 dosáhl slightly superior performance než Span NLI BERT
- Mixtral měl nižší výkon na evidence extraction
- MAP vypočítána průměrováním precision na recall levels s relevantními tokeny

### Internal Dataset (JPMorgan)

**Celková přesnost: 96.46%**

Accuracy vypočítána jako:
```
Accuracy = (Správně identifikované koncepty) / (Celkový počet konceptů)
```

Kde každý koncept je klasifikován jako:
- Entailed
- Contradicted
- Neutral

**Performance podle jednotlivých klauzulí:**

| Clause | Accuracy |
|--------|----------|
| Assignment | ~99% |
| Audits | ~97% |
| Indemnity | ~96% |
| Insurance | ~96% |
| Limitations of Liability | ~96% |
| Representations & Warranties | ~98% |
| Source Code Escrow | ~99% |
| System Modifications | ~99% |
| **Red Flags** | **~88%** (nejnižší) |

**Klíčová zjištění:**
- Velmi vysoká přesnost napříč všemi klauzulemi
- Red Flags clause měla nejnižší accuracy (nejkomplexnější)
- Model dokázal generovat natural language explanations rozdílů

### Sample Output

**Template Concept:**
```
The deliverables will not contain any malware, malicious programs
and will not store any data on computers, systems, or network.
```

**GPT-4 Analysis:**
```
The document does not explicitly state that the deliverables will
not contain any malware... However, it does mention that the
supplier will comply with certain security and risk management
policies, and that the supplier is responsible for assessing and
remediating security vulnerabilities.
```

**Sources:**
- Section 5.10 Application Security
- Section 5.5 Critical Vulnerabilities

---

## Inovace a přínosy

### Výzkumné přínosy
1. **První NLI-based approach** pro direct comparison legal contracts vs templates
2. **Bidirectional comparison** - entailment testování v obou směrech
3. **Natural language output** - první systém generující textové vysvětlení rozdílů
4. **Clause library creation** - automatické vytváření katalogu schválených termínů

### Praktické přínosy
- Výrazné zkrácení času pro contract negotiations
- Automatická identifikace clause variations
- Podpora pro vytváření pre-negotiated Master Service Agreements (MSAs)
- Zlepšení konzistence a kvality kontraktů
- Handling komplexních amendments

### Technické inovace
- Kombinace DiT + EasyOCR pro OCR (lepší než Tesseract)
- RAG s cross-reference resolution
- Intelligent document chunking
- Concept-level granularity (ne jen sentence-level)

---

## Srovnání s předchozími pracemi

| Aspekt | Koreeda & Manning 2021 | Roegiest et al. 2023 | Lam et al. 2023 | Tento paper |
|--------|----------------------|---------------------|-----------------|-------------|
| Task | Sentence-level NLI | Legal Q&A | Clause drafting | Document comparison |
| Model | Span NLI BERT | Embedding-based | Multi-step method | GPT-4 + Mixtral |
| Output | Labels + spans | Structured answers | Clause suggestions | Natural language |
| Dataset | ContractNLI | Legal questions | LEDGAR (SEC) | ContractNLI + internal |
| Application | Contract review | Question answering | Clause modification | Contract negotiation |

---

## Limitace a rizika

### Identifikované limitace
1. **Annotation quality**: Anotátoři byli zkušení s legal dokumentací, ale ne trained lawyers
   - Potenciální impact na accuracy v komplexních případech

2. **Neutral class**: Na ContractNLI nebyla evaluována neutral třída
   - Zaměření pouze na Entailment/Contradiction

3. **Confidentiality**: Nemožnost zveřejnit specifické clause variations z internal datasetu

4. **OCR errors**: Možné chyby při OCR scannovaných dokumentů

### Praktická aplikace
- Model dosahuje 96.46% accuracy, což je dostatečné pro practical applications
- Vhodné jako nástroj pro legal professionals, ne jako kompletní náhrada
- Výstupy vyžadují review kvalifikovaným právníkem

---

## Závěr

Paper demonstruje úspěšnou aplikaci LLMs (zejména GPT-4) pro automatizaci contract review procesu. Klíčovým přínosem je:

1. **Superior performance**: Překonání všech předchozích metod na ContractNLI datasetu
2. **High accuracy**: 96.46% na real-world internal datasetu
3. **Practical applicability**: První systém produkující natural language comparison
4. **Comprehensive approach**: Handling amendments, OCR, RAG, evidence extraction

Výsledky ukazují významný potenciál pro:
- Automatizaci legal document analysis
- Zrychlení contract negotiations
- Zlepšení konzistence kontraktů
- Redukci manuální práce právníků

Tento přístup představuje významný krok vpřed v intersekci NLP a legal technology, s přímou aplikovatelností v enterprise prostředí.
