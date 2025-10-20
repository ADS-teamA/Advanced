# Jak spustit RAG Pipeline

## 🚀 Rychlý Start

### 1. Nastav API klíč v `.env`

```bash
# Otevři .env soubor
nano .env
```

```bash
# Minimální konfigurace (Claude + BGE-M3-v2 LOCAL):
ANTHROPIC_API_KEY=sk-ant-...    # Pro LLM (summaries)

# BGE-M3-v2 běží LOKÁLNĚ na tvém M1 Macu - žádný API klíč nepotřebuje! 🚀
```

**Získání API klíče:**
- Claude: https://console.anthropic.com/

**Proč BGE-M3-v2?**
- ✅ Běží lokálně na M1 (MPS acceleration)
- ✅ Žádné API klíče, žádné náklady
- ✅ Multilingual (100+ jazyků včetně češtiny)
- ✅ SOTA performance (close to commercial APIs)

### 2. Spusť pipeline

```bash
# Základní použití
python run_pipeline.py <cesta_k_pdf>

# Příklad
python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"
```

### 3. Najdi výstupy

Všechny výstupy jsou v:
```
output/<název_dokumentu>/<timestamp>/
├── phase1_extraction.json      # Struktura dokumentu
├── phase2_summaries.json       # Generované summaries
├── phase3_chunks.json          # Multi-layer chunky
└── phase4_vector_store/        # FAISS indexy
    ├── layer1.index
    ├── layer2.index
    ├── layer3.index
    └── metadata.pkl
```

---

## ⚙️ Konfigurace Modelů

### Výběr LLM (pro summaries)

Edituj `.env`:

```bash
# Claude Sonnet 4.5 (default, doporučeno)
LLM_PROVIDER=claude
LLM_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_API_KEY=sk-ant-...

# Claude Haiku 4.5 (rychlejší, levnější)
LLM_PROVIDER=claude
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI GPT-4o Mini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### Výběr Embedding Modelu

```bash
# BGE-M3-v2 (DEFAULT, doporučeno pro M1 Mac) ⭐
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=bge-m3
# Žádný API klíč - běží lokálně na M1 s MPS acceleration!
# Features: 1024D, multilingual (Czech), 8192 tokens, ZDARMA

# === Alternativy (vyžadují API klíče) ===

# Kanon 2 (#1 na MLEB 2025)
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=kanon-2
VOYAGE_API_KEY=pa-...

# Voyage 3 Large (#2 na MLEB)
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-3-large
VOYAGE_API_KEY=pa-...

# Voyage Law 2 (legal-optimized)
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-law-2
VOYAGE_API_KEY=pa-...

# OpenAI text-embedding-3-large
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=sk-...
```

---

## 📊 Výstup Pipeline

### PHASE 1: Extraction (phase1_extraction.json)

```json
{
  "document_id": "GRI_306_2016",
  "num_sections": 118,
  "hierarchy_depth": 4,
  "sections": [
    {
      "section_id": "sec_001",
      "title": "Introduction",
      "level": 1,
      "depth": 0,
      "path": "Introduction",
      "page_number": 1
    }
  ]
}
```

### PHASE 2: Summaries (phase2_summaries.json)

```json
{
  "document_id": "GRI_306_2016",
  "document_summary": "Technical specification for waste management...",
  "section_summaries": [
    {
      "section_id": "sec_001",
      "title": "Introduction",
      "summary": "Overview of waste management procedures..."
    }
  ]
}
```

### PHASE 3: Chunks (phase3_chunks.json)

```json
{
  "chunking_stats": {
    "layer1_count": 1,
    "layer2_count": 118,
    "layer3_count": 123,
    "total_chunks": 242,
    "layer3_avg_size": 450,
    "sac_avg_overhead": 150
  },
  "layer3": [
    {
      "chunk_id": "GRI_306_2016_L3_sec_001_chunk_0",
      "content": "[DOC SUMMARY] ... [CHUNK] The primary...",
      "raw_content": "The primary cooling system...",
      "metadata": {
        "layer": 3,
        "section_title": "Introduction",
        "page_number": 1
      }
    }
  ]
}
```

### PHASE 4: Vector Store (phase4_vector_store/)

```
phase4_vector_store/
├── layer1.index        # FAISS index pro Layer 1 (1 vektor)
├── layer2.index        # FAISS index pro Layer 2 (118 vektorů)
├── layer3.index        # FAISS index pro Layer 3 (123 vektorů)
└── metadata.pkl        # Metadata pro všechny chunky
```

---

## 🧪 Testování

### Test na GRI 306 dokumentu

```bash
# Stáhni testovací dokument
# Už máš v: data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf

# Spusť pipeline
python run_pipeline.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"

# Očekávané výsledky:
# - 118 sekcí
# - Hloubka hierarchie: 4
# - 242 celkových chunků (1 + 118 + 123)
# - ~2.9 MB vector store (pro Kanon 2: 1024D)
```

### Test s vlastním dokumentem

```bash
# 1. Vlož PDF do data/
cp /path/to/your.pdf data/

# 2. Spusť pipeline
python run_pipeline.py data/your.pdf

# 3. Najdi výstupy v:
ls -la output/your/<timestamp>/
```

---

## 🔍 Kontrola Výstupů

### Prohlédni JSON výstupy

```bash
# PHASE 1 - Struktura
cat output/*/phase1_extraction.json | jq '.sections[] | {title, level, depth}'

# PHASE 2 - Summaries
cat output/*/phase2_summaries.json | jq '.document_summary'

# PHASE 3 - Chunking stats
cat output/*/phase3_chunks.json | jq '.chunking_stats'

# PHASE 3 - První 3 chunky
cat output/*/phase3_chunks.json | jq '.layer3[0:3] | .[] | {chunk_id, section_title}'
```

### Zkontroluj Vector Store

```bash
# Velikost indexů
du -h output/*/phase4_vector_store/

# Struktura
ls -la output/*/phase4_vector_store/
```

---

## 💰 Náklady

### Claude Sonnet 4.5 (summaries)
- **Vstup:** $3 / 1M tokens
- **Výstup:** $15 / 1M tokens
- **GRI 306 (15 stran):** ~$0.02

### Claude Haiku 4.5 (summaries)
- **Vstup:** $0.80 / 1M tokens
- **Výstup:** $4 / 1M tokens
- **GRI 306 (15 stran):** ~$0.005

### BGE-M3-v2 (embeddings - LOCAL)
- **Cena:** ZDARMA 🎉
- **Běží lokálně na M1 Macu**
- **Žádné API volání, žádné náklady**

**Celkem pro GRI 306:**
- Claude Sonnet + BGE-M3-v2 (LOCAL): ~$0.02 ⭐
- Claude Haiku + BGE-M3-v2 (LOCAL): ~$0.005 ⭐

**Úspora vs cloud embeddings:**
- Vs Kanon 2: $0.003 saved per document
- Vs OpenAI: $0.002 saved per document
- Pro 1000 dokumentů: **$2-3 ušetřeno!**

---

## ⚠️ Troubleshooting

### "ANTHROPIC_API_KEY required"

```bash
# Nastav Claude API klíč
export ANTHROPIC_API_KEY="sk-ant-..."
# nebo edituj .env
```

### "VOYAGE_API_KEY required"

```bash
# Nastav Voyage AI klíč
export VOYAGE_API_KEY="pa-..."
# nebo edituj .env

# Nebo přepni na OpenAI embeddings:
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
```

### "voyageai package required"

```bash
# Nainstaluj Voyage AI SDK
uv pip install voyageai
```

### "anthropic package required"

```bash
# Nainstaluj Anthropic SDK
uv pip install anthropic
```

### "File not found"

```bash
# Zkontroluj cestu k PDF
ls -la "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"

# Pozor na mezery - escapuj je nebo použij uvozovky
python run_pipeline.py "data/my file.pdf"
```

---

## 📖 Další Informace

- **Kompletní dokumentace:** `README.md`
- **PHASE 1-2 detaily:** `IMPLEMENTATION_SUMMARY.md`
- **PHASE 3 detaily:** `PHASE3_COMPLETE.md`
- **PHASE 4 detaily:** `PHASE4_COMPLETE.md`
- **Research foundation:** `PIPELINE.md`

---

## 🎯 Next Steps (PHASE 5-7)

Po dokončení PHASE 1-4 můžeš implementovat:

**PHASE 5: Query & Retrieval**
- Embedding queries
- Hierarchical search (K=6)
- DRM prevention

**PHASE 6: Context Assembly**
- Strip SAC summaries
- Concatenate chunks
- Add citations

**PHASE 7: Answer Generation**
- Claude/GPT-4 integration
- Citation formatting
- Answer validation

---

**Aktuální Status:** PHASE 1-4 COMPLETED ✅
**Updated:** 2025-10-20
