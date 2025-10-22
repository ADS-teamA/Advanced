# Windows Quick Start Guide

**TL;DR:** Windows má problémy s PyTorch DLL. Doporučení: Použij cloud embeddings místo lokálních.

---

## ⚡ Nejrychlejší Řešení (Doporučené)

Vyhni se problémům s PyTorch úplně - použij cloud embeddings:

```bash
# 1. Nainstaluj základní závislosti
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv sync

# 2. Zkopíruj a uprav .env
copy .env.example .env
```

**V `.env` nastav:**
```bash
ANTHROPIC_API_KEY=sk-ant-your_key_here
OPENAI_API_KEY=sk-your_key_here
EMBEDDING_MODEL=text-embedding-3-large  # Cloud embedding - BEZ PROBLÉMŮ!
```

**Spusť:**
```bash
python run_pipeline.py data\dokument.pdf
```

**Hotovo!** ✅ Žádné DLL chyby, vše funguje.

---

## 🔧 Pokud Máš DLL Error

**Chyba:**
```
OSError: [WinError 1114] Error loading "C:\...\torch\lib\c10.dll"
```

**Řešení 1: Visual C++ Redistributables (nejčastější)**
1. Stáhni: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Nainstaluj
3. Restartuj PowerShell/CMD
4. Zkus znovu

**Řešení 2: Reinstall PyTorch**
```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Řešení 3: Použij cloud embeddings (doporučené)**
```bash
# V .env změň:
EMBEDDING_MODEL=text-embedding-3-large
```

---

## 🆚 Cloud vs Local Embeddings

### Cloud (Doporučené pro Windows)

**Výhody:**
- ✅ Žádné instalační problémy
- ✅ Rychlé
- ✅ Vysoká kvalita
- ✅ Funguje všude

**Nevýhody:**
- ❌ Vyžaduje API klíč ($$)
- ❌ Síťové připojení

**Modely:**
- `text-embedding-3-large` (OpenAI, 3072D)
- `voyage-3-large` (Voyage AI, 1024D, nejlepší kvalita)

### Local (BGE-M3)

**Výhody:**
- ✅ Zdarma
- ✅ Offline
- ✅ Multilingual

**Nevýhody:**
- ❌ Složitá instalace na Windows
- ❌ Pomalé na CPU
- ❌ DLL problémy

**Doporučení:** Pokud nemáš NVIDIA GPU, použij cloud embeddings.

---

## 📋 Kompletní Windows Setup (Krok za Krokem)

### 1. Python a uv

```bash
# PowerShell (jako Admin)
# Zkontroluj Python verzi (3.10+ required)
python --version

# Nainstaluj uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restartuj PowerShell
```

### 2. Stáhni Projekt

```bash
git clone <repository-url>
cd MY_SUJBOT
```

### 3. Nainstaluj PyTorch (DŮLEŽITÉ!)

```bash
# MUSÍ být PŘED uv sync!
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Ověř instalaci
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
```

**Pokud chyba:** Nainstaluj Visual C++ Redistributables (viz výše)

### 4. Nainstaluj Aplikaci

```bash
uv sync
```

### 5. Konfigurace

```bash
# Zkopíruj template
copy .env.example .env

# Uprav .env v Notepadu
notepad .env
```

**Minimální .env pro Windows:**
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Pro cloud embeddings (DOPORUČENO)
OPENAI_API_KEY=sk-your_key_here
EMBEDDING_MODEL=text-embedding-3-large

# Nebo pro Voyage AI (nejlepší kvalita)
# VOYAGE_API_KEY=your_voyage_key_here
# EMBEDDING_MODEL=voyage-3-large
```

### 6. Testuj

```bash
# Test import
python -c "from src.indexing_pipeline import IndexingPipeline; print('OK')"

# Test pipeline
python run_pipeline.py --help
```

**Pokud vše OK, můžeš indexovat dokumenty:**
```bash
python run_pipeline.py data\dokument.pdf
```

---

## 🎯 Které API Klíče Potřebuji?

**Minimální konfigurace:**
```bash
ANTHROPIC_API_KEY=...        # REQUIRED (summaries)
OPENAI_API_KEY=...          # REQUIRED (pro text-embedding-3-large)
```

**Alternativa s Voyage AI:**
```bash
ANTHROPIC_API_KEY=...        # REQUIRED (summaries)
VOYAGE_API_KEY=...          # Pro voyage-3-large embeddings
```

**Kde získat klíče:**
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys
- Voyage AI: https://www.voyageai.com/

---

## 🐛 Časté Problémy

### ImportError: No module named 'torch'

**Řešení:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### ModuleNotFoundError: No module named 'sentence_transformers'

**Důvod:** Zkoušíš použít BGE-M3 local embeddings.

**Řešení:**
```bash
# Možnost 1: Nainstaluj sentence-transformers
uv pip install sentence-transformers

# Možnost 2: Použij cloud embeddings (jednodušší)
# V .env nastav: EMBEDDING_MODEL=text-embedding-3-large
```

### Pipeline spadne s "API key not found"

**Řešení:**
```bash
# Zkontroluj .env soubor
type .env

# Ujisti se, že klíče jsou správně nastavené
# Restartuj PowerShell/CMD po úpravě .env
```

### Docling extrakce je pomalá

**Normální:** Docling používá ML modely pro layout detection, je CPU-intensive.

**Tipy:**
- První run je pomalý (stahuje modely)
- Další runs jsou rychlejší (modely v cache)
- Běží na CPU (GPU by pomohl, ale není nutný)

---

## 📖 Další Dokumentace

- **[INSTALL.md](INSTALL.md)** - Kompletní instalační návod pro všechny platformy
- **[README.md](README.md)** - Přehled projektu
- **[PIPELINE.md](PIPELINE.md)** - Technické detaily a research
- **[CLAUDE.md](CLAUDE.md)** - Development guide

---

## 💡 Doporučení

**Pro Windows uživatele:**
1. ✅ Použij `text-embedding-3-large` nebo `voyage-3-large`
2. ✅ Nainstaluj Visual C++ Redistributables
3. ✅ Instaluj PyTorch PŘED `uv sync`
4. ❌ Vyhni se BGE-M3 pokud nemáš NVIDIA GPU

**Nejjednodušší setup:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-large
```

**Žádné problémy, funguje spolehlivě. 🚀**
