# Installation Guide - MY_SUJBOT RAG Pipeline

Cross-platform installation instructions for Windows, macOS, and Linux.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Linux Installation](#linux-installation)
- [Troubleshooting](#troubleshooting)
- [Embedding Model Options](#embedding-model-options)

---

## Prerequisites

**Required:**
- Python 3.10 or higher
- `uv` package manager (recommended) or `pip`
- 8GB+ RAM (16GB recommended for local embeddings)

**Install uv:**
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Windows Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MY_SUJBOT
```

### Step 2: Install PyTorch (CPU version)

**IMPORTANT:** Windows requires PyTorch to be installed separately BEFORE other dependencies.

```bash
# Install PyTorch CPU version (works on all Windows machines)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

**For NVIDIA GPU support (optional):**
```bash
# Check CUDA version first
nvidia-smi

# Install PyTorch with CUDA 11.8 (adjust version as needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Application Dependencies
```bash
# Install core dependencies
uv sync

# OR with pip
pip install -e .
```

### Step 4: Choose Embedding Model

**Option A: Cloud-based (Recommended for Windows)**
```bash
# No additional installation needed
# Set API keys in .env (see Configuration section)
```

**Option B: Local BGE-M3 (CPU - slower)**
```bash
uv pip install sentence-transformers
```

### Step 5: Configuration
```bash
# Copy environment template
copy .env.example .env

# Edit .env with your API keys
notepad .env
```

Add to `.env`:
```bash
# For cloud embeddings (recommended)
ANTHROPIC_API_KEY=sk-ant-xxxxx  # Required for summaries
OPENAI_API_KEY=sk-xxxxx         # For OpenAI embeddings (optional)
VOYAGE_API_KEY=xxxxx            # For Voyage AI embeddings (optional)

# Embedding model selection
EMBEDDING_MODEL=text-embedding-3-large  # or "voyage-3-large" or "bge-m3"
```

### Step 6: Verify Installation
```bash
python -c "from src.indexing_pipeline import IndexingPipeline; print('✓ Installation successful!')"
```

---

## macOS Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MY_SUJBOT
```

### Step 2: Install Dependencies

**For Apple Silicon (M1/M2/M3):**
```bash
# PyTorch with MPS (Metal Performance Shaders) acceleration
uv sync

# Install local embeddings support (optional, runs on Apple Silicon GPU)
uv pip install sentence-transformers

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**For Intel Macs:**
```bash
# Standard installation
uv sync

# For local embeddings (CPU only)
uv pip install sentence-transformers
```

### Step 3: Configuration
```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

Add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx         # Optional
VOYAGE_API_KEY=xxxxx            # Optional

# For Apple Silicon - use local BGE-M3 for free GPU-accelerated embeddings
EMBEDDING_MODEL=bge-m3          # Runs locally on MPS

# Or use cloud embeddings
# EMBEDDING_MODEL=text-embedding-3-large
```

### Step 4: Verify Installation
```bash
python run_pipeline.py --help
```

---

## Linux Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MY_SUJBOT
```

### Step 2: Install PyTorch

**For CPU-only:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**For NVIDIA GPU (CUDA):**
```bash
# Check CUDA version
nvcc --version

# Install PyTorch with CUDA 11.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies
```bash
uv sync

# Optional: Local embeddings
uv pip install sentence-transformers
```

### Step 4: Configuration
```bash
cp .env.example .env
nano .env
```

### Step 5: Verify Installation
```bash
python -c "import torch; from src.indexing_pipeline import IndexingPipeline; print('✓ Success')"
```

---

## Troubleshooting

### Windows: DLL Load Failed Error

**Error:**
```
OSError: [WinError 1114] Error loading "C:\...\torch\lib\c10.dll"
```

**Solutions:**

1. **Install Visual C++ Redistributables:**
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run installer
   - Restart terminal

2. **Reinstall PyTorch:**
   ```bash
   uv pip uninstall torch torchvision
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Use cloud embeddings instead:**
   - Set `EMBEDDING_MODEL=text-embedding-3-large` in `.env`
   - Requires `OPENAI_API_KEY`

### macOS: MPS Not Available

**If MPS is not detected on Apple Silicon:**
```bash
# Check macOS version (requires macOS 12.3+)
sw_vers

# Update PyTorch
uv pip install --upgrade torch torchvision

# Fall back to CPU if needed
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_pipeline.py
```

### Linux: CUDA Out of Memory

```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Or use cloud embeddings
# Set EMBEDDING_MODEL=text-embedding-3-large in .env
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:**
```bash
# BGE-M3 local embeddings require sentence-transformers
uv pip install sentence-transformers

# OR use cloud embeddings (no installation needed)
# Set EMBEDDING_MODEL=text-embedding-3-large in .env
```

### Docling Extraction Errors

**Error:** PyTorch-related errors during document extraction

**Solution:**
Docling requires PyTorch for layout detection. This is unavoidable but only needs CPU version:
```bash
# Windows
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Linux
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# macOS
uv sync  # Already includes correct PyTorch
```

---

## Embedding Model Options

The pipeline supports multiple embedding models. Choose based on your needs:

### 1. OpenAI (Cloud - Recommended for Windows)
**Pros:**
- No local installation issues
- High quality (3072 dimensions)
- Fast API
- Works on all platforms

**Cons:**
- Requires API key ($$$)
- Network latency

**Setup:**
```bash
# .env
OPENAI_API_KEY=sk-xxxxx
EMBEDDING_MODEL=text-embedding-3-large
```

### 2. Voyage AI (Cloud - Best Quality)
**Pros:**
- SOTA performance (MLEB #1)
- Legal/technical optimized models
- Fast API

**Cons:**
- Requires API key ($$$)

**Setup:**
```bash
# .env
VOYAGE_API_KEY=xxxxx
EMBEDDING_MODEL=voyage-3-large  # or kanon-2 or voyage-law-2

# Install client
uv pip install voyageai
```

### 3. BGE-M3 (Local - Best for Apple Silicon)
**Pros:**
- FREE - runs locally
- Multilingual (100+ languages including Czech)
- GPU acceleration on Apple Silicon (MPS)
- No API keys needed

**Cons:**
- Requires PyTorch installation
- Slower on CPU
- Windows installation can be tricky

**Setup:**
```bash
# Install
uv pip install sentence-transformers

# .env
EMBEDDING_MODEL=bge-m3

# No API key needed!
```

### Comparison Table

| Model | Platform | Cost | Quality | Speed | Installation |
|-------|----------|------|---------|-------|--------------|
| **text-embedding-3-large** | All | $$$ | High | Fast | Easy |
| **voyage-3-large** | All | $$$ | Highest | Fast | Easy |
| **voyage-law-2** | All | $$$ | High (legal) | Fast | Easy |
| **bge-m3** (Mac GPU) | macOS M1+ | FREE | Good | Fast | Medium |
| **bge-m3** (CPU) | All | FREE | Good | Slow | Hard (Windows) |

### Recommendation by Platform

**Windows:**
- **Best:** `text-embedding-3-large` (OpenAI) - avoid PyTorch issues
- **Alternative:** `voyage-3-large` - best quality
- **Avoid:** `bge-m3` - installation issues

**macOS (Apple Silicon):**
- **Best:** `bge-m3` - FREE, GPU-accelerated
- **Alternative:** `text-embedding-3-large` - if you prefer cloud

**macOS (Intel):**
- **Best:** `text-embedding-3-large` - avoid slow CPU inference
- **Alternative:** `voyage-3-large`

**Linux (with NVIDIA GPU):**
- **Best:** `bge-m3` - FREE, GPU-accelerated
- **Alternative:** `text-embedding-3-large`

**Linux (CPU only):**
- **Best:** `text-embedding-3-large` - avoid slow CPU inference
- **Alternative:** `voyage-3-large`

---

## Quick Start After Installation

### Single Document
```bash
python run_pipeline.py data/document.pdf
```

### Batch Processing
```bash
python run_pipeline.py data/documents/
```

### Python API
```python
from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from pathlib import Path

config = IndexingConfig(
    embedding_model="text-embedding-3-large",  # or bge-m3, voyage-3-large
    enable_knowledge_graph=False,  # Optional
)

pipeline = IndexingPipeline(config)
result = pipeline.index_document(Path("document.pdf"))

# Save results
result["vector_store"].save("output/vector_store")
```

---

## Getting API Keys

### Anthropic (Required for Summaries)
1. Visit: https://console.anthropic.com/
2. Sign up / Log in
3. Go to API Keys
4. Create new key
5. Copy to `.env` as `ANTHROPIC_API_KEY`

### OpenAI (Optional - Embeddings)
1. Visit: https://platform.openai.com/api-keys
2. Sign up / Log in
3. Create new secret key
4. Copy to `.env` as `OPENAI_API_KEY`

### Voyage AI (Optional - Best Embeddings)
1. Visit: https://www.voyageai.com/
2. Sign up
3. Get API key
4. Copy to `.env` as `VOYAGE_API_KEY`

---

## Support

**Documentation:**
- `README.md` - Project overview
- `PIPELINE.md` - Technical details and research
- `CLAUDE.md` - Development guide

**Common Issues:**
- Windows DLL errors → Use cloud embeddings
- macOS MPS not available → Update macOS/PyTorch
- Linux CUDA errors → Use CPU version
- Import errors → Check virtual environment activation

**For More Help:**
Open an issue on GitHub with:
- Operating system and version
- Python version (`python --version`)
- Full error message
- Output of `uv pip list`
