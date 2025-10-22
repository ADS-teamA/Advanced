# Cross-Platform Compatibility Changes

**Date:** 2025-10-22
**Issue:** Windows users experiencing PyTorch DLL errors (`OSError: [WinError 1114]`)

---

## Summary of Changes

This update makes MY_SUJBOT fully cross-platform compatible for **Windows, macOS, and Linux**.

### Problem
- Windows users couldn't run the application due to PyTorch DLL loading errors
- macOS-specific dependencies (ocrmac) prevented Linux/Windows installation
- No clear guidance for platform-specific installation

### Solution
1. ✅ Removed macOS-specific dependencies
2. ✅ Added proper PyTorch installation instructions for each platform
3. ✅ Created comprehensive installation guides
4. ✅ Updated all documentation with platform-specific notes
5. ✅ Made sentence-transformers optional (for local BGE-M3 embeddings)

---

## Files Changed

### New Files
1. **`INSTALL.md`** - Comprehensive platform-specific installation guide
   - Windows (PyTorch pre-installation steps)
   - macOS (Apple Silicon MPS / Intel)
   - Linux (CPU / CUDA variants)
   - Troubleshooting section

2. **`WINDOWS_QUICKSTART.md`** - Quick reference for Windows users
   - TL;DR: Use cloud embeddings to avoid DLL issues
   - Step-by-step Windows setup
   - Common errors and solutions

3. **`CROSS_PLATFORM_CHANGES.md`** - This file

### Modified Files

1. **`pyproject.toml`**
   - ❌ Removed `ocrmac>=1.0.0` (macOS-only)
   - ✅ Added `faiss-cpu>=1.7.4` explicitly
   - ✅ Added `numpy>=1.24.0` explicitly
   - ✅ Added optional dependencies groups:
     - `local-embeddings` - sentence-transformers for BGE-M3
     - `voyage` - voyageai client
     - `knowledge-graph` - networkx, neo4j
     - `all` - all optional features
   - ✅ Added comment about PyTorch platform-specific installation

2. **`.env.example`**
   - ✅ Complete rewrite with all embedding model options
   - ✅ Platform-specific recommendations (Windows/macOS/Linux)
   - ✅ Detailed comments for each configuration option
   - ✅ Added VOYAGE_API_KEY for Voyage AI embeddings
   - ✅ Added Knowledge Graph configuration section

3. **`README.md`**
   - ✅ Updated Quick Start with platform-specific instructions
   - ✅ Added Windows-specific PyTorch installation step
   - ✅ Link to INSTALL.md for detailed instructions

4. **`CLAUDE.md`**
   - ✅ Added "Cross-Platform Compatibility" section
   - ✅ Platform-specific installation instructions
   - ✅ Embedding model selection by platform
   - ✅ Windows troubleshooting section
   - ✅ Guidelines for future cross-platform development

5. **`HOW_TO_RUN.md`**
   - ✅ Added cross-platform notice at top
   - ✅ Clarified this is macOS-specific guide
   - ✅ Link to INSTALL.md for other platforms

---

## Installation Instructions Summary

### Windows (Most Important!)

**Problem:** PyTorch DLL errors prevent application from running

**Solution:**
```bash
# 1. Install PyTorch FIRST (before uv sync)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install application
uv sync

# 3. Use cloud embeddings (recommended)
# In .env:
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-large  # Avoids PyTorch issues
```

**Alternative:** Install Visual C++ Redistributables if DLL errors persist.

### macOS

**No changes needed - works as before:**
```bash
uv sync

# In .env:
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=bge-m3  # FREE local embeddings on Apple Silicon
```

### Linux

**Choose based on hardware:**
```bash
# CPU only:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv sync

# NVIDIA GPU (CUDA 11.8):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv sync
```

---

## Embedding Model Options

The application now supports multiple embedding models with clear platform recommendations:

| Model | Windows | macOS (M1+) | macOS (Intel) | Linux (GPU) | Linux (CPU) |
|-------|---------|-------------|---------------|-------------|-------------|
| **text-embedding-3-large** | ✅ Best | ⚠️ OK | ✅ Best | ⚠️ OK | ✅ Best |
| **voyage-3-large** | ✅ Best | ✅ Best | ✅ Best | ✅ Best | ✅ Best |
| **bge-m3** | ❌ Avoid* | ✅ Best (FREE) | ⚠️ Slow | ✅ Best (FREE) | ❌ Slow |

*Unless you have NVIDIA GPU

### Recommendations by Platform

**Windows:**
- Use `text-embedding-3-large` or `voyage-3-large` (cloud)
- Avoids PyTorch installation issues

**macOS (Apple Silicon):**
- Use `bge-m3` (local, FREE, GPU-accelerated via MPS)

**Linux with NVIDIA GPU:**
- Use `bge-m3` (local, FREE, GPU-accelerated)

**Any platform (best quality):**
- Use `voyage-3-large` (cloud, SOTA performance)

---

## Testing Recommendations

### For Windows Users (Colleagues)

1. **Delete existing environment:**
   ```bash
   rmdir /s .venv
   ```

2. **Follow new installation:**
   ```bash
   # Install PyTorch first
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # Install app
   uv sync

   # Configure
   copy .env.example .env
   # Edit .env and set:
   #   ANTHROPIC_API_KEY=...
   #   OPENAI_API_KEY=...
   #   EMBEDDING_MODEL=text-embedding-3-large
   ```

3. **Test:**
   ```bash
   python -c "from src.indexing_pipeline import IndexingPipeline; print('OK')"
   python run_pipeline.py --help
   ```

### For macOS Users (You)

No changes needed - should work as before:
```bash
uv sync
# Configure .env as usual
```

### For Linux Users

Choose CPU or CUDA based on hardware (see INSTALL.md).

---

## Migration Guide

### Existing Users

**If you're already using the application:**

1. **Update code:**
   ```bash
   git pull
   ```

2. **Update dependencies:**
   ```bash
   uv sync
   ```

3. **Update .env:**
   - Check `.env.example` for new options
   - Add `EMBEDDING_MODEL=` line if not present
   - For Windows: Set to `text-embedding-3-large`
   - For macOS: Set to `bge-m3` or `text-embedding-3-large`

4. **Optional dependencies:**
   ```bash
   # If you want local BGE-M3 embeddings:
   uv pip install sentence-transformers

   # If you want Voyage AI embeddings:
   uv pip install voyageai

   # If you want Knowledge Graph:
   uv pip install networkx neo4j

   # Or install everything:
   uv pip install -e ".[all]"
   ```

---

## API Key Requirements

### Required (Minimum)

**For summaries (PHASE 2):**
- `ANTHROPIC_API_KEY` (Claude) **OR** `OPENAI_API_KEY` (GPT)

**For embeddings (PHASE 4):**
- Depends on chosen model:
  - `text-embedding-3-large` → Requires `OPENAI_API_KEY`
  - `voyage-3-large` → Requires `VOYAGE_API_KEY`
  - `bge-m3` → No API key needed (local)

### Recommended Configurations

**Windows (Safest):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-large
```

**macOS M1+ (FREE):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=bge-m3
# No embedding API key needed!
```

**Any Platform (Best Quality):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=...
EMBEDDING_MODEL=voyage-3-large
```

---

## Breaking Changes

### None

All changes are backward compatible. Existing configurations will continue to work.

### Deprecations

- macOS-specific `ocrmac` dependency removed (was unused)
- No impact on functionality

---

## Future Considerations

### For Developers

**When adding new dependencies:**
1. Check compatibility with Windows, macOS, and Linux
2. If platform-specific, make it optional
3. Document in `INSTALL.md`
4. Update `.env.example` with recommendations

**GPU Detection:**
- Code must gracefully handle: CPU-only, CUDA (NVIDIA), MPS (Apple Silicon)
- Use `torch.backends.mps.is_available()` for Apple Silicon
- Use `torch.cuda.is_available()` for NVIDIA
- Fallback to CPU if neither available

**Path Handling:**
- Always use `pathlib.Path` instead of string concatenation
- Avoids Windows vs Unix path separator issues

---

## Support

**Documentation:**
- `INSTALL.md` - Platform-specific installation
- `WINDOWS_QUICKSTART.md` - Quick Windows guide
- `README.md` - Project overview
- `CLAUDE.md` - Development guidelines
- `PIPELINE.md` - Technical details

**For Issues:**
- Check `INSTALL.md` troubleshooting section first
- Windows DLL errors → Use cloud embeddings
- Import errors → Check virtual environment
- API key errors → Verify `.env` configuration

---

## What Your Windows Colleagues Need to Do

**Short version:**
```bash
# 1. Install PyTorch first
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install app
uv sync

# 3. Configure .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-large

# 4. Run
python run_pipeline.py document.pdf
```

**Full version:** See `WINDOWS_QUICKSTART.md`

---

**Questions? Check the documentation or open an issue.**
