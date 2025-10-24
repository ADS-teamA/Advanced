# Centrální databáze dokumentů (vector_db/)

Tento systém poskytuje **centrální databázi** pro všechny zaindexované dokumenty místo izolovaných vector stores pro každý dokument.

## Výhody centrální databáze

✅ **Jeden zdroj pravdy** - Všechny dokumenty na jednom místě
✅ **Incremental indexing** - Přidávej dokumenty postupně bez přeindexování
✅ **Snadná správa** - Jediná databáze místo desítek složek
✅ **Agent-ready** - Připraveno pro použití s RAG Agent CLI
✅ **Hybrid search** - Automatická podpora BM25 + Dense + RRF fusion

## Quick Start

### 1. Vytvoření databáze z existujícího vector store

Pokud už máš zaindexovaný dokument, migruj ho do centrální databáze:

```bash
# Migruj existující vector store
uv run python manage_vector_db.py migrate output/BZ_VR1/20251024_164925/phase4_vector_store

# Zobraz statistiky
uv run python manage_vector_db.py stats
```

### 2. Přidání nového dokumentu

```bash
# Přidej nový dokument (automaticky vytvoří databázi, pokud neexistuje)
uv run python manage_vector_db.py add data/new_document.pdf

# Zkontroluj, že byl přidán
uv run python manage_vector_db.py stats
```

### 3. Použití s RAG Agent CLI

```bash
# Spusť agenta s centrální databází
uv run python -m src.agent.cli --vector-store vector_db

# Agent nyní má přístup ke VŠEM dokumentům v databázi!
```

## Příkazy

### `add` - Přidat dokument

Zaindexuje nový dokument a přidá ho do centrální databáze.

```bash
uv run python manage_vector_db.py add <cesta_k_dokumentu>

# Příklady
uv run python manage_vector_db.py add data/dokument.pdf
uv run python manage_vector_db.py add data/pravidla.docx
```

**Co se stane:**
1. Dokument se zaindexuje (PHASE 1-6)
2. Vytvoří se vector store pro tento dokument
3. Vector store se přidá (merge) do centrální databáze
4. Databáze se uloží na disk

### `migrate` - Migrovat existující vector store

Přidá existující vector store do centrální databáze.

```bash
uv run python manage_vector_db.py migrate <cesta_k_vector_store>

# Příklad
uv run python manage_vector_db.py migrate output/BZ_VR1/20251024_164925/phase4_vector_store
```

**Co se stane:**
1. Načte se existující vector store
2. Přidá se (merge) do centrální databáze
3. Databáze se uloží na disk

### `stats` - Zobrazit statistiky

Zobrazí informace o centrální databázi.

```bash
uv run python manage_vector_db.py stats
```

**Výstup:**
```
Umístění: vector_db

Dokumenty:        2
Celkem vektorů:   4523

FAISS:
  Layer 1 (Doc):  2
  Layer 2 (Sec):  1523
  Layer 3 (Chnk): 3000
  Dimenze:        3072D

BM25:
  Layer 1:        2
  Layer 2:        1523
  Layer 3:        3000

Hybrid Search:    True
RRF Fusion k:     60
```

### `init` - Vytvořit prázdnou databázi

Vytvoří novou databázi (volitelně z existujícího store).

```bash
# Vytvoř prázdnou databázi
uv run python manage_vector_db.py init

# Vytvoř databázi z existujícího store
uv run python manage_vector_db.py init --from output/existing_store
```

## Workflow

### Scénář 1: Začínám od nuly

```bash
# 1. Přidej první dokument (vytvoří databázi)
uv run python manage_vector_db.py add data/dokument1.pdf

# 2. Přidej další dokumenty
uv run python manage_vector_db.py add data/dokument2.pdf
uv run python manage_vector_db.py add data/dokument3.pdf

# 3. Zkontroluj statistiky
uv run python manage_vector_db.py stats

# 4. Použij s agentem
uv run python -m src.agent.cli --vector-store vector_db
```

### Scénář 2: Mám už zaindexované dokumenty

```bash
# 1. Migruj existující vector stores
uv run python manage_vector_db.py migrate output/doc1/phase4_vector_store
uv run python manage_vector_db.py migrate output/doc2/phase4_vector_store
uv run python manage_vector_db.py migrate output/doc3/phase4_vector_store

# 2. Zkontroluj statistiky
uv run python manage_vector_db.py stats

# 3. Přidávej nové dokumenty
uv run python manage_vector_db.py add data/new_doc.pdf

# 4. Použij s agentem
uv run python -m src.agent.cli --vector-store vector_db
```

### Scénář 3: Migrace z BZ_VR1 do centrální databáze

```bash
# Migruj stávající BZ_VR1 vector store
uv run python manage_vector_db.py migrate output/BZ_VR1/20251024_164925/phase4_vector_store

# Ověř migraci
uv run python manage_vector_db.py stats

# Použij s agentem
uv run python -m src.agent.cli --vector-store vector_db
```

## Technické detaily

### Struktura databáze

```
vector_db/
├── layer1.index              # FAISS Layer 1 (Document)
├── layer2.index              # FAISS Layer 2 (Section)
├── layer3.index              # FAISS Layer 3 (Chunk)
├── metadata.pkl              # FAISS metadata
├── bm25_layer1.pkl           # BM25 Layer 1
├── bm25_layer2.pkl           # BM25 Layer 2
├── bm25_layer3.pkl           # BM25 Layer 3
└── hybrid_config.pkl         # Hybrid search config
```

### Merge strategie

Když přidáváš dokument do existující databáze, systém použije **incremental indexing**:

1. **FAISS indexes**: Vektory se přidají do existujících indexů pomocí `faiss.add()`
2. **BM25 indexes**: Corpus se rozšíří a BM25 se přebuduje
3. **Metadata**: Metadata se přidají s offsety pro správné indexování
4. **Document mapping**: Mapování `doc_id → indices` se aktualizuje

**Důležité:**
- Embedding dimenze musí být stejná (3072D pro `text-embedding-3-large`)
- Fusion k parametr (60) se bere z existující databáze
- Merge je **aditivní** - dokumenty se přidávají, nikdy nemazou

### Performance

- **Merge rychlost**: ~1-2 sekundy pro dokument s 1000 chunky
- **Load rychlost**: ~150-200ms pro databázi s 10 dokumenty
- **Search rychlost**: Nezávislá na počtu dokumentů (díky FAISS)

### Limity

- **Paměť**: Celá databáze se načítá do paměti (limitace FAISS)
- **Velikost**: Pro >10K dokumentů zvažte disk-based FAISS index
- **Dimenze**: Nelze kombinovat různé embedding modely (musí být stejné dimenze)

## FAQ

**Q: Mohu smazat originální vector stores po migraci?**
A: Ano, centrální databáze obsahuje všechna data. Můžeš archivovat `output/` složku.

**Q: Co se stane, když přidám stejný dokument dvakrát?**
A: Systém ho přidá znovu (duplikáty nejsou detekovány). Nedoporučuje se.

**Q: Mohu upravit už zaindexovaný dokument?**
A: Ne, systém nepodporuje update. Musíš smazat databázi a přeindexovat.

**Q: Jak smažu databázi?**
A: Jednoduše smaž `vector_db/` složku: `rm -rf vector_db`

**Q: Funguje to s Knowledge Graphem?**
A: Ano! Knowledge Graph se **NE**ukládá do centrální databáze (pouze vector store). Pro KG použij individuální `<document_id>_kg.json` soubory.

**Q: Mohu používat centrální databázi a individuální stores současně?**
A: Ano, jsou nezávislé. Agent lze spustit s libovolným vector store.

## Troubleshooting

### Chyba: "Cannot merge stores with different dimensions"

**Problém:** Snažíš se sloučit vector stores s různými embedding modely.

**Řešení:**
1. Zkontroluj embedding model v `.env`: `EMBEDDING_MODEL=text-embedding-3-large`
2. Ujisti se, že všechny dokumenty používají stejný model
3. Případně vytvoř novou databázi s jednotným modelem

### Chyba: "Vector store nenalezen"

**Problém:** Cesta k vector store je chybná.

**Řešení:**
1. Zkontroluj, že složka existuje: `ls output/BZ_VR1/20251024_164925/phase4_vector_store`
2. Ujisti se, že obsahuje FAISS indexy: `ls output/.../layer1.index`

### Warning: "Store není HybridVectorStore"

**Problém:** Starší vector store bez BM25 indexů.

**Info:** Systém automaticky wrappuje do HybridVectorStore s prázdným BM25. Funguje, ale nebude mít sparse retrieval.

**Řešení:** Přeindexuj dokument s `enable_hybrid_search=True`.

## Integrace s agentem

Centrální databáze je navržena pro použití s RAG Agent CLI:

```bash
# Spusť agenta s centrální databází
uv run python -m src.agent.cli --vector-store vector_db

# Agent nyní může dotazovat VŠECHNY dokumenty současně
> What are the waste disposal requirements across all indexed documents?

# Agent má přístup k:
# - Všem 26 specialized tools
# - Hybrid search (BM25 + Dense + RRF)
# - Cross-document queries
# - Multi-hop reasoning
```

## Next Steps

Po nastavení centrální databáze můžeš:

1. **Pravidelně přidávat dokumenty**: `manage_vector_db.py add`
2. **Používat s agentem**: `src.agent.cli --vector-store vector_db`
3. **Monitorovat velikost**: `manage_vector_db.py stats`
4. **Backupovat databázi**: `tar -czf vector_db_backup.tar.gz vector_db/`

---

**Happy indexing! 🚀**
