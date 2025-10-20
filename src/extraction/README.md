# IBM Docling Document Extraction

Komplexní implementace IBM Docling frameworku pro extrakci struktury z právních dokumentů s 97.9% přesností pro tabulky a 100% věrností textu.

## 🚀 Přehled

Tento modul poskytuje špičkovou extrakci dokumentové struktury pomocí IBM Docling, optimalizovanou pro právní dokumenty včetně smluv, zákonů, NDA a policy dokumentů.

### Klíčové funkce

- ✅ **97.9% přesnost** při extrakci tabulek (TableFormer)
- ✅ **100% věrnost textu** při extrakci
- ✅ **Hierarchická struktura** s vnořenými sekcemi
- ✅ **Multi-formát**: PDF, DOCX, PPTX, XLSX, HTML, obrázky
- ✅ **OCR podpora** včetně Apple Silicon (MLX)
- ✅ **GPU akcelerace** (volitelná)
- ✅ **Právní analýza** - klauzule, entity, citace
- ✅ **Lokální běh** bez cloudových závislostí

## 📦 Instalace

```bash
# Nainstalovat závislosti
pip install -r requirements.txt

# Pro GPU podporu (volitelné)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Rychlý start

### 1. Základní extrakce

```python
from src.extraction.docling_extractor import DoclingExtractor, ExtractionConfig

# Vytvořit extraktor
config = ExtractionConfig(
    enable_ocr=True,
    extract_tables=True,
    generate_markdown=True
)
extractor = DoclingExtractor(config)

# Extrahovat dokument
result = extractor.extract("smlouva.pdf")

# Použít výsledky
print(result.markdown)  # Markdown s hierarchií
print(f"Sections: {result.num_sections}")
print(f"Tables: {result.num_tables}")

# Export tabulek
for table in result.tables:
    df = table.data  # Pandas DataFrame
    print(f"Table {table.table_id}: {table.num_rows}x{table.num_cols}")
```

### 2. Integrace s LawGPT

```python
from src.extraction.document_processor import DocumentProcessor
from LawGPT.src.core.models import DocumentType

# Vytvořit procesor
processor = DocumentProcessor()

# Zpracovat dokument
document = processor.process(
    "contract.pdf",
    doc_type=DocumentType.CONTRACT
)

# Použít LawGPT modely
print(f"Document ID: {document.metadata.document_id}")
print(f"Type: {document.metadata.document_type}")
print(f"Sections: {len(document.structure.sections)}")

# Přístup k sekcím
for section in document.structure.get_top_level_sections():
    print(f"[Level {section.level}] {section.title}")
```

### 3. Právní analýza

```python
from src.extraction.legal_analyzer import LegalDocumentAnalyzer, RiskLevel

# Vytvořit analyzátor
analyzer = LegalDocumentAnalyzer(language="en")

# Analyzovat dokument
analysis = analyzer.analyze(
    text=document.text,
    document_id=document.metadata.document_id
)

# Výsledky analýzy
print(f"Clauses: {len(analysis.clauses)}")
print(f"Entities: {len(analysis.entities)}")
print(f"Citations: {len(analysis.citations)}")

# High-risk klauzule
high_risk = [c for c in analysis.clauses
             if c.risk_level == RiskLevel.HIGH]
for clause in high_risk:
    print(f"⚠️ {clause.clause_type}: {clause.title}")

# Pokrytí standardních klauzulí
coverage = analyzer.analyze_clause_coverage(analysis)
print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
```

## 🔧 Konfigurace

### ExtractionConfig

```python
config = ExtractionConfig(
    # OCR nastavení
    enable_ocr=True,              # Povolit OCR
    use_mac_ocr=True,             # Použít macOS OCR (Apple Silicon)
    ocr_engine="easyocr",         # "easyocr", "tesseract", "mac"

    # Extrakce tabulek
    table_mode=TableFormerMode.ACCURATE,  # ACCURATE nebo FAST
    extract_tables=True,

    # Extrakce obrázků
    extract_images=True,

    # Struktura
    extract_hierarchy=True,
    preserve_reading_order=True,

    # Výkon
    use_gpu=False,                # GPU akcelerace
    batch_size=1,

    # Výstupní formáty
    generate_markdown=True,
    generate_json=True,
    generate_html=False
)
```

## 📚 Moduly

### DoclingExtractor

Hlavní třída pro extrakci dokumentů.

**Metody:**
- `extract(source, document_id)` - Extrahovat jeden dokument
- `extract_batch(sources, document_ids)` - Batch zpracování
- `extract_tables_only(source)` - Pouze tabulky (rychlejší)
- `extract_text_only(source)` - Pouze text
- `get_supported_formats()` - Seznam podporovaných formátů

**Příklad:**

```python
# Extrakce pouze tabulek
tables = extractor.extract_tables_only("report.pdf")
for table in tables:
    print(table.caption)

# Extrakce textu se strukturou
text = extractor.extract_text_only("document.pdf", preserve_structure=True)
```

### DocumentProcessor

Integrace Docling s LawGPT datovými modely.

**Metody:**
- `process(source, document_id, doc_type)` - Zpracovat dokument
- `process_batch(sources, document_ids, doc_types)` - Batch zpracování
- `get_extraction_metadata(source)` - Pouze metadata (rychlé)

**Funkce:**
- Auto-detekce typu dokumentu
- Extrakce stran z kontraktů/NDA
- Detekce jazyka
- Konverze na LawGPT Document objekty

### LegalDocumentAnalyzer

Pokročilá analýza právních dokumentů.

**Extrahuje:**
- **Klauzule**: Confidentiality, Termination, Liability, atd.
- **Entity**: Organizace, osoby, soudy
- **Citace**: Zákony, judikáty, články
- **Datumy**: Effective date, expiration, signature
- **Klíčové termíny**: Frekvence právních termínů

**Hodnocení rizik:**
- LOW - Standardní klauzule
- MEDIUM - Běžné klauzule vyžadující pozornost
- HIGH - Rizikové klauzule (unlimited liability, atd.)
- CRITICAL - Kritické klauzule (irrevocable, perpetual)

## 🎮 Demo skripty

### 1. Základní demo

```bash
python scripts/demo_docling_basic.py
```

Demonstruje:
- Základní extrakci textu
- Detekci tabulek
- Export do Markdown/JSON
- Hierarchickou strukturu

### 2. Právní analýza

```bash
python scripts/demo_docling_legal.py
```

Demonstruje:
- Integraci s LawGPT
- Auto-detekci typu dokumentu
- Extrakci klauzulí a hodnocení rizik
- Analýzu entit a citací
- Pokrytí standardních klauzulí

### 3. Batch zpracování

```bash
python scripts/demo_docling_batch.py
```

Demonstruje:
- Zpracování více dokumentů najednou
- Srovnávací analýzu
- Agregované statistiky
- HTML report

## 📊 Podporované formáty

| Formát | Přípona | OCR podpora | Tabulky | Hierarchie |
|--------|---------|-------------|---------|------------|
| PDF | .pdf | ✅ | ✅ | ✅ |
| Word | .docx | ➖ | ✅ | ✅ |
| PowerPoint | .pptx | ➖ | ✅ | ✅ |
| Excel | .xlsx | ➖ | ✅ | ➖ |
| HTML | .html, .htm | ➖ | ✅ | ✅ |
| Obrázky | .jpg, .png, .tiff | ✅ | ✅ | ➖ |

## 🔍 Detekované typy klauzulí

- **CONFIDENTIALITY** - Mlčenlivost, důvěrnost
- **TERMINATION** - Ukončení, výpověď
- **INDEMNIFICATION** - Odškodnění
- **LIABILITY** - Odpovědnost, náhrada škody
- **JURISDICTION** - Jurisdikce, rozhodné právo
- **DISPUTE_RESOLUTION** - Řešení sporů, arbitráž
- **PAYMENT** - Platby, odměna
- **INTELLECTUAL_PROPERTY** - Duševní vlastnictví
- **WARRANTY** - Záruka, prohlášení
- **FORCE_MAJEURE** - Vyšší moc
- **AMENDMENT** - Změny a doplňky
- **SEVERABILITY** - Oddělitelnost ustanovení
- **ENTIRE_AGREEMENT** - Úplnost ujednání
- **NOTICE** - Oznámení
- **ASSIGNMENT** - Postoupení práv

## ⚡ Optimalizace výkonu

### GPU akcelerace

```python
config = ExtractionConfig(use_gpu=True)
extractor = DoclingExtractor(config)
```

Doporučeno pro:
- Batch zpracování více dokumentů
- Velké dokumenty (>100 stran)
- Dokumenty s mnoha tabulkami

### Apple Silicon optimalizace

```python
config = ExtractionConfig(
    use_mac_ocr=True,  # Native macOS OCR
    use_gpu=False       # MLX místo CUDA
)
```

### Rychlé zpracování

```python
# Pouze text (bez tabulek a OCR)
text = extractor.extract_text_only("doc.pdf", preserve_structure=False)

# Pouze metadata (nejrychlejší)
metadata = processor.get_extraction_metadata("doc.pdf")
```

## 🧪 Testování

```bash
# Spustit všechny testy
pytest tests/test_docling_extraction.py -v

# Pouze unit testy
pytest tests/test_docling_extraction.py::TestDoclingExtractor -v

# S coverage
pytest tests/test_docling_extraction.py --cov=src/extraction --cov-report=html
```

## 📈 Příklady výstupu

### JSON struktura

```json
{
  "document_id": "contract_001",
  "num_pages": 15,
  "num_sections": 23,
  "num_tables": 3,
  "total_chars": 45230,
  "sections": [
    {
      "section_id": "sec_1",
      "title": "Definitions",
      "level": 0,
      "children_ids": ["sec_2", "sec_3"]
    }
  ],
  "tables": [
    {
      "table_id": "table_1",
      "caption": "Payment Schedule",
      "num_rows": 12,
      "num_cols": 4
    }
  ]
}
```

### Markdown s hierarchií

```markdown
# CONTRACT AGREEMENT

## 1. Definitions

### 1.1 General Terms

The following terms shall have the meanings set forth below...

### 1.2 Specific Terms

"Confidential Information" means...

## 2. Obligations

### 2.1 Payment Terms
...
```

## 🐛 Troubleshooting

### OCR nefunguje na macOS

```python
# Zkusit fallback na EasyOCR
config = ExtractionConfig(use_mac_ocr=False, ocr_engine="easyocr")
```

### Pomalé zpracování velkých PDF

```python
# Vypnout OCR pro již digitální dokumenty
config = ExtractionConfig(enable_ocr=False)

# Nebo použít fast mode pro tabulky
config = ExtractionConfig(table_mode=TableFormerMode.FAST)
```

### Out of memory při GPU zpracování

```python
# Snížit batch size nebo vypnout GPU
config = ExtractionConfig(use_gpu=False)
```

## 🔗 Další zdroje

- [IBM Docling GitHub](https://github.com/DS4SD/docling)
- [Docling dokumentace](https://ds4sd.github.io/docling/)
- [LawGPT dokumentace](../README.md)

## 📝 Poznámky k implementaci

### Verze a kompatibilita

- Python 3.9+
- Docling 2.57.0+
- macOS (Apple Silicon) / Linux / Windows
- GPU: CUDA 11.8+ nebo Apple MLX

### Omezení

- Naskenované dokumenty vyžadují OCR (pomalejší)
- Tabulky se složitým layoutem mohou vyžadovat ruční review
- Auto-detekce typu dokumentu není 100% přesná
- NER pro entity je zjednodušené (pro produkci zvážit LexNLP)

### Budoucí vylepšení

- [ ] Integrace LexNLP pro pokročilé NER
- [ ] Podpora více jazyků (čeština, němčina, atd.)
- [ ] Fine-tuning na české právní dokumenty
- [ ] Integrace s RAG pipeline
- [ ] Web UI pro vizualizaci
- [ ] API endpoint

## 📄 Licence

Tento modul je součástí LawGPT projektu a používá IBM Docling (Apache 2.0 License).

## 👥 Autoři

Vytvořeno pro projekt LawGPT s podporou Claude Code.
