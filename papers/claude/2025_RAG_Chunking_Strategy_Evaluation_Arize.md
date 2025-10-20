# Benchmarking Evaluation of LLM Retrieval Augmented Generation (RAG)

**Zdroj:** Arize AI Blog
**URL:** https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/
**Datum:** 2024/2025
**Autor:** Arize AI Research Team

---

## Executive Summary

Tento výzkum poskytuje systematické empirické hodnocení různých konfigurací systémů Retrieval Augmented Generation (RAG) s využitím produktové dokumentace Arize AI jako benchmarkového datasetu. Studie identifikuje optimální parametry pro velikost chunků (300-500 tokenů), počet získaných chunků (K=4-6), a porovnává různé chunking strategie a retrieval metody z hlediska přesnosti, kvality odpovědí a latence.

---

## 1. Úvod a Kontext

### 1.1 Problematika

Knowledge bases umožňují Large Language Models (LLM) přístup k informacím mimo jejich trénovací data. Bez rigorózního hodnocení však praktici postrádají konkrétní vodítka pro optimální konfigurace RAG systémů pro jejich specifické datasety.

### 1.2 Cíle Studie

Studie hodnotí RAG systémy testováním různých konfigurací s měřením tří klíčových výkonnostních dimenzí:

1. **Precision of Context Retrieved** - Přesnost relevance získaného kontextu z vector store
2. **Accuracy of LLM Output** - Přesnost a kontextová správnost odpovědí generovaných LLM
3. **System Latency** - Odezva systému (reakční čas)

### 1.3 Metodologie

- Benchmark dataset: Arize AI produktová dokumentace
- Evaluační framework: Phoenix LLM evals library (open source)
- Reprodukovatelné testovací skripty dostupné v Colab notebooku
- Systematické testování multiple konfigurací

---

## 2. Chunking Strategie - Detailní Analýza

### 2.1 Uniform Chunking (Uniformní Dělení)

**Definice:**
Rozděluje data do konzistentních velikostí měřených v tokenech. V angličtině přibližně 1 token = 4 znaky.

**Výhody:**
- Jednoduchá implementace
- Předvídatelné velikosti chunků
- Nízké výpočetní nároky

**Nevýhody:**
- **Riziko fragmentace:** "risks dividing individual pieces of information across multiple chunks, which might lead to incomplete or incorrect responses"
- Nereflektuje sémantickou strukturu textu
- Může rozdělit související koncepty

**Použití:**
Vhodné pro uniformní, strukturované dokumenty kde je rychlost prioritou.

---

### 2.2 Sentence-Based Chunking (Dělení Podle Vět)

**Definice:**
Segmentuje data na strukturálních hranicích jako jsou tečky, line breaks nebo odstavce.

**Výhody:**
- Lepší sémantická koherence než uniform chunking
- Zachovává integritu vět
- Přirozenější textové jednotky

**Nevýhody:**
- Stále riskuje rozdělení souvisejícího obsahu napříč chunky
- Vyžaduje NLP knihovny pro optimální výsledky
- Variabilní velikosti chunků mohou komplikovat retrieval

**Implementace:**
Obvykle využívá sentence tokenizery nebo regex pattern matching na interpunkci.

---

### 2.3 Recursive Chunking (Rekurzivní Dělení)

**Definice:**
Iterativně rozděluje text, dokud chunky nesplní definované podmínky velikosti nebo struktury.

**Výhody:**
- Produkuje kontextově koherentní chunky
- Vyvažuje granularitu s kontextem
- Adaptivní na strukturu dokumentu

**Nevýhody:**
- **Více výpočetně náročné** než jednodušší metody
- Pomalejší zpracování
- Komplexnější implementace

**Mechanismus:**
Rekurzivně aplikuje splitting pravidla (např. nejdřív odstavce, pak věty, pak slova) dokud nesplní size constraints.

---

### 2.4 Parent Document Retrieval (Pokročilá Strategie)

**Definice:**
Používá malé chunky pro sémantické matchování v query fázi, poté získává větší "parent" chunky obsahující matchované segmenty.

**Výhody:**
- **Kombinuje výhody obou světů:** Přesnost malých chunků + kontext velkých chunků
- Vylepšuje context enrichment
- Implementováno v LangChain a LlamaIndex

**Mechanismus:**
1. **Query fáze:** Systém matchuje user queries na relevantní informace ve vector store pomocí přesných, sémanticky bohatých menších chunků
2. **Retrieval fáze:** Jakmile jsou malé chunky identifikovány, systém získává větší okolní kontext (parent chunks)

**Výhody:**
- Řeší trade-off mezi granularitou a kontextem
- Zlepšuje kvalitu odpovědí díky většímu kontextu
- Zachovává přesnost matchování malých chunků

---

## 3. Experimentální Výsledky a Nálezy

### 3.1 Optimální Velikost Chunků

**Klíčový nález:**
Výzkum identifikoval "sweet spot" mezi menším kontextem 100 tokenů a většími kontexty 1000 tokenů.

**Doporučení:**
- ✅ **Optimální: 300-500 tokenů**
- ⚠️ Větší chunky (>1000 tokenů): způsobují "decline in response accuracy"
- ⚠️ Příliš malé chunky (<100 tokenů): nedostatečný kontext

**Citace:**
> "Chunk sizes of 300/500 tokens seem to be a good target; going bigger has negative results."

---

### 3.2 K Hodnota (Počet Získaných Chunků)

**Testované hodnoty:** K = 4, 5, 6, 10

**Výsledky:**
- ✅ **K=4: Optimální volba** - nejlepší balance mezi performance a latencí
- K=5-6: Přijatelná alternativa s mírně vyšší latencí
- K=10: Významné zvýšení latence bez proporcionálního zlepšení přesnosti

**Latence Impact:**
- K=4 (standard retrieval): baseline latence
- K=10 (standard retrieval): ~2× zvýšení latence
- K s re-rankingem: **extrémně vysoké zpoždění** ("latency skyrockets")

**Doporučení:**
Pro interaktivní aplikace zůstat u K=4-6 s jednoduchým retrievalem.

---

### 3.3 Retrieval Metody - Komparativní Analýza

#### 3.3.1 Standard Retrieval + Simple Embedding

**Výhody:**
- ✅ Nejrychlejší možnost
- ✅ Vhodné pro interaktivní aplikace
- ✅ Dobrý performance/speed trade-off

**Nevýhody:**
- Nižší přesnost než pokročilé metody

---

#### 3.3.2 HyDE (Hypothetical Document Embeddings) + Re-ranking

**Výhody:**
- ✅ **Nejvyšší přesnost:** "does outperform most of the other options"
- ✅ Superior accuracy

**Nevýhody:**
- ❌ **Velmi pomalé:** "very slow"
- ❌ Významné latency costs
- ❌ Nevhodné pro real-time aplikace

**Použití:**
Vhodné pro batch processing nebo aplikace kde je přesnost kritičtější než rychlost.

---

#### 3.3.3 Re-ranking Samostatně

**Překvapivý nález:**
Re-ranking samostatně underperformoval očekávání.

**Problém:**
- Někdy posouvá optimální chunky z pozice #1 na pozice #2-#4
- Může snižovat precision místo zlepšování

**Doporučení:**
Re-ranking používat primárně v kombinaci s HyDE, ne jako standalone řešení.

---

## 4. Implementační Detaily

### 4.1 Phoenix LLM Evals Library

**Framework použitý ve studii:**
- Open source nástroj od Arize AI
- Umožňuje systematické hodnocení RAG systémů
- Podporuje custom metriky a evaluace

**Dostupné zdroje:**
- Reprodukovatelné testovací skripty
- Colab notebook s demonstrací
- Parametrizace na custom dokumentaci

---

### 4.2 Evaluační Metriky - Detail

#### Precision of Context Retrieved
- Měří relevanci získaných chunků k dotazu
- Identifikuje "hallucination risk"

#### Accuracy of LLM Output
- Hodnotí kvalitu finální odpovědi
- Kontroluje kontextovou správnost
- Měří coherenci

#### System Latency
- End-to-end response time
- Kritické pro user experience
- Trade-off s přesností

---

## 5. Best Practices a Doporučení

### 5.1 Rychlý Start - Doporučené Nastavení

Pro většinu aplikací začít s:
```
Chunk size: 400 tokenů
K value: 4-5
Retrieval: Standard embedding (bez re-ranking)
Chunking: Sentence-based nebo Recursive
```

### 5.2 Optimalizace Pro Různé Use Cases

#### Interaktivní Aplikace (Chatbots, Search)
- **Priorita:** Nízká latence
- **Chunk size:** 300-400 tokenů
- **K value:** 4
- **Retrieval:** Standard embedding
- **Chunking:** Uniform nebo Sentence-based

#### Batch Processing / Analýza
- **Priorita:** Vysoká přesnost
- **Chunk size:** 400-500 tokenů
- **K value:** 5-6
- **Retrieval:** HyDE + Re-ranking
- **Chunking:** Recursive nebo Parent Document

#### Komplexní Technická Dokumentace
- **Priorita:** Kontext a přesnost
- **Chunk size:** 500 tokenů
- **K value:** 5-6
- **Retrieval:** Parent Document Retrieval
- **Chunking:** Recursive

---

### 5.3 Balancing Act: Precision vs. Latency

**Trade-off Matrix:**

| Konfigurace | Precision | Latency | Use Case |
|-------------|-----------|---------|----------|
| Small chunks (100-200) + K=4 | Střední | Nízká | Rychlé vyhledávání |
| Medium chunks (300-500) + K=4-5 | ✅ Vysoká | ✅ Nízká | **Recommended default** |
| Large chunks (>1000) + K=6+ | Nízká | Střední | ❌ Nedoporučeno |
| HyDE + Re-rank + K=6 | Nejvyšší | Vysoká | Kritické aplikace |

---

## 6. Kritická Upozornění a Limitace

### 6.1 Dataset Specificity

**Zásadní poznámka od autorů:**

> "Experimentování s vaším konkrétním datasetem zůstává zásadní. I když naše benchmarky poskytují vodítko, jednotlivé případy použití vyžadují customizované testování."

**Důvody:**
- Různé domény mají různé charakteristiky
- Struktura dokumentace se liší
- Query patterns závisí na use case

---

### 6.2 Testování na Vlastních Datech

**Doporučený proces:**
1. Začít s baseline (recommended settings)
2. Systematicky testovat varianty
3. Měřit všechny tři metriky (precision, accuracy, latency)
4. A/B testování s real users
5. Iterativní optimalizace

---

## 7. Pokročilé Techniky a Budoucí Směry

### 7.1 Parent Document Retrieval

Implementace v populárních frameworks:
- **LangChain:** ParentDocumentRetriever
- **LlamaIndex:** Recursive retrieval

**Výhody pro production:**
- Škálovatelnost
- Lepší kontext preservation
- Flexibilita v konfiguraci

---

### 7.2 Hybrid Approaches

**Perspektivní směry:**
- Kombinace multiple chunking strategií
- Dynamic chunk sizing based on query
- Multi-stage retrieval pipelines

---

## 8. Závěr

### Klíčová Doporučení (TL;DR)

1. ✅ **Chunk size:** 300-500 tokenů (sweet spot)
2. ✅ **K value:** 4-6 (optimální balance)
3. ✅ **Retrieval:** Simple embedding pro rychlost, HyDE+Re-rank pro přesnost
4. ✅ **Chunking:** Recursive nebo Sentence-based pro většinu případů
5. ⚠️ **Vždy testovat na vlastních datech**

### Praktické Takeaways

- **Pro začátek:** 400 tokenů, K=4, standard retrieval
- **Pro optimalizaci:** Systematicky testovat varianty
- **Pro produkci:** Parent document retrieval s monitoringem
- **Pro evaluaci:** Phoenix LLM evals library

---

## 9. Zdroje a Další Čtení

- **Originální článek:** https://arize.com/blog-course/evaluation-of-llm-rag-chunking-strategy/
- **Phoenix Library:** Open-source evaluation framework od Arize AI
- **Colab Notebook:** Reprodukovatelné experimenty s custom dokumentací
- **LangChain Documentation:** ParentDocumentRetriever implementace
- **LlamaIndex Documentation:** Recursive retrieval patterns

---

## 10. Poznámky k Implementaci

### Typický RAG Pipeline

```
1. Document Ingestion
   ↓
2. Chunking Strategy (300-500 tokens)
   ↓
3. Embedding Generation
   ↓
4. Vector Store Indexing
   ↓
5. Query Processing
   ↓
6. Retrieval (K=4-5 chunks)
   ↓
7. (Optional) Re-ranking
   ↓
8. Context Assembly
   ↓
9. LLM Generation
   ↓
10. Response Delivery
```

### Performance Monitoring

**Metriky k tracking:**
- Query latency percentiles (p50, p95, p99)
- Context relevance scores
- User satisfaction ratings
- Failure rates (no results found)

---

**Dokument připraven:** October 2025
**Zpracováno pro:** MY_SUJBOT Project
**Účel:** Reference pro RAG chunking strategy implementation
