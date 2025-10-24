"""
Správa centrální databáze dokumentů (vector_db/)

Tento skript umožňuje:
1. Vytvořit novou centrální databázi
2. Přidat dokument do centrální databáze
3. Migrovat existující vector store do centrální databáze
4. Zobrazit statistiky centrální databáze

Použití:
    # Přidat nový dokument (automaticky vytvoří databázi, pokud neexistuje)
    python manage_vector_db.py add data/dokument.pdf

    # Migrovat existující vector store
    python manage_vector_db.py migrate output/BZ_VR1/20251024_164925/phase4_vector_store

    # Zobrazit statistiky
    python manage_vector_db.py stats

    # Vytvořit prázdnou databázi
    python manage_vector_db.py init
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from src.hybrid_search import HybridVectorStore

# Centrální databáze v root složce projektu
CENTRAL_DB_PATH = Path("vector_db")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_central_db() -> Optional[HybridVectorStore]:
    """
    Načte centrální databázi, pokud existuje.

    Returns:
        HybridVectorStore instance nebo None, pokud databáze neexistuje
    """
    if not CENTRAL_DB_PATH.exists():
        logger.info(f"Centrální databáze neexistuje: {CENTRAL_DB_PATH}")
        return None

    try:
        logger.info(f"Načítám centrální databázi z {CENTRAL_DB_PATH}")
        db = HybridVectorStore.load(CENTRAL_DB_PATH)
        logger.info(f"✓ Databáze načtena: {db.get_stats()}")
        return db
    except Exception as e:
        logger.error(f"✗ Chyba při načítání databáze: {e}")
        return None


def create_central_db(initial_store_path: Optional[Path] = None) -> HybridVectorStore:
    """
    Vytvoří novou centrální databázi.

    Args:
        initial_store_path: Volitelná cesta k existujícímu vector store pro inicializaci

    Returns:
        Nová HybridVectorStore instance
    """
    logger.info(f"Vytvářím novou centrální databázi v {CENTRAL_DB_PATH}")

    if initial_store_path:
        # Načti existující store a použij jako výchozí
        logger.info(f"Inicializuji z existujícího store: {initial_store_path}")
        db = HybridVectorStore.load(initial_store_path)
    else:
        # Vytvoř prázdnou databázi (bude vytvořena při prvním přidání dokumentu)
        logger.warning("Vytváření prázdné databáze - musí se přidat alespoň 1 dokument")
        return None

    # Ulož do centrálního umístění
    CENTRAL_DB_PATH.mkdir(parents=True, exist_ok=True)
    db.save(CENTRAL_DB_PATH)

    logger.info(f"✓ Centrální databáze vytvořena: {db.get_stats()}")
    return db


def add_document_to_db(document_path: Path) -> HybridVectorStore:
    """
    Přidá dokument do centrální databáze.

    Pokud databáze neexistuje, vytvoří ji.

    Args:
        document_path: Cesta k dokumentu (PDF, DOCX, atd.)

    Returns:
        Aktualizovaná HybridVectorStore instance
    """
    document_path = Path(document_path)

    if not document_path.exists():
        raise FileNotFoundError(f"Dokument nenalezen: {document_path}")

    logger.info("="*80)
    logger.info(f"Přidávám dokument do centrální databáze: {document_path.name}")
    logger.info("="*80)

    # 1. Indexuj nový dokument
    logger.info("KROK 1: Indexování dokumentu...")
    config = IndexingConfig(
        enable_hybrid_search=True,
        enable_knowledge_graph=False,  # Knowledge Graph vypnutý (rychlejší indexování)
        enable_reranking=False  # Můžeš zapnout, pokud chceš
    )
    pipeline = IndexingPipeline(config)
    result = pipeline.index_document(document_path)
    new_store = result["vector_store"]

    logger.info(f"✓ Dokument zaindexován: {new_store.get_stats()}")

    # 2. Načti nebo vytvoř centrální databázi
    logger.info("KROK 2: Načítání centrální databáze...")
    central_db = get_central_db()

    if central_db is None:
        # Databáze neexistuje - vytvoř ji z nového store
        logger.info("Centrální databáze neexistuje - vytvářím novou...")
        CENTRAL_DB_PATH.mkdir(parents=True, exist_ok=True)
        new_store.save(CENTRAL_DB_PATH)
        central_db = new_store
    else:
        # Databáze existuje - merge nový store do ní
        logger.info("KROK 3: Přidávám dokument do existující databáze...")
        central_db.merge(new_store)

        # Ulož aktualizovanou databázi
        logger.info("KROK 4: Ukládám aktualizovanou databázi...")
        central_db.save(CENTRAL_DB_PATH)

    logger.info("="*80)
    logger.info(f"✓ Dokument přidán do centrální databáze!")
    logger.info(f"  Celkem dokumentů: {central_db.get_stats()['documents']}")
    logger.info(f"  Celkem vektorů: {central_db.get_stats()['total_vectors']}")
    logger.info(f"  Umístění: {CENTRAL_DB_PATH}")
    logger.info("="*80)

    return central_db


def migrate_store_to_db(store_path: Path) -> HybridVectorStore:
    """
    Migruje existující vector store do centrální databáze.

    Args:
        store_path: Cesta k existujícímu vector store

    Returns:
        Aktualizovaná HybridVectorStore instance
    """
    store_path = Path(store_path)

    if not store_path.exists():
        raise FileNotFoundError(f"Vector store nenalezen: {store_path}")

    logger.info("="*80)
    logger.info(f"Migruji vector store do centrální databáze: {store_path}")
    logger.info("="*80)

    # 1. Načti existující store
    logger.info("KROK 1: Načítání vector store...")
    try:
        migrating_store = HybridVectorStore.load(store_path)
        logger.info(f"✓ Store načten: {migrating_store.get_stats()}")
    except Exception as e:
        logger.error(f"✗ Chyba při načítání store: {e}")
        logger.info("Pokus o načtení jako FAISSVectorStore...")

        # Pokud to není HybridVectorStore, zkus FAISSVectorStore
        from src.faiss_vector_store import FAISSVectorStore
        faiss_store = FAISSVectorStore.load(store_path)
        logger.warning(
            "Store není HybridVectorStore - je to jen FAISS bez BM25. "
            "Centrální databáze používá hybrid search, takže tento store "
            "nebude mít BM25 indexy."
        )
        # Wrap do HybridVectorStore s prázdným BM25
        from src.hybrid_search import BM25Store
        migrating_store = HybridVectorStore(
            faiss_store=faiss_store,
            bm25_store=BM25Store(),
            fusion_k=60
        )

    # 2. Načti nebo vytvoř centrální databázi
    logger.info("KROK 2: Načítání centrální databáze...")
    central_db = get_central_db()

    if central_db is None:
        # Databáze neexistuje - vytvoř ji z migrovaného store
        logger.info("Centrální databáze neexistuje - vytvářím novou...")
        CENTRAL_DB_PATH.mkdir(parents=True, exist_ok=True)
        migrating_store.save(CENTRAL_DB_PATH)
        central_db = migrating_store
    else:
        # Databáze existuje - merge migrovaný store do ní
        logger.info("KROK 3: Přidávám store do existující databáze...")
        central_db.merge(migrating_store)

        # Ulož aktualizovanou databázi
        logger.info("KROK 4: Ukládám aktualizovanou databázi...")
        central_db.save(CENTRAL_DB_PATH)

    logger.info("="*80)
    logger.info(f"✓ Vector store migrován do centrální databáze!")
    logger.info(f"  Celkem dokumentů: {central_db.get_stats()['documents']}")
    logger.info(f"  Celkem vektorů: {central_db.get_stats()['total_vectors']}")
    logger.info(f"  Umístění: {CENTRAL_DB_PATH}")
    logger.info("="*80)

    return central_db


def show_stats():
    """Zobrazí statistiky centrální databáze."""
    logger.info("="*80)
    logger.info("Statistiky centrální databáze")
    logger.info("="*80)

    central_db = get_central_db()

    if central_db is None:
        logger.warning("Centrální databáze neexistuje!")
        logger.info(f"Vytvoř ji pomocí: python manage_vector_db.py add <dokument.pdf>")
        return

    stats = central_db.get_stats()

    logger.info(f"Umístění: {CENTRAL_DB_PATH}")
    logger.info(f"")
    logger.info(f"Dokumenty:        {stats['documents']}")
    logger.info(f"Celkem vektorů:   {stats['total_vectors']}")
    logger.info(f"")
    logger.info(f"FAISS:")
    logger.info(f"  Layer 1 (Doc):  {stats['layer1_count']}")
    logger.info(f"  Layer 2 (Sec):  {stats['layer2_count']}")
    logger.info(f"  Layer 3 (Chnk): {stats['layer3_count']}")
    logger.info(f"  Dimenze:        {stats['dimensions']}D")
    logger.info(f"")
    logger.info(f"BM25:")
    logger.info(f"  Layer 1:        {stats['bm25_layer1_count']}")
    logger.info(f"  Layer 2:        {stats['bm25_layer2_count']}")
    logger.info(f"  Layer 3:        {stats['bm25_layer3_count']}")
    logger.info(f"")
    logger.info(f"Hybrid Search:    {stats['hybrid_enabled']}")
    logger.info(f"RRF Fusion k:     {stats['fusion_k']}")

    logger.info("="*80)


def main():
    """Hlavní funkce."""
    parser = argparse.ArgumentParser(
        description="Správa centrální databáze dokumentů (vector_db/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklady použití:
  # Přidat nový dokument
  python manage_vector_db.py add data/dokument.pdf

  # Migrovat existující vector store
  python manage_vector_db.py migrate output/BZ_VR1/20251024_164925/phase4_vector_store

  # Zobrazit statistiky
  python manage_vector_db.py stats

  # Vytvořit prázdnou databázi z existujícího store
  python manage_vector_db.py init --from output/existing_store
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Příkaz")

    # Příkaz: add
    parser_add = subparsers.add_parser("add", help="Přidat dokument do databáze")
    parser_add.add_argument("document", type=Path, help="Cesta k dokumentu")

    # Příkaz: migrate
    parser_migrate = subparsers.add_parser("migrate", help="Migrovat vector store do databáze")
    parser_migrate.add_argument("store_path", type=Path, help="Cesta k vector store")

    # Příkaz: stats
    subparsers.add_parser("stats", help="Zobrazit statistiky databáze")

    # Příkaz: init
    parser_init = subparsers.add_parser("init", help="Vytvořit novou databázi")
    parser_init.add_argument(
        "--from",
        dest="from_store",
        type=Path,
        help="Volitelná cesta k existujícímu store pro inicializaci"
    )

    args = parser.parse_args()

    if args.command == "add":
        add_document_to_db(args.document)

    elif args.command == "migrate":
        migrate_store_to_db(args.store_path)

    elif args.command == "stats":
        show_stats()

    elif args.command == "init":
        create_central_db(args.from_store)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
