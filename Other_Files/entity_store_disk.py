"""
Disk-based Entity Store using SQLite
=====================================
Drop-in replacement for KGPREntityStore that reads from SQLite instead of 
loading everything into RAM.

Two steps:
  1. BUILD the database (one-time, offline):
     python entity_store_disk.py build entity_linking_results/msmarco_passage_with_id.jsonl passage_entities.db

  2. USE in your experiment:
     Replace passage_el_path with the .db file path, and use DiskEntityStore instead.

Memory usage: ~50 MB (SQLite cache) vs ~5-10 GB (full dict in RAM)
Speed: ~0.1ms per lookup (negligible compared to MonoT5 scoring)
Disk usage: ~2-4 GB for the .db file
"""

import os
import json
import sqlite3
from typing import Dict, Set, List
from tqdm import tqdm


class DiskEntityStore:
    """
    SQLite-backed entity store. Same interface as KGPREntityStore
    but reads from disk instead of loading everything into RAM.
    
    Usage:
        store = DiskEntityStore("passage_entities.db")
        names = store.get_passage_names("12345")
        mids  = store.get_passage_mids("12345")
    """
    
    def __init__(self, db_path: str, query_el_path: str = None):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite database not found: {db_path}\n"
                f"Build it first: python entity_store_disk.py build <jsonl_path> {db_path}"
            )
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = -50000")  # ~50 MB cache
        self.conn.execute("PRAGMA journal_mode = WAL")
        
        # Count entries
        count = self.conn.execute("SELECT COUNT(*) FROM passage_entities").fetchone()[0]
        print(f"  DiskEntityStore: {count:,} passages (SQLite: {db_path})")
        
        # Optional: load query EL into memory (tiny, usually <100 queries)
        self._query_data: Dict[str, Dict[str, Set[str]]] = {}
        if query_el_path and os.path.exists(query_el_path):
            self._load_query_el(query_el_path)
    
    def _load_query_el(self, path: str):
        """Load query entities into memory (small, usually <100 entries)."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = str(item["id"])
                names = {n.lower().strip() for n in item.get("entity_name", []) if len(n.strip()) >= 2}
                mids = set(item.get("entity", []))
                if names or mids:
                    self._query_data[qid] = {"names": names, "mids": mids}
        print(f"    + {len(self._query_data)} query entities loaded")
    
    def _get_passage(self, docno: str):
        """Fetch one passage's entity data from SQLite."""
        row = self.conn.execute(
            "SELECT names, mids FROM passage_entities WHERE docno = ?",
            (str(docno),)
        ).fetchone()
        if row is None:
            return None
        names_str, mids_str = row
        return {
            "names": set(json.loads(names_str)) if names_str else set(),
            "mids": set(json.loads(mids_str)) if mids_str else set(),
        }
    
    def get_passage_names(self, docno: str) -> Set[str]:
        data = self._get_passage(docno)
        return data["names"] if data else set()
    
    def get_passage_mids(self, docno: str) -> Set[str]:
        data = self._get_passage(docno)
        return data["mids"] if data else set()
    
    def get_query_names(self, qid: str) -> Set[str]:
        data = self._query_data.get(str(qid))
        return data["names"] if data else set()
    
    def get_query_mids(self, qid: str) -> Set[str]:
        data = self._query_data.get(str(qid))
        return data["mids"] if data else set()
    
    def has_passage(self, docno: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM passage_entities WHERE docno = ? LIMIT 1",
            (str(docno),)
        ).fetchone()
        return row is not None
    
    def passage_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM passage_entities").fetchone()[0]
    
    def get_coverage(self, docnos: List[str]) -> Dict:
        total = len(docnos)
        covered = sum(1 for d in docnos if self.has_passage(d))
        return {
            'total': total,
            'covered': covered,
            'missing': total - covered,
            'coverage_pct': f"{100*covered/total:.1f}%" if total > 0 else "N/A"
        }
    
    def close(self):
        self.conn.close()


def build_database(jsonl_path: str, db_path: str):
    """
    Build SQLite database from JSONL entity linking file.
    
    Usage:
        python entity_store_disk.py build input.jsonl output.db
    """
    print(f"Building SQLite database...")
    print(f"  Input:  {jsonl_path}")
    print(f"  Output: {db_path}")
    
    # Count lines first for progress bar
    print("  Counting lines...")
    total_lines = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"  Total lines: {total_lines:,}")
    
    # Remove existing db
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE passage_entities (
            docno TEXT PRIMARY KEY,
            names TEXT,
            mids TEXT
        )
    """)
    
    batch = []
    batch_size = 10000
    inserted = 0
    skipped = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="  Building DB"):
            line = line.strip()
            if not line:
                continue
            
            item = json.loads(line)
            doc_id = str(item["id"])
            
            names = [n.lower().strip() for n in item.get("entity_name", []) if len(n.strip()) >= 2]
            mids = list(item.get("entity", []))
            
            if not names and not mids:
                skipped += 1
                continue
            
            batch.append((
                doc_id,
                json.dumps(names),
                json.dumps(mids),
            ))
            
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT OR IGNORE INTO passage_entities (docno, names, mids) VALUES (?, ?, ?)",
                    batch
                )
                conn.commit()
                inserted += len(batch)
                batch = []
    
    # Final batch
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO passage_entities (docno, names, mids) VALUES (?, ?, ?)",
            batch
        )
        conn.commit()
        inserted += len(batch)
    
    # Create index (speeds up lookups significantly)
    print("  Creating index...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docno ON passage_entities(docno)")
    conn.commit()
    
    # Stats
    count = conn.execute("SELECT COUNT(*) FROM passage_entities").fetchone()[0]
    conn.close()
    
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"\n  Done!")
    print(f"  Inserted: {inserted:,}")
    print(f"  Skipped (no entities): {skipped:,}")
    print(f"  Total in DB: {count:,}")
    print(f"  DB size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Build DB:  python entity_store_disk.py build <jsonl_path> <db_path>")
        print("  Test DB:   python entity_store_disk.py test <db_path>")
        print()
        print("Examples:")
        print("  python entity_store_disk.py build entity_linking_results/msmarco_passage_with_id.jsonl passage_entities.db")
        print("  python entity_store_disk.py test passage_entities.db")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "build":
        if len(sys.argv) < 4:
            print("Usage: python entity_store_disk.py build <jsonl_path> <db_path>")
            sys.exit(1)
        build_database(sys.argv[2], sys.argv[3])
    
    elif cmd == "test":
        if len(sys.argv) < 3:
            print("Usage: python entity_store_disk.py test <db_path>")
            sys.exit(1)
        
        store = DiskEntityStore(sys.argv[2])
        
        # Test with first few entries
        rows = store.conn.execute("SELECT docno FROM passage_entities LIMIT 5").fetchall()
        for (docno,) in rows:
            names = store.get_passage_names(docno)
            mids = store.get_passage_mids(docno)
            print(f"  {docno}: names={names}, mids={mids}")
        
        store.close()
        print("✓ Test complete!")