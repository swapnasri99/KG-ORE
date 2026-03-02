"""
Build MMEAD Cache DB (Fast)
===========================

Uses bulk DuckDB JOIN queries — finishes in ~5 minutes.

Output: mmead_cache.db containing:
    - passage_entities: pid → entity_ids, entity_names
    - entity_embeddings: entity_name → 300d normalized vector

Usage:
    python build_mmead_cache.py --output mmead_cache.db
"""

import argparse
import json
import sqlite3
import time
import struct
import numpy as np
from collections import defaultdict


def build_cache(output_path: str, emb_dim: int = 300, linker: str = 'rel'):
    print("=" * 70)
    print("Building MMEAD Cache DB")
    print("=" * 70)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")

    # ============================================================
    # Part 1: Entity Links
    # ============================================================
    print("\n[1/3] Exporting entity links...")
    start = time.time()

    conn.execute("DROP TABLE IF EXISTS passage_entities")
    conn.execute("""
        CREATE TABLE passage_entities (
            pid TEXT PRIMARY KEY,
            entity_ids BLOB,
            entity_names TEXT
        )
    """)

    from mmead import get_links
    links = get_links('v1', 'passage', linker=linker)
    table_name = f"msmarco_v1_passage_links_{linker}"
    duckdb_cursor = links.cursor

    print("  Reading all passages from DuckDB...")
    all_rows = duckdb_cursor.execute(f"""
        SELECT pid, entity_id, entity
        FROM {table_name}
        ORDER BY pid
    """).fetchall()
    print(f"  Got {len(all_rows):,} entity rows")

    pid_data = defaultdict(lambda: (set(), set()))
    for pid, eid, name in all_rows:
        pid_str = str(pid)
        if eid is not None:
            pid_data[pid_str][0].add(int(eid))
        if name:
            pid_data[pid_str][1].add(name)

    print(f"  Writing {len(pid_data):,} passages to SQLite...")
    batch = []
    for pid_str, (ids, names) in pid_data.items():
        id_list = sorted(ids)
        id_blob = struct.pack(f'{len(id_list)}i', *id_list)
        names_json = json.dumps(sorted(names))
        batch.append((pid_str, id_blob, names_json))
        if len(batch) >= 50000:
            conn.executemany("INSERT INTO passage_entities VALUES (?, ?, ?)", batch)
            batch = []
    if batch:
        conn.executemany("INSERT INTO passage_entities VALUES (?, ?, ?)", batch)
    conn.commit()

    elapsed = time.time() - start
    print(f"  ✓ {len(pid_data):,} passages in {elapsed:.1f}s")

    # ============================================================
    # Part 2: Collect unique entity names
    # ============================================================
    print("\n[2/3] Collecting unique entity names...")
    all_entity_names = set()
    for ids, names in pid_data.values():
        all_entity_names.update(names)
    print(f"  Found {len(all_entity_names):,} unique entity names")

    del pid_data
    del all_rows

    # ============================================================
    # Part 3: Bulk extract embeddings via temp table JOIN
    # ============================================================
    print(f"\n[3/3] Extracting embeddings ({emb_dim}d) via bulk JOIN...")
    start = time.time()

    conn.execute("DROP TABLE IF EXISTS entity_embeddings")
    conn.execute("""
        CREATE TABLE entity_embeddings (
            entity_name TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)

    from mmead import get_embeddings
    emb = get_embeddings(emb_dim)
    emb_cursor = emb.cursor

    # Convert entity names to DuckDB keys: "Manhattan Project" → "ENTITY/Manhattan_Project"
    entity_list = sorted(all_entity_names)
    keys = ["ENTITY/" + name.replace(" ", "_") for name in entity_list]
    # Map key back to original name for storage
    key_to_name = {k: n for k, n in zip(keys, entity_list)}

    print(f"  Creating temp table with {len(keys):,} keys...")
    emb_cursor.execute("CREATE TEMPORARY TABLE tmp_keys (key VARCHAR)")
    emb_cursor.executemany("INSERT INTO tmp_keys VALUES (?)", [(k,) for k in keys])

    print(f"  Running bulk JOIN on wiki2vec_{emb_dim}d...")
    results = emb_cursor.execute(f"""
        SELECT w.key, w.embedding FROM wiki2vec_{emb_dim}d w
        INNER JOIN tmp_keys t ON w.key = t.key
    """).fetchall()

    emb_cursor.execute("DROP TABLE tmp_keys")

    elapsed_query = time.time() - start
    print(f"  Found {len(results):,}/{len(keys):,} embeddings in {elapsed_query:.1f}s")

    # Write to SQLite
    print(f"  Writing to SQLite...")
    batch = []
    for key, vec_data in results:
        name = key_to_name.get(key)
        if name is None:
            continue

        # Convert to numpy, normalize
        vec = np.array(vec_data, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec_blob = vec.tobytes()
        batch.append((name, vec_blob))

        if len(batch) >= 50000:
            conn.executemany("INSERT OR IGNORE INTO entity_embeddings VALUES (?, ?)", batch)
            batch = []

    if batch:
        conn.executemany("INSERT OR IGNORE INTO entity_embeddings VALUES (?, ?)", batch)
    conn.commit()

    not_found = len(all_entity_names) - len(results)
    elapsed = time.time() - start
    print(f"  ✓ {len(results):,} embeddings written, {not_found:,} not found ({elapsed:.1f}s)")

    # ============================================================
    # Index
    # ============================================================
    print("\nCreating indexes...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_name ON entity_embeddings(entity_name)")
    conn.commit()

    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n{'=' * 70}")
    print(f"DONE: {output_path}")
    print(f"  Passages: {conn.execute('SELECT COUNT(*) FROM passage_entities').fetchone()[0]:,}")
    print(f"  Embeddings: {conn.execute('SELECT COUNT(*) FROM entity_embeddings').fetchone()[0]:,}")
    print(f"  DB size: {size_mb:.0f} MB")
    print(f"{'=' * 70}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="mmead_cache.db")
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--linker", type=str, default="rel")
    args = parser.parse_args()
    build_cache(args.output, args.emb_dim, args.linker)