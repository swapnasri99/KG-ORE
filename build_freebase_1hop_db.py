"""
Build disk-backed Freebase 1-hop edges SQLite DB.

The subgraph_1hop_triples.npy contains integer IDs, not MID strings.
We need ent2id.pickle to map IDs back to MIDs like "m.01fz8s".

Usage:
    python build_freebase_1hop_db.py \
        --triples_npy freebase/subgraph_1hop_triples.npy \
        --ent2id_pickle freebase/ent2id.pickle \
        --out_db freebase_1hop.db

This stores BIDIRECTIONAL edges (h→t AND t→h) so connectivity
checks only need one query direction.
"""
import argparse
import sqlite3
import pickle
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_npy", required=True, help="Path to subgraph_1hop_triples.npy")
    ap.add_argument("--ent2id_pickle", required=True, help="Path to ent2id.pickle")
    ap.add_argument("--out_db", required=True, help="Output SQLite db path")
    ap.add_argument("--batch", type=int, default=200000)
    args = ap.parse_args()

    # Load ent2id mapping: mid_string -> integer_id
    print("Loading ent2id.pickle...")
    with open(args.ent2id_pickle, 'rb') as f:
        ent2id = pickle.load(f)

    # Reverse mapping: integer_id -> mid_string
    id2ent = {v: k for k, v in ent2id.items()}
    print(f"  {len(ent2id):,} entities in mapping")

    # Load triples: shape (N, 3) with [head_id, relation_id, tail_id]
    print("Loading triples...")
    triples = np.load(args.triples_npy)
    print(f"  {len(triples):,} triples")

    # Build SQLite DB
    conn = sqlite3.connect(args.out_db)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("DROP TABLE IF EXISTS edges;")
    cur.execute("CREATE TABLE edges (h TEXT NOT NULL, t TEXT NOT NULL);")

    buf = []
    skipped = 0
    inserted = 0

    for row in tqdm(triples, desc="Building edges"):
        h_id = int(row[0])
        t_id = int(row[2])

        h_mid = id2ent.get(h_id)
        t_mid = id2ent.get(t_id)

        if h_mid is None or t_mid is None:
            skipped += 1
            continue

        # Store BOTH directions for bidirectional lookup
        buf.append((h_mid, t_mid))
        buf.append((t_mid, h_mid))

        if len(buf) >= args.batch:
            cur.executemany("INSERT INTO edges(h, t) VALUES (?, ?)", buf)
            conn.commit()
            inserted += len(buf)
            buf = []

    if buf:
        cur.executemany("INSERT INTO edges(h, t) VALUES (?, ?)", buf)
        conn.commit()
        inserted += len(buf)

    # Create index AFTER all inserts (much faster)
    print("Creating index on h column...")
    cur.execute("CREATE INDEX idx_h ON edges(h);")
    conn.commit()

    # Stats
    total_rows = cur.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()

    db_size_mb = __import__('os').path.getsize(args.out_db) / (1024 * 1024)
    print(f"\nDone: {args.out_db}")
    print(f"  Rows: {total_rows:,} (bidirectional)")
    print(f"  Skipped (missing in ent2id): {skipped:,}")
    print(f"  DB size: {db_size_mb:.1f} MB")


if __name__ == "__main__":
    main()