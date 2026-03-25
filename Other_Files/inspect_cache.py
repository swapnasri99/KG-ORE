"""
Deep debug: Why are 86% of PIDs missing from MMEAD REL?
Run this on your machine: python debug_coverage.py
"""

import ir_datasets

# ============================================================
# STEP 1: What text do the MISSING passages actually contain?
# ============================================================
print("=" * 70)
print("STEP 1: Passage text for test PIDs")
print("=" * 70)

ds = ir_datasets.load('msmarco-passage')
docstore = ds.docs_store()

test_pids = {
    "3539483": "MISSING",
    "357584": "MISSING", 
    "8832562": "MISSING",
    "8412682": "MISSING",
    "527690": "MISSING",
    "8184298": "MISSING",
    "1568086": "FOUND",
    "7267248": "FOUND",
}

for pid, status in test_pids.items():
    try:
        text = docstore.get(pid).text[:200]
        print(f"\n  [{status}] pid={pid}: {text}")
    except Exception as e:
        print(f"\n  [{status}] pid={pid}: ERROR - {e}")

# ============================================================
# STEP 2: Check DuckDB directly — are these PIDs truly absent?
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: DuckDB REL direct lookup")
print("=" * 70)

from mmead import get_links
links = get_links('v1', 'passage', linker='rel')
cursor = links.cursor

for pid, status in test_pids.items():
    r = cursor.execute(
        f"SELECT pid, entity, entity_id FROM msmarco_v1_passage_links_rel WHERE pid = {pid}"
    ).fetchall()
    print(f"  pid={pid} [{status}]: {len(r)} entities -> {r[:3]}")

# ============================================================
# STEP 3: Check nearby PIDs for MISSING ones
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Nearby PIDs (are there gaps?)")
print("=" * 70)

for pid in ["3539483", "357584", "527690"]:
    p = int(pid)
    rows = cursor.execute(f"""
        SELECT DISTINCT pid FROM msmarco_v1_passage_links_rel 
        WHERE pid BETWEEN {p-5} AND {p+5}
        ORDER BY pid
    """).fetchall()
    print(f"\n  Around pid={pid}: {[r[0] for r in rows]}")

# ============================================================
# STEP 4: Coverage distribution across PID ranges
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Coverage by PID range (1M buckets)")
print("=" * 70)

total_msmarco = 8841823  # total MS MARCO passages
for start in range(0, 9000000, 1000000):
    count = cursor.execute(f"""
        SELECT COUNT(DISTINCT pid) FROM msmarco_v1_passage_links_rel 
        WHERE pid >= {start} AND pid < {start + 1000000}
    """).fetchone()[0]
    pct = count / 1000000 * 100
    print(f"  {start:>8,} - {start+999999:>8,}: {count:>7,} passages ({pct:.1f}%)")

# ============================================================
# STEP 5: What does MMEAD *actually* provide for these passages?
# Try loading via the load_links_from_docid API
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Try MMEAD API directly (load_links_from_docid)")
print("=" * 70)

for pid in ["3539483", "357584", "1568086"]:
    try:
        result = links.load_links_from_docid(int(pid))
        print(f"  pid={pid}: {result}")
    except Exception as e:
        print(f"  pid={pid}: ERROR - {type(e).__name__}: {e}")

# ============================================================
# STEP 6: Check if 'field' column matters
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: What values does 'field' column have?")
print("=" * 70)

fields = cursor.execute("""
    SELECT DISTINCT field, COUNT(*) as cnt 
    FROM msmarco_v1_passage_links_rel 
    GROUP BY field
""").fetchall()
print(f"  Fields: {fields}")

# Maybe entities are stored under a different field?
for pid in ["3539483"]:
    r = cursor.execute(f"""
        SELECT * FROM msmarco_v1_passage_links_rel WHERE pid = {pid}
    """).fetchall()
    print(f"  ALL columns for pid={pid}: {r}")