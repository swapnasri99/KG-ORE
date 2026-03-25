"""
QUICK CHECK: Compare the two KGPR entity files
================================================
Checks if passage_test (BM25 top-1000) and msmarco_passage (full) 
give SAME entities for the SAME passages.

USAGE:
  python check_kgpr_files.py \
    --bm25 entity_linking_results/passage_test_with_id_bm25rank1000.jsonl \
    --full entity_linking_results/msmarco_passage_with_id.jsonl

Also checks your MMEAD DB to see what entity format it uses.
"""

import json
import argparse
import os
import sqlite3
from collections import Counter


def load_kgpr_sample(filepath, max_n=1000):
    """Load first N entries from KGPR JSONL"""
    data = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            obj = json.loads(line.strip())
            pid = obj.get('id', '')
            data[str(pid)] = {
                'entity_name': obj.get('entity_name', []),
                'entity': obj.get('entity', []),  # Freebase MIDs
                'wikipedia_entity': obj.get('wikipedia_entity', []),
                'mention': obj.get('mention', []),
            }
    return data


def check_kgpr_files(bm25_path, full_path):
    """Compare entities between BM25-1000 and full msmarco files"""
    print(f"\n{'='*70}")
    print(f"COMPARING KGPR FILES")
    print(f"{'='*70}")
    print(f"BM25 file: {bm25_path}")
    print(f"Full file: {full_path}")
    
    # Load sample from BM25 file
    bm25_data = load_kgpr_sample(bm25_path, max_n=500)
    print(f"\nLoaded {len(bm25_data)} passages from BM25 file")
    
    # Look up same passage IDs in full file
    target_pids = set(bm25_data.keys())
    full_data = {}
    
    print(f"Searching for these {len(target_pids)} passages in full file...")
    with open(full_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            pid = str(obj.get('id', ''))
            if pid in target_pids:
                full_data[pid] = {
                    'entity_name': obj.get('entity_name', []),
                    'entity': obj.get('entity', []),
                    'wikipedia_entity': obj.get('wikipedia_entity', []),
                    'mention': obj.get('mention', []),
                }
                if len(full_data) >= len(target_pids):
                    break
    
    found = len(full_data)
    not_found = len(target_pids) - found
    print(f"Found {found} / {len(target_pids)} passages in full file")
    print(f"NOT found: {not_found}")
    
    if not_found > 0:
        missing = target_pids - set(full_data.keys())
        print(f"  Missing PIDs (first 10): {list(missing)[:10]}")
    
    # Compare entities for shared passages
    same_count = 0
    diff_count = 0
    
    print(f"\n--- Entity Comparison for Shared Passages ---")
    shown = 0
    for pid in list(target_pids)[:200]:
        if pid in bm25_data and pid in full_data:
            b_ents = set(bm25_data[pid]['entity'])
            f_ents = set(full_data[pid]['entity'])
            b_names = set(bm25_data[pid]['entity_name'])
            f_names = set(full_data[pid]['entity_name'])
            
            if b_ents == f_ents:
                same_count += 1
            else:
                diff_count += 1
                if shown < 5:
                    print(f"\n  PID {pid}: DIFFERENT entities!")
                    print(f"    BM25 MIDs:  {sorted(b_ents)[:5]}")
                    print(f"    Full MIDs:  {sorted(f_ents)[:5]}")
                    print(f"    BM25 names: {sorted(b_names)[:5]}")
                    print(f"    Full names: {sorted(f_names)[:5]}")
                    print(f"    Only in BM25: {b_ents - f_ents}")
                    print(f"    Only in Full: {f_ents - b_ents}")
                    shown += 1
    
    print(f"\n  Passages with IDENTICAL entities: {same_count}")
    print(f"  Passages with DIFFERENT entities: {diff_count}")
    
    if same_count > 0 and diff_count == 0:
        print(f"\n  >>> Both files give IDENTICAL entities for shared passages.")
        print(f"  >>> The issue is NOT in the entity data itself.")
        print(f"  >>> Check: Is your scorer loading the right file?")
        print(f"  >>> Check: Is the passage ID lookup working correctly?")


def check_freebase_mid_format(kgpr_path, freebase_db_path):
    """Check if KGPR MIDs match the format in Freebase DB"""
    print(f"\n{'='*70}")
    print(f"CHECKING FREEBASE MID FORMAT COMPATIBILITY")
    print(f"{'='*70}")
    
    # Get sample MIDs from KGPR
    kgpr_mids = set()
    with open(kgpr_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            obj = json.loads(line.strip())
            for mid in obj.get('entity', []):
                kgpr_mids.add(mid)
    
    print(f"\nSample KGPR MIDs (from entity file):")
    sample_mids = list(kgpr_mids)[:10]
    for mid in sample_mids:
        print(f"  '{mid}'")
    
    # Check format: do they start with 'm.' or '/m/'?
    m_dot_count = sum(1 for m in kgpr_mids if m.startswith('m.'))
    slash_m_count = sum(1 for m in kgpr_mids if m.startswith('/m/'))
    print(f"\nFormat: 'm.xxx' style: {m_dot_count}")
    print(f"Format: '/m/xxx' style: {slash_m_count}")
    
    if not os.path.exists(freebase_db_path):
        print(f"\nFreebase DB not found: {freebase_db_path}")
        return
    
    # Check what format the Freebase DB uses
    conn = sqlite3.connect(freebase_db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cur.fetchall()]
    
    for table in tables:
        cur.execute(f"PRAGMA table_info([{table}])")
        cols = [c[1] for c in cur.fetchall()]
        
        for col in cols:
            cur.execute(f"SELECT [{col}] FROM [{table}] LIMIT 10")
            vals = [str(r[0]) for r in cur.fetchall()]
            
            has_m_dot = any(v.startswith('m.') for v in vals)
            has_slash_m = any(v.startswith('/m/') for v in vals)
            
            if has_m_dot or has_slash_m:
                print(f"\nFreebase DB column '{table}.{col}' samples:")
                for v in vals[:5]:
                    print(f"  '{v}'")
                
                if has_m_dot and slash_m_count > 0:
                    print(f"  WARNING: DB uses 'm.xxx' but KGPR file uses '/m/xxx'!")
                elif has_slash_m and m_dot_count > 0:
                    print(f"  WARNING: DB uses '/m/xxx' but KGPR file uses 'm.xxx'!")
                elif has_m_dot and m_dot_count > 0:
                    print(f"  OK: Both use 'm.xxx' format")
                elif has_slash_m and slash_m_count > 0:
                    print(f"  OK: Both use '/m/xxx' format")
    
    # Try looking up some KGPR MIDs in Freebase DB
    print(f"\n--- Lookup Test: Can we find KGPR entities in Freebase DB? ---")
    for table in tables:
        cur.execute(f"PRAGMA table_info([{table}])")
        cols = [c[1] for c in cur.fetchall()]
        
        for mid in sample_mids[:5]:
            for col in cols:
                # Try exact match
                cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col}] = ?", (mid,))
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  FOUND: '{mid}' in {table}.{col} ({count} rows)")
                    break
                
                # Try with /m/ prefix
                mid_slash = '/m/' + mid[2:] if mid.startswith('m.') else mid
                cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col}] = ?", (mid_slash,))
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  FOUND: '{mid_slash}' (converted) in {table}.{col} ({count} rows)")
                    break
                
                # Try without /m/ prefix
                mid_no_slash = mid.replace('/m/', 'm.') if mid.startswith('/m/') else mid
                cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col}] = ?", (mid_no_slash,))
                count = cur.fetchone()[0]
                if count > 0:
                    print(f"  FOUND: '{mid_no_slash}' (converted) in {table}.{col} ({count} rows)")
                    break
            else:
                print(f"  NOT FOUND: '{mid}' in any column of {table}")
    
    conn.close()


def check_mmead_vs_kgpr_entities(mmead_path, kgpr_path):
    """Compare what MMEAD/REL finds vs what KGPR/ELQ finds for same passages"""
    print(f"\n{'='*70}")
    print(f"MMEAD vs KGPR: Same Passages, Different Entities?")
    print(f"{'='*70}")
    
    if not os.path.exists(mmead_path):
        print(f"MMEAD DB not found: {mmead_path}")
        return
    
    conn = sqlite3.connect(mmead_path)
    cur = conn.cursor()
    
    # Show MMEAD structure
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cur.fetchall()]
    print(f"\nMEAD tables: {tables}")
    
    for table in tables:
        cur.execute(f"PRAGMA table_info([{table}])")
        cols = [(c[1], c[2]) for c in cur.fetchall()]
        print(f"\n  {table} columns: {cols}")
        
        col_names = [c[0] for c in cols]
        cur.execute(f"SELECT * FROM [{table}] LIMIT 3")
        rows = cur.fetchall()
        print(f"  Sample rows:")
        for row in rows:
            row_dict = dict(zip(col_names, row))
            for k, v in row_dict.items():
                if isinstance(v, str) and len(v) > 80:
                    row_dict[k] = v[:80] + '...'
            print(f"    {row_dict}")
    
    # Get some passage IDs from KGPR
    kgpr_pids = []
    with open(kgpr_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            obj = json.loads(line.strip())
            kgpr_pids.append(str(obj.get('id', '')))
    
    # Try finding them in MMEAD
    print(f"\n--- Looking up KGPR passage IDs in MMEAD DB ---")
    for table in tables:
        cur.execute(f"PRAGMA table_info([{table}])")
        all_cols = [c[1] for c in cur.fetchall()]
        
        # Try each column that might be a PID
        for col in all_cols:
            found = 0
            for pid in kgpr_pids[:10]:
                cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col}] = ?", (pid,))
                if cur.fetchone()[0] > 0:
                    found += 1
                else:
                    # Try as integer
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col}] = ?", (int(pid),))
                        if cur.fetchone()[0] > 0:
                            found += 1
                    except:
                        pass
            
            if found > 0:
                print(f"  Column {table}.{col}: Found {found}/10 KGPR passage IDs")
                
                # Show side-by-side for a found passage
                for pid in kgpr_pids[:5]:
                    cur.execute(f"SELECT * FROM [{table}] WHERE [{col}] = ? OR [{col}] = ? LIMIT 5",
                               (pid, int(pid) if pid.isdigit() else pid))
                    rows = cur.fetchall()
                    if rows:
                        print(f"\n    PID {pid} in MMEAD ({table}):")
                        for row in rows[:3]:
                            row_dict = dict(zip(all_cols, row))
                            for k, v in row_dict.items():
                                if isinstance(v, str) and len(v) > 60:
                                    row_dict[k] = v[:60] + '...'
                            print(f"      {row_dict}")
                        break
    
    conn.close()


def check_which_entities_your_scorer_uses(kgpr_path):
    """Analyze what your scorer actually needs"""
    print(f"\n{'='*70}")
    print(f"WHAT YOUR KG SCORER NEEDS")
    print(f"{'='*70}")
    
    # Load a sample passage
    with open(kgpr_path, 'r') as f:
        obj = json.loads(f.readline().strip())
    
    print(f"""
Your KGPR entity file has these fields per passage:
  - 'entity_name':       {obj.get('entity_name', [])[:3]}  (canonical names)
  - 'entity' (MIDs):     {obj.get('entity', [])[:3]}  (Freebase MIDs like m.01fz8s)
  - 'wikipedia_entity':  {obj.get('wikipedia_entity', [])[:2]}  (Wiki page titles)
  - 'mention':           {obj.get('mention', [])[:3]}  (surface forms in text)

YOUR SCORER USES:
  1. Entity Overlap:  Compares 'entity_name' between two passages
     -> Uses exact string matching on canonical names
     -> "William Bradford" == "William Bradford" -> match!
     
  2. KG Connectivity: Uses 'entity' MIDs to look up Freebase DB
     -> Checks if m.01fz8s and m.079lm0p are connected in Freebase
     -> MID format MUST match what's in freebase_1hop_clean.db!

KEY QUESTION: Does your scorer load from the KGPR JSONL file 
or from the MMEAD SQLite DB?
  - If KGPR JSONL -> entities have Freebase MIDs -> Freebase lookup works
  - If MMEAD DB   -> entities have Wiki IDs      -> Freebase lookup FAILS

Also check: Are the MIDs in format 'm.01fz8s' or '/m/01fz8s'?
  Your KGPR file uses: 'm.01fz8s' (no leading slash)
  If your Freebase DB uses '/m/01fz8s' -> lookups will fail!
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25", type=str, default=None,
                       help="KGPR BM25-1000 entity file")
    parser.add_argument("--full", type=str, default=None,
                       help="KGPR full msmarco entity file")
    parser.add_argument("--freebase", type=str, default=None,
                       help="Freebase SQLite DB")
    parser.add_argument("--mmead", type=str, default=None,
                       help="MMEAD SQLite DB")
    args = parser.parse_args()
    
    kgpr_path = args.bm25 or args.full
    
    # Compare BM25 vs Full KGPR files
    if args.bm25 and args.full:
        if os.path.exists(args.bm25) and os.path.exists(args.full):
            check_kgpr_files(args.bm25, args.full)
    
    # Check MID format compatibility with Freebase DB
    if kgpr_path and args.freebase:
        if os.path.exists(kgpr_path):
            check_freebase_mid_format(kgpr_path, args.freebase)
    
    # Compare MMEAD vs KGPR entities
    if args.mmead and kgpr_path:
        if os.path.exists(kgpr_path):
            check_mmead_vs_kgpr_entities(args.mmead, kgpr_path)
    
    # Show what scorer needs
    if kgpr_path and os.path.exists(kgpr_path):
        check_which_entities_your_scorer_uses(kgpr_path)


if __name__ == "__main__":
    main()