"""
MINI EXPERIMENT: See actual KG scores for real passages
========================================================
This does what your experiment does but for just 2-3 queries,
and prints EVERY score so you can see what's happening.

USAGE (compare both sources):

  # Using BM25-1000 JSONL (the one that works well):
  python mini_experiment.py \
    --passage_el entity_linking_results/passage_test_with_id_bm25rank1000.jsonl \
    --freebase_dir freebase

  # Using the full DB (the one that gives worse results):
  python mini_experiment.py \
    --passage_el_db passage_entities.db \
    --freebase_dir freebase

Run both and compare the output!
"""

import sys
import os
import json
import math
import numpy as np
import argparse

sys.path.insert(0, '.')

from kg_scorer_unified import (
    KGPREntityStore, KGScorerUnified, FreebaseGraph, 
    DiskEntityStore, create_scorer
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_el", type=str, default=None,
                       help="JSONL entity file (in-memory)")
    parser.add_argument("--passage_el_db", type=str, default=None,
                       help="SQLite entity DB (disk)")
    parser.add_argument("--freebase_dir", type=str, default="freebase")
    parser.add_argument("--graph_dir", type=str, default=None,
                       help="Corpus graph dir (optional, to get real neighbors)")
    args = parser.parse_args()

    if not args.passage_el and not args.passage_el_db:
        print("Provide one of: --passage_el <jsonl> OR --passage_el_db <db>")
        return

    # ================================================================
    # 1. Load entity store
    # ================================================================
    print("=" * 70)
    if args.passage_el_db:
        print(f"MODE: DiskEntityStore (DB: {args.passage_el_db})")
        entity_store = DiskEntityStore(args.passage_el_db)
        source_label = "DB"
    else:
        print(f"MODE: KGPREntityStore (JSONL: {args.passage_el})")
        entity_store = KGPREntityStore(passage_el_path=args.passage_el)
        source_label = "JSONL"
    print("=" * 70)

    # ================================================================
    # 2. Load Freebase graph
    # ================================================================
    fb = FreebaseGraph(args.freebase_dir)

    # ================================================================
    # 3. Pick sample passages to test
    # ================================================================
    # Get some passage IDs that have entities
    if args.passage_el:
        # From JSONL, grab first passages with entities
        sample_pids = []
        with open(args.passage_el, 'r') as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                obj = json.loads(line.strip())
                pid = str(obj['id'])
                if obj.get('entity', []):
                    sample_pids.append(pid)
    else:
        # From DB, grab some passages
        import sqlite3
        conn = sqlite3.connect(args.passage_el_db)
        rows = conn.execute(
            "SELECT docno FROM passage_entities WHERE mids != '[]' LIMIT 20"
        ).fetchall()
        sample_pids = [r[0] for r in rows]
        conn.close()

    print(f"\nSample passages: {sample_pids[:10]}")

    # ================================================================
    # 4. For each passage, show entities and test scoring
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ENTITY DATA FROM {source_label}")
    print(f"{'='*70}")

    for pid in sample_pids[:5]:
        names = entity_store.get_passage_names(pid)
        mids = entity_store.get_passage_mids(pid)
        
        # Check how many MIDs exist in Freebase
        valid_mids = {m for m in mids if fb.has_entity(m)}
        
        print(f"\n  PID {pid}:")
        print(f"    Names ({len(names)}): {sorted(names)[:5]}")
        print(f"    MIDs  ({len(mids)}): {sorted(mids)[:5]}")
        print(f"    Valid in Freebase: {len(valid_mids)}/{len(mids)}")

    # ================================================================
    # 5. Score pairs of passages — this is what your experiment does
    # ================================================================
    print(f"\n{'='*70}")
    print(f"KG SCORING: Passage Pairs (gamma=1.0, alpha=0, beta=0)")
    print(f"{'='*70}")

    def kg_score_log(mids1, mids2):
        """Same as your scorer with mode='log'"""
        if not mids1 or not mids2:
            return 0.0, 0, 0, 0
        valid1 = {m for m in mids1 if fb.has_entity(m)}
        valid2 = {m for m in mids2 if fb.has_entity(m)}
        if not valid1 or not valid2:
            return 0.0, len(valid1), len(valid2), 0
        connected = 0
        for m1 in valid1:
            for m2 in valid2:
                if m1 != m2 and fb.are_connected(m1, m2):
                    connected += 1
        score = min(math.log2(connected + 1) / 5.0, 1.0) if connected > 0 else 0.0
        return score, len(valid1), len(valid2), connected

    # Score all pairs
    print(f"\n  Scoring {len(sample_pids[:10])} passages against each other:\n")
    print(f"  {'Doc_A':>10} vs {'Doc_B':>10}  |  KG_Score  Valid_A  Valid_B  Connected")
    print(f"  {'-'*75}")
    
    non_zero_scores = 0
    zero_scores = 0
    
    for i, pid_a in enumerate(sample_pids[:10]):
        mids_a = entity_store.get_passage_mids(pid_a)
        for j, pid_b in enumerate(sample_pids[:10]):
            if j <= i:
                continue
            mids_b = entity_store.get_passage_mids(pid_b)
            score, v1, v2, conn = kg_score_log(mids_a, mids_b)
            
            marker = " <<<" if score > 0 else ""
            print(f"  {pid_a:>10} vs {pid_b:>10}  |  {score:.4f}   {v1:>5}    {v2:>5}    {conn:>5}{marker}")
            
            if score > 0:
                non_zero_scores += 1
            else:
                zero_scores += 1

    print(f"\n  Non-zero KG scores: {non_zero_scores}")
    print(f"  Zero KG scores:     {zero_scores}")
    print(f"  Ratio with signal:  {100*non_zero_scores/(non_zero_scores+zero_scores):.1f}%")

    # ================================================================
    # 6. Now the CRITICAL test: what happens with a passage and
    #    its ACTUAL neighbors from the corpus graph?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"NEIGHBOR COVERAGE TEST")
    print(f"{'='*70}")
    print(f"\n  Testing: for a given passage's neighbors, how many have entities?")
    
    # We don't have the corpus graph here, so we'll simulate with
    # a range of passage IDs around each sample
    for pid in sample_pids[:3]:
        pid_int = int(pid)
        # Simulate neighbors: nearby passage IDs (in real experiment, 
        # these come from corpus graph)
        fake_neighbors = [str(pid_int + offset) for offset in range(-50, 51) if offset != 0]
        
        has_entities = 0
        has_mids = 0
        has_valid_mids = 0
        
        for n_pid in fake_neighbors:
            names = entity_store.get_passage_names(n_pid)
            mids = entity_store.get_passage_mids(n_pid)
            if names or mids:
                has_entities += 1
            if mids:
                has_mids += 1
                valid = {m for m in mids if fb.has_entity(m)}
                if valid:
                    has_valid_mids += 1
        
        print(f"\n  PID {pid}: checking {len(fake_neighbors)} nearby passages")
        print(f"    With any entities:    {has_entities}/{len(fake_neighbors)}")
        print(f"    With Freebase MIDs:   {has_mids}/{len(fake_neighbors)}")
        print(f"    With VALID FB MIDs:   {has_valid_mids}/{len(fake_neighbors)}")
        
        if source_label == "DB":
            print(f"    → DB covers ALL passages, so neighbors get KG scores even if irrelevant")
        else:
            print(f"    → JSONL only covers BM25-1000 passages, so most neighbors get score=0")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY ({source_label})")
    print(f"{'='*70}")
    print(f"""
  Entity source: {source_label}
  Passages with entities: {entity_store.passage_count():,}
  
  When gamma=1.0 (KG only):
  - Score = KG connectivity between document and neighbor entities
  - If a neighbor has NO entities → score = 0.0 → pushed to bottom
  - If a neighbor HAS entities  → score > 0 possible → can be selected
  
  KEY DIFFERENCE:
  - JSONL (207K passages): Most neighbors outside BM25-1000 → score=0 → filtered out
  - DB (8.8M passages): ALL neighbors have entities → noisy ones get scores too
  
  Run this script with BOTH sources and compare the "Non-zero KG scores" count!
""")


if __name__ == "__main__":
    main()