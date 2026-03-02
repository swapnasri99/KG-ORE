"""
MMEAD Setup and Testing for KG-ORE
===================================

This script tests MMEAD installation and compares its entity linking
quality against your current KGPR entity linking files.

INSTALLATION (run these in your terminal first):
    
    # Step 1: Activate your virtual environment
    cd ~/KG-ORE
    source .venv/bin/activate

    # Step 2: Install MMEAD and its dependency DuckDB
    pip install mmead --break-system-packages
    # OR if you're in a venv (which you are):
    pip install mmead

    # Step 3: Run this script to test
    python test_mmead.py

WHAT MMEAD DOWNLOADS (first run only, stored in cache after):
    - Entity links for MS MARCO v1 passages (~2-3 GB)
    - Wikipedia2Vec embeddings 300d (~1-2 GB)
    - Mappings (~small)
    Total: ~4-5 GB first download, then cached

NOTE: First run will be slow (downloading + building DuckDB). 
      After that, loading takes seconds.
"""

import sys
import json
import time
import os


def test_installation():
    """Step 1: Check if MMEAD is installed."""
    print("=" * 70)
    print("STEP 1: Checking MMEAD installation")
    print("=" * 70)
    
    try:
        import mmead
        print("  ✓ mmead imported successfully")
    except ImportError:
        print("  ✗ mmead not installed!")
        print("  Run: pip install mmead")
        return False

    try:
        import duckdb
        print(f"  ✓ duckdb version: {duckdb.__version__}")
    except ImportError:
        print("  ✗ duckdb not installed!")
        print("  Run: pip install duckdb")
        return False

    try:
        import numpy as np
        print(f"  ✓ numpy version: {np.__version__}")
    except ImportError:
        print("  ✗ numpy not installed!")
        return False

    return True


def test_entity_links():
    """Step 2: Load entity links and check a few passages."""
    print("\n" + "=" * 70)
    print("STEP 2: Loading entity links (first time downloads ~2-3 GB)")
    print("=" * 70)

    from mmead import get_links

    print("  Loading REL entity links for MS MARCO v1 passages...")
    print("  (This may take a few minutes on first run)")
    
    start = time.time()
    links = get_links('v1', 'passage', linker='rel')
    elapsed = time.time() - start
    print(f"  ✓ Loaded in {elapsed:.1f}s")

    # Test with the William Bradford passages we know
    test_pids = [7067032, 2495755, 2495759, 4309131, 123]

    for pid in test_pids:
        print(f"\n  --- Passage {pid} ---")
        try:
            result = links.load_links_from_docid(pid)
            
            # Parse the JSON result
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
            
            entities = data.get('passage', [])
            print(f"  Found {len(entities)} entities:")
            for ent in entities:
                name = ent.get('entity', 'unknown')
                eid = ent.get('entity_id', '?')
                tag = ent.get('details', {}).get('tag', '?')
                score = ent.get('details', {}).get('md_score', 0)
                print(f"    - {name} (id={eid}, type={tag}, confidence={score:.3f})")
        except Exception as e:
            print(f"  Error: {e}")

    return links


def test_embeddings():
    """Step 3: Load Wikipedia2Vec embeddings and test similarity."""
    print("\n" + "=" * 70)
    print("STEP 3: Loading Wikipedia2Vec embeddings (first time downloads ~1-2 GB)")
    print("=" * 70)

    from mmead import get_embeddings
    import numpy as np

    print("  Loading 300d embeddings...")
    start = time.time()
    emb = get_embeddings(300)
    elapsed = time.time() - start
    print(f"  ✓ Loaded in {elapsed:.1f}s")

    # Test entity embeddings
    test_pairs = [
        ("William Bradford", "Plymouth Colony"),      # should be high
        ("William Bradford", "Mayflower"),             # should be high
        ("Plymouth Colony", "Mayflower"),              # should be high
        ("William Bradford", "Albert Einstein"),       # should be low
        ("Alaska", "United States"),                   # should be high
        ("Alaska", "William Shakespeare"),             # should be low
    ]

    print("\n  Entity similarity scores (dot product):")
    print(f"  {'Entity A':<25} {'Entity B':<25} {'Score':>10}")
    print(f"  {'-'*25} {'-'*25} {'-'*10}")

    for ent_a, ent_b in test_pairs:
        try:
            vec_a = emb.load_entity_embedding(ent_a)
            vec_b = emb.load_entity_embedding(ent_b)
            
            if vec_a is not None and vec_b is not None:
                score = float(np.dot(vec_a, vec_b))
                # Also compute cosine similarity
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                cosine = score / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                print(f"  {ent_a:<25} {ent_b:<25} dot={score:>8.2f}  cosine={cosine:.3f}")
            else:
                print(f"  {ent_a:<25} {ent_b:<25} NOT FOUND")
        except Exception as e:
            print(f"  {ent_a:<25} {ent_b:<25} Error: {e}")

    return emb


def test_mappings():
    """Step 4: Test entity ID <-> name mappings."""
    print("\n" + "=" * 70)
    print("STEP 4: Testing mappings")
    print("=" * 70)

    from mmead import get_mappings

    m = get_mappings()

    # Test some entities
    test_entities = ["Manhattan Project", "World War II", "Plymouth Colony", "Montreal"]

    for entity in test_entities:
        try:
            eid = m.get_id_from_entity(entity)
            back = m.get_entity_from_id(eid)
            print(f"  {entity} → id={eid} → {back}")
        except Exception as e:
            print(f"  {entity} → Error: {e}")

    return m


def compare_with_kgpr(links):
    """Step 5: Compare MMEAD entities vs KGPR entities for same passages."""
    print("\n" + "=" * 70)
    print("STEP 5: Comparing MMEAD vs KGPR entity linking")
    print("=" * 70)

    kgpr_path = "entity_linking_results/passage_test_with_id_bm25rank1000.jsonl"
    if not os.path.exists(kgpr_path):
        print(f"  KGPR file not found: {kgpr_path}")
        print("  Skipping comparison")
        return

    # Load first 5 KGPR passages
    kgpr_passages = {}
    with open(kgpr_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            item = json.loads(line.strip())
            kgpr_passages[str(item['id'])] = item

    # Compare
    for pid, kgpr_data in kgpr_passages.items():
        print(f"\n  {'█' * 60}")
        print(f"  PASSAGE {pid}")
        print(f"  {'█' * 60}")
        
        # Show passage text (truncated)
        text = kgpr_data.get('text', '')[:200]
        print(f"  Text: {text}...")

        # KGPR entities
        kgpr_ents = list(set(e.lower().strip() for e in kgpr_data.get('entity_name', [])))
        print(f"\n  KGPR entities ({len(kgpr_ents)}):")
        for e in sorted(kgpr_ents):
            print(f"    - {e}")

        # MMEAD entities
        try:
            result = links.load_links_from_docid(int(pid))
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            mmead_ents = []
            mmead_ids = []
            for ent in data.get('passage', []):
                name = ent.get('entity', '').lower().strip()
                eid = ent.get('entity_id', '')
                if name and name not in mmead_ents:
                    mmead_ents.append(name)
                    mmead_ids.append(eid)

            print(f"\n  MMEAD entities ({len(mmead_ents)}):")
            for name, eid in zip(mmead_ents, mmead_ids):
                print(f"    - {name} (id={eid})")

            # Comparison
            kgpr_set = set(kgpr_ents)
            mmead_set = set(mmead_ents)
            shared = kgpr_set & mmead_set
            only_kgpr = kgpr_set - mmead_set
            only_mmead = mmead_set - kgpr_set

            print(f"\n  COMPARISON:")
            print(f"    Shared:      {len(shared)} → {shared or 'none'}")
            print(f"    Only KGPR:   {len(only_kgpr)} → {only_kgpr or 'none'}")
            print(f"    Only MMEAD:  {len(only_mmead)} → {only_mmead or 'none'}")

        except Exception as e:
            print(f"\n  MMEAD error: {e}")



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("MMEAD Setup and Testing for KG-ORE")
    print("=" * 70)

    # Step 1: Check installation
    if not test_installation():
        print("\nPlease install MMEAD first:")
        print("  pip install mmead")
        sys.exit(1)

    # Step 2: Test entity links
    links = test_entity_links()

    # Step 3: Test embeddings
    emb = test_embeddings()

    # Step 4: Test mappings
    m = test_mappings()

    # Step 5: Compare with KGPR
    if links:
        compare_with_kgpr(links)

    

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)