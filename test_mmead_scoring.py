"""
Test MMEAD embeddings with disambiguated entity names
and demonstrate how entity overlap + embedding similarity scoring works.

Usage:
    python test_mmead_scoring.py
"""

import json
import numpy as np
import time


def test_embedding_lookup():
    """Test which entity names have embeddings."""
    from mmead import get_embeddings, get_links, get_mappings

    print("=" * 70)
    print("PART 1: Which entity names have embeddings?")
    print("=" * 70)

    emb = get_embeddings(300)
    links = get_links('v1', 'passage', linker='rel')
    mappings = get_mappings()

    # Get entities from passage 7067032
    result = links.load_links_from_docid(7067032)
    if isinstance(result, str):
        data = json.loads(result)
    else:
        data = result

    print("\n  Passage 7067032 entities:")
    for ent in data.get('passage', []):
        name = ent['entity']
        eid = ent['entity_id']

        # Try loading embedding with entity name as-is
        try:
            vec = emb.load_entity_embedding(name)
            found = "✓ FOUND" if vec is not None else "✗ NOT FOUND"
        except:
            found = "✗ NOT FOUND"
            vec = None

        print(f"    {name:<40} (id={eid}) → {found}")

        # If not found, try without disambiguation brackets
        if vec is None and "(" in name:
            base_name = name.split("(")[0].strip()
            try:
                vec2 = emb.load_entity_embedding(base_name)
                found2 = "✓ FOUND" if vec2 is not None else "✗"
                print(f"      → Tried '{base_name}' → {found2}")
            except:
                print(f"      → Tried '{base_name}' → ✗")

    # Test more entity names
    print("\n  Testing various entity name formats:")
    test_names = [
        "Plymouth Colony",
        "Mayflower Compact",
        "Mayflower",
        "England",
        "Netherlands",
        "Leiden",
        "William Bradford (governor)",
        "William Bradford (Attorney General)",
        "William Bradford",
        "New World",
        "English Dissenters",
        "United States",
        "Alaska",
        "Manhattan Project",
        "World War II",
        "Bradford",
    ]

    found_count = 0
    not_found = []
    for name in test_names:
        try:
            vec = emb.load_entity_embedding(name)
            if vec is not None:
                norm = np.linalg.norm(vec)
                print(f"    ✓ {name:<40} norm={norm:.2f}")
                found_count += 1
            else:
                print(f"    ✗ {name:<40} (returned None)")
                not_found.append(name)
        except Exception as e:
            print(f"    ✗ {name:<40} ({e})")
            not_found.append(name)

    print(f"\n  Found: {found_count}/{len(test_names)}")
    if not_found:
        print(f"  Not found: {not_found}")


def demo_scoring():
    """Demonstrate how entity overlap + embedding similarity scoring works."""
    from mmead import get_embeddings, get_links

    print("\n" + "=" * 70)
    print("PART 2: Scoring demo — two passages compared")
    print("=" * 70)

    emb = get_embeddings(300)
    links = get_links('v1', 'passage', linker='rel')

    # Load two passages about William Bradford
    passages = {}
    for pid in [7067032, 2495755]:
        result = links.load_links_from_docid(pid)
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        # Extract unique entities with their IDs
        entities = {}
        for ent in data.get('passage', []):
            eid = ent['entity_id']
            if eid not in entities:
                entities[eid] = ent['entity']
        passages[pid] = entities

    # Show entities
    for pid, ents in passages.items():
        print(f"\n  Passage {pid} entities:")
        for eid, name in ents.items():
            print(f"    id={eid:<12} {name}")

    # --- β score: Entity ID overlap ---
    ids_a = set(passages[7067032].keys())
    ids_b = set(passages[2495755].keys())

    shared_ids = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    beta_score = len(shared_ids) / min(len(ids_a), len(ids_b)) if min(len(ids_a), len(ids_b)) > 0 else 0

    print(f"\n  --- β SCORE (Entity ID Overlap) ---")
    print(f"  Shared IDs ({len(shared_ids)}):")
    for eid in shared_ids:
        name_a = passages[7067032].get(eid, '?')
        name_b = passages[2495755].get(eid, '?')
        print(f"    id={eid}: '{name_a}' == '{name_b}'")
    print(f"  Only in A: {only_a}")
    print(f"  Only in B: {only_b}")
    print(f"  β = {len(shared_ids)} / min({len(ids_a)}, {len(ids_b)}) = {beta_score:.3f}")

    # --- γ score: Embedding similarity for NON-overlapping entities ---
    print(f"\n  --- γ SCORE (Embedding Similarity) ---")

    # Get names for non-overlapping entities
    ents_only_a = {eid: passages[7067032][eid] for eid in only_a}
    ents_only_b = {eid: passages[2495755][eid] for eid in only_b}

    print(f"  Non-overlapping from A: {list(ents_only_a.values())}")
    print(f"  Non-overlapping from B: {list(ents_only_b.values())}")

    # Compute pairwise cosine similarities
    similarities = []
    print(f"\n  Pairwise cosine similarities:")
    for eid_a, name_a in ents_only_a.items():
        for eid_b, name_b in ents_only_b.items():
            try:
                vec_a = emb.load_entity_embedding(name_a)
                vec_b = emb.load_entity_embedding(name_b)

                if vec_a is not None and vec_b is not None:
                    dot = float(np.dot(vec_a, vec_b))
                    norm_a = np.linalg.norm(vec_a)
                    norm_b = np.linalg.norm(vec_b)
                    cosine = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                    similarities.append(cosine)
                    print(f"    {name_a:<30} vs {name_b:<30} cosine={cosine:.3f}")
                else:
                    missing = name_a if vec_a is None else name_b
                    print(f"    {name_a:<30} vs {name_b:<30} SKIP (no embedding for '{missing}')")
            except Exception as e:
                print(f"    {name_a:<30} vs {name_b:<30} Error: {e}")

    if similarities:
        gamma_score = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        print(f"\n  γ (avg cosine) = {gamma_score:.3f}")
        print(f"  γ (max cosine) = {max_sim:.3f}")
    else:
        print(f"\n  γ = 0.0 (no embeddings found for non-overlapping entities)")

    # --- Combined score ---
    print(f"\n  --- COMBINED SCORE ---")
    beta_weight = 0.5
    gamma_weight = 0.5
    gamma_val = gamma_score if similarities else 0.0
    combined = beta_weight * beta_score + gamma_weight * gamma_val
    print(f"  combined = {beta_weight} × β({beta_score:.3f}) + {gamma_weight} × γ({gamma_val:.3f}) = {combined:.3f}")


def show_all_passage_entities(n=5):
    """Show entities for first n passages from BM25 top-1000 to understand coverage."""
    from mmead import get_links

    print("\n" + "=" * 70)
    print(f"PART 3: MMEAD entity coverage for first {n} BM25 passages")
    print("=" * 70)

    links = get_links('v1', 'passage', linker='rel')

    # Load passage IDs from BM25 file
    kgpr_path = "entity_linking_results/passage_test_with_id_bm25rank1000.jsonl"
    pids = []
    with open(kgpr_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line.strip())
            pids.append(int(item['id']))

    total_ents = 0
    passages_with_ents = 0

    for pid in pids:
        try:
            result = links.load_links_from_docid(pid)
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            ents = data.get('passage', [])
            unique_ids = set(e['entity_id'] for e in ents)
            total_ents += len(unique_ids)
            if unique_ids:
                passages_with_ents += 1

            print(f"\n  Passage {pid}: {len(unique_ids)} unique entities")
            for e in ents:
                eid = e['entity_id']
                name = e['entity']
                # Only print unique
                if eid in unique_ids:
                    unique_ids.discard(eid)
                    print(f"    id={eid:<12} {name}")

        except Exception as e:
            print(f"\n  Passage {pid}: Error - {e}")

    print(f"\n  Summary: {passages_with_ents}/{n} passages have entities")
    print(f"  Average entities per passage: {total_ents/n:.1f}")


if __name__ == "__main__":
    test_embedding_lookup()
    demo_scoring()
    show_all_passage_entities(n=10)

    print("\n" + "=" * 70)
    print("DONE — If embeddings work, next step is building mmead_scorer.py")
    print("=" * 70)