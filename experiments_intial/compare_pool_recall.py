"""
Upper-Bound Pool Recall Comparison: Baseline ORE (LAFF) vs Union
================================================================
Computes R(rel=2)@budget style recall for both pools.
Reports per-query and average.

Usage:
  python compare_pool_recall.py --dl 19 --passage_el_db passage_entities.db --freebase_dir freebase

  Or with JSONL:
  python compare_pool_recall.py --dl 19 --passage_el entity_linking_results/passage_test_with_id_bm25rank1000.jsonl --freebase_dir freebase
"""
import argparse
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta

from kg_scorer_unified_patched import create_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=int, default=19)
    ap.add_argument("--bm25_depth", type=int, default=100)
    ap.add_argument("--lk", type=int, default=128)
    ap.add_argument("--neighbor_k", type=int, default=16)
    ap.add_argument("--passage_el", type=str, default=None)
    ap.add_argument("--passage_el_db", type=str, default=None)
    ap.add_argument("--query_el", type=str, default=None)
    ap.add_argument("--freebase_dir", type=str, default=None)
    args = ap.parse_args()

    if not args.passage_el and not args.passage_el_db:
        raise ValueError("Provide either --passage_el or --passage_el_db")
    if not args.freebase_dir:
        raise ValueError("Provide --freebase_dir")

    if not pt.started():
        pt.java.init()

    ds = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics_df = ds.get_topics()
    qrels_df = ds.get_qrels().copy()

    # R(rel=2): relevant = label >= 2
    qrels_rel_map = (
        qrels_df[qrels_df["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25",
        num_results=args.bm25_depth,
    )
    laff_graph = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    # KG scorer for union: gamma=1.0, alpha=0, beta=0, minmax
    kg_scorer = create_scorer(
        alpha=0.0, beta=0.3, gamma=0.7,
        kg_score_mode="minmax",
        freebase_dir=args.freebase_dir,
        passage_el_path=args.passage_el,
        query_el_path=args.query_el,
        passage_el_db=args.passage_el_db,
        debug=False,
    )

    bm25_res = bm25.transform(topics_df)

    print(f"\n{'qid':>10s} | {'total_rel':>9s} | {'bm25_rel':>8s} | {'baseline':>8s} | {'union':>8s} | {'base_R':>7s} | {'union_R':>7s} | {'kg_only_new':>11s}")
    print("-" * 95)

    baseline_recalls = []
    union_recalls = []
    kg_only_relevant_counts = []

    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        bm25_docs = group.sort_values("rank")["docno"].astype(str).tolist()
        relevant_docs = qrels_rel_map.get(qid, set())
        num_rel_total = len(relevant_docs)

        if num_rel_total == 0:
            continue

        # === Baseline pool: BM25 + raw LAFF top-k ===
        baseline_seen = set()
        for d in bm25_docs:
            baseline_seen.add(d)

        for d in bm25_docs:
            try:
                neighbors, _ = laff_graph.neighbours(d, weights=True)
                for n in neighbors[:args.neighbor_k]:
                    baseline_seen.add(str(n))
            except Exception:
                pass

        # === Union pool: BM25 + union(LAFF top-k, KG top-k) ===
        union_seen = set()
        for d in bm25_docs:
            union_seen.add(d)

        kg_only_neighbors = set()  # neighbors in union but NOT in baseline

        for d in bm25_docs:
            try:
                neighbors, weights = laff_graph.neighbours(d, weights=True)
                neighbor_docnos = [str(n) for n in neighbors]

                # LAFF top-k
                laff_top = set(neighbor_docnos[:args.neighbor_k])

                # KG top-k
                rescored = kg_scorer.rescore_neighbors(
                    d, neighbor_docnos, weights, qid=qid
                )
                kg_top = set(n for n, _, _ in rescored[:args.neighbor_k])

                # Union
                union_neighbors = laff_top | kg_top
                for n in union_neighbors:
                    union_seen.add(n)

                # Track KG-only additions (in KG top but not in LAFF top)
                for n in (kg_top - laff_top):
                    kg_only_neighbors.add(n)

            except Exception:
                pass

        # Compute recall
        baseline_rel = len(baseline_seen & relevant_docs)
        union_rel = len(union_seen & relevant_docs)
        bm25_rel = len(set(bm25_docs) & relevant_docs)

        baseline_recall = baseline_rel / num_rel_total
        union_recall = union_rel / num_rel_total

        # How many KG-only neighbors are relevant?
        kg_only_relevant = len(kg_only_neighbors & relevant_docs)

        baseline_recalls.append(baseline_recall)
        union_recalls.append(union_recall)
        kg_only_relevant_counts.append(kg_only_relevant)

        print(f"{qid:>10s} | {num_rel_total:>9d} | {bm25_rel:>8d} | {baseline_rel:>8d} | {union_rel:>8d} | {baseline_recall:>7.4f} | {union_recall:>7.4f} | {kg_only_relevant:>11d}")

    print("-" * 95)
    print(f"\n{'AVERAGES':>10s} | {'':>9s} | {'':>8s} | {'':>8s} | {'':>8s} | {np.mean(baseline_recalls):>7.4f} | {np.mean(union_recalls):>7.4f} | {np.mean(kg_only_relevant_counts):>11.1f}")

    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"  Queries evaluated:              {len(baseline_recalls)}")
    print(f"  Avg Baseline Pool R(rel=2):     {np.mean(baseline_recalls):.4f}")
    print(f"  Avg Union Pool R(rel=2):        {np.mean(union_recalls):.4f}")
    print(f"  Avg gain (union - baseline):    {np.mean(union_recalls) - np.mean(baseline_recalls):.4f}")
    print(f"  Queries where union > baseline: {sum(1 for b, u in zip(baseline_recalls, union_recalls) if u > b)}")
    print(f"  Queries where union = baseline: {sum(1 for b, u in zip(baseline_recalls, union_recalls) if u == b)}")
    print(f"  Queries where union < baseline: {sum(1 for b, u in zip(baseline_recalls, union_recalls) if u < b)}")
    print(f"  Avg KG-only relevant docs:      {np.mean(kg_only_relevant_counts):.1f}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()