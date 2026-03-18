import argparse
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta
from scipy.optimize import lsq_linear
from kg_scorer_unified_patched import create_scorer


def main():
    parser = argparse.ArgumentParser(description="Learn alpha and gamma on DL19 (beta fixed to 0)")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed_k", type=int, default=10)
    parser.add_argument("--lk", type=int, default=128)
    parser.add_argument("--kg_neighbor_k", type=int, default=16)
    parser.add_argument("--passage_el", type=str, default=None)
    parser.add_argument("--passage_el_db", type=str, default=None)
    parser.add_argument("--freebase_dir", type=str, default=None)
    parser.add_argument("--kg_mode", type=str, default="minmax")
    parser.add_argument("--sample_neg_per_query", type=int, default=100)
    parser.add_argument("--output", type=str, default="learned_ag_dl19.txt")
    parser.add_argument("--query_el", type=str, default=None, help="Path to query EL JSONL (activates query-conditioned entity filtering)")

    # --- Option B: per-feature lower bound floor for gamma ---
    parser.add_argument(
        "--gamma_floor", type=float, default=0.05,
        help="Minimum allowed gamma value in lsq_linear bounds. "
             "Prevents KG signal collapsing to 0 when Freebase coverage is sparse. "
             "Justified by Experiment 1 recall diagnostics showing KG neighbors are valid."
    )

    # --- Option C: only keep samples where KG signal is actually present ---
    parser.add_argument(
        "--kg_signal_filter", action="store_true",
        help="If set, only include training samples where kg_connectivity > 0. "
             "Prevents the sparse-KG majority from drowning out valid KG signal in regression."
    )
    parser.add_argument(
        "--kg_signal_filter_neg_only", action="store_true",
        help="Softer version of --kg_signal_filter: filter KG signal only for negatives. "
             "Always keeps all positives regardless of KG signal."
    )

    args = parser.parse_args()

    if not args.passage_el and not args.passage_el_db:
        raise ValueError("Provide either --passage_el or --passage_el_db")

    if not pt.started():
        pt.java.init()

    eval_dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")
    topics = eval_dataset.get_topics()
    qrels = eval_dataset.get_qrels()

    qrels_map = (
        qrels[qrels["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(set)
        .to_dict()
    )

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", num_results=args.budget
    )

    laff_graph = (
        pta.Artifact.from_hf("macavaney/msmarco-passage.corpusgraph.bm25.128.laff")
        .to_limit_k(args.lk)
    )

    scorer = create_scorer(
        alpha=0.6,
        beta=0.0,
        gamma=0.4,
        kg_score_mode=args.kg_mode,
        freebase_dir=args.freebase_dir,
        passage_el_path=args.passage_el,
        passage_el_db=args.passage_el_db,
        query_el_path=args.query_el,
        debug=False,
    )

    bm25_res = bm25(topics)

    X = []
    y = []

    # --- Diagnostic counters ---
    total_pairs_seen = 0
    pairs_with_kg_signal = 0       # kg_connectivity > 0
    pairs_kg_pos = 0               # kg > 0 AND relevant
    pairs_kg_neg = 0               # kg > 0 AND not relevant
    pairs_filtered_out = 0         # dropped by Option C filter

    for qid, df in bm25_res.groupby("qid"):
        qid = str(qid)
        rel_set = qrels_map.get(qid, set())
        seed_docs = df.sort_values("score", ascending=False)["docno"].tolist()[:args.seed_k]

        seen = set()
        neg_count = 0

        for seed in seed_docs:
            neighbors, weights = laff_graph.neighbours(seed, weights=True)
            rescored = scorer.rescore_neighbors(
                docno=str(seed),
                neighbor_docnos=[str(n) for n in neighbors],
                laff_weights=weights,
                qid=qid,
            )

            for neighbor_docno, _, comp in rescored[:args.kg_neighbor_k]:
                if neighbor_docno in seen:
                    continue
                seen.add(neighbor_docno)

                total_pairs_seen += 1
                label = 1.0 if neighbor_docno in rel_set else 0.0
                has_kg_signal = comp.kg_connectivity > 0.0

                if has_kg_signal:
                    pairs_with_kg_signal += 1
                    if label == 1.0:
                        pairs_kg_pos += 1
                    else:
                        pairs_kg_neg += 1

                feat = [comp.laff_norm, comp.kg_connectivity]

                # --- Option C filtering logic ---
                if args.kg_signal_filter:
                    # Hard filter: only keep pairs where KG signal is present
                    if not has_kg_signal:
                        pairs_filtered_out += 1
                        continue
                elif args.kg_signal_filter_neg_only:
                    # Soft filter: always keep positives, filter negatives without KG signal
                    if label == 0.0 and not has_kg_signal:
                        pairs_filtered_out += 1
                        continue

                if label == 1.0:
                    X.append(feat)
                    y.append(label)
                elif neg_count < args.sample_neg_per_query:
                    X.append(feat)
                    y.append(label)
                    neg_count += 1

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        raise RuntimeError("No samples collected. Try relaxing --kg_signal_filter or check your EL coverage.")

    # --- Diagnostic printout ---
    print()
    print("=" * 60)
    print("SAMPLE DIAGNOSTICS")
    print("=" * 60)
    print(f"  Total neighbor pairs evaluated : {total_pairs_seen:,}")
    print(f"  Pairs with kg_connectivity > 0 : {pairs_with_kg_signal:,} "
          f"({100*pairs_with_kg_signal/max(total_pairs_seen,1):.1f}%)")
    print(f"    of which relevant (label=1)  : {pairs_kg_pos:,}")
    print(f"    of which not relevant (label=0): {pairs_kg_neg:,}")
    print(f"  Pairs filtered out (Option C)  : {pairs_filtered_out:,}")
    print(f"  Samples kept for regression    : {len(y):,}  "
          f"(pos={int(y.sum())}, neg={int((y==0).sum())})")
    print()
    print(f"  mean laff      = {X[:, 0].mean():.4f}   nonzero = {(X[:, 0] > 0).mean():.2%}")
    print(f"  mean kg_conn   = {X[:, 1].mean():.4f}   nonzero = {(X[:, 1] > 0).mean():.2%}")

    pos = X[y == 1]
    neg = X[y == 0]
    if len(pos) > 0 and len(neg) > 0:
        print()
        print(f"  POS mean [laff, kg] = {pos.mean(axis=0)}")
        print(f"  NEG mean [laff, kg] = {neg.mean(axis=0)}")
        kg_sep = pos[:, 1].mean() - neg[:, 1].mean()
        print(f"  KG separation (pos_mean - neg_mean) = {kg_sep:+.4f}")
        if kg_sep <= 0:
            print("  WARNING: KG signal not separating pos/neg — "
                  "consider checking entity linking coverage or Freebase connectivity.")

    # --- Option B: per-feature bounds, gamma has a floor ---
    # alpha: [0.0, 1.0]
    # gamma: [gamma_floor, 1.0]
    lb = [0.0, args.gamma_floor]
    ub = [1.0, 1.0]

    print()
    print("=" * 60)
    print("REGRESSION")
    print("=" * 60)
    print(f"  Bounds: alpha=[0.0, 1.0], gamma=[{args.gamma_floor}, 1.0]  (Option B)")
    if args.kg_signal_filter:
        print("  Sample filter: KG signal required for ALL samples  (Option C hard)")
    elif args.kg_signal_filter_neg_only:
        print("  Sample filter: KG signal required for NEGATIVE samples only  (Option C soft)")
    else:
        print("  Sample filter: none (all pairs kept)")

    result = lsq_linear(X, y, bounds=(lb, ub), lsq_solver="exact")
    a_raw, g_raw = result.x

    # Normalise so they sum to 1
    total = a_raw + g_raw
    if total > 0:
        a, g = a_raw / total, g_raw / total
    else:
        a, g = 1.0, 0.0

    print()
    print(f"  Raw from solver : alpha={a_raw:.6f}, gamma={g_raw:.6f}")
    print(f"  After normalise : alpha={a:.6f}, gamma={g:.6f}")
    print()
    print(f"alpha={a:.6f}")
    print("beta =0.000000")
    print(f"gamma={g:.6f}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"alpha={a:.6f}\n")
        f.write("beta=0.000000\n")
        f.write(f"gamma={g:.6f}\n")

    print()
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()