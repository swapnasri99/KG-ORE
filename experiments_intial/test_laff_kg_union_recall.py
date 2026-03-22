import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from kg_scorer_unified_patched import create_scorer


def main():
    """
    Why am I doing this?
    --------------------
    I want to test whether KG is adding useful relevant documents
    beyond plain LAFF within the same LAFF neighbor pool.

    This script compares, for each query:
      1) LAFF-only top-k neighbors
      2) KG-reranked top-k neighbors from the SAME candidate pool
      3) UNION of LAFF and KG relevant docs

    This answers:
      - Are LAFF and KG mostly retrieving the same relevant docs?
      - Does KG add extra relevant docs that LAFF misses?
      - Does the union improve recall?

    Important:
      KG here does NOT fetch totally new documents from outside LAFF's
      top-lk candidate pool. It only changes ranking within that pool.
      So any gain comes from promoting relevant docs already present in
      the LAFF candidate set.
    """

    ap = argparse.ArgumentParser(description="Test LAFF vs KG vs UNION recall")
    ap.add_argument("--dl", type=int, default=19, help="TREC DL year: 19 or 20")
    ap.add_argument("--seed_k", type=int, default=5, help="Top BM25 docs used as seeds")
    ap.add_argument("--neighbor_k", type=int, default=16, help="Top-k neighbors kept per method")
    ap.add_argument("--lk", type=int, default=128, help="LAFF candidate pool size before reranking")

    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--kg_mode", type=str, default="minmax")

    ap.add_argument("--passage_el", type=str, default=None)
    ap.add_argument("--passage_el_db", type=str, default=None)
    ap.add_argument("--query_el", type=str, default=None)
    ap.add_argument("--freebase_dir", type=str, default=None)

    ap.add_argument("--use_qid", action="store_true", help="Pass qid into KG scorer for query conditioning")
    ap.add_argument("--out_csv", type=str, default="test_laff_kg_union_recall.csv")

    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    eval_dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics = eval_dataset.get_topics()
    qrels = eval_dataset.get_qrels()

    # rel >= 2
    qrels_map = (
        qrels[qrels["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(set)
        .to_dict()
    )

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage",
        "terrier_stemmed",
        wmodel="BM25",
        num_results=max(100, args.seed_k)
    )

    bm25_res = bm25(topics)

    laff_graph = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    kg_scorer = create_scorer(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        kg_score_mode=args.kg_mode,
        freebase_dir=args.freebase_dir,
        passage_el_path=args.passage_el,
        query_el_path=args.query_el,
        passage_el_db=args.passage_el_db,
        debug=False,
    )

    rows = []

    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        relset = qrels_map.get(qid, set())

        if len(relset) == 0:
            continue

        seeds = (
            group.sort_values("score", ascending=False)["docno"]
            .astype(str)
            .tolist()[:args.seed_k]
        )

        laff_union_docs = set()
        kg_union_docs = set()

        for seed in seeds:
            neighbors, weights = laff_graph.neighbours(seed, weights=True)
            neighbor_docnos = [str(n) for n in neighbors]
            weights = np.array(list(weights), dtype=float)

            # LAFF-only ranking
            laff_pairs = sorted(
                zip(neighbor_docnos, weights.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            laff_topk = [docno for docno, _ in laff_pairs[:args.neighbor_k]]
            laff_union_docs.update(laff_topk)

            # KG reranking on same pool
            rescored = kg_scorer.rescore_neighbors(
                docno=seed,
                neighbor_docnos=neighbor_docnos,
                laff_weights=weights,
                qid=qid if args.use_qid else None,
            )
            kg_topk = [docno for docno, _, _ in rescored[:args.neighbor_k]]
            kg_union_docs.update(kg_topk)

        laff_rel = laff_union_docs & relset
        kg_rel = kg_union_docs & relset
        both_rel = laff_rel & kg_rel
        union_rel = laff_rel | kg_rel

        laff_only_rel = laff_rel - kg_rel
        kg_extra_rel = kg_rel - laff_rel

        total_rel = len(relset)

        rows.append({
            "qid": qid,
            "total_relevant_docs": total_rel,

            "laff_relevant_count": len(laff_rel),
            "kg_relevant_count": len(kg_rel),
            "both_relevant_count": len(both_rel),
            "union_relevant_count": len(union_rel),

            "laff_only_count": len(laff_only_rel),
            "kg_extra_count": len(kg_extra_rel),

            "laff_recall": len(laff_rel) / total_rel,
            "kg_recall": len(kg_rel) / total_rel,
            "union_recall": len(union_rel) / total_rel,

            "laff_relevant_docs": " ".join(sorted(laff_rel)),
            "kg_relevant_docs": " ".join(sorted(kg_rel)),
            "both_relevant_docs": " ".join(sorted(both_rel)),
            "union_relevant_docs": " ".join(sorted(union_rel)),
            "laff_only_docs": " ".join(sorted(laff_only_rel)),
            "kg_extra_docs": " ".join(sorted(kg_extra_rel)),
        })

    df = pd.DataFrame(rows).sort_values("qid")
    df.to_csv(args.out_csv, index=False)

    pd.set_option("display.max_colwidth", 200)

    table = df[[
        "qid",
        "total_relevant_docs",
        "laff_relevant_count",
        "kg_relevant_count",
        "both_relevant_count",
        "union_relevant_count",
        "laff_only_count",
        "kg_extra_count",
        "laff_recall",
        "kg_recall",
        "union_recall",
    ]].reset_index(drop=True)

    table.insert(0, "No", table.index + 1)

    print("=" * 130)
    print("TEST: LAFF vs KG vs UNION RECALL")
    print("=" * 130)
    print(table.to_string(index=False))

    print("\n" + "=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(f"Queries                     : {len(df)}")
    print(f"Mean LAFF recall            : {df['laff_recall'].mean():.4f}")
    print(f"Mean KG recall              : {df['kg_recall'].mean():.4f}")
    print(f"Mean UNION recall           : {df['union_recall'].mean():.4f}")
    print(f"Queries where UNION > LAFF  : {(df['union_recall'] > df['laff_recall']).sum()}")
    print(f"Queries where UNION > KG    : {(df['union_recall'] > df['kg_recall']).sum()}")
    print(f"Queries where KG > LAFF     : {(df['kg_recall'] > df['laff_recall']).sum()}")
    print(f"Queries where LAFF > KG     : {(df['kg_recall'] < df['laff_recall']).sum()}")
    print(f"Saved CSV                   : {args.out_csv}")

    # Best union improvement over LAFF
    df["union_gain_over_laff"] = df["union_recall"] - df["laff_recall"]
    gain_df = df[df["union_gain_over_laff"] > 0].sort_values("union_gain_over_laff", ascending=False)

    print("\n" + "=" * 130)
    print("BEST UNION GAIN CASE")
    print("=" * 130)
    if not gain_df.empty:
        row = gain_df.iloc[0]
        print(f"QID                    : {row['qid']}")
        print(f"LAFF recall            : {row['laff_recall']:.4f}")
        print(f"KG recall              : {row['kg_recall']:.4f}")
        print(f"UNION recall           : {row['union_recall']:.4f}")
        print(f"LAFF-only relevant     : {row['laff_only_count']}")
        print(f"KG-extra relevant      : {row['kg_extra_count']}")
        print(f"Both relevant          : {row['both_relevant_count']}")
    else:
        print("No query where UNION improved over LAFF.")

    # Most overlap case
    df["overlap_ratio"] = df["both_relevant_count"] / df[["laff_relevant_count", "kg_relevant_count"]].max(axis=1).replace(0, np.nan)
    overlap_df = df.sort_values("overlap_ratio", ascending=False)

    print("\n" + "=" * 130)
    print("MOST OVERLAPPING CASE")
    print("=" * 130)
    if not overlap_df.empty:
        row = overlap_df.iloc[0]
        print(f"QID                    : {row['qid']}")
        #print(f"LAFF relevant count    : {row['laff_relevant_count']}")
        #print(f"KG relevant count      : {row['kg_relevant_count']}")
        #print(f"Both relevant count    : {row['both_relevant_count']}")
        print(f"Overlap ratio          : {row['overlap_ratio']:.4f}" if pd.notna(row["overlap_ratio"]) else "Overlap ratio          : NaN")


if __name__ == "__main__":
    main()