"""
Experiment 1: Baseline Upper Bound
Pool = BM25 top-100 + raw LAFF top-16 neighbors per seed (no KG rescoring).
Oracle picks the best 16 from the entire pool by qrel grade.
Reports: pool recall and upper-bound recall@16.
"""
import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


def dcg(grades, k):
    """Compute DCG@k from a list of grades."""
    grades = grades[:k]
    return sum(g / np.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(pool_grades, all_query_grades, k):
    """
    Compute oracle nDCG@k: best possible nDCG if we pick the top-k
    highest-graded docs from pool.
    """
    # Oracle ranking: sort pool docs by grade descending
    oracle = sorted(pool_grades, reverse=True)
    # Ideal ranking: sort ALL judged docs by grade descending
    ideal = sorted(all_query_grades, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(oracle, k) / ideal_dcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=int, default=19)
    ap.add_argument("--bm25_depth", type=int, default=100)
    ap.add_argument("--lk", type=int, default=128)
    ap.add_argument("--neighbor_k", type=int, default=16)
    ap.add_argument("--topk", type=int, default=16)
    args = ap.parse_args()

    if not pt.started():
        pt.java.init()

    ds = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics_df = ds.get_topics()
    qrels_df = ds.get_qrels().copy()

    # Relevant = label >= 2
    qrels_rel_map = (
        qrels_df[qrels_df["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )
    # Full grade map for nDCG
    qrels_grade_map = (
        qrels_df.groupby("qid")
        .apply(lambda x: dict(zip(x["docno"].astype(str), x["label"])))
        .to_dict()
    )

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25",
        num_results=args.bm25_depth,
    )
    laff_graph = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    bm25_res = bm25.transform(topics_df)

    rows = []
    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        bm25_docs = group.sort_values("rank")["docno"].astype(str).tolist()
        relevant_docs = qrels_rel_map.get(qid, set())
        grade_map = qrels_grade_map.get(qid, {})

        # Build pool: BM25 top-100 + raw LAFF top-16 per seed
        seen = set()
        pool = []
        for d in bm25_docs:
            if d not in seen:
                pool.append(d)
                seen.add(d)

        for d in bm25_docs:
            try:
                neighbors, _ = laff_graph.neighbours(d, weights=True)
                for n in neighbors[:args.neighbor_k]:
                    n = str(n)
                    if n not in seen:
                        pool.append(n)
                        seen.add(n)
            except Exception:
                pass

        num_rel_total = len(relevant_docs)
        num_rel_bm25 = len(set(bm25_docs) & relevant_docs)
        num_rel_pool = len(set(pool) & relevant_docs)

        # Oracle top-k: pick best topk docs from pool by grade
        pool_grades_sorted = sorted(
            [grade_map.get(d, 0) for d in pool], reverse=True
        )
        oracle_top = pool_grades_sorted[:args.topk]
        num_rel_oracle = sum(1 for g in oracle_top if g >= 2)

        recall_oracle = num_rel_oracle / num_rel_total if num_rel_total > 0 else 0.0

        # Oracle nDCG@topk
        all_grades = list(grade_map.values())
        oracle_ndcg = ndcg_at_k(
            [grade_map.get(d, 0) for d in pool],
            all_grades, args.topk
        )

        rows.append({
            "qid": qid,
            "num_rel_total": num_rel_total,
            "bm25_rel": num_rel_bm25,
            "pool_rel": num_rel_pool,
            "oracle_top16_rel": num_rel_oracle,
            "oracle_recall@16": recall_oracle,
            "oracle_nDCG@16": oracle_ndcg,
            "pool_size": len(pool),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: BASELINE UPPER BOUND")
    print(f"  Pool = BM25 top-{args.bm25_depth} + raw LAFF top-{args.neighbor_k}")
    print("=" * 60)
    print(f"  Queries:                        {len(df)}")
    print(f"  Avg pool size:                  {df['pool_size'].mean():.1f}")
    print(f"  Avg relevant in BM25-{args.bm25_depth}:       {df['bm25_rel'].mean():.3f}")
    print(f"  Avg relevant in full pool:      {df['pool_rel'].mean():.3f}")
    print(f"  Avg relevant in oracle top-{args.topk}:  {df['oracle_top16_rel'].mean():.3f}")
    print(f"  Avg oracle Recall@{args.topk}:          {df['oracle_recall@16'].mean():.4f}")
    print(f"  Avg oracle nDCG@{args.topk}:            {df['oracle_nDCG@16'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()