"""
Experiment 2: KG+LAFF Upper Bound
Pool = BM25 top-100 + top-16 KG+LAFF rescored neighbors per seed.
User provides --alpha, --beta, --gamma. Uses minmax KG mode.
Oracle picks the best 16 from the entire pool by qrel grade.
Reports: pool recall and upper-bound recall@16.
"""
import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from kg_scorer_unified_patched import create_scorer


def dcg(grades, k):
    grades = grades[:k]
    return sum(g / np.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(pool_grades, all_query_grades, k):
    oracle = sorted(pool_grades, reverse=True)
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
    ap.add_argument("--kg_neighbor_k", type=int, default=16)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--passage_el", type=str, default=None)
    ap.add_argument("--passage_el_db", type=str, default=None)
    ap.add_argument("--query_el", type=str, default=None)
    ap.add_argument("--freebase_dir", type=str, default=None)
    args = ap.parse_args()

    if not args.passage_el and not args.passage_el_db:
        raise ValueError("Provide either --passage_el or --passage_el_db")
    if args.gamma > 0 and not args.freebase_dir:
        raise ValueError("--gamma > 0 requires --freebase_dir")

    if not pt.started():
        pt.java.init()

    ds = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics_df = ds.get_topics()
    qrels_df = ds.get_qrels().copy()

    qrels_rel_map = (
        qrels_df[qrels_df["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )
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

    kg_scorer = create_scorer(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        kg_score_mode="minmax",
        freebase_dir=args.freebase_dir,
        passage_el_path=args.passage_el,
        query_el_path=args.query_el,
        passage_el_db=args.passage_el_db,
        debug=False,
    )

    bm25_res = bm25.transform(topics_df)

    rows = []
    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        bm25_docs = group.sort_values("rank")["docno"].astype(str).tolist()
        relevant_docs = qrels_rel_map.get(qid, set())
        grade_map = qrels_grade_map.get(qid, {})

        seen = set()
        pool = []
        for d in bm25_docs:
            if d not in seen:
                pool.append(d)
                seen.add(d)

        for d in bm25_docs:
            try:
                neighbors, weights = laff_graph.neighbours(d, weights=True)
                rescored = kg_scorer.rescore_neighbors(
                    d, [str(n) for n in neighbors], weights, qid=qid
                )
                for neighbor_docno, _, _ in rescored[:args.kg_neighbor_k]:
                    neighbor_docno = str(neighbor_docno)
                    if neighbor_docno not in seen:
                        pool.append(neighbor_docno)
                        seen.add(neighbor_docno)
            except Exception:
                pass

        num_rel_total = len(relevant_docs)
        num_rel_bm25 = len(set(bm25_docs) & relevant_docs)
        num_rel_pool = len(set(pool) & relevant_docs)

        pool_grades_sorted = sorted(
            [grade_map.get(d, 0) for d in pool], reverse=True
        )
        oracle_top = pool_grades_sorted[:args.topk]
        num_rel_oracle = sum(1 for g in oracle_top if g >= 2)
        recall_oracle = num_rel_oracle / num_rel_total if num_rel_total > 0 else 0.0

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
    print("EXPERIMENT 2: KG+LAFF UPPER BOUND")
    print(f"  Pool = BM25 top-{args.bm25_depth} + KG+LAFF rescored top-{args.kg_neighbor_k}")
    print(f"  Weights: α={args.alpha}, β={args.beta}, γ={args.gamma} (minmax)")
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