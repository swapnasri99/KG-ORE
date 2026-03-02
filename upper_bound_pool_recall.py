import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import ir_datasets
import sqlite3
import json
import ast
import re


def load_topics(ds_name: str) -> pd.DataFrame:
    ds = ir_datasets.load(ds_name)
    rows = [{"qid": str(q.query_id), "query": q.text} for q in ds.queries_iter()]
    return pd.DataFrame(rows)


def load_qrels_rel2(ds_name: str, rel_level: int = 2):
    ds = ir_datasets.load(ds_name)
    rel = {}
    for qr in ds.qrels_iter():
        if int(qr.relevance) >= rel_level:
            rel.setdefault(str(qr.query_id), set()).add(str(qr.doc_id))
    return rel


def pool_recall(bm25_run: pd.DataFrame, qrels: dict, seed_k: int, laff_graph, neigh_k: int):
    bm25_run = bm25_run[bm25_run["rank"] < seed_k]

    recalls = []
    pool_sizes = []

    for qid, group in bm25_run.groupby("qid"):
        relset = qrels.get(str(qid), set())
        if not relset:
            continue

        seed_docnos = [str(x) for x in group["docno"].tolist()]
        pool = set(seed_docnos)

        for d in seed_docnos:
            neigh = laff_graph.neighbors(d, k=neigh_k)
            # some implementations return (docno, weight)
            if neigh and isinstance(neigh[0], (tuple, list)):
                neigh = [str(x[0]) for x in neigh]
            else:
                neigh = [str(x) for x in neigh]
            pool.update(neigh)

        hit = len(pool.intersection(relset))
        recalls.append(hit / len(relset))
        pool_sizes.append(len(pool))

    return float(np.mean(recalls)), float(np.mean(pool_sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=int, default=19, choices=[19, 20])
    ap.add_argument("--index", type=str, required=True, help="Path to PyTerrier index, e.g. ./indices/msmarco_passage")
    ap.add_argument("--seed_k", type=int, default=50, help="BM25 seed depth (top-k docs)")
    ap.add_argument("--ks", type=str, default="16,32,64,128", help="Neighbor k values, comma-separated")
    args = ap.parse_args()

    pt.java.init()

    if args.dl == 19:
        ds_name = "msmarco-passage/trec-dl-2019/judged"
    else:
        ds_name = "msmarco-passage/trec-dl-2020/judged"

    topics = load_topics(ds_name)
    qrels = load_qrels_rel2(ds_name, rel_level=2)

    index = pt.IndexFactory.of(args.index)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    # BM25 run once (no algorithm)
    bm25_run = bm25.transform(topics)

    # Load LAFF graph once (full 128), we control neighbor_k at query time
    laff_graph = pta.Artifact.from_hf("macavaney/msmarco-passage.corpusgraph.bm25.128.laff")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    rows = []

    for k in ks:
        rec, avg_pool = pool_recall(
            bm25_run=bm25_run,
            qrels=qrels,
            seed_k=args.seed_k,
            laff_graph=laff_graph,
            neigh_k=k
        )
        rows.append({"neighbor_k": k, "pool_recall_rel2": rec, "avg_pool_size": avg_pool})
        print(f"k={k:3d}  pool_recall(rel>=2)={rec:.4f}  avg_pool_size={avg_pool:.1f}")

    df = pd.DataFrame(rows).sort_values("neighbor_k")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()