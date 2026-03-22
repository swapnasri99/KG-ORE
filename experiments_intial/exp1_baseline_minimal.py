import argparse
import pyterrier as pt
import pyterrier_alpha as pta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=int, default=19)
    ap.add_argument("--bm25_depth", type=int, default=100)
    ap.add_argument("--lk", type=int, default=128)
    ap.add_argument("--neighbor_k", type=int, default=16)
    args = ap.parse_args()

    if not pt.started():
        pt.java.init()

    ds = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics = ds.get_topics()
    qrels = ds.get_qrels().copy()

    qrels_rel = (
        qrels[qrels["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage",
        "terrier_stemmed",
        wmodel="BM25",
        num_results=args.bm25_depth,
    )

    laff = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    res = bm25.transform(topics)

    print("qid      | bm25_rel100 | pool_rel | new_rel")
    print("-" * 42)

    for qid, group in res.groupby("qid"):
        qid = str(qid)
        rel = qrels_rel.get(qid, set())

        bm25_docs = group.sort_values("rank")["docno"].astype(str).tolist()

        seen = set()
        pool = []

        for d in bm25_docs:
            if d not in seen:
                pool.append(d)
                seen.add(d)

        for d in bm25_docs:
            try:
                neighbors, weights = laff.neighbours(d, weights=True)
                for n in neighbors[:args.neighbor_k]:
                    n = str(n)
                    if n not in seen:
                        pool.append(n)
                        seen.add(n)
            except Exception:
                pass

        bm25_rel = len(set(bm25_docs) & rel)
        pool_rel = len(set(pool) & rel)
        new_rel = pool_rel - bm25_rel

        print(f"{qid:8s} | {bm25_rel:11d} | {pool_rel:8d} | {new_rel:7d}")


if __name__ == "__main__":
    main()