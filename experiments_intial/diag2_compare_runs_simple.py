#!/usr/bin/env python3
import argparse
import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyterrier as pt


def parse_args():
    ap = argparse.ArgumentParser(
        description="Simple comparison of Run A vs Run B final ranks for chosen queries/docs"
    )
    ap.add_argument("--dl", type=int, default=19, help="TREC-DL year: 19 or 20")
    ap.add_argument("--run_a", required=True, help="Run A directory or run file")
    ap.add_argument("--run_b", required=True, help="Run B directory or run file")
    ap.add_argument("--qid", action="append", default=[], help="Query id to inspect. Can be repeated.")
    ap.add_argument("--docs", nargs="*", default=None,
                    help="Docnos of interest. If omitted, all relevant docs for the query are shown.")
    ap.add_argument("--rels", type=int, default=2,
                    help="Minimum qrel label to treat as relevant (default: 2)")
    ap.add_argument("--topk", type=int, default=50,
                    help="Only report docs within this rank as FOUND; otherwise MISS")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional path to save comparison table as CSV")
    return ap.parse_args()


def resolve_run_file(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    candidates = []
    patterns = ["*.res.gz", "*.res", "*.txt", "*.tsv", "*.gz"]
    for pattern in patterns:
        candidates.extend(p.glob(pattern))
        candidates.extend(p.rglob(pattern))

    # prefer files that look like run outputs
    preferred = [c for c in candidates if any(x in c.name.lower() for x in ["run", "res", "results", "output"])]
    chosen_pool = preferred if preferred else candidates
    if not chosen_pool:
        raise FileNotFoundError(f"No candidate run files found under: {p}")

    chosen = max(chosen_pool, key=lambda x: x.stat().st_mtime)
    return chosen


def try_read_run(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # TREC run: qid Q0 docno rank score tag
            if len(parts) >= 6 and parts[1] == "Q0":
                qid, _, docno, rank, score = parts[:5]
                rows.append({
                    "qid": str(qid),
                    "docno": str(docno),
                    "rank": int(float(rank)),
                    "score": float(score),
                })
                continue
            # simple TSV/space format fallback: qid docno rank score
            if len(parts) >= 4:
                try:
                    qid, docno = parts[0], parts[1]
                    rank = int(float(parts[2]))
                    score = float(parts[3])
                    rows.append({
                        "qid": str(qid),
                        "docno": str(docno),
                        "rank": rank,
                        "score": score,
                    })
                    continue
                except Exception:
                    pass
    if not rows:
        raise ValueError(f"Could not parse run file: {path}")
    df = pd.DataFrame(rows)
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    return df


def build_rank_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    return {(str(r.qid), str(r.docno)): int(r.rank) for r in df.itertuples(index=False)}


def load_topics_and_qrels(dl: int, rels: int):
    if not pt.started():
        pt.java.init()
    eval_dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{dl}/judged")
    topics = eval_dataset.get_topics()[["qid", "query"]].copy()
    topics["qid"] = topics["qid"].astype(str)
    topic_map = dict(zip(topics["qid"], topics["query"]))

    qrels = eval_dataset.get_qrels().copy()
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels = qrels[qrels["label"] >= rels]

    qrels_map: Dict[str, Dict[str, int]] = {}
    for row in qrels.itertuples(index=False):
        qrels_map.setdefault(str(row.qid), {})[str(row.docno)] = int(row.label)
    return topic_map, qrels_map


def label_from_rank(rank: Optional[int], topk: int) -> str:
    if rank is None:
        return "MISS"
    return f"FOUND@{rank}" if rank <= topk else f"BELOW@{rank}"


def main():
    args = parse_args()

    run_a_file = resolve_run_file(args.run_a)
    run_b_file = resolve_run_file(args.run_b)

    df_a = try_read_run(run_a_file)
    df_b = try_read_run(run_b_file)
    rank_a = build_rank_lookup(df_a)
    rank_b = build_rank_lookup(df_b)

    topic_map, qrels_map = load_topics_and_qrels(args.dl, args.rels)

    qids = [str(q) for q in args.qid]
    if not qids:
        raise ValueError("Please provide at least one --qid")

    out_rows = []

    print("=" * 100)
    print("EXPERIMENT 2: RUN A vs RUN B DOC RANK COMPARISON")
    print("=" * 100)
    print(f"Run A file : {run_a_file}")
    print(f"Run B file : {run_b_file}")
    print(f"Relevant label threshold : {args.rels}")
    print(f"Top-k threshold         : {args.topk}")

    for qid in qids:
        query = topic_map.get(qid, "")
        rel_docs_map = qrels_map.get(qid, {})
        rel_docs_sorted = sorted(rel_docs_map.keys(), key=lambda d: (-rel_docs_map[d], d))
        docs_of_interest = [str(d) for d in args.docs] if args.docs else rel_docs_sorted

        print("\n" + "-" * 100)
        print(f"QID   : {qid}")
        print(f"Query : {query}")
        print(f"All relevant docs (label>={args.rels}): {rel_docs_sorted}")
        print("-" * 100)

        table_rows = []
        for docno in docs_of_interest:
            qrel = rel_docs_map.get(docno, "-")
            ra = rank_a.get((qid, docno))
            rb = rank_b.get((qid, docno))

            if ra is None and rb is None:
                delta = "-"
            elif ra is None:
                delta = "NEW_IN_B"
            elif rb is None:
                delta = "LOST_IN_B"
            else:
                delta = ra - rb  # positive means better in B

            row = {
                "docno": docno,
                "qrel": qrel,
                "runA": label_from_rank(ra, args.topk),
                "runB": label_from_rank(rb, args.topk),
                "rankA": ra if ra is not None else "MISS",
                "rankB": rb if rb is not None else "MISS",
                "delta(A-B)": delta,
            }
            table_rows.append(row)
            out_rows.append({"qid": qid, "query": query, **row})

        print(pd.DataFrame(table_rows).to_string(index=False))

        improved = [r["docno"] for r in table_rows if isinstance(r["delta(A-B)"], int) and r["delta(A-B)"] > 0]
        worsened = [r["docno"] for r in table_rows if isinstance(r["delta(A-B)"], int) and r["delta(A-B)"] < 0]
        new_in_b = [r["docno"] for r in table_rows if r["delta(A-B)"] == "NEW_IN_B"]

        print("\nQuick summary")
        print(f"  Improved in Run B : {improved if improved else 'None'}")
        print(f"  Worsened in Run B : {worsened if worsened else 'None'}")
        print(f"  New in Run B      : {new_in_b if new_in_b else 'None'}")

    if args.out_csv:
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(args.out_csv, index=False)
        print("\nSaved CSV:", args.out_csv)


if __name__ == "__main__":
    main()
