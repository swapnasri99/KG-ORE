import os
import gzip
import argparse
from collections import defaultdict

import ir_datasets


def read_trec_run(run_path, topk=50):
    """
    Reads a TREC run file:
    qid Q0 docno rank score tag

    Returns:
        dict[qid] = set(docno) for top-k retrieved docs
    """
    result = defaultdict(set)
    open_fn = gzip.open if run_path.endswith(".gz") else open

    with open_fn(run_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            qid, _, docno, rank, _, _ = parts[:6]

            try:
                rank = int(rank)
            except ValueError:
                continue

            if rank < topk:
                result[str(qid)].add(docno)

    return result


def load_qrels(dataset_name, min_rel=2):
    """
    Load qrels from ir_datasets and keep only docs with relevance >= min_rel.
    Returns:
        dict[qid] = set(relevant_docnos)
    """
    dataset = ir_datasets.load(dataset_name)
    qrels = defaultdict(set)

    for qrel in dataset.qrels_iter():
        if qrel.relevance >= min_rel:
            qrels[str(qrel.query_id)].add(str(qrel.doc_id))

    return qrels


def load_c50_runs(folder):
    """
    Load only c50 run files from folder.
    Returns:
        dict[system_name] = dict[qid] = set(docnos)
    """
    runs = {}

    for fname in sorted(os.listdir(folder)):
        if not (fname.endswith(".res") or fname.endswith(".res.gz")):
            continue

        if ".c50." not in fname and "c50" not in fname:
            continue

        system_name = fname.replace(".res.gz", "").replace(".res", "")
        run_path = os.path.join(folder, fname)
        runs[system_name] = read_trec_run(run_path, topk=50)

    return runs


def qid_sort_key(qid):
    try:
        return int(qid)
    except ValueError:
        return qid


def write_count_log(runs, qrels, out_file, title):
    all_qids = sorted(qrels.keys(), key=qid_sort_key)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{title}\n")
        f.write("Count of relevant documents retrieved per query (top-50)\n")
        f.write("=" * 100 + "\n\n")

        f.write("Systems included:\n")
        for sys_name in runs:
            f.write(f"  - {sys_name}\n")
        f.write("\n")

        for qid in all_qids:
            f.write(f"QID: {qid}\n")
            rel_docs = qrels.get(qid, set())
            f.write(f"  Total relevant in qrels (rel>=2): {len(rel_docs)}\n")

            for sys_name in runs:
                retrieved = runs[sys_name].get(qid, set())
                captured = retrieved.intersection(rel_docs)
                f.write(f"  {sys_name:<24}: {len(captured)}\n")

            f.write("\n")

    print(f"Saved log to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Count relevant documents retrieved per query for each c=50 system."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing significance run files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="ir_datasets name, e.g. msmarco-passage/trec-dl-2019/judged"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output log file"
    )
    parser.add_argument(
        "--min_rel",
        type=int,
        default=2,
        help="Minimum qrel relevance to count as relevant"
    )

    args = parser.parse_args()

    runs = load_c50_runs(args.folder)
    if not runs:
        print("No c50 run files found.")
        return

    qrels = load_qrels(args.dataset, min_rel=args.min_rel)
    write_count_log(runs, qrels, args.out, title=args.dataset)


if __name__ == "__main__":
    main()