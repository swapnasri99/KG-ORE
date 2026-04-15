"""
Find queries where KG-ORE captures relevant documents that GAR / QuAM missed.

Usage:
    python find_kg_wins.py --dataset dl20
    python find_kg_wins.py --dataset dl19
    python find_kg_wins.py --dataset both
"""

import argparse, gzip, os, collections
import pyterrier as pt

if not pt.started():
    pt.java.init()

BASE_DIR = "runs/adaptive"

SYSTEMS = {
    "KG_ORE": "KG_ORE_best",
    "GAR":    "GAR",
    "QuAM":   "QuAM",
}


def load_qrels_pt(dataset_name, min_rel=2):
    ds = pt.get_dataset(dataset_name)
    qrels_df = ds.get_qrels()
    rel_df = qrels_df[qrels_df['label'] >= min_rel]
    return rel_df.groupby('qid')['docno'].apply(set).to_dict()


def load_run(path):
    run = collections.defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            run[parts[0]].append(parts[2])
    return dict(run)


def analyse(dataset, cutoff):
    sig_dir = os.path.join(BASE_DIR, dataset, "significance")
    year = "2019" if dataset == "dl19" else "2020"
    ds = pt.get_dataset(f"irds:msmarco-passage/trec-dl-{year}/judged")
    qrels = load_qrels_pt(f"irds:msmarco-passage/trec-dl-{year}/judged", min_rel=2)
    topics = ds.get_topics()
    qid_to_query = dict(zip(topics['qid'], topics['query']))

    runs = {}
    for label, stem in SYSTEMS.items():
        fpath = os.path.join(sig_dir, f"{stem}.c{cutoff}.res.gz")
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing {fpath}")
            continue
        runs[label] = load_run(fpath)

    if "KG_ORE" not in runs:
        print("  [ERROR] KG_ORE run not found."); return []

    kg_run = runs["KG_ORE"]
    common_qids = set(kg_run.keys()) & set(qrels.keys())
    for label in ["GAR", "QuAM"]:
        if label in runs:
            common_qids &= set(runs[label].keys())

    results = []
    for qid in sorted(common_qids):
        rel = qrels[qid]
        kg_rel = set(kg_run[qid]) & rel

        gar_rel = set(runs["GAR"][qid]) & rel if "GAR" in runs else set()
        quam_rel = set(runs["QuAM"][qid]) & rel if "QuAM" in runs else set()

        extra_vs_gar = kg_rel - gar_rel
        extra_vs_quam = kg_rel - quam_rel
        extra_vs_both = kg_rel - (gar_rel | quam_rel)

        if not extra_vs_gar and not extra_vs_quam:
            continue

        results.append({
            "qid": qid,
            "query": qid_to_query.get(qid, "?"),
            "total_rel": len(rel),
            "kg": len(kg_rel),
            "gar": len(gar_rel),
            "quam": len(quam_rel),
            "extra_vs_gar": len(extra_vs_gar),
            "extra_vs_quam": len(extra_vs_quam),
            "extra_vs_both": len(extra_vs_both),
        })

    results.sort(key=lambda r: r["extra_vs_both"], reverse=True)
    return results


def print_report(results, dataset, cutoff, top_n=8):
    print(f"\n  {dataset.upper()} @{cutoff} — Queries where KG-ORE retrieves extra relevant docs\n")

    if not results:
        print("  No improvement queries found.\n"); return

    print(f"  {'QID':<10} {'Query':<40} {'Rel':>4} {'KG':>4} {'GAR':>4} {'QuAM':>5} {'Δ GAR':>6} {'Δ QuAM':>7} {'Δ Both':>7}")
    print(f"  {'─'*10} {'─'*40} {'─'*4} {'─'*4} {'─'*4} {'─'*5} {'─'*6} {'─'*7} {'─'*7}")

    for row in results[:top_n]:
        q = row["query"][:38] + ".." if len(row["query"]) > 40 else row["query"]
        print(f"  {row['qid']:<10} {q:<40} {row['total_rel']:>4} {row['kg']:>4} "
              f"{row['gar']:>4} {row['quam']:>5} {'+'+str(row['extra_vs_gar']):>6} "
              f"{'+'+str(row['extra_vs_quam']):>7} {'+'+str(row['extra_vs_both']):>7}")

    remaining = len(results) - top_n
    if remaining > 0:
        print(f"\n  ... {remaining} more queries with gains.")

    total_vs_both = sum(r["extra_vs_both"] for r in results)
    n_improved = sum(1 for r in results if r["extra_vs_both"] > 0)
    print(f"\n  {n_improved} queries where KG-ORE found docs missed by both GAR and QuAM "
          f"({total_vs_both} extra docs total).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dl20", choices=["dl19", "dl20", "both"])
    parser.add_argument("--cutoff", type=int, default=50, choices=[50, 100])
    parser.add_argument("--top_n", type=int, default=8)
    args = parser.parse_args()

    datasets = ["dl19", "dl20"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        results = analyse(ds, args.cutoff)
        print_report(results, ds, args.cutoff, top_n=args.top_n)