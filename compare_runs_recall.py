"""
Per query: how many relevant docs found in each run.

Usage:
  python compare_runs_recall.py --dl 19 --top_k 50
  python compare_runs_recall.py --dl 20 --top_k 100
"""

import argparse, gzip, os
from collections import defaultdict
import pyterrier as pt

parser = argparse.ArgumentParser()
parser.add_argument('--dl', type=int, required=True, choices=[19, 20])
parser.add_argument('--top_k', type=int, required=True, help='Cutoff: 50 or 100')
parser.add_argument('--runs_dir', type=str, default=None)
parser.add_argument('--min_rel', type=int, default=2)
args = parser.parse_args()

if args.runs_dir is None:
    args.runs_dir = f'runs/adaptive/dl{args.dl}/significance'

if not pt.started():
    pt.java.init()

eval_ds = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')
qrels_df = eval_ds.get_qrels()

qrels_rel = defaultdict(set)
for _, row in qrels_df.iterrows():
    if int(row['label']) >= args.min_rel:
        qrels_rel[str(row['qid'])].add(str(row['docno']))

tag = f'c{args.top_k}'
run_files = sorted([f for f in os.listdir(args.runs_dir)
                    if f.endswith('.res.gz') and tag in f.lower()])

if not run_files:
    print(f"No .res.gz files with '{tag}' found in {args.runs_dir}")
    exit(1)

runs = {}
for fname in run_files:
    run_docs = defaultdict(list)
    with gzip.open(os.path.join(args.runs_dir, fname), 'rt') as f:
        for line in f:
            parts = line.strip().split()
            qid, docno, rank = str(parts[0]), str(parts[2]), int(parts[3])
            if rank < args.top_k:
                run_docs[qid].append(docno)
    label = fname.replace('.res.gz', '').replace(f'.{tag}', '')
    runs[label] = run_docs

all_qids = sorted(set().union(*(r.keys() for r in runs.values())), key=lambda x: int(x))
labels = list(runs.keys())

print(f"\nDL20{args.dl}  |  Relevant docs in top-{args.top_k}  |  min_rel={args.min_rel}\n")
header = f"{'qid':<12} {'total':>5}"
for label in labels:
    header += f"  {label:>10}"
print(header)
print("─" * len(header))

totals = {l: 0 for l in labels}
for qid in all_qids:
    rel = qrels_rel.get(qid, set())
    if not rel:
        continue
    row = f"{qid:<12} {len(rel):>5}"
    for label in labels:
        docs = set(runs[label].get(qid, []))
        found = len(rel & docs)
        totals[label] += found
        row += f"  {found:>10}"
    print(row)

print("─" * len(header))
total_rel = sum(len(qrels_rel.get(q, set())) for q in all_qids if qrels_rel.get(q))
row = f"{'TOTAL':<12} {total_rel:>5}"
for label in labels:
    row += f"  {totals[label]:>10}"
print(row)