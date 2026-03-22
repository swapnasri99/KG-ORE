"""
Per-Query Recall Diagnostic
============================
Runs the full ORE pipeline and prints per-query R(rel=2)@budget
so you can compare against the pool upper bound.

Usage:
  python run_perquery_recall.py --dl 19 --budget 50 --ce 4 --s1 10 --s2 15 \
    --alpha 0.0 --beta 0.3 --gamma 0.7 --kg_mode minmax \
    --passage_el_db passage_entities.db --freebase_dir freebase \
    --neighbor_mode union
"""
import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import R
import os
from pyterrier_dr import FlexIndex, TasB
from pyterrier_t5 import MonoT5ReRanker

import torch
import argparse
import pandas as pd
import numpy as np
import random
random.seed(42)

parser = argparse.ArgumentParser(description='Per-Query Recall Diagnostic')

parser.add_argument('--dl', type=int, default=19)
parser.add_argument('--budget', type=int, default=50)
parser.add_argument('--ce', type=int, default=4)
parser.add_argument('--s', type=int, default=10)
parser.add_argument('--s1', type=int, default=10)
parser.add_argument('--s2', type=int, default=15)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_bm25_calls', type=int, default=25)
parser.add_argument('--kg_neighbor_k', type=int, default=16)
parser.add_argument('--use_kg_in_cer', action='store_true')
parser.add_argument('--kg_cer_weight_init', type=float, default=0.20)

parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--kg_mode', type=str, default='minmax')

parser.add_argument('--passage_el', type=str, default=None)
parser.add_argument('--passage_el_db', type=str, default=None)
parser.add_argument('--query_el', type=str, default=None)
parser.add_argument('--freebase_dir', type=str, default=None)
parser.add_argument('--lk', type=int, default=128)
parser.add_argument('--neighbor_mode', type=str, default='union', choices=['kg_laff', 'union'])
parser.add_argument('--mode', type=str, default='reuse', choices=['overwrite', 'reuse'])

args = parser.parse_args()

if not args.passage_el and not args.passage_el_db:
    print('ERROR: Provide either --passage_el or --passage_el_db')
    exit(1)

if not pt.started():
    pt.java.init()

dataset = pt.get_dataset('irds:msmarco-passage')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TasB.dot(batch_size=1, device=device)
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)

scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

from ore_kg_unified_union import create_ore_kg

ore_kg = create_ore_kg(
    dual_encoder=model,
    scorer=scorer,
    corpus_index=idx,
    graph=graph,
    laff_graph=laff_graph,
    kg_alpha=args.alpha,
    kg_beta=args.beta,
    kg_gamma=args.gamma,
    kg_score_mode=args.kg_mode,
    freebase_dir=args.freebase_dir,
    passage_el_path=args.passage_el,
    query_el_path=args.query_el,
    passage_el_db=args.passage_el_db,
    budget=args.budget,
    cross_enc_budget=args.ce,
    top_s=args.s1,
    top_s2=args.s2,
    verbose=args.verbose,
    param_bounds=(0.25, 0.95),
    num_bm25_calls=args.num_bm25_calls,
    kg_neighbor_k=args.kg_neighbor_k,
    use_kg_in_cer=args.use_kg_in_cer,
    kg_cer_weight_init=args.kg_cer_weight_init,
    neighbor_mode=args.neighbor_mode,
)

# Build experiment name to match saved run
parts = []
if args.alpha > 0:
    parts.append(f'L{args.alpha}')
if args.beta > 0:
    parts.append(f'E{args.beta}')
if args.gamma > 0:
    parts.append(f'K{args.gamma}')
if args.use_kg_in_cer:
    parts.append('CERKG')
if args.neighbor_mode == 'union':
    parts.append('UNION')
exp_name = f"ORE_{'_'.join(parts)}.c{args.budget}.DL{args.dl}"

kg_tag = '_'.join(parts) if parts else 'KG'
save_dir = f'runs/adaptive/dl{args.dl}/kg_ore/{kg_tag}/'
os.makedirs(save_dir, exist_ok=True)

# Run with perquery=True to get per-query metrics
result = pt.Experiment(
    [bm25 >> ore_kg],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [R(rel=2)@args.budget],
    names=[exp_name],
    save_dir=save_dir,
    save_mode=args.mode,
    perquery=True,
)

# Process per-query results
qrels_df = eval_dataset.get_qrels().copy()
qrels_rel_map = (
    qrels_df[qrels_df["label"] >= 2]
    .groupby("qid")["docno"]
    .apply(lambda x: set(x.astype(str)))
    .to_dict()
)

print(f"\n{'='*60}")
print(f"Per-Query R(rel=2)@{args.budget} — {exp_name}")
print(f"{'='*60}")
print(f"{'qid':>10s} | {'total_rel':>9s} | {'R@'+str(args.budget):>8s}")
print("-" * 35)

recalls = []
for _, row in result.iterrows():
    qid = str(row['qid'])
    recall_val = row['value']
    total_rel = len(qrels_rel_map.get(qid, set()))
    recalls.append(recall_val)
    print(f"{qid:>10s} | {total_rel:>9d} | {recall_val:>8.4f}")

print("-" * 35)
print(f"{'AVG':>10s} | {'':>9s} | {np.mean(recalls):>8.4f}")
print(f"{'='*60}")
print(f"\nThis is the ACTUAL system recall (after CER scheduling + MonoT5).")
print(f"Compare against pool upper-bound recall to see how much CER misses.")