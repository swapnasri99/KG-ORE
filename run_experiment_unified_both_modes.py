import argparse
import os
import random

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import torch
from ir_measures import R, nDCG
from pyterrier_dr import FlexIndex, TasB
from pyterrier_t5 import MonoT5ReRanker

random.seed(42)

parser = argparse.ArgumentParser(description='ORE baseline + union expansion with total relevant docs in final top 50')
parser.add_argument('--dl', type=int, default=19, help='TREC-DL year: 19 or 20')
parser.add_argument('--budget', type=int, default=100, help='Re-ranking budget')
parser.add_argument('--ce', type=int, default=7, help='Cross-encoder batch budget')
parser.add_argument('--s1', type=int, default=25, help='Top-S1 cluster heads')
parser.add_argument('--s2', type=int, default=15, help='Top-S2 expansion')
parser.add_argument('--batch', type=int, default=16, help='MonoT5 batch size')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--kg_neighbor_k', type=int, default=16, help='LAFF top-k and KG top-k size inside union expansion')
parser.add_argument('--alpha', type=float, default=0.0, help='LAFF weight in KG scorer')
parser.add_argument('--beta', type=float, default=0.3, help='Entity overlap weight')
parser.add_argument('--gamma', type=float, default=0.7, help='KG connectivity weight')
parser.add_argument('--kg_mode', type=str, default='minmax', help='KG scoring: log, binary, count, coverage, minmax')
parser.add_argument('--passage_el', type=str, default=None, help='Path to passage entity linking JSONL')
parser.add_argument('--freebase_dir', type=str, default=None, help='Path to freebase dir')
parser.add_argument('--query_el', type=str, default=None, help='Path to query EL JSONL (optional)')
parser.add_argument('--mode', type=str, default='overwrite', choices=['overwrite', 'reuse'])
parser.add_argument('--correction', type=str, default=None, choices=['bonferroni', 'holm'])
parser.add_argument('--baseline', type=int, default=None)
parser.add_argument('--lk', type=int, default=16)
parser.add_argument('--passage_el_db', type=str, default=None, help='Path to SQLite .db for full corpus EL')
parser.add_argument('--max_queries', type=int, default=None, help='Only run first N queries for debugging')
parser.add_argument('--skip_queries', type=int, default=0, help='Skip first N queries')
args = parser.parse_args()

if not pt.started():
    pt.java.init()

dataset = pt.get_dataset('irds:msmarco-passage')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = TasB.dot(batch_size=1, device=device)
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff')


scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')
topics_df = eval_dataset.get_topics()
qrels_df = eval_dataset.get_qrels()

if args.skip_queries > 0 or args.max_queries is not None:
    all_qids = topics_df['qid'].tolist()
    all_qids = all_qids[args.skip_queries:]
    if args.max_queries is not None:
        all_qids = all_qids[:args.max_queries]
    selected_qids = all_qids
    topics_df = topics_df[topics_df['qid'].isin(selected_qids)].copy()
    qrels_df = qrels_df[qrels_df['qid'].isin(selected_qids)].copy()
    print(f'[DEBUG] Running {len(selected_qids)} queries (skip={args.skip_queries}): {selected_qids}')

qrels_map = qrels_df[qrels_df['label'] >= 2].groupby('qid')['docno'].apply(set).to_dict()
print(f'[DEBUG] qrels_map built for {len(qrels_map)} queries (rel>=2).')
print('✓ All components loaded\n')

from ore_kg_unified_both_modes import create_ore_kg_union_on_baseline

print('=' * 60)
print('Experiment Configuration')
print('=' * 60)
print(f'  TREC-DL: 20{args.dl}')
print(f'  Budget: {args.budget}')
print(f'  Weights: α={args.alpha} (LAFF), β={args.beta} (Entity), γ={args.gamma} (KG)')
print(f'  Passage EL: {args.passage_el or args.passage_el_db}')
print(f'  Freebase: {args.freebase_dir or "DISABLED"}')
print(f'  KG Mode: {args.kg_mode}')
print('  num_bm25_calls: fixed to 0 in code')
print(f'  kg_neighbor_k: {args.kg_neighbor_k}')
print('  selection path: original ORE baseline')
print('  expansion path: LAFF union KG-only')

ore_union = create_ore_kg_union_on_baseline(
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
    param_bounds=(0.25, 0.9),
    num_bm25_calls=0,
    kg_neighbor_k=args.kg_neighbor_k,
    qrels_map=qrels_map,
)

print('\n' + '=' * 60)
print('Running Experiment')
print('=' * 60)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

parts = []
if args.alpha > 0:
    parts.append(f'L{args.alpha}')
if args.beta > 0:
    parts.append(f'E{args.beta}')
if args.gamma > 0:
    parts.append(f'K{args.gamma}')
parts.append('UNION_ON_BASELINE_TOP50TOTAL')
exp_name = f"ORE_{'_'.join(parts)}.c{args.budget}.DL{args.dl}"

kg_tag = '_'.join(parts)
save_dir = f'runs/adaptive/dl{args.dl}/kg_ore/{kg_tag}/'
os.makedirs(save_dir, exist_ok=True)

experiment_kwargs = {
    'save_dir': save_dir,
    'save_mode': args.mode,
    'verbose': args.verbose,
}
if args.correction:
    experiment_kwargs['correction'] = args.correction
if args.baseline is not None:
    experiment_kwargs['baseline'] = args.baseline

result = pt.Experiment(
    [bm25 >> ore_union],
    topics_df,
    qrels_df,
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[exp_name],
    **experiment_kwargs,
)

print('\n' + '=' * 60)
print(f'Results: Union-on-Baseline DL20{args.dl}')
print('=' * 60)
print(result.T)