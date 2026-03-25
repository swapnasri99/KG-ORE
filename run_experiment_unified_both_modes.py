import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
import os
from pyterrier_dr import FlexIndex, TasB
from pyterrier_t5 import MonoT5ReRanker

import torch
import argparse
import pandas as pd
import random
random.seed(42)

parser = argparse.ArgumentParser(description='KG-Enhanced ORE')

parser.add_argument('--dl', type=int, default=19, help='TREC-DL year: 19 or 20')
parser.add_argument('--budget', type=int, default=100, help='Re-ranking budget')
parser.add_argument('--ce', type=int, default=7, help='Cross-encoder batch budget')
parser.add_argument('--s', type=int, default=30, help='Top S docs for set affinity (currently unused in ORE core)')
parser.add_argument('--s1', type=int, default=25, help='Top-S1 cluster heads')
parser.add_argument('--s2', type=int, default=15, help='Top-S2 expansion')
parser.add_argument('--batch', type=int, default=16, help='MonoT5 batch size')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_bm25_calls', type=int, default=25, help='How many newly expanded docs to BM25-score per iteration')
parser.add_argument('--kg_neighbor_k', type=int, default=16, help='Keep top-k neighbors after KG filtering from LAFF-128')
parser.add_argument('--use_kg_in_cer', action='store_true', help='Add separate KG term into CER')
parser.add_argument('--kg_cer_weight_init', type=float, default=0.20, help='Initial lambda for KG term in CER before fitting')

parser.add_argument('--alpha', type=float, default=0.5, help='LAFF weight')
parser.add_argument('--beta', type=float, default=0.3, help='Entity overlap weight')
parser.add_argument('--gamma', type=float, default=0.2, help='KG connectivity weight (0 = no Freebase)')
parser.add_argument('--kg_mode', type=str, default='log', help='KG scoring: log, binary, count, coverage')

parser.add_argument('--passage_el', type=str, default=None, help='Path to passage entity linking JSONL')
parser.add_argument('--freebase_dir', type=str, default=None, help='Path to kgpr_data/freebase/ (required if gamma > 0)')
parser.add_argument('--query_el', type=str, default=None, help='Path to query EL JSONL (optional)')
parser.add_argument('--mode', type=str, default='overwrite', choices=['overwrite', 'reuse'])
parser.add_argument('--correction', type=str, default=None, choices=['bonferroni', 'holm'])
parser.add_argument('--baseline', type=int, default=None)
parser.add_argument('--lk', type=int, default=128)
parser.add_argument('--passage_el_db', type=str, default=None, help='Path to SQLite .db for full corpus EL (low RAM mode)')
parser.add_argument('--neighbor_mode', type=str, default='kg_laff', choices=['kg_laff', 'union'],
                    help="'kg_laff' = KG+LAFF rescored top-k, 'union' = union of LAFF top-k and KG top-k")
parser.add_argument('--max_queries', type=int, default=None, help='Only run first N queries for debugging')
parser.add_argument('--skip_queries', type=int, default=0, help='Skip first N queries')

args = parser.parse_args()
if not args.passage_el and not args.passage_el_db:
    print('ERROR: Provide either --passage_el (JSONL) or --passage_el_db (SQLite)')
    exit(1)

if args.gamma > 0 and not args.freebase_dir:
    print('ERROR: --gamma > 0 requires --freebase_dir')
    exit(1)

if not pt.started():
    pt.java.init()

dataset = pt.get_dataset('irds:msmarco-passage')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading Components')
print('=' * 60)

print('[1/6] TasB encoder...')
model = TasB.dot(batch_size=1, device=device)

print('[2/6] FlexIndex...')
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

print('[3/6] BM25 retriever...')
bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

print('[4/6] Corpus graphs...')
graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)

print('[5/6] MonoT5 scorer...')
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

print(f'[6/6] Evaluation dataset (TREC-DL 20{args.dl})...')
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')
topics_df = eval_dataset.get_topics()
qrels_df = eval_dataset.get_qrels()

if args.skip_queries > 0 or args.max_queries is not None:
    all_qids = topics_df['qid'].tolist()
    all_qids = all_qids[args.skip_queries:]  # skip first N
    if args.max_queries is not None:
        all_qids = all_qids[:args.max_queries]  # then take next M
    selected_qids = all_qids
    topics_df = topics_df[topics_df['qid'].isin(selected_qids)].copy()
    qrels_df = qrels_df[qrels_df['qid'].isin(selected_qids)].copy()
    print(f'[DEBUG] Running {len(selected_qids)} queries (skip={args.skip_queries}): {selected_qids}')

qrels_map = qrels_df[qrels_df['label'] >= 2].groupby('qid')['docno'].apply(set).to_dict()
print(f'[DEBUG] qrels_map built for {len(qrels_map)} queries (rel>=2).')
print('✓ All components loaded\n')

from ore_kg_unified_both_modes import create_ore_kg

print('=' * 60)
print('Experiment Configuration')
print('=' * 60)
print(f'  TREC-DL: 20{args.dl}')
print(f'  Budget: {args.budget}')
print(f'  Weights: α={args.alpha} (LAFF), β={args.beta} (Entity), γ={args.gamma} (KG)')
print(f'  Passage EL: {args.passage_el or args.passage_el_db}')
print(f'  Freebase: {args.freebase_dir or "DISABLED"}')
print(f'  KG Mode: {args.kg_mode}')
print(f'  num_bm25_calls: {args.num_bm25_calls}')
print(f'  kg_neighbor_k: {args.kg_neighbor_k}')
print(f'  use_kg_in_cer: {args.use_kg_in_cer}')
print(f'  neighbor_mode: {args.neighbor_mode}')

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
    qrels_map=qrels_map,  # Pass qrels_map for debugging purposes
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
if args.use_kg_in_cer:
    parts.append('CERKG')
if args.neighbor_mode == 'union':
    parts.append('UNION')
exp_name = f"ORE_{'_'.join(parts)}.c{args.budget}.DL{args.dl}"

kg_tag = '_'.join(parts) if parts else 'KG'
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
    [bm25 >> ore_kg],
    topics_df,
    qrels_df,
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[exp_name],
    **experiment_kwargs,
)

print('\n' + '=' * 60)
print(f'Results: {exp_name}')
print('=' * 60)
print(result.T)
print(f'\nScorer stats: {ore_kg.kg_scorer.get_stats()}')
print('\nExperiment complete')
print(f'Run saved to: {save_dir}')