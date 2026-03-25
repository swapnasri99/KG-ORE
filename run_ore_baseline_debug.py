import argparse
import os
import random

import pyterrier as pt
import pyterrier_alpha as pta
import torch
from ir_measures import R, nDCG
from pyterrier_dr import FlexIndex, TasB
from pyterrier_t5 import MonoT5ReRanker

random.seed(42)

parser = argparse.ArgumentParser(description='ORE Baseline with total relevant docs in final top 50')
parser.add_argument('--dl', type=int, default=19)
parser.add_argument('--budget', type=int, default=100)
parser.add_argument('--ce', type=int, default=7)
parser.add_argument('--s1', type=int, default=25)
parser.add_argument('--s2', type=int, default=15)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--max_queries', type=int, default=None)
parser.add_argument('--skip_queries', type=int, default=0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--mode', type=str, default='overwrite', choices=['overwrite', 'reuse'])
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
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(16)

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

from ore_adaptive_debug import OREAdaptive

ore = OREAdaptive(
    model,
    scorer,
    idx,
    graph,
    laff_graph,
    budget=args.budget,
    cross_enc_budget=args.ce,
    param_bounds=(0.25, 0.9),
    num_bm25_calls=0,
    verbose=args.verbose,
    top_s=args.s1,
    top_s2=args.s2,
    qrels_map=qrels_map,
)

save_dir = f'runs/adaptive/dl{args.dl}/ore_top50total/'
os.makedirs(save_dir, exist_ok=True)

result = pt.Experiment(
    [bm25 >> ore],
    topics_df,
    qrels_df,
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[f'ore_baseline.c{args.budget}'],
    save_dir=save_dir,
    save_mode=args.mode,
    verbose=args.verbose,
)

print('\n' + '=' * 60)
print(f'Results: ORE Baseline DL20{args.dl}')
print('=' * 60)
print(result.T)