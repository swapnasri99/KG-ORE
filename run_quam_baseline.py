import argparse
import os
import random

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import torch
from ir_measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

random.seed(42)

parser = argparse.ArgumentParser(description='QuAM baseline runner')
parser.add_argument('--dl', type=int, default=19, help='TREC-DL year: 19 or 20')
parser.add_argument('--budget', type=int, default=100, help='Re-ranking budget c')
parser.add_argument('--ce', type=int, default=7, help='Cross-encoder batch budget')
parser.add_argument('--s', type=int, default=30, help='Top-s docs for set affinity (QuAM S)')
parser.add_argument('--batch', type=int, default=16, help='MonoT5 batch size')
parser.add_argument('--lk', type=int, default=16, help='LAFF graph limit k')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--mode', type=str, default='overwrite', choices=['overwrite', 'reuse'])
parser.add_argument('--correction', type=str, default=None, choices=['bonferroni'])
args = parser.parse_args()

if not pt.started():
    pt.java.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

dataset = pt.get_dataset('irds:msmarco-passage')

retriever = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

laff_graph = pta.Artifact.from_hf(
    'macavaney/msmarco-passage.corpusgraph.bm25.128.laff'
).to_limit_k(args.lk)

scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(
    verbose=args.verbose, batch_size=args.batch
)

eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

print('✓ All components loaded\n')

from baselines.quam import QUAM

print('=' * 60)
print('QuAM Baseline Configuration')
print('=' * 60)
print(f'  TREC-DL: 20{args.dl}')
print(f'  Budget: {args.budget}')
print(f'  Cross-encoder budget (ce): {args.ce}')
print(f'  Top-s (set affinity): {args.s}')
print(f'  LAFF limit-k: {args.lk}')
print(f'  Batch size: {args.batch}')
print('=' * 60)

exp_name = f'QuAM.c{args.budget}.s{args.s}.ce{args.ce}.lk{args.lk}'
save_dir = f'runs/adaptive/dl{args.dl}/quam/c{args.budget}_s{args.s}_ce{args.ce}_lk{args.lk}/'
os.makedirs(save_dir, exist_ok=True)

class RemoveColon(pt.Transformer):
    def transform(self, topics):
        topics = topics.copy()
        topics["query"] = topics["query"].str.replace(":", " ", regex=False)
        return topics


experiment_kwargs = {
    'save_dir': save_dir,
    'save_mode': args.mode,
    'verbose': args.verbose,
}
if args.correction:
    experiment_kwargs['correction'] = args.correction

"""result = pt.Experiment(
    [
        retriever >> QUAM(
            scorer=scorer,
            corpus_graph=laff_graph,
            num_results=args.budget,
            cross_enc_budget=args.ce,
            top_k_docs=args.s,
            batch_size=args.batch,
            verbose=args.verbose,
        ),
    ],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG @ 10, nDCG @ args.budget, R(rel=2) @ args.budget],
    names=[exp_name],
    **experiment_kwargs,
)
"""
result = pt.Experiment(
    [
        RemoveColon() >> retriever >> QUAM(
            scorer=scorer,
            corpus_graph=laff_graph,
            num_results=args.budget,
            cross_enc_budget=args.ce,
            top_k_docs=args.s,
            batch_size=args.batch,
            verbose=args.verbose,
        ),
    ],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG @ 10, nDCG @ args.budget, R(rel=2) @ args.budget],
    names=[exp_name],
    **experiment_kwargs,
)

print('\n' + '=' * 60)
print(f'Results: QuAM Baseline DL20{args.dl}')
print('=' * 60)
print(result.T)

