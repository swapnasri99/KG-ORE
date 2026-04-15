"""
Significance test — clean short output with bold for significant results.

Examples:
  python run_sig_clean.py --dl 19 --budget 50 --systems GAR ore --baseline 0
  python run_sig_clean.py --dl 19 --budget 50 --systems GAR KG_ORE_best --baseline 0
"""

import argparse
import os

import pandas as pd
import pyterrier as pt
from ir_measures import nDCG, R

parser = argparse.ArgumentParser()
parser.add_argument('--dl', type=int, required=True, choices=[19, 20])
parser.add_argument('--budget', type=int, required=True, choices=[50, 100])
parser.add_argument('--systems', nargs='+', required=True)
parser.add_argument('--baseline', type=int, required=True)
args = parser.parse_args()

if not pt.java.started():
    pt.java.init()

sig_dir = f'runs/adaptive/dl{args.dl}/significance'
names = [f'{s}.c{args.budget}' for s in args.systems]

for name in names:
    path = os.path.join(sig_dir, f'{name}.res.gz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing: {path}')

dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')
dummy = pt.apply.generic(lambda df: df)

metrics = [nDCG @ 10, nDCG @ args.budget, R(rel=2) @ args.budget]
metric_names = [f'nDCG@10', f'nDCG@{args.budget}', f'R(rel=2)@{args.budget}']

result = pt.Experiment(
    [dummy] * len(names),
    dataset.get_topics(),
    dataset.get_qrels(),
    metrics,
    names=names,
    save_dir=sig_dir,
    save_mode='reuse',
    correction='bonferroni',
    baseline=args.baseline,
)


BOLD = '\033[1m'
GREEN = '\033[92m'
RESET = '\033[0m'

baseline_name = names[args.baseline]

print()
print(f'{BOLD}{"=" * 72}{RESET}')
print(f'{BOLD}  DL20{args.dl} | c={args.budget} | Baseline: {baseline_name}{RESET}')
print(f'  Paired t-test, p<0.05, Bonferroni correction')
print(f'{BOLD}{"=" * 72}{RESET}')
print(f'  {"System":<22} {"nDCG@10":>10} {"nDCG@"+str(args.budget):>10} {"R@"+str(args.budget):>10}')
print(f'  {"-" * 62}')

for _, row in result.iterrows():
    name = row['name']
    vals = []
    for m in metric_names:
        score = row[m]
        reject_col = f'{m} reject'
        is_sig = row.get(reject_col, False)
        if is_sig is True:
            vals.append(f'{GREEN}{BOLD}{score:.3f}*{RESET}')
        else:
            vals.append(f'{score:.3f} ')

    
    line = f'  {name:<22}'
    for v in vals:
        if '*' in v:
            line += f'{v:>25}'  
        else:
            line += f'{v:>11}'
    print(line)

print()
print(f'  * = significantly different from {baseline_name} (p<0.05, Bonferroni)')
print()