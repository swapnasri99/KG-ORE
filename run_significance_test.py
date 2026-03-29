"""
Significance test: KG-ORE vs GAR & QuAM with Bonferroni correction.
Uses saved .res.gz files (save_mode='reuse')

Usage:
  python run_significance_test.py --dl 19 --budget 50
  python run_significance_test.py --dl 20 --budget 50
  python run_significance_test.py --dl 19 --budget 100
  python run_significance_test.py --dl 20 --budget 100
"""

import argparse
import os
import shutil

import pandas as pd
import pyterrier as pt
from ir_measures import nDCG, R

parser = argparse.ArgumentParser(description='Significance test: KG-ORE vs GAR/QuAM (Bonferroni)')
parser.add_argument('--dl', type=int, default=19, help='TREC-DL year: 19 or 20')
parser.add_argument('--budget', type=int, default=50, help='Budget: 50 or 100')
parser.add_argument('--baseline', type=int, default=2,
                    help='Index of baseline system for paired test (0=GAR, 1=QuAM, 2=KG-ORE)')
args = parser.parse_args()

if not pt.started():
    pt.java.init()

# ---------- paths ----------
baseline_dir = f'runs/adaptive/dl{args.dl}/gbm25'
kg_dir = f'runs/adaptive/dl{args.dl}/kg_ore/E0.6_K0.4_UNION_ON_BASELINE_TOP50TOTAL'
compare_dir = f'runs/adaptive/dl{args.dl}/significance'
os.makedirs(compare_dir, exist_ok=True)

gar_name = f'GAR.c{args.budget}'
quam_name = f'QuAM.c{args.budget}'
kg_name = f'KG_ORE.c{args.budget}'

kg_saved = f'ORE_E0.6_K0.4_UNION_ON_BASELINE_TOP50TOTAL.c{args.budget}.DL{args.dl}'


copies = [
    (os.path.join(baseline_dir, f'{gar_name}.res.gz'),
     os.path.join(compare_dir, f'{gar_name}.res.gz')),
    (os.path.join(baseline_dir, f'{quam_name}.res.gz'),
     os.path.join(compare_dir, f'{quam_name}.res.gz')),
    (os.path.join(kg_dir, f'{kg_saved}.res.gz'),
     os.path.join(compare_dir, f'{kg_name}.res.gz')),
]

for src, dst in copies:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'  ✓ {src}')
    else:
        raise FileNotFoundError(f'Missing saved run: {src}')

# ---------- dataset ----------
dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

dummy = pt.apply.generic(lambda df: df)

names = [gar_name, quam_name, kg_name]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ---------- experiment ----------
result = pt.Experiment(
    [dummy, dummy, dummy],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG @ 10, nDCG @ args.budget, R(rel=2) @ args.budget],
    names=names,
    save_dir=compare_dir,
    save_mode='reuse',
    correction='bonferroni',
    baseline=args.baseline,
)


keep = ['name']
for col in result.columns:
    if col == 'name':
        continue
    if ' ' not in col or 'p-value corrected' in col or col.endswith('reject'):
        keep.append(col)

clean = result[keep].copy()


clean.columns = [c.replace(' p-value corrected', ' p').replace(' reject', ' sig') for c in clean.columns]

print()
print('=' * 70)
print(f'  Significance Test — DL20{args.dl}  c={args.budget}')
print(f'  Baseline: {names[args.baseline]}')
print(f'  Correction: Bonferroni')
print('=' * 70)
print(clean.to_string(index=False))
print()