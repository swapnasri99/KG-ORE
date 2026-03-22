"""
Statistical Significance Test: KG-ORE vs Baselines
Clean table output.
"""
import pyterrier as pt
import pandas as pd
from ir_measures import nDCG, R

if not pt.started():
    pt.java.init()

eval_dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

def load_run(path):
    df = pd.read_csv(path, sep='\s+', header=None,
                      names=['qid','Q0','docno','rank','score','tag'])
    df['qid'] = df['qid'].astype(str)
    df['docno'] = df['docno'].astype(str)
    return df

runs = [
    load_run('runs/adaptive/dl19/gbm25/GAR.c50.res.gz'),
    load_run('runs/adaptive/dl19/gbm25/QuAM.c50.res.gz'),
    load_run('runs/adaptive/dl19/kg_ore/E0.3_K0.7_UNION/ORE_E0.3_K0.7_UNION.c50.DL19.res.gz'),
]

names = ['GAR', 'QuAM', 'KG-ORE (b=0.3,g=0.7)']

result = pt.Experiment(
    runs,
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG@10, nDCG@50, R(rel=2)@50],
    names=names,
    baseline=1,  # QuAM is baseline
    correction='bonferroni',
)

# Clean results table
print("\n" + "=" * 70)
print("RESULTS TABLE (Baseline: QuAM, Correction: Bonferroni)")
print("=" * 70)

header = f"{'Method':<28s} | {'nDCG@10':>8s} | {'nDCG@50':>8s} | {'R@50':>8s}"
print(header)
print("-" * 70)

for _, row in result.iterrows():
    name = row['name']
    n10 = row['nDCG@10']
    n50 = row['nDCG@50']
    r50 = row['R(rel=2)@50']
    print(f"{name:<28s} | {n10:>8.4f} | {n50:>8.4f} | {r50:>8.4f}")

print("-" * 70)

# Significance details for KG-ORE vs QuAM
print("\n" + "=" * 70)
print("SIGNIFICANCE: KG-ORE vs QuAM (Bonferroni corrected)")
print("=" * 70)

kg_row = result[result['name'] == 'KG-ORE (b=0.3,g=0.7)'].iloc[0]
quam_row = result[result['name'] == 'QuAM'].iloc[0]

for metric in ['nDCG@10', 'nDCG@50', 'R(rel=2)@50']:
    kg_val = kg_row[metric]
    quam_val = quam_row[metric]
    diff = kg_val - quam_val
    pct = (diff / quam_val) * 100 if quam_val > 0 else 0

    p_col = f'{metric} p-value corrected'
    reject_col = f'{metric} reject'

    if p_col in kg_row and pd.notna(kg_row[p_col]):
        p_val = kg_row[p_col]
        reject = kg_row[reject_col]
        sig_mark = "YES *" if reject else "no"
        print(f"  {metric:<15s}: QuAM={quam_val:.4f}  KG-ORE={kg_val:.4f}  diff={diff:+.4f} ({pct:+.1f}%)  p={p_val:.4f}  significant={sig_mark}")
    else:
        print(f"  {metric:<15s}: QuAM={quam_val:.4f}  KG-ORE={kg_val:.4f}  diff={diff:+.4f} ({pct:+.1f}%)  (baseline)")

print("=" * 70)