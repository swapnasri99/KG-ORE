
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


parser = argparse.ArgumentParser(description="KG-Enhanced ORE")

# ORE parameters
parser.add_argument("--dl", type=int, default=19, help="TREC-DL year: 19 or 20")
parser.add_argument("--budget", type=int, default=100, help="Re-ranking budget")
parser.add_argument("--ce", type=int, default=7, help="Cross-encoder batch budget")
parser.add_argument("--s", type=int, default=30, help="Top S docs for set affinity")
parser.add_argument("--s1", type=int, default=25, help="Top-S1 cluster heads")
parser.add_argument("--s2", type=int, default=15, help="Top-S2 expansion")
parser.add_argument("--batch", type=int, default=16, help="MonoT5 batch size")
parser.add_argument("--verbose", action="store_true")

# KG scoring
parser.add_argument("--alpha", type=float, default=0.5, help="LAFF weight")
parser.add_argument("--beta", type=float, default=0.3, help="Entity overlap weight")
parser.add_argument("--gamma", type=float, default=0.2, help="KG connectivity weight (0 = no Freebase)")
parser.add_argument("--kg_mode", type=str, default="log", help="KG scoring: log, binary, count, coverage")

# Data paths (REQUIRED)
parser.add_argument("--passage_el", type=str, default=None,
                    help="Path to passage entity linking JSONL")
parser.add_argument("--freebase_dir", type=str, default=None,
                    help="Path to kgpr_data/freebase/ (required if gamma > 0)")
parser.add_argument("--query_el", type=str, default=None,
                    help="Path to query EL JSONL (optional)")

parser.add_argument("--mode", type=str, default="overwrite",
                    choices=["overwrite", "reuse"],
                    help="overwrite=run fully, reuse=load saved runs")
parser.add_argument("--correction", type=str, default=None,
                    choices=["bonferroni", "holm"],
                    help="Multiple testing correction for significance")
parser.add_argument("--baseline", type=int, default=None,
                    help="Index of baseline system for significance testing (0-based)")

parser.add_argument("--passage_el_db", type=str, default=None,help="Path to SQLite .db for full corpus EL (low RAM mode)")

args = parser.parse_args()
if not args.passage_el and not args.passage_el_db:
    print("ERROR: Provide either --passage_el (JSONL) or --passage_el_db (SQLite)")
    exit(1)

# Validates
if args.gamma > 0 and not args.freebase_dir:
    print("ERROR: --gamma > 0 requires --freebase_dir")
    print("Either set --gamma 0.0 or provide --freebase_dir kgpr_data/freebase")
    exit(1)

# ============================================================
# Initialize PyTerrier
# ============================================================

if not pt.started():
    pt.java.init()

dataset = pt.get_dataset('irds:msmarco-passage')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("Loading Components")
print("=" * 60)

print("[1/6] TasB encoder...")
model = TasB.dot(batch_size=1, device=device)

print("[2/6] FlexIndex...")
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

print("[3/6] BM25 retriever...")
bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

print("[4/6] Corpus graphs...")
graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff')

print("[5/6] MonoT5 scorer...")
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

print("[6/6] Evaluation dataset (TREC-DL 20{})...".format(args.dl))
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

print("✓ All components loaded\n")

# ============================================================
# Create KG-Enhanced ORE
# ============================================================

from ore_kg_unified import create_ore_kg

print("=" * 60)
print("Experiment Configuration")
print("=" * 60)
print(f"  TREC-DL: 20{args.dl}")
print(f"  Budget: {args.budget}")
print(f"  Weights: α={args.alpha} (LAFF), β={args.beta} (Entity), γ={args.gamma} (KG)")
print(f"  Passage EL: {args.passage_el}")
print(f"  Freebase: {args.freebase_dir or 'DISABLED'}")
print(f"  KG Mode: {args.kg_mode}")

ore_kg = create_ore_kg(
    dual_encoder=model,
    scorer=scorer,
    corpus_index=idx,
    graph=graph,
    laff_graph=laff_graph,
    # KG
    kg_alpha=args.alpha,
    kg_beta=args.beta,
    kg_gamma=args.gamma,
    kg_score_mode=args.kg_mode,
    # Data
    freebase_dir=args.freebase_dir,
    passage_el_path=args.passage_el,
    query_el_path=args.query_el,
    passage_el_db=args.passage_el_db,
    # ORE
    budget=args.budget,
    cross_enc_budget=args.ce,
    top_s=args.s,
    top_s2=args.s2,
    verbose=args.verbose,
    param_bounds=(0.25, 0.95),
    num_bm25_calls=0
)

# ============================================================
# Run
# ============================================================

print("\n" + "=" * 60)
print("Running Experiment")
print("=" * 60)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Name
parts = []
if args.alpha > 0: parts.append(f"L{args.alpha}")
if args.beta > 0: parts.append(f"E{args.beta}")
if args.gamma > 0: parts.append(f"K{args.gamma}")
exp_name = f"ORE_{'_'.join(parts)}.c{args.budget}.DL{args.dl}"


kg_tag = "_".join(parts) if parts else "KG"
save_dir = f"runs/adaptive/dl{args.dl}/kg_ore/{kg_tag}/"
os.makedirs(save_dir, exist_ok=True)

print(f"  Experiment name: {exp_name}")
print(f"  Save dir: {save_dir}")
print(f"  Save mode: {args.mode}")

experiment_kwargs = {
    'save_dir': save_dir,
    'save_mode': args.mode,
    'verbose': True,
}

if args.correction:
    experiment_kwargs['correction'] = args.correction
if args.baseline is not None:
    experiment_kwargs['baseline'] = args.baseline

result = pt.Experiment(
    [bm25 >> ore_kg],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[exp_name],
    **experiment_kwargs,
    
)

print("\n" + "=" * 60)
print(f"Results: {exp_name}")
print("=" * 60)
print(result.T)


print(f"\nScorer stats: {ore_kg.kg_scorer.get_stats()}")

print("\nExperiment complete")
print(f"Run saved to: {save_dir}")