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

parser = argparse.ArgumentParser(description="KG-Enhanced ORE (NO-LAFF-WEIGHT, BM25.128 graph)")

# ORE parameters
parser.add_argument("--dl", type=int, default=19)
parser.add_argument("--budget", type=int, default=100)
parser.add_argument("--ce", type=int, default=7)
parser.add_argument("--s", type=int, default=30)
parser.add_argument("--s1", type=int, default=25)
parser.add_argument("--s2", type=int, default=15)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--verbose", action="store_true")

# KG scoring
parser.add_argument("--alpha", type=float, default=0.0, help="Graph weight (set 0 to disable)")
parser.add_argument("--beta", type=float, default=0.6, help="Entity overlap weight")
parser.add_argument("--gamma", type=float, default=0.4, help="KG connectivity weight")
parser.add_argument("--kg_mode", type=str, default="log", help="binary,count,log,coverage,ratio,raw")

# Data paths (REQUIRED)
parser.add_argument("--passage_el", type=str, default=None)
parser.add_argument("--freebase_1hop_db", type=str, default=None, help="SQLite db built from subgraph_1hop_triples.npy")
parser.add_argument("--query_el", type=str, default=None)
parser.add_argument("--passage_el_db", type=str, default=None, help="SQLite db for full corpus EL")

parser.add_argument("--mode", type=str, default="overwrite", choices=["overwrite", "reuse"])
args = parser.parse_args()

if not args.passage_el and not args.passage_el_db:
    print("ERROR: Provide either --passage_el (JSONL) or --passage_el_db (SQLite)")
    exit(1)

if args.gamma > 0 and not args.freebase_1hop_db:
    print("ERROR: --gamma > 0 requires --freebase_1hop_db (SQLite KG db)")
    exit(1)


if not pt.started():
    pt.java.init()

dataset = pt.get_dataset('irds:msmarco-passage')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("[1/5] TasB encoder...")
model = TasB.dot(batch_size=1, device=device)

print("[2/5] FlexIndex...")
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

print("[3/5] BM25 retriever...")
bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)

print("[4/5] Corpus graphs...")

graph16 = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
graph128 = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128')  # NOT LAFF

print("[5/5] MonoT5 scorer...")
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

from no_laff_try.ore_kg_nolaff import create_ore_kg

ore_kg = create_ore_kg(
    dual_encoder=model,
    scorer=scorer,
    corpus_index=idx,
    graph=graph16,
    laff_graph=graph128,   # bm25.128 used for neighbors+weights
    kg_alpha=args.alpha,
    kg_beta=args.beta,
    kg_gamma=args.gamma,
    kg_score_mode=args.kg_mode,
    freebase_1hop_db=args.freebase_1hop_db,
    passage_el_db=args.passage_el_db,
    budget=args.budget,
    cross_enc_budget=args.ce,
    top_s=args.s,
    top_s2=args.s2,
    verbose=args.verbose,
    param_bounds=(0.25, 0.95),
    num_bm25_calls=0,
)

exp_name = f"ORE_G{args.alpha}_E{args.beta}_K{args.gamma}.c{args.budget}.DL{args.dl}"
save_dir = f"runs/adaptive/dl{args.dl}/kg_ore/nolaff_bm25graph128/"
os.makedirs(save_dir, exist_ok=True)

result = pt.Experiment(
    [bm25 >> ore_kg],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[exp_name],
    save_dir=save_dir,
    save_mode=args.mode,
    verbose=True,
)

print(result.T)
print("\nExperiment complete:", exp_name)
print("Saved to:", save_dir)