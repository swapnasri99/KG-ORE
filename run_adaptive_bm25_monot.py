import pyterrier as pt
from ir_measures import nDCG, R
from pyterrier_t5 import MonoT5ReRanker

import torch
import argparse
import pandas as pd
import random
random.seed(42)

if not pt.started():
    pt.init()

parser = argparse.ArgumentParser()
parser.add_argument("--dl", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--batch", type=int, default=16, help="batch size for MonoT5")

args = parser.parse_args()

dataset = pt.get_dataset('irds:msmarco-passage')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# BM25 retriever
retriever = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')

# MonoT5 scorer
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=True, batch_size=args.batch)

# Load evaluation dataset
dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)

print(f"\nRunning BM25 + MonoT5 on TREC-DL-20{args.dl}, budget={args.budget}")

result = pt.Experiment(
        [   
            retriever % args.budget >> scorer,   # BM25 + MonoT5
        ],
        dataset.get_topics(),
        dataset.get_qrels(),
        [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
        names=[
            f"BM25_MonoT5.c{args.budget}",
        ],
        verbose=True
    )

print("\n--- Results ---")
print(result.T)