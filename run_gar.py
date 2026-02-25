"""
Minimal GAR (Graph Adaptive Re-ranking) experiment
With SSL workaround for corporate networks

Requirements:
    pip install pyterrier-adaptive pyterrier-t5 pyterrier-alpha

Usage:
    python run_gar_minimal_ssl_fix.py --dl 19 --budget 100
"""

# =============================================================================
# SSL WORKAROUND - Add this BEFORE any other imports
# =============================================================================
import os
import ssl
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Disable SSL verification for HuggingFace Hub
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Disable SSL verification for urllib
ssl._create_default_https_context = ssl._create_unverified_context

# For requests library
import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

# =============================================================================
# Now the actual imports
# =============================================================================
import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
from baselines.gar import GAR
from pyterrier_t5 import MonoT5ReRanker

import torch
import argparse
import pandas as pd
import random
random.seed(42)

# Initialize PyTerrier (REQUIRED)
if not pt.started():
    pt.init()

parser = argparse.ArgumentParser()
parser.add_argument("--dl", type=int, default=19, help="TREC-DL year: 19 or 20")
parser.add_argument("--budget", type=int, default=100, help="Re-ranking budget (c)")
parser.add_argument("--batch", type=int, default=16, help="Batch size for MonoT5")
parser.add_argument("--graph_k", type=int, default=8, help="Number of neighbors in corpus graph (default: 8)")
parser.add_argument("--verbose", action="store_true", help="Show progress bars")

args = parser.parse_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# 1. Load MS MARCO passage dataset (for text retrieval)
# =============================================================================
dataset = pt.get_dataset('irds:msmarco-passage')

# =============================================================================
# 2. BM25 Retriever (using pre-built index from PyTerrier)
# =============================================================================
print("Loading BM25 retriever...")
retriever = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')

# =============================================================================
# 3. MonoT5 Scorer (reranker)
# =============================================================================
print("Loading MonoT5 reranker...")
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=args.verbose, batch_size=args.batch)

# =============================================================================
# 4. Load Pre-built Corpus Graph from HuggingFace
#    This is a BM25-based nearest neighbor graph with k=16
#    We can limit it to fewer neighbors using to_limit_k()
# =============================================================================
print(f"Loading corpus graph (k={args.graph_k}) from HuggingFace...")
# This downloads ~2GB on first run, then cached
graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')

# Limit to requested k neighbors
if args.graph_k < 16:
    graph = graph.to_limit_k(args.graph_k)



# =============================================================================
# 5. Load evaluation dataset (TREC-DL 2019 or 2020)
# =============================================================================
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

# =============================================================================
# 6. Run Experiment: BM25+MonoT5 vs BM25+GAR(MonoT5)
# =============================================================================
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)

print(f"\n{'='*60}")
print(f"Running experiment on TREC-DL-20{args.dl}")
print(f"Budget: {args.budget}, Graph neighbors: {args.graph_k}")
print(f"{'='*60}\n")

result = pt.Experiment(
    [   
        # Baseline: BM25 top-c → MonoT5 rerank
        retriever % args.budget >> scorer,
        
        # GAR: BM25 → GAR(MonoT5, corpus_graph) with budget c
        retriever >> GAR(scorer, graph, num_results=args.budget, verbose=args.verbose),
    ],
    eval_dataset.get_topics(),
    eval_dataset.get_qrels(),
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[
        f"BM25_MonoT5.c{args.budget}",
        f"BM25_GAR.c{args.budget}.k{args.graph_k}",
    ],
    verbose=True
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(result.T)

