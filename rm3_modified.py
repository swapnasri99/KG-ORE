import pyterrier as pt
from pyterrier.measures import *
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd
import argparse
import os
from ir_measures import R, nDCG

class RemoveColon(pt.Transformer):
    def transform(self, topics):
        topics = topics.copy()
        topics["query"] = topics["query"].str.replace(":", " ", regex=False)
        return topics

parser = argparse.ArgumentParser()
parser.add_argument("--dl", type=int, required=True, help="19 or 20")
parser.add_argument("--budget", type=int, required=True, help="50 or 100")
parser.add_argument("--mode", type=str, default="reuse", help="reuse or overwrite")
args = parser.parse_args()

# dataset + qrels
dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")

# BM25 index

pt.java.init()

bm25 = pt.terrier.Retriever.from_dataset(
    "msmarco_passage",
    "terrier_stemmed",
    wmodel="BM25"
)

existing_index = pt.IndexFactory.of(bm25.indexref)

# first-stage BM25
bm25 = pt.terrier.Retriever(existing_index, wmodel="BM25")

# MonoT5 reranker
text_getter = pt.text.get_text(pt.get_dataset("irds:msmarco-passage"), "text")
scorer = text_getter >> MonoT5ReRanker(verbose=False, batch_size=16)

c = args.budget

# RM3 pipeline: BM25 -> RM3 -> BM25
rm3_pipe = (
    bm25 % c
    >> pt.rewrite.RM3(existing_index, fb_terms=10, fb_docs=10, fb_lambda=0.6)
    >> bm25 % c
)

# save directory
save_dir = f"runs/rm3/dl{args.dl}/c{c}/"
os.makedirs(save_dir, exist_ok=True)

result = pt.Experiment(

    [
        RemoveColon() >> rm3_pipe % c >> pt.rewrite.reset() >> scorer,
    ],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@c, R(rel=2)@c],
    names=[
        f"BM25_RM3_MonoT5.c{c}",
    ],
    save_dir=save_dir,
    save_mode=args.mode,
    verbose=True,
)

print(result.T)