import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_t5 import MonoT5ReRanker
from pyterrier.measures import *
from ir_measures import nDCG, R
from baselines.gar import GAR
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if not pt.started():
    pt.init()

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

# BM25
bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
)

# MonoT5
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(batch_size=16)

from pyterrier_adaptive import CorpusGraph
artifact = pta.Artifact.from_hf("macavaney/msmarco-passage.corpusgraph.bm25.16")
graph_path = artifact.path           # local cached directory
corpus_graph = CorpusGraph.load(graph_path)

gar = GAR(
    scorer=scorer,
    corpus_graph=corpus_graph,
    num_results=100,
    cross_enc_budget=4,
    verbose=True
)

result = pt.Experiment(
    [bm25 % 100, gar % 100],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@100, R(rel=2)@100],
    names=["BM25", "GAR"]
)

print(result)