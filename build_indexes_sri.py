# This script initializes PyTerrier and prepares the BM25 index for the MSMARCO passage dataset.
# It can be used to build the necessary indexes for the ORE experiments.
import pyterrier as pt
if not pt.started():
    pt.init()


print("PyTerrier initialized successfully!")
bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
print("BM25 index ready!")

