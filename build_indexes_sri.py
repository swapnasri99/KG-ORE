#import torch
#from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex
import pyterrier as pt
if not pt.started():
    pt.init()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("PyTerrier initialized successfully!")
### to create bm25 index. The bm25 index will created at path like  "/home/user_name/.pyterrier/corpora/msmarco_passage/index/terrier_stemmed"


bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
print("BM25 index ready!")

