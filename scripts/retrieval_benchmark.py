import json,time
from pathlib import Path
import joblib,numpy as np
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn
from ber.pipeline import sample_queries
from ber.text import normalize_text
v=joblib.load('work/vectorizers_v1.joblib')
q=sample_queries('work/splits_v1/train/source1.tsv',100)
cache=next(p for p in Path('work/retrieval_cache_v1/train/source2').iterdir() if (p/'manifest.json').exists())
for b in v:
 t=time.monotonic();x=sparse.load_npz(cache/f'000000.{b}.npz').tocsr();y=v[b].transform([normalize_text(z['business_address' if b=='address_char' else 'business_name']) for z in q]);xt=x.T.tocsr()
 r=sp_matmul_topn(y,xt,top_n=21,threshold=0.,sort=True,n_threads=1)
 ties=sum(len(r.data[r.indptr[i]:r.indptr[i+1]])>20 and r.data[r.indptr[i]+19]==r.data[r.indptr[i]+20] for i in range(len(q)))
 print(json.dumps(dict(branch=b,seconds=time.monotonic()-t,ties=int(ties),queries=len(q),targets=x.shape[0],nnz=x.nnz)),flush=True)
from ber.retrieval import _multiply_topk,_merge_topk
for b in v:
 x=sparse.load_npz(cache/f'000000.{b}.npz').tocsr();y=v[b].transform([normalize_text(z['business_address' if b=='address_char' else 'business_name']) for z in q]);xt=x.T.tocsr();ids=np.arange(x.shape[0]);t=time.monotonic()
 for a,bscore in _multiply_topk(y,x,xt,ids,20,4):
  _merge_topk(np.full(20,-1),np.zeros(20),a,bscore,20)
 print(json.dumps(dict(branch=b,full_seconds=time.monotonic()-t,queries=len(q))),flush=True)
from ber.retrieval import distinctive_query_matrix
for limit in (4,8,12):
 for b in v:
  x=sparse.load_npz(cache/f'000000.{b}.npz').tocsr();y=v[b].transform([normalize_text(z['business_address' if b=='address_char' else 'business_name']) for z in q]);y=distinctive_query_matrix(y,v[b].idf_,limit);xt=x.T.tocsr();ids=np.arange(x.shape[0]);t=time.monotonic()
  for a,bscore in _multiply_topk(y,x,xt,ids,20,4):_merge_topk(np.full(20,-1),np.zeros(20),a,bscore,20)
  print(json.dumps(dict(branch=b,query_features=limit,full_seconds=time.monotonic()-t,queries=len(q))),flush=True)
