import json,time
from pathlib import Path
import joblib,numpy as np
from ber.pipeline import sample_queries,read_truth,write_json,event
from ber.retrieval import retrieve_sparse
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=12);p.add_argument("--k",type=int,default=20);p.add_argument("--vectorizers",default="work/vectorizers_v1.joblib");p.add_argument("--cache",default="work/retrieval_cache_v1");p.add_argument("--prefix",default="approx_probe");p.add_argument("--cache-workers",type=int,default=1);p.add_argument("--rerank-pool",type=int);args=p.parse_args()
    tag=f"{args.prefix}_q{args.limit}_k{args.k}"+(f"_r{args.rerank_pool}" if args.rerank_pool else "")
    start=time.monotonic()
    queries=sample_queries('work/splits_v1/train/source1.tsv',1000)
    truth=read_truth('work/splits_v1/train/ground_truth.tsv',queries)
    v=joblib.load(args.vectorizers)
    result=retrieve_sparse(queries,'work/splits_v1/train/source2.tsv',v,f'experiments/{tag}_s2',k=args.k,threads=2,target_batch=250000,query_batch=512,cache_dir=Path(args.cache)/'train/source2',cache_workers=args.cache_workers,rerank_pool=args.rerank_pool,query_feature_limit=args.limit,progress=lambda m:event(**m))
    for attempt in range(20):
     if all(Path(f'experiments/pilot_v1/train/retrieval_s2/{b}.npz').exists() for b in v):break
     time.sleep(3)
    base={b:np.load(f'experiments/pilot_v1/train/retrieval_s2/{b}.npz') for b in v}
    lookup={str(q):i for i,q in enumerate(base['name_char']['query_ids'])}
    counts={name:{'true':0,'retrieved':0,'lost_vs_base':0,'gained_vs_base':0} for name in ('overall','US','India')}
    for i,q in enumerate(queries):
     row=lookup[q['entity_id']];t={int(z.split('-')[1]) for z in truth[q['entity_id']] if z.startswith('S2-')}
     a=set().union(*(set(result[b]['target_ids'][i].tolist()) for b in v));b=set().union(*(set(base[b]['target_ids'][row].tolist()) for b in v))
     for name in ('overall',q['country']):
      c=counts[name];c['true']+=len(t);c['retrieved']+=len(t&a);c['lost_vs_base']+=len((t&b)-a);c['gained_vs_base']+=len((t&a)-b)
    for c in counts.values():c['recall']=c['retrieved']/c['true'] if c['true'] else None
    report={'scope':'1000 training anchors; entire S2 training pool; sparse branches only; not validation','query_feature_limit':args.limit,'k':args.k,'runtime_seconds':time.monotonic()-start,'counts':counts}
    write_json(f'experiments/{tag}_report.json',report);print(json.dumps(report),flush=True)

if __name__=="__main__":main()
