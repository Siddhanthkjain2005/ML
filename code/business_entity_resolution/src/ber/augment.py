"""Append label-independent alternate-text features to a saved pair dataset."""
import json,sqlite3
from pathlib import Path
import joblib,numpy as np
from .advanced_features import EXTRA_NAMES,extra_pair_features
from .pair_dataset import _lookup_records,_array_digest
from .pipeline import write_json,event
from .retrieval import _digest,BRANCHES
from .text import prepare_record
from .romanization import representation_metadata


def augment(candidate_dir,feature_dir,roman_vectorizers_path,output_dir,batch_size=10000):
    cand,base,out=map(Path,(candidate_dir,feature_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
    base_meta=json.loads((base/'metadata.json').read_text());vectors=joblib.load(roman_vectorizers_path)
    profile={'base_X':_digest(base/'X.npy'),'candidates':{f:_digest(cand/f) for f in ('anchors.npy','target_codes.npy','queries.jsonl')},'roman_vectors':_digest(Path(roman_vectorizers_path)),'code':{f:_digest(Path(__file__).with_name(f)) for f in ('augment.py','advanced_features.py','romanization.py')},'representation':representation_metadata(),'batch_size':batch_size}
    names=base_meta['feature_names']+list(EXTRA_NAMES)+['roman_'+b+'_cosine' for b in BRANCHES]
    X=np.load(base/'X.npy',mmap_mode='r');a=np.load(cand/'anchors.npy',mmap_mode='r');codes=np.load(cand/'target_codes.npy',mmap_mode='r')
    queries=[prepare_record(json.loads(line)) for line in (cand/'queries.jsonl').open()]
    meta_path=out/'metadata.json';path=out/'X.npy'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['profile']!=profile:raise ValueError('Augmentation inputs changed')
        if meta['complete']:
            if _digest(path)!=meta['sha256']:raise ValueError('Augmented matrix corrupted')
            return meta
    else:
        meta={'profile':profile,'feature_names':names,'shape':[len(a),len(names)],'complete':False,'rows_completed':0,'batches':[]}
        write_json(meta_path,meta)
    Y=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(a),len(names)))
    for b in meta['batches']:
        if _array_digest(Y[b['start']:b['end']])!=b['sha256']:raise ValueError('Augmentation checkpoint corrupted')
    lookup_path=base/'targets.sqlite'
    if not lookup_path.exists():lookup_path=Path(base_meta['target_lookup']['shared_source_index'])
    db=sqlite3.connect(f'file:{lookup_path.resolve()}?mode=ro',uri=True)
    try:
        for start in range(meta['rows_completed'],len(a),batch_size):
            end=min(start+batch_size,len(a));batch=np.empty((end-start,len(names)),dtype=np.float32);batch[:,:X.shape[1]]=X[start:end]
            unique_codes,inverse=np.unique(codes[start:end],return_inverse=True);targets=_lookup_records(db,unique_codes);records=[targets[int(c)] for c in unique_codes]
            unique_a,qi=np.unique(a[start:end],return_inverse=True);left=[queries[int(i)] for i in unique_a]
            for offset,(anchor,code) in enumerate(zip(a[start:end],codes[start:end])):
                batch[offset,X.shape[1]:X.shape[1]+len(EXTRA_NAMES)]=extra_pair_features(queries[int(anchor)],targets[int(code)])
            for j,b in enumerate(BRANCHES):
                field='address' if b=='address_char' else 'name'
                lv=vectors[b].transform([getattr(r,field).normalized for r in left]);rv=vectors[b].transform([getattr(r,field).normalized for r in records])
                batch[:,-3+j]=np.clip(np.asarray(lv[qi].multiply(rv[inverse]).sum(axis=1)).ravel(),0,1)
            if not np.isfinite(batch).all():raise ValueError('Nonfinite augmented features')
            Y[start:end]=batch;Y.flush();meta['batches'].append({'start':start,'end':end,'sha256':_array_digest(batch)});meta['rows_completed']=end;write_json(meta_path,meta)
            event(stage='augmented_features',output=str(out),rows=end,total=len(a))
    finally:db.close();Y.flush()
    meta.update(complete=True,sha256=_digest(path));write_json(meta_path,meta);return meta
