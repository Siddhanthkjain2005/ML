"""Label-free evidence from competing raw Source1 reference records."""
import csv,json,math,os,sqlite3
from pathlib import Path
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from .text import legal_name_alternates,normalize_text
from .romanization import romanize
from .advanced_features import view,jaccard
from .pipeline import write_json,event
from .retrieval import _digest
from .pair_dataset import _lookup_records,_array_digest

NAMES=tuple(f'reference_{kind}_{stat}' for kind in ('name','core') for stat in ('other_count_log','overflow','other_address_ratio','other_address_token_set','other_address_token_sort','other_number_jaccard','other_joint','address_margin'))


def keys(name):
 return legal_name_alternates(romanize(name))


def build_reference_index(source1,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);db=output/'reference.sqlite';manifest=output/'manifest.json'
 config={'input_sha256':_digest(Path(source1)),'implementation':_digest(Path(__file__)),'text':_digest(Path(__file__).with_name('text.py')),'romanization':_digest(Path(__file__).with_name('romanization.py'))}
 if manifest.exists():
  meta=json.loads(manifest.read_text())
  if meta['config']!=config or _digest(db)!=meta['sha256']:raise ValueError('Reference index changed')
  return db
 temp=output/'reference.building.sqlite'
 if temp.exists():temp.unlink()
 conn=sqlite3.connect(temp);conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF')
 conn.execute('CREATE TABLE records(id TEXT PRIMARY KEY,name_key TEXT,core_key TEXT,address TEXT,country TEXT)');batch=[];count=0
 try:
  with Path(source1).open(encoding='utf-8',newline='') as f:
   for r in csv.DictReader(f,delimiter='\t'):
    name,core=keys(r['business_name']);batch.append((r['entity_id'],name,core,r['business_address'],r['country']));count+=1
    if len(batch)==20000:conn.executemany('INSERT INTO records VALUES(?,?,?,?,?)',batch);batch=[]
  if batch:conn.executemany('INSERT INTO records VALUES(?,?,?,?,?)',batch)
  conn.execute('CREATE INDEX names ON records(name_key)');conn.execute('CREATE INDEX cores ON records(core_key)');conn.commit()
 finally:conn.close()
 os.replace(temp,db);write_json(manifest,{'config':config,'sha256':_digest(db),'rows':count});return db


def owner_features(query,target,owners,overflow=False):
 # Caller excludes this query's own ID, so it is never its own competitor.
 qa=view(query.address.normalized)[1];ta=view(target.address.normalized)[1];tn=set(view(target.address.normalized)[2])
 current=fuzz.token_sort_ratio(qa,ta)/100 if qa and ta else 0.
 maxima=np.zeros(5,dtype=np.float32)
 for _,address,country in owners:
  _,a,n,_=view(address)
  if not ta or not a:continue
  ratios=np.array([fuzz.ratio(ta,a)/100,fuzz.token_set_ratio(ta,a)/100,fuzz.token_sort_ratio(ta,a)/100,jaccard(tn,set(n))],dtype=np.float32)
  # Country equality is open-set, with no country-specific interpretation.
  joint=float(ratios[[0,2,3]].mean())*float(normalize_text(country)==target.country)
  maxima=np.maximum(maxima,[*ratios,joint])
 return [math.log1p(len(owners)),float(overflow),*maxima,current-float(maxima[2])]


def augment_reference(candidate_dir,base_feature_dir,raw_feature_dir,source1,output_dir,index_dir,batch_size=10000):
 cand,base,raw,out=map(Path,(candidate_dir,base_feature_dir,raw_feature_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
 index=build_reference_index(source1,index_dir);refs=sqlite3.connect(f'file:{index.resolve()}?mode=ro',uri=True)
 lookup_path=raw/'targets.sqlite'
 if not lookup_path.exists():lookup_path=Path(json.loads((raw/'metadata.json').read_text())['target_lookup']['shared_source_index'])
 targets=sqlite3.connect(f'file:{lookup_path.resolve()}?mode=ro',uri=True)
 from .text import prepare_record
 queries=[prepare_record(json.loads(line)) for line in (cand/'queries.jsonl').open()]
 a=np.load(cand/'anchors.npy',mmap_mode='r');codes=np.load(cand/'target_codes.npy',mmap_mode='r');X=np.load(base/'X.npy',mmap_mode='r')
 names=json.loads((base/'metadata.json').read_text())['feature_names']+list(NAMES)
 config={'base_X':_digest(base/'X.npy'),'reference':_digest(index),'implementation':_digest(Path(__file__)),'candidates':{f:_digest(cand/f) for f in ('anchors.npy','target_codes.npy','queries.jsonl')}}
 path=out/'X.npy';mp=out/'metadata.json'
 if mp.exists():
  meta=json.loads(mp.read_text())
  if meta['config']!=config:raise ValueError('Reference context changed')
  if meta['complete']:
   if _digest(path)!=meta['sha256']:raise ValueError('Reference features corrupted')
   return meta
 else:meta={'config':config,'feature_names':names,'shape':[len(a),len(names)],'rows_completed':0,'batches':[],'complete':False};write_json(mp,meta)
 Y=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(a),len(names)))
 for b in meta['batches']:
  if _array_digest(Y[b['start']:b['end']])!=b['sha256']:raise ValueError('Reference feature checkpoint corrupted')
 @lru_cache(maxsize=30000)
 def lookup(kind,key):
  if not key:return (),False
  rows=refs.execute(f'SELECT id,address,country FROM records WHERE {kind}_key=? LIMIT 101',(key,)).fetchall()
  return ((),True) if len(rows)>100 else (tuple(rows),False)
 try:
  for start in range(meta['rows_completed'],len(a),batch_size):
   end=min(start+batch_size,len(a));records=_lookup_records(targets,np.unique(codes[start:end]));batch=np.empty((end-start,len(names)),dtype=np.float32);batch[:,:X.shape[1]]=X[start:end]
   for offset,(anchor,code) in enumerate(zip(a[start:end],codes[start:end])):
    q,t=queries[int(anchor)],records[int(code)];values=[]
    for kind,key in zip(('name','core'),keys(t.name.normalized)):
     owners,overflow=lookup(kind,key);owners=[r for r in owners if r[0]!=q.entity_id]
     values.extend(owner_features(q,t,owners,overflow))
    batch[offset,X.shape[1]:]=values
   Y[start:end]=batch;Y.flush();meta['rows_completed']=end;meta['batches'].append({'start':start,'end':end,'sha256':_array_digest(batch)});write_json(mp,meta);event(stage='reference_context',rows=end,total=len(a),output=str(out))
 finally:targets.close();refs.close();Y.flush()
 meta.update(complete=True,sha256=_digest(path));write_json(mp,meta);return meta
