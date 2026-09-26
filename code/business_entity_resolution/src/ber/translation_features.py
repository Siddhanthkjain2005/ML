"""Frozen training-only token translation comparisons; no labels are read."""
import json,sqlite3
from functools import lru_cache
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein,JaroWinkler
from .token_translation import name_tokens,nonlatin,translate_name
from .romanization import romanize
from .pair_dataset import _lookup_records,_array_digest
from .pipeline import write_json,event
from .retrieval import _digest

NAMES=tuple('translated_name_'+n for n in ('exact','edit','jaro','ratio','partial','token_set','token_sort','jaccard','containment','left_coverage','right_coverage','right_nonlatin_fraction'))


def augment_translation(candidate_dir,base_dir,raw_dir,lexicon_path,output_dir,batch_size=10000):
 cand,base,raw,out=map(Path,(candidate_dir,base_dir,raw_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
 lexicon=json.loads(Path(lexicon_path).read_text());mapping=lexicon['mappings']
 if not mapping:raise ValueError('Translation lexicon has no accepted mappings')
 names=json.loads((base/'metadata.json').read_text())['feature_names']+list(NAMES)
 config={'base_X':_digest(base/'X.npy'),'lexicon':_digest(Path(lexicon_path)),'code':{f:_digest(Path(__file__).with_name(f)) for f in ('translation_features.py','token_translation.py','romanization.py')},'candidates':{f:_digest(cand/f) for f in ('anchors.npy','target_codes.npy','queries.jsonl')}}
 path=out/'X.npy';mp=out/'metadata.json'
 if mp.exists():
  meta=json.loads(mp.read_text())
  if meta['config']!=config:raise ValueError('Translation feature inputs changed')
  if meta['complete']:
   if _digest(path)!=meta['sha256']:raise ValueError('Translation features corrupted')
   return meta
 X=np.load(base/'X.npy',mmap_mode='r');a=np.load(cand/'anchors.npy',mmap_mode='r');codes=np.load(cand/'target_codes.npy',mmap_mode='r')
 queries=[json.loads(line)['business_name'] for line in (cand/'queries.jsonl').open()]
 if not mp.exists():
  meta={'config':config,'feature_names':names,'shape':[len(a),len(names)],'rows_completed':0,'batches':[],'complete':False};write_json(mp,meta)
 Y=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(a),len(names)))
 for b in meta['batches']:
  if _array_digest(Y[b['start']:b['end']])!=b['sha256']:raise ValueError('Translation checkpoint corrupted')
 lookup=raw/'targets.sqlite'
 if not lookup.exists():lookup=Path(json.loads((raw/'metadata.json').read_text())['target_lookup']['shared_source_index'])
 db=sqlite3.connect(f'file:{lookup.resolve()}?mode=ro',uri=True)
 @lru_cache(maxsize=50000)
 def view(text):
  tokens=name_tokens(text);foreign=sum(nonlatin(t) for t in tokens);mapped=sum(t in mapping for t in tokens)
  translated=romanize(translate_name(text,mapping))
  return translated,set(translated.split()),mapped/max(1,foreign),foreign/max(1,len(tokens))
 try:
  for start in range(meta['rows_completed'],len(a),batch_size):
   end=min(start+batch_size,len(a));targets=_lookup_records(db,np.unique(codes[start:end]));batch=np.empty((end-start,len(names)),np.float32);batch[:,:X.shape[1]]=X[start:end]
   for offset,(anchor,code) in enumerate(zip(a[start:end],codes[start:end])):
    left,lt,lc,_=view(queries[int(anchor)]);right,rt,rc,rf=view(targets[int(code)].name.normalized)
    if left and right:
     overlap=len(lt&rt);values=[float(left==right),Levenshtein.normalized_similarity(left,right),JaroWinkler.normalized_similarity(left,right),fuzz.ratio(left,right)/100,fuzz.partial_ratio(left,right)/100,fuzz.token_set_ratio(left,right)/100,fuzz.token_sort_ratio(left,right)/100,overlap/max(1,len(lt|rt)),overlap/max(1,min(len(lt),len(rt)))]
    else:values=[0.]*9
    batch[offset,X.shape[1]:]=values+[lc,rc,rf]
   if not np.isfinite(batch).all():raise ValueError('Nonfinite translation features')
   Y[start:end]=batch;Y.flush();meta['rows_completed']=end;meta['batches'].append({'start':start,'end':end,'sha256':_array_digest(batch)});write_json(mp,meta);event(stage='translation_features',rows=end,total=len(a),output=str(out))
 finally:db.close();Y.flush()
 meta.update(complete=True,sha256=_digest(path));write_json(mp,meta);return meta
