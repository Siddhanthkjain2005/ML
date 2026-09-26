import csv,json,time
from pathlib import Path
from ber.pipeline import sample_queries,read_truth,event,write_json
from ber.token_translation import fit_translation,nonlatin,translate_name


def main():
 start=time.monotonic();queries=sample_queries('work/splits_v1/train/source1.tsv',150000,42);truth=read_truth('work/splits_v1/train/ground_truth.tsv',queries)
 byid={q['entity_id']:q['business_name'] for q in queries}
 owners={t:(byid[q],q) for q,targets in truth.items() for t in targets};pairs=[]
 for s in (2,3):
  with Path(f'work/splits_v1/train/source{s}.tsv').open(encoding='utf-8',newline='') as f:
   for r in csv.DictReader(f,delimiter='\t'):
    if r['entity_id'] in owners and nonlatin(r['business_name']):pairs.append((owners[r['entity_id']][0],r['business_name'],owners[r['entity_id']][1]))
  event(stage='translation_pairs',source=s,pairs=len(pairs))
 model=fit_translation(pairs,'work/token_translation_v1.json')
 from rapidfuzz import fuzz
 rows=[{'left':a,'right':b,'translated':translate_name(b,model['mappings']),'before':fuzz.token_sort_ratio(a.casefold(),b.casefold()),'after':fuzz.token_sort_ratio(a.casefold(),translate_name(b,model['mappings']))} for a,b,_ in pairs[:50]]
 write_json('experiments/translation_fit.json',{'training_only':True,'seconds':time.monotonic()-start,'sampled_anchors':len(queries),'positive_nonlatin_pairs':len(pairs),'mapped_tokens':len(model['mappings']),'examples':rows})
 event(stage='translation_fit_complete',mapped_tokens=len(model['mappings']),seconds=time.monotonic()-start)

if __name__=='__main__':main()
