"""Bounded, resumable final inference. No ground truth is read."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import joblib
import numpy as np
from .candidates import build_candidates, decode_target_code
from .pair_dataset import build_pair_dataset
from .retrieval import retrieve_sparse
from .model import predict_scores,load_model
from .pipeline import write_json, event


def digest(path):
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): sha.update(block)
    return sha.hexdigest()


def write_partition(candidate_dir, scores, threshold, output):
    folder=Path(candidate_dir); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    queries=[json.loads(line) for line in (folder/'queries.jsonl').open()]
    anchors=np.load(folder/'anchors.npy',mmap_mode='r')
    codes=np.load(folder/'target_codes.npy',mmap_mode='r')
    if len(scores)!=len(codes) or not np.isfinite(scores).all(): raise ValueError('Invalid scores')
    offsets=np.searchsorted(anchors,np.arange(len(queries)+1))
    paths={}
    for name,column,selected in [('candidate_pairs.tsv','candidate_entity_ids',False),('matching_results.tsv','matched_entity_ids',True)]:
        path=output/name; temp=path.with_suffix('.tmp')
        with temp.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['source1_entity_id',column])
            for i,q in enumerate(queries):
                start,end=offsets[i:i+2]
                values=codes[start:end]
                if selected: values=values[scores[start:end]>=threshold]
                w.writerow([q['entity_id'],','.join(decode_target_code(code) for code in values)])
        os.replace(temp,path);paths[name]=digest(path)
    return {'rows':len(queries),'candidate_pairs':len(codes),'selected_pairs':int((scores>=threshold).sum()),'sha256':paths}


def run_inference(test_dir,vectorizers_path,model_path,threshold,output_dir,cache_dir,*,
                  batch_anchors=10000,threads=4,k=20,target_batch=250000,keep_intermediates=False,
                  query_feature_limit=None,shared_index=True,feature_profile='base',roman_vectorizers_path=None):
    if not math.isfinite(threshold) or threshold<0: raise ValueError('Invalid threshold')
    if batch_anchors<1: raise ValueError('batch_anchors must be positive')
    if feature_profile not in ('base','advanced','reference'):raise ValueError('Unknown feature profile')
    if feature_profile!='base' and roman_vectorizers_path is None:raise ValueError('Romanized vectorizers required for enhanced profiles')
    from .pair_dataset import PAIR_FEATURE_NAMES
    feature_count=len(PAIR_FEATURE_NAMES)
    if feature_profile!='base':
        from .advanced_features import EXTRA_NAMES
        feature_count+=len(EXTRA_NAMES)+3
    if feature_profile=='reference':
        from .reference_context import NAMES
        feature_count+=len(NAMES)
    fitted=load_model(model_path)
    if fitted.n_features_in_!=feature_count:raise ValueError('Model and requested feature profile do not match')
    test=Path(test_dir); output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    sources={s:test/f'test_source{s}.tsv' for s in (1,2,3)}
    config={'inputs':{str(s):digest(p) for s,p in sources.items()},'vectorizers':digest(vectorizers_path),
            'model':digest(model_path),'threshold':threshold,'batch_anchors':batch_anchors,'threads':threads,
            'k':k,'target_batch':target_batch,'keep_intermediates':keep_intermediates,
            'query_feature_limit':query_feature_limit,'shared_index':shared_index,
            'feature_profile':feature_profile,
            'roman_vectorizers':digest(roman_vectorizers_path) if roman_vectorizers_path else None,
            'code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    configuration=output/'config.json'
    if configuration.exists() and json.loads(configuration.read_text())!=config:
        raise ValueError('Inference configuration changed; use a new output directory')
    write_json(configuration,config)
    vectorizers=joblib.load(vectorizers_path)
    index_path=None
    if shared_index:
        from .source_index import build_source_index
        index_path=build_source_index({s:sources[s] for s in (2,3)},Path(cache_dir)/'raw_records')['path']
    partitions=[]
    with sources[1].open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        for index in itertools.count():
            queries=list(itertools.islice(reader,batch_anchors))
            if not queries: break
            part=output/'partitions'/f'{index:06d}';part.mkdir(parents=True,exist_ok=True)
            done=part/'complete.json'
            query_hash=hashlib.sha256(json.dumps(queries,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if done.exists():
                meta=json.loads(done.read_text())
                if meta['query_hash']!=query_hash:raise ValueError('Query partition changed')
                for filename,checksum in meta['sha256'].items():
                    if digest(part/filename)!=checksum:raise ValueError('Partition output corrupted')
            else:
                retrieval={s:retrieve_sparse(queries,sources[s],vectorizers,part/f'retrieval_s{s}',k=k,
                    threads=threads,target_batch=target_batch,query_batch=512,
                    cache_dir=Path(cache_dir)/f'source{s}',progress=lambda m:event(partition=index,**m),
                    query_feature_limit=query_feature_limit) for s in (2,3)}
                build_candidates(queries,{s:sources[s] for s in (2,3)},retrieval,part/'candidates',source_index_path=index_path)
                features=build_pair_dataset(part/'candidates',{s:sources[s] for s in (2,3)},vectorizers,part/'features',source_index_path=index_path)
                feature_path=Path(features['X_path'])
                if feature_profile!='base':
                    from .augment import augment
                    augment(part/'candidates',part/'features',roman_vectorizers_path,part/'features_advanced')
                    feature_path=part/'features_advanced/X.npy'
                if feature_profile=='reference':
                    from .reference_context import augment_reference
                    augment_reference(part/'candidates',part/'features_advanced',part/'features',sources[1],
                        part/'features_reference',Path(cache_dir)/'reference_index')
                    feature_path=part/'features_reference/X.npy'
                X=np.load(feature_path,mmap_mode='r')
                scores=predict_scores(fitted,X)
                meta=write_partition(part/'candidates',scores,threshold,part)
                np.save(part/'scores.npy',scores)
                meta['sha256']['scores.npy']=digest(part/'scores.npy')
                meta['query_hash']=query_hash
                write_json(done,meta)
                del X,scores,retrieval
            partitions.append(meta)
            if not keep_intermediates:
                # Only task-owned regenerable directories; committed TSVs are the
                # complete candidate/prediction record, scores retain their order.
                for name in ('features','features_advanced','features_reference','candidates','retrieval_s2','retrieval_s3'):
                    if (part/name).exists():shutil.rmtree(part/name)
            event(stage='inference_partition_complete',partition=index,rows=meta['rows'])
    for filename in ('matching_results.tsv','candidate_pairs.tsv'):
        temp=output/(filename+'.tmp')
        with temp.open('wb') as dest:
            for index in range(len(partitions)):
                with (output/'partitions'/f'{index:06d}'/filename).open('rb') as source:
                    header=source.readline()
                    if index==0:dest.write(header)
                    shutil.copyfileobj(source,dest)
        os.replace(temp,output/filename)
    result={'complete':True,'rows':sum(p['rows'] for p in partitions),
            'candidate_pairs':sum(p['candidate_pairs'] for p in partitions),
            'selected_pairs':sum(p['selected_pairs'] for p in partitions),
            'sha256':{f:digest(output/f) for f in ('matching_results.tsv','candidate_pairs.tsv')},
            'validation_status':'Must run strict and official membership validators before packaging'}
    write_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('test-dir','vectorizers','model','output','cache'):p.add_argument('--'+name,required=True)
    p.add_argument('--threshold',type=float,required=True)
    p.add_argument('--batch-anchors',type=int,default=10000)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--k',type=int,default=20)
    p.add_argument('--target-batch',type=int,default=250000)
    p.add_argument('--keep-intermediates',action='store_true')
    p.add_argument('--query-feature-limit',type=int)
    p.add_argument('--no-shared-index',action='store_true')
    p.add_argument('--feature-profile',choices=['base','advanced','reference'],default='base')
    p.add_argument('--roman-vectorizers')
    a=p.parse_args()
    print(json.dumps(run_inference(a.test_dir,a.vectorizers,a.model,a.threshold,a.output,a.cache,
        batch_anchors=a.batch_anchors,threads=a.threads,k=a.k,target_batch=a.target_batch,
        keep_intermediates=a.keep_intermediates,query_feature_limit=a.query_feature_limit,
        shared_index=not a.no_shared_index,feature_profile=a.feature_profile,
        roman_vectorizers_path=a.roman_vectorizers),indent=2))

if __name__=='__main__':main()
