import csv
import pytest
from ber.source_index import build_source_index,iter_selected_records


def test_index_exact_union_and_required_ids(tmp_path):
    paths={}
    for s in (2,3):
        p=tmp_path/f'source{s}.tsv';paths[s]=p
        with p.open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country'])
            w.writerows([(f'S{s}-1','Acme','12 Road','France'),(f'S{s}-2','nan','','India'),(f'S{s}-3','Other','23 Street','US')])
    meta=build_source_index(paths,tmp_path/'index')
    assert build_source_index(paths,tmp_path/'index')==meta
    rows=list(iter_selected_records(meta['path'],2,[2*10**12+2],['acme'],['12 road']))
    assert [r['entity_id'] for r in rows]==['S2-1','S2-2']
    assert rows[1]['business_name']=='nan'
    with paths[2].open('a') as f:f.write('S2-4\tNew\t\tUS\n')
    with pytest.raises(ValueError):build_source_index(paths,tmp_path/'index')


def test_shared_index_candidates_and_features_equal_streaming(tmp_path):
    import numpy as np
    from ber.retrieval import build_vectorizers,retrieve_sparse
    from ber.candidates import build_candidates
    from ber.pair_dataset import build_pair_dataset
    paths={}
    for s in (2,3):
        p=tmp_path/f'source{s}.tsv';paths[s]=p
        with p.open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country'])
            w.writerows([(f'S{s}-1','Acme Labs','12 Road','France'),(f'S{s}-2','nan','','India'),(f'S{s}-3','Other Labs','23 Street','US')])
    queries=[dict(entity_id='S1-1',business_name='Acme Labs',business_address='12 Road',country='France'),dict(entity_id='S1-2',business_name='nan',business_address='',country='India')]
    v=build_vectorizers(list(paths.values()),tmp_path/'v.joblib',min_df=1,max_df=1.0)
    r={s:retrieve_sparse(queries,p,v,tmp_path/f'r{s}',k=2,threads=1) for s,p in paths.items()}
    index=build_source_index(paths,tmp_path/'index')
    a=tmp_path/'a';b=tmp_path/'b'
    build_candidates(queries,paths,r,a)
    build_candidates(queries,paths,r,b,source_index_path=index['path'])
    for name in ('anchors','target_codes','masks','ranks','scores'):
        np.testing.assert_array_equal(np.load(a/f'{name}.npy'),np.load(b/f'{name}.npy'))
    x=build_pair_dataset(a,paths,v,tmp_path/'fa')
    y=build_pair_dataset(b,paths,v,tmp_path/'fb',source_index_path=index['path'])
    np.testing.assert_array_equal(np.load(x['X_path']),np.load(y['X_path']))
