import json
import csv
import numpy as np
import pytest
from ber.inference import write_partition
from ber.error_analysis import analyze


def fixture(tmp_path):
    folder=tmp_path/'candidates';folder.mkdir()
    queries=[dict(entity_id=f'S1-{i}',business_name='Business',business_address='1 Road',country='France') for i in range(1,4)]
    (folder/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
    for name,value in {'anchors':np.array([0,0,1],dtype=np.uint32),'target_codes':np.array([2*10**12+1,3*10**12+1,2*10**12+2],dtype=np.uint64),'labels':np.array([1,0,0],dtype=np.uint8),'truth_counts':np.array([2,0,0],dtype=np.uint32)}.items():
        np.save(folder/f'{name}.npy',value)
    return folder


def test_output_full_coverage_subset_and_multi_source(tmp_path):
    folder=fixture(tmp_path); out=tmp_path/'outputs'
    write_partition(folder,np.array([.9,.8,.1]),.5,out)
    with (out/'matching_results.tsv').open() as f: rows=list(csv.reader(f,delimiter='\t'))
    assert rows[1]==['S1-1','S2-1,S3-1']
    assert rows[2:]==[['S1-2',''],['S1-3','']]
    with (out/'candidate_pairs.tsv').open() as f: candidates=list(csv.reader(f,delimiter='\t'))
    assert candidates[2]==['S1-2','S2-2']
    with pytest.raises(ValueError):write_partition(folder,np.array([float('nan'),.8,.1]),.5,out)


def test_analysis_counts_unretrieved_positives_and_empty_singletons(tmp_path):
    folder=fixture(tmp_path); scores=tmp_path/'scores.npy';np.save(scores,np.array([.9,.1,.8]))
    result=analyze(folder,scores,.5,tmp_path/'analysis')
    assert result['overall']['false_positives']==1
    assert result['overall']['false_negatives']==1
    assert result['overall']['unretrieved_true_links']==1
    assert result['overall']['macro_f0_5']==pytest.approx((1.25/1.5+0+1)/3)
