import json,time
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from ber.pipeline import write_json,load_arrays,event
from ber.model import select_decision,predict_scores
from ber.metrics import iter_threshold_array_metrics


def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument("--feature-dir",default="features");parser.add_argument("--output",default="experiments/scorer_refinement_v1");args=parser.parse_args()
 root=Path('experiments/pilot_v1');out=Path(args.output);out.mkdir(exist_ok=True)
 X=np.load(root/'train'/args.feature_dir/'X.npy',mmap_mode='r');y=np.load(root/'train/candidates/labels.npy',mmap_mode='r')
 cal=load_arrays(root/'calibration/candidates');XC=np.load(root/'calibration'/args.feature_dir/'X.npy',mmap_mode='r')
 variants={'depth7':dict(n_estimators=1000,max_depth=7,learning_rate=.05,min_child_weight=5,reg_lambda=3.),'depth5_long':dict(n_estimators=1400,max_depth=5,learning_rate=.05,min_child_weight=5,reg_lambda=3.),'depth9':dict(n_estimators=800,max_depth=9,learning_rate=.06,min_child_weight=10,reg_lambda=5.)}
 decisions={}
 for name,params in variants.items():
  folder=out/name;folder.mkdir(exist_ok=True);model=folder/'model.json';done=folder/'calibration.json'
  if done.exists():decisions[name]=json.loads(done.read_text());continue
  start=time.monotonic();m=XGBClassifier(**params,tree_method='hist',objective='binary:logistic',n_jobs=4,random_state=42,subsample=.9,colsample_bytree=.9)
  if model.exists():m.load_model(model)
  else:
   with threadpool_limits(limits=4):m.fit(X,y)
   m.save_model(model)
  scores=predict_scores(m,XC);np.save(folder/'calibration_scores.npy',scores)
  d=select_decision(cal['truth_counts'],cal['anchors'],cal['target_codes'],cal['labels'],scores)
  decisions[name]=d;write_json(done,d);write_json(folder/'metadata.json',{'feature_directory':args.feature_dir,'feature_names':json.loads((root/'train'/args.feature_dir/'metadata.json').read_text())['feature_names'],'parameters':params,'license':'Apache-2.0','training_partition':'train','selection_partition':'calibration','seconds':time.monotonic()-start})
  event(stage='scorer_calibration',name=name,**d)
 best=max(decisions,key=lambda n:decisions[n]['macro_f0_5'])
 write_json(out/'frozen_selection.json',{'name':best,'decisions':decisions,'basis':'calibration only'})
 if args.feature_dir in ('features_advanced_v1','features_reference_v1'):
  from ber.augment import augment
  augment(root/'validation/candidates',root/'validation/features','work/vectorizers_romanized_v1.joblib',root/'validation/features_advanced_v1')
 if args.feature_dir=='features_reference_v1':
  from ber.reference_context import augment_reference
  augment_reference(root/'validation/candidates',root/'validation/features_advanced_v1',root/'validation/features','work/splits_v1/validation/source1.tsv',root/'validation/features_reference_v1','work/reference_indices/validation')
 val=load_arrays(root/'validation/candidates');XV=np.load(root/'validation'/args.feature_dir/'X.npy',mmap_mode='r')
 scores=predict_scores(out/best/'model.json',XV);np.save(out/best/'validation_scores.npy',scores)
 result=next(iter_threshold_array_metrics(val['truth_counts'],val['anchors'],val['target_codes'],val['labels'],scores,[decisions[best]['threshold']]))
 write_json(out/'result.json',{'selected_model':best,'validation':result,'calibration':decisions[best]});event(stage='refinement_complete',selected_model=best,validation=result)

if __name__=='__main__':main()
