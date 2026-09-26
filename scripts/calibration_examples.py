import json,sqlite3
from pathlib import Path
import numpy as np
root=Path('experiments/pilot_v1/calibration');q=[json.loads(s) for s in (root/'candidates/queries.jsonl').open()]
a=np.load(root/'candidates/anchors.npy');c=np.load(root/'candidates/target_codes.npy');y=np.load(root/'candidates/labels.npy');p=np.load(root/'xgboost_scores.npy');db=sqlite3.connect(root/'features/targets.sqlite')
result={}
for group,indices in [('false_positive',np.flatnonzero((y==0)&(p>=.366))),('false_negative',np.flatnonzero((y==1)&(p<.366)))]:
 selected=indices[np.argsort(-p[indices] if group=='false_positive' else p[indices])[:25]]
 result[group]=[dict(query=q[int(a[i])],target=db.execute('SELECT name,address,country FROM records WHERE code=?',(int(c[i]),)).fetchone(),score=float(p[i])) for i in selected]
Path('experiments/pilot_v1/calibration_examples.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({k:v[:6] for k,v in result.items()},indent=2,ensure_ascii=False))
