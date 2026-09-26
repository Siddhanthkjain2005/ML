import time,json
from ber.retrieval import build_vectorizers
from ber.romanization import representation_metadata,romanize
start=time.monotonic()
print(json.dumps(representation_metadata()),flush=True)
for example in ('कृष्ण हॉस्पिटैलिटी प्राइवेट लिमिटेड','ईस्ट कंसल्टेंसी प्राइवेट लिमिटेड','ഗോൾഡൻ ഇന്റർനാഷണൽ പ്രൈവറ്റ് ലിമിറ്റഡ്','Café Moderne'):
 print(json.dumps({'raw':example,'romanized':romanize(example)},ensure_ascii=False),flush=True)
v=build_vectorizers([f'work/splits_v1/train/source{s}.tsv' for s in (1,2,3)],'work/vectorizers_romanized_v1.joblib',representation='romanized')
print(json.dumps({'complete':True,'seconds':time.monotonic()-start,'vocabularies':{b:len(x.vocabulary_) for b,x in v.items()}}),flush=True)
