import json,time
from pathlib import Path
from ber.source_index import build_source_index


def main():
 start=time.monotonic()
 r=build_source_index({s:Path('work/test')/f'test_source{s}.tsv' for s in (2,3)},'work/test_cache/raw_records')
 print(json.dumps({'complete':True,'seconds':time.monotonic()-start,'index':r}),flush=True)

if __name__=='__main__':main()
