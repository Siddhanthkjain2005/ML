"""Extract only the three official test TSV members, without path traversal."""
import argparse,json,os,shutil,zipfile
from pathlib import Path
from .retrieval import _digest
from .pipeline import write_json


def extract_test(archive,output_dir):
 output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);manifest=output/'input_manifest.json'
 archive_hash=_digest(Path(archive))
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['archive_sha256']!=archive_hash:raise ValueError('Input archive changed')
  for name,data in m['files'].items():
   if _digest(output/name)!=data['sha256']:raise ValueError('Extracted test file corrupted')
  return m
 files={}
 with zipfile.ZipFile(archive) as z:
  for source in (1,2,3):
   name=f'test_source{source}.tsv'
   members=[n for n in z.namelist() if n.endswith('/dataset/test/'+name) and not n.startswith('__MACOSX/')]
   if len(members)!=1:raise ValueError(f'Expected exactly one official test member for {name}')
   target=output/name;temp=output/(name+'.tmp')
   with z.open(members[0]) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
   os.replace(temp,target);files[name]={'member':members[0],'bytes':target.stat().st_size,'sha256':_digest(target)}
 m={'archive_sha256':archive_hash,'files':files};write_json(manifest,m);return m


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-zip',required=True);p.add_argument('--output',default='work/test');a=p.parse_args();print(json.dumps(extract_test(a.data_zip,a.output),indent=2))

if __name__=='__main__':main()
