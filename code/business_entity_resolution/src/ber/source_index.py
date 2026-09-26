"""Reusable raw-record/exact-key index for many inference query batches."""
import csv
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import pickle
from .text import normalize_text,prepare_record
from .candidates import encode_target_id,decode_target_code
from .retrieval import _digest
from .pipeline import write_json


def build_source_index(target_files,output_dir):
    """Build once from both complete target sources; atomic and content-checked."""
    if set(target_files)!={2,3}:raise ValueError('Expected both target sources')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    config={str(s):{'path':str(Path(p).resolve()),'sha256':_digest(Path(p))} for s,p in target_files.items()}
    config['normalizer_sha256']=_digest(Path(__file__).with_name('text.py'))
    config['schema_version']=2
    manifest=output/'manifest.json';db=output/'records.sqlite'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['config']!=config or _digest(db)!=meta['sha256']:raise ValueError('Source index changed or corrupted')
        return meta
    with tempfile.TemporaryDirectory(prefix='.building-',dir=output) as temporary:
        stage=Path(temporary)/'records.sqlite';conn=sqlite3.connect(stage)
        counts={}
        try:
            conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF')
            conn.execute('PRAGMA cache_size=-65536')
            conn.execute('CREATE TABLE records(code INTEGER PRIMARY KEY,name TEXT,address TEXT,country TEXT,name_key TEXT,address_key TEXT,prepared BLOB)')
            for s,p in target_files.items():
                count=0;batch=[]
                with Path(p).open(encoding='utf-8',newline='') as f:
                    r=csv.DictReader(f,delimiter='\t')
                    if r.fieldnames!=['entity_id','business_name','business_address','country']:raise ValueError('Unexpected source header')
                    for row in r:
                        code=encode_target_id(row['entity_id'])
                        if code//10**12!=s:raise ValueError('Wrong target source')
                        name,address=row['business_name'],row['business_address']
                        # This trusted, locally generated cache avoids repeating
                        # Unicode/token preparation for each candidate batch.
                        prepared=pickle.dumps(prepare_record(row),protocol=5)
                        batch.append((code,name,address,row['country'],normalize_text(name),normalize_text(address),prepared))
                        count+=1
                        if len(batch)==20000:
                            conn.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?)',batch);batch=[]
                if batch:conn.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?)',batch)
                conn.commit();counts[str(s)]=count
            conn.execute('CREATE INDEX name_keys ON records(name_key)')
            conn.execute('CREATE INDEX address_keys ON records(address_key)');conn.commit()
        finally:conn.close()
        # Detect a concurrent raw-input edit before publishing the index.
        for s,p in target_files.items():
            if _digest(Path(p))!=config[str(s)]['sha256']:raise ValueError('Target input changed during indexing')
        os.replace(stage,db)
    meta={'config':config,'path':str(db.resolve()),'counts':counts,'sha256':_digest(db)}
    write_json(manifest,meta)
    return meta


def iter_selected_records(index_path,source,required_codes=(),name_keys=(),address_keys=()):
    """Yield each exact-key or required-ID hit once without a full raw scan."""
    if source not in (2,3):raise ValueError('Invalid source')
    conn=sqlite3.connect(f'file:{Path(index_path).resolve()}?mode=ro',uri=True)
    try:
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('CREATE TEMP TABLE chosen(code INTEGER PRIMARY KEY)')
        conn.executemany('INSERT OR IGNORE INTO chosen VALUES(?)',((int(c),) for c in required_codes))
        for column,keys in (('name_key',name_keys),('address_key',address_keys)):
            for key in set(keys):
                if key:conn.execute(f'INSERT OR IGNORE INTO chosen SELECT code FROM records WHERE {column}=? AND code>=? AND code<?',(key,source*10**12,(source+1)*10**12))
        for code,name,address,country in conn.execute('SELECT r.code,r.name,r.address,r.country FROM chosen c JOIN records r ON r.code=c.code WHERE r.code>=? AND r.code<? ORDER BY r.code',(source*10**12,(source+1)*10**12)):
            yield {'entity_id':decode_target_code(code),'business_name':name,'business_address':address,'country':country}
    finally:conn.close()
