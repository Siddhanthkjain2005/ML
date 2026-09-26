"""Prepare identity-disjoint folds and training-only retrieval vocabulary."""
import argparse
import json
from pathlib import Path
from .data import prepare_splits
from .retrieval import build_vectorizers, _digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-zip',required=True)
    p.add_argument('--work',default='work')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();root=Path(args.work);splits=root/'splits_v1'
    checksum=_digest(Path(args.data_zip))
    manifest=splits/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['input']['sha256']!=checksum or meta['split_config']['seed']!=args.seed:
            raise ValueError('Existing split differs from input/seed; use a new work directory')
    else:
        prepare_splits(args.data_zip,splits,seed=args.seed,input_sha256=checksum,
                       progress=lambda value:print(value,flush=True))
    vocab=root/'vectorizers_v1.joblib'
    if vocab.exists():
        info=json.loads(Path(str(vocab)+'.manifest.json').read_text())
        if info['artifact_sha256']!=_digest(vocab) or info['seed']!=args.seed:
            raise ValueError('Existing vocabulary differs/corrupted')
        expected={str((splits/'train'/f'source{s}.tsv').resolve()) for s in (1,2,3)}
        if {item['path'] for item in info['training_files']}!=expected:
            raise ValueError('Vocabulary was fitted on different input files')
        for item in info['training_files']:
            if _digest(Path(item['path']))!=item['sha256']:raise ValueError('Training file changed')
    else:
        build_vectorizers([splits/'train'/f'source{s}.tsv' for s in (1,2,3)],vocab,seed=args.seed)
    print(json.dumps({'splits':str(splits),'vectorizers':str(vocab),'input_sha256':checksum}))

if __name__=='__main__':main()
