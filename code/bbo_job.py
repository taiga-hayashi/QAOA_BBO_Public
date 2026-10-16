"""Bounded pueue entry point; validate frozen inputs and retain failures."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

CODE=Path(__file__).resolve().parent
OUT=CODE/'results/multilayer_filter_bbo_pilot5'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calculate','analyze']);parser.add_argument('--seed',type=int);args=parser.parse_args()
    protocol=json.loads((OUT/'json/protocol.json').read_text());assert protocol['status']=='ready'
    tag=args.stage
    if args.stage=='calculate':
        assert str(args.seed) in protocol['per_seed_configs']
        cfg=OUT/f'json/run_config_seed{args.seed}.json'
        assert sha(cfg)==protocol['per_seed_configs'][str(args.seed)]
        tag=f'calculate_seed{args.seed}'
    else:
        cfg=OUT/'json/run_config.json';assert sha(cfg)==protocol['config_sha256']
    for name,digest in protocol['sources'].items():assert sha(CODE/name)==digest,name
    settings=json.loads(cfg.read_text())
    assert sha(CODE/settings['problem']['data_path'])==protocol['dataset_sha256']
    assert sha(CODE/settings['problem']['config_path'])==protocol['schema_sha256']
    assert json.loads((OUT/'json/prevalidation.json').read_text())['status']=='passed'
    target='main.py' if args.stage=='calculate' else 'analyze_bbo.py'
    log=OUT/f'logs/{tag}.log';status=OUT/f'json/{tag}_execution.json'
    state={'status':'running','started_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
           'command':[sys.executable,'-u',str(CODE/target),'--config',str(cfg)],'cwd':str(CODE),
           'protocol_sha256':sha(OUT/'json/protocol.json')}
    status.write_text(json.dumps(state,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR=str(OUT/'logs/mplconfig'),VECLIB_MAXIMUM_THREADS='1')
    started=time.monotonic()
    try:
        with log.open('a') as stream:
            subprocess.run(state['command'],cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,
                           timeout=172800 if args.stage=='calculate' else 3600,check=True)
        state['status']='completed'
    except BaseException:
        state['status']='failed';state['traceback']=traceback.format_exc()
        with log.open('a') as stream:stream.write(state['traceback'])
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        state['ended_at']=datetime.now(timezone(timedelta(hours=9))).isoformat()
        status.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
