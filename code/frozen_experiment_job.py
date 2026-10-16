"""Generic bounded job for a locally frozen experiment protocol."""
import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--stage',required=True);args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text());assert protocol['status']=='ready'
    cwd=Path(protocol['cwd']);out=Path(protocol['output_dir']);stage=protocol['stages'][args.stage]
    for name,digest in protocol['files'].items():assert sha(cwd/name)==digest,name
    for name,digest in protocol['frozen_files'].items():assert sha(out/name)==digest,name
    (out/'logs').mkdir(exist_ok=True)
    status_path=out/f'json/{args.stage}_execution.json';log_path=out/f'logs/{args.stage}.log'
    state={'status':'running','started_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
           'command':stage['command'],'cwd':str(cwd),'protocol_sha256':sha(args.protocol)}
    def save():
        tmp=status_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');tmp.replace(status_path)
    save();start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(out/'logs/mplconfig'))
    try:
        with log_path.open('a') as log:
            subprocess.run(stage['command'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=stage['timeout_seconds'],check=True)
        state['status']='completed'
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc())
        with log_path.open('a') as log:log.write(state['traceback'])
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-start
        state['ended_at']=datetime.now(timezone(timedelta(hours=9))).isoformat();save()


if __name__=='__main__':main()
