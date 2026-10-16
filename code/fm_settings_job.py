"""Bounded pueue task for the frozen195-fit study and dependent analysis."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path

CODE=Path(__file__).resolve().parent
OUT=CODE/'results/fm_training_settings_fixed40'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def reports():
    d=json.loads((OUT/'json/summary.json').read_text());cfg=json.loads((OUT/'json/frozen_config.json').read_text())
    assert d['status']=='audited' and d['fit_count']==195
    lines=['# FM training settings:fixed40','',
           '195 fits audited:13 one-factor-at-a-time settings,5 fixed training datasets,3 weight initializations.',
           'Each row uses median over the3 initializations within each data seed,then median [Q25,Q75] over5 data seeds.',
           '', '| Setting | Unseen RMSE | Unseen Spearman | True regret of FM-selected candidate |',
           '|---|---|---|---|']
    for name,metrics in d['settings'].items():
        vals=[]
        for metric in cfg['primary_metrics']:
            x=metrics[metric]
            vals.append(f"{x['median']:.6g} [{x['q25']:.6g},{x['q75']:.6g}]" if 'median' in x else 'undefined;all records retained')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['','Paired differences and36 Bonferroni-corrected Wilcoxon tests are stored in json/summary.json.',
              'All195 run records,checkpoints,predictions and csv/metrics.csv are retained.',
              'The4056 unseen candidates are exploratory evaluation data,not an independent final test after hyperparameter selection.',
              'Figures:fm_rank,fm_learning_rate,fm_weight_decay,fm_epochs,each PDF/SVG/PNG. Lines show5-data-seed medians,bands25–75th percentiles. Each figure changes one factor;all other factors remain at baseline. First panel shows train/unseen MSE on a log y-axis;second shows unseen Spearman;third shows true normalized regret of the exact prediction-selected candidate. Learning-rate and epoch x-axes are logarithmic;weight-decay uses symlog to include zero;rank is linear. No titles or bottom annotations.',
              'Manual visual review remains pending. No SA,QAOA,new optical simulation or BBO was executed by this study.']
    (OUT/'md').mkdir(exist_ok=True);(OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'md/DISCUSSION.md').write_text('# Interpretation limits\n\nFixed40 training points per data seed;all settings reuse the same five datasets. Model initialization repeats are not independent data seeds. This OFAT study does not identify interactions or a joint optimum. Evaluation-based setting selection needs confirmation on new data seeds. Lower prediction error,better ranking and better candidate quality are distinct outcomes. No sampler or closed-loop optimization advantage follows from these results. All failures and undefined correlations are retained. Rendered figure review is pending.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','audit']);args=parser.parse_args()
    protocol=json.loads((OUT/'json/execution_protocol.json').read_text());assert protocol['status']=='ready'
    for name,digest in protocol['files'].items():assert sha(CODE/name)==digest,name
    for name,digest in protocol['frozen_files'].items():assert sha(OUT/name)==digest,name
    cfg=OUT/'json/frozen_config.json';action='run' if args.stage=='train' else 'analyze'
    cmd=[sys.executable,'-u',str(CODE/'evaluate_fm_settings.py'),'--config',str(cfg),'--action',action]
    log=OUT/f'logs/{args.stage}.log';statepath=OUT/f'json/{args.stage}_execution.json'
    state={'status':'running','started_at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'argv':cmd,'cwd':str(CODE)}
    save(statepath,state);started=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(OUT/'logs/mplconfig'))
    try:
        with log.open('a') as f:
            subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=172800 if args.stage=='train' else 3600)
        if args.stage=='audit':reports()
        state['status']='completed'
    except BaseException:
        state.update(status='failed',traceback=traceback.format_exc())
        with log.open('a') as f:f.write(state['traceback'])
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        state['ended_at']=datetime.now(timezone(timedelta(hours=9))).isoformat();save(statepath,state)


if __name__=='__main__':main()
