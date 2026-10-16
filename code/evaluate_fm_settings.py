"""Paired fixed-data FM training evaluation; reuses the production FM learner.

Default action is plan only. Run/analyze must be requested explicitly.
No SA, QAOA, optical simulation, or closed-loop BBO is invoked.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from config import CODE_DIR
from evaluator import load_problem
from fm_model import train_fm, model_sha256


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def resolve(path):return (CODE_DIR/Path(path)).resolve()

def variants(cfg):
    base=cfg['baseline'];out=[('baseline',dict(base),'baseline',None)]
    for field,values in cfg['grid'].items():
        if field not in {'rank','learning_rate','weight_decay','epochs'}:raise ValueError(field)
        for value in values:
            if value==base[field]:continue
            params=dict(base);params[field]=value
            if params['rank']<1 or params['epochs']<1 or params['learning_rate']<=0 or params['weight_decay']<0:
                raise ValueError('Invalid training parameters')
            out.append((f'{field}_{value:g}',params,field,value))
    if len({v[0] for v in out})!=len(out):raise ValueError('Duplicate configurations')
    return out

def data_split(ids,n):
    ids=np.asarray(ids,dtype=int)
    if ids.ndim!=1 or len(ids)!=40 or len(set(ids))!=40 or np.any((ids<0)|(ids>=n)):
        raise ValueError('Exactly40 distinct in-range training IDs required')
    unseen=np.setdiff1d(np.arange(n),ids)
    return ids,unseen

def evaluate(pred,truth,train_ids):
    pred=np.asarray(pred,float);truth=np.asarray(truth,float)
    if pred.shape!=truth.shape or pred.ndim!=1 or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Finite prediction/truth vectors required')
    train,unseen=data_split(train_ids,len(truth));result={}
    for tag,ids in [('train',train),('unseen',unseen)]:
        error=pred[ids]-truth[ids]
        result[tag+'_mse']=float(np.mean(error**2));result[tag+'_rmse']=float(np.sqrt(np.mean(error**2)))
        result[tag+'_mae']=float(np.mean(abs(error)))
    rho=spearmanr(pred[unseen],truth[unseen]).statistic if np.ptp(pred[unseen])>0 and np.ptp(truth[unseen])>0 else None
    result['unseen_spearman']=float(rho) if rho is not None and np.isfinite(rho) else None
    # Selection sees only predictions, excludes training points, ties by candidate ID.
    chosen=int(unseen[np.lexsort((unseen,pred[unseen]))[0]])
    span=float(np.ptp(truth));fopt=float(min(truth))
    if span<=0:raise ValueError('Nonconstant true objective required')
    result.update(selected_id=chosen,selected_true_fom=float(truth[chosen]),
                  selected_true_regret=float((truth[chosen]-fopt)/span),
                  selected_true_rank=int(1+np.sum(truth[unseen]<truth[chosen])),
                  best_after_one_true_fom=float(min(min(truth[train]),truth[chosen])))
    result['best_after_one_regret']=(result['best_after_one_true_fom']-fopt)/span
    for fraction in [.01,.05]:
        k=max(1,math.ceil(fraction*len(unseen)))
        true_top=unseen[np.lexsort((unseen,truth[unseen]))[:k]]
        pred_top=unseen[np.lexsort((unseen,pred[unseen]))[:k]]
        result[f'top{int(fraction*100)}_recall']=len(set(true_top)&set(pred_top))/k
    return result

def setup(cfg):
    p=load_problem(cfg['problem']);initial=json.loads(resolve(cfg['initial_sets_path']).read_text())
    if initial['problem_id']!=p.problem_id:raise ValueError('Initial-set problem mismatch')
    for seed in cfg['seeds']:data_split(initial['initial_ids'][str(seed)],len(p.categories))
    if cfg['initialization_repeats']<1:raise ValueError('Positive repeat count required')
    if len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Unique data seeds required')
    sources={name:sha(CODE_DIR/name) for name in ['evaluate_fm_settings.py','fm_model.py','evaluator.py','config.py']}
    identity={'config':cfg,'dataset_sha256':p.data_sha256,'schema_sha256':sha(p.config_path),
              'initial_sets_sha256':sha(resolve(cfg['initial_sets_path'])),'sources':sources}
    identity['experiment_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return p,initial,identity

def plan(cfg):
    p,initial,identity=setup(cfg);out=resolve(cfg['output_dir']);v=variants(cfg)
    plan={'status':'prepared_only','identity':identity,'training_count':len(v)*len(cfg['seeds'])*cfg['initialization_repeats'],
          'variants':[{'id':i,'fm':params,'factor':field,'value':value} for i,params,field,value in v],
          'statistics':'median over initialization repeats within each data seed,then median/IQR over data seeds;paired against baseline;Bonferroni over12variants x3primary metrics',
          'evaluation':'4056 remaining candidates,exploratory,not independent final test after hyperparameter selection',
          'scope':'FM predictive and exact-surrogate candidate quality;no SA/QAOA/BBO',
          'initial_ids':initial['initial_ids'],'python':sys.version,'platform':platform.platform()}
    target=out/'json/plan.json'
    if target.exists() and json.loads(target.read_text())['identity']!=identity:raise ValueError('Existing experiment differs;choose a new output_dir')
    save(target,plan);print(f"Prepared {len(v)} configurations x {len(cfg['seeds'])} data seeds x {cfg['initialization_repeats']} initializations = {plan['training_count']} fits. No fits started.")
    return p,initial,identity,out

def run(cfg):
    p,initial,identity,out=plan(cfg)
    for setting,params,_,_ in variants(cfg):
        for data_seed in cfg['seeds']:
            ids=initial['initial_ids'][str(data_seed)]
            # Only these40 values enter the production learner.
            x=p.patterns[ids].astype(np.float32);y=[p.evaluate_id(i) for i in ids]
            for rep in range(cfg['initialization_repeats']):
                stem=f'{setting}_seed{data_seed}_init{rep}';path=out/f'json/runs/{stem}.json'
                if path.exists():
                    old=json.loads(path.read_text());assert old['experiment_sha256']==identity['experiment_sha256']
                    if old['status']=='completed':
                        assert sha(out/old['model_file'])==old['model_file_sha256']
                        assert sha(out/old['prediction_file'])==old['prediction_file_sha256']
                        continue
                    if old['status']!='completed':
                        save(out/f"json/failures/{stem}_{time.time_ns()}.json",old)
                model_seed=int(data_seed)+rep*cfg['initialization_seed_stride']
                record={'status':'running','setting':setting,'data_seed':data_seed,'initialization_rep':rep,
                        'model_seed':model_seed,'fm':params,'train_ids':ids,'experiment_sha256':identity['experiment_sha256']}
                save(path,record);started=time.perf_counter()
                try:
                    model=train_fm(x,y,p.n_bits,params,model_seed)
                    with torch.no_grad():pred=model(torch.tensor(p.patterns,dtype=torch.float32)).numpy().astype(float)
                    # All remaining true values are accessed only AFTER fitting and predicting.
                    truth=np.array([p.evaluate_id(i) for i in range(len(p.categories))])
                    metrics=evaluate(pred,truth,ids)
                    model_path=out/f'checkpoints/{stem}.pt';model_path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(model.state_dict(),model_path)
                    pred_path=out/f'data/{stem}.npz';pred_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(pred_path,predictions=pred)
                    record.update(status='completed',metrics=metrics,model_sha256=model_sha256(model),
                                  model_file=str(model_path.relative_to(out)),model_file_sha256=sha(model_path),
                                  prediction_file=str(pred_path.relative_to(out)),prediction_file_sha256=sha(pred_path))
                except Exception:
                    record.update(status='failed',traceback=traceback.format_exc());raise
                finally:
                    record['elapsed_seconds']=time.perf_counter()-started;save(path,record)
                print(stem,'completed',flush=True)

def summarize(cfg):
    p,initial,identity=setup(cfg);out=resolve(cfg['output_dir']);records=[]
    for setting,params,_,_ in variants(cfg):
        for seed in cfg['seeds']:
            for rep in range(cfg['initialization_repeats']):
                r=json.loads((out/f'json/runs/{setting}_seed{seed}_init{rep}.json').read_text())
                assert r['experiment_sha256']==identity['experiment_sha256'] and r['status']=='completed'
                assert r['train_ids']==initial['initial_ids'][str(seed)] and r['fm']==params
                assert sha(out/r['prediction_file'])==r['prediction_file_sha256']
                assert sha(out/r['model_file'])==r['model_file_sha256']
                pred=np.load(out/r['prediction_file'])['predictions']
                check=evaluate(pred,np.array([p.evaluate_id(i) for i in range(len(p.categories))]),r['train_ids'])
                assert check==r['metrics'];records.append(r)
    fields=[k for k in records[0]['metrics'] if k!='selected_id'];grouped={};summary={}
    for setting,_,_,_ in variants(cfg):
        grouped[setting]={};summary[setting]={}
        for metric in fields:
            values=[]
            for seed in cfg['seeds']:
                raw=[r['metrics'][metric] for r in records if r['setting']==setting and r['data_seed']==seed]
                values.append(float(np.median(raw)) if all(x is not None for x in raw) else None)
            grouped[setting][metric]=values
            if any(x is None for x in values):summary[setting][metric]={'status':'undefined','per_seed':values}
            else:
                q1,med,q3=np.percentile(values,[25,50,75]);summary[setting][metric]={'median':med,'q25':q1,'q75':q3,'per_seed':values}
    tests=[];correction=(len(variants(cfg))-1)*len(cfg['primary_metrics'])
    for setting,_,_,_ in variants(cfg)[1:]:
        for metric in cfg['primary_metrics']:
            a,b=grouped[setting][metric],grouped['baseline'][metric]
            if any(v is None for v in a+b):tests.append({'setting':setting,'metric':metric,'status':'undefined;all rows retained'});continue
            diff=np.array(a)-np.array(b);pv=1. if np.all(abs(diff)<1e-12) else float(wilcoxon(diff).pvalue)
            tests.append({'setting':setting,'metric':metric,'per_seed_differences':diff.tolist(),'p_raw':pv,'p_bonferroni':min(1.,pv*correction)})
    save(out/'json/summary.json',{'status':'audited','experiment_sha256':identity['experiment_sha256'],'settings':summary,'paired_tests':tests,'fit_count':len(records),'visual_review':'pending after rendering'})
    (out/'csv').mkdir(exist_ok=True)
    rows=[{k:r[k] for k in ['setting','data_seed','initialization_rep','model_seed']}|r['fm']|r['metrics'] for r in records]
    with (out/'csv/metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    render(cfg,summary,out)
    print('All fits audited;summary,paired tests,CSV and figures generated;manual visual review pending.')

def render(cfg,summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field,grid in cfg['grid'].items():
        configs=[]
        for setting,params,_,_ in variants(cfg):
            if all(params[k]==v for k,v in cfg['baseline'].items() if k!=field):configs.append((params[field],setting))
        configs=sorted(set(configs));xs=[x for x,_ in configs]
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for ax,metrics,ylabel in zip(axes,[['train_mse','unseen_mse'],['unseen_spearman'],['selected_true_regret']],['MSE','Unseen Spearman correlation','True regret of FM-selected candidate']):
            for metric in metrics:
                ds=[summary[name][metric] for _,name in configs]
                if any('median' not in d for d in ds):continue
                color='#1f77b4' if metric=='train_mse' else '#ff7f0e'
                ax.plot(xs,[d['median'] for d in ds],marker='o',color=color,label=metric)
                ax.fill_between(xs,[d['q25'] for d in ds],[d['q75'] for d in ds],color=color,alpha=.15)
            ax.set(xlabel=field,ylabel=ylabel);ax.grid(alpha=.15)
            if field in ['learning_rate','epochs']:ax.set_xscale('log')
            elif field=='weight_decay':ax.set_xscale('symlog',linthresh=.001)
        axes[0].set_yscale('log');axes[0].legend(frameon=False)
        fig.tight_layout()
        for fmt in ['pdf','svg','png']:
            (out/fmt).mkdir(exist_ok=True);fig.savefig(out/fmt/f'fm_{field}.{fmt}',dpi=300,bbox_inches='tight')
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CODE_DIR/'config_fm_training_fixed40.json')
    parser.add_argument('--action',choices=['plan','run','analyze'],default='plan');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());{'plan':plan,'run':run,'analyze':summarize}[args.action](cfg)

if __name__=='__main__':main()
