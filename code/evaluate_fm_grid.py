"""Full factorial FM study,reusing the learner,metrics and fixed40 datasets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
import evaluate_fm_settings as shared
from fm_model import model_sha256,TorchFM
from fm_target_scaling import train_scaled_fm,predict_original

CODE=shared.CODE_DIR
DEFAULT_CONFIG=CODE/'config_fm_training_grid_fixed40.json'


def variants(cfg):
    items=[]
    for rank,lr,wd,std in itertools.product(cfg['grid']['rank'],cfg['grid']['learning_rate'],cfg['grid']['weight_decay'],cfg['target_standardization']):
        if isinstance(rank,bool) or int(rank)!=rank or rank<1 or not np.isfinite(lr) or lr<=0 or not np.isfinite(wd) or wd<0 or not isinstance(std,bool):
            raise ValueError('Invalid factorial setting')
        params=dict(cfg['baseline'],rank=int(rank),learning_rate=float(lr),weight_decay=float(wd))
        name=f'r{rank}_lr{lr:g}_wd{wd:g}_'+('standardized' if std else 'raw')
        items.append((name,params,std))
    if len({x[0] for x in items})!=len(items):raise ValueError('Duplicate factorial settings')
    if cfg['baseline']['epochs']<1:raise ValueError('Positive fixed epoch count required')
    return items


def baseline_name(cfg):
    matches=[name for name,params,std in variants(cfg) if params==cfg['baseline'] and not std]
    if len(matches)!=1:raise ValueError('Original raw baseline must appear exactly once')
    return matches[0]


def setup(cfg):
    p,initial,identity=shared.setup(cfg)
    identity.pop('experiment_sha256')
    identity['sources'].update({name:shared.sha(CODE/name) for name in ['evaluate_fm_grid.py','fm_target_scaling.py']})
    identity['experiment_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    baseline_name(cfg)
    return p,initial,identity,shared.resolve(cfg['output_dir'])


def contrasts(cfg):
    cases=variants(cfg);base=baseline_name(cfg)
    baseline=[('vs_original_baseline',name,base) for name,_,_ in cases if name!=base]
    matched=[('standardized_minus_raw',name,name.removesuffix('standardized')+'raw') for name,_,std in cases if std]
    return baseline+matched


def plan(cfg):
    p,initial,identity,out=setup(cfg)
    data={'status':'prepared_only','identity':identity,'fits':len(variants(cfg))*len(cfg['seeds'])*cfg['initialization_repeats'],
          'settings':[{'id':name,'fm':params,'standardized':std} for name,params,std in variants(cfg)],
          'baseline':baseline_name(cfg),'initial_ids':initial['initial_ids'],
          'test_count':len(contrasts(cfg))*len(cfg['primary_metrics']),
          'analysis_policy':'3 initialization medians within each data seed,then5 data seed median/IQR;all predeclared contrasts share one Bonferroni family',
          'scope':'fixed40 surrogate learning;4056 unseen points are exploratory evaluation;no QAOA,SA,BBO or new optical simulations',
          'python':sys.version,'platform':platform.platform()}
    path=out/'json/plan.json'
    if path.exists() and json.loads(path.read_text())['identity']!=identity:raise ValueError('Existing experiment differs;use new output_dir')
    shared.save(path,data)
    print(f"Plan ready:{len(variants(cfg))} settings,{data['fits']} fits,{data['test_count']} paired primary tests",flush=True)
    return p,initial,identity,out


def run(cfg):
    p,initial,identity,out=plan(cfg)
    for setting,params,std in variants(cfg):
        for data_seed in cfg['seeds']:
            ids=initial['initial_ids'][str(data_seed)]
            x=p.patterns[ids].astype(np.float32);y=[p.evaluate_id(i) for i in ids]
            for rep in range(cfg['initialization_repeats']):
                stem=f'{setting}_seed{data_seed}_init{rep}';path=out/f'json/runs/{stem}.json'
                if path.exists():
                    old=json.loads(path.read_text());assert old['experiment_sha256']==identity['experiment_sha256']
                    if old['status']=='completed':
                        assert shared.sha(out/old['checkpoint'])==old['checkpoint_sha256']
                        assert shared.sha(out/old['predictions'])==old['predictions_sha256']
                        continue
                    shared.save(out/f'json/failures/{stem}_{time.time_ns()}.json',old)
                seed=int(data_seed)+rep*cfg['initialization_seed_stride']
                rec={'status':'running','setting':setting,'fm':params,'standardized':std,'data_seed':data_seed,
                     'initialization_rep':rep,'model_seed':seed,'train_ids':ids,'experiment_sha256':identity['experiment_sha256']}
                shared.save(path,rec);started=time.perf_counter()
                try:
                    model,transform=train_scaled_fm(x,y,p.n_bits,params,seed,std)
                    pred=predict_original(model,p.patterns,transform)
                    # Full truth is diagnostic only,accessed after fitting/prediction.
                    truth=np.array([p.evaluate_id(i) for i in range(len(p.categories))])
                    metrics=shared.evaluate(pred,truth,ids)
                    cp=out/f'checkpoints/{stem}.pt';cp.parent.mkdir(parents=True,exist_ok=True)
                    torch.save({'state_dict':model.state_dict(),'target_transform':transform,'fm':params,'model_seed':seed},cp)
                    predictions=out/f'data/{stem}.npz';predictions.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(predictions,predictions=pred)
                    rec.update(status='completed',target_transform=transform,model_sha256=model_sha256(model),metrics=metrics,
                               checkpoint=str(cp.relative_to(out)),checkpoint_sha256=shared.sha(cp),
                               predictions=str(predictions.relative_to(out)),predictions_sha256=shared.sha(predictions))
                except Exception:
                    rec.update(status='failed',traceback=traceback.format_exc());raise
                finally:
                    rec['elapsed_seconds']=time.perf_counter()-started;shared.save(path,rec)
                print(stem,'completed',flush=True)


def read_and_audit(cfg):
    p,initial,identity,out=setup(cfg);truth=np.array([p.evaluate_id(i) for i in range(len(p.categories))]);records=[]
    for name,params,std in variants(cfg):
        for seed in cfg['seeds']:
            ids=initial['initial_ids'][str(seed)];ys=truth[ids]
            for rep in range(cfg['initialization_repeats']):
                r=json.loads((out/f'json/runs/{name}_seed{seed}_init{rep}.json').read_text())
                assert r['status']=='completed' and r['experiment_sha256']==identity['experiment_sha256']
                assert r['setting']==name and r['fm']==params and r['standardized']==std
                assert r['data_seed']==seed and r['initialization_rep']==rep
                assert r['model_seed']==seed+rep*cfg['initialization_seed_stride'] and r['train_ids']==ids
                assert shared.sha(out/r['checkpoint'])==r['checkpoint_sha256']
                assert shared.sha(out/r['predictions'])==r['predictions_sha256']
                pred=np.load(out/r['predictions'])['predictions']
                checkpoint=torch.load(out/r['checkpoint'],map_location='cpu',weights_only=True)
                assert checkpoint['target_transform']==r['target_transform'] and checkpoint['fm']==params
                assert checkpoint['model_seed']==r['model_seed']
                model=TorchFM(p.n_bits,params['rank']);model.load_state_dict(checkpoint['state_dict']);model.eval()
                assert model_sha256(model)==r['model_sha256']
                np.testing.assert_allclose(predict_original(model,p.patterns,r['target_transform']),pred,atol=1e-10,rtol=0)
                from fm_target_scaling import fit_target_transform
                _,expected=fit_target_transform(ys,std);assert expected==r['target_transform']
                check=shared.evaluate(pred,truth,ids);assert check==r['metrics']
                records.append(r)
    return records,identity,out


def summarize(cfg):
    records,identity,out=read_and_audit(cfg);metrics=[k for k in records[0]['metrics'] if k!='selected_id'];summary={}
    for setting,_,_ in variants(cfg):
        summary[setting]={}
        for metric in metrics:
            values=[]
            for seed in cfg['seeds']:
                raw=[r['metrics'][metric] for r in records if r['setting']==setting and r['data_seed']==seed]
                assert len(raw)==cfg['initialization_repeats']
                values.append(float(np.median(raw)) if all(v is not None for v in raw) else None)
            entry={'per_seed':dict(zip(map(str,cfg['seeds']),values))}
            if any(v is None for v in values):entry['status']='undefined;all runs retained'
            else:
                q1,med,q3=np.percentile(values,[25,50,75]);entry.update(median=float(med),q25=float(q1),q75=float(q3))
            summary[setting][metric]=entry
    tests=[];correction=len(contrasts(cfg))*len(cfg['primary_metrics'])
    for kind,a,b in contrasts(cfg):
        for metric in cfg['primary_metrics']:
            av=list(summary[a][metric]['per_seed'].values());bv=list(summary[b][metric]['per_seed'].values())
            test={'kind':kind,'setting':a,'reference':b,'metric':metric}
            if any(v is None for v in av+bv):test['status']='undefined;all runs retained'
            else:
                diff=np.array(av)-np.array(bv);pv=1. if np.all(abs(diff)<1e-12) else float(wilcoxon(diff).pvalue)
                test.update(per_seed_differences=dict(zip(map(str,cfg['seeds']),diff.tolist())),median_difference=float(np.median(diff)),p_raw=pv,p_bonferroni=min(1.,pv*correction))
            tests.append(test)
    result={'status':'audited','fit_count':len(records),'setting_count':len(summary),'experiment_sha256':identity['experiment_sha256'],
            'settings':summary,'paired_tests':tests,'bonferroni_family_size':correction,'visual_review':'pending after rendering'}
    shared.save(out/'json/summary.json',result)
    (out/'csv').mkdir(exist_ok=True)
    rows=[{k:r[k] for k in ['setting','data_seed','initialization_rep','model_seed','standardized']}|r['fm']|r['target_transform']|r['metrics'] for r in records]
    with (out/'csv/metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render(cfg,summary,tests,out);reports(cfg,result,out)
    print(f'All{len(records)} fits audited;statistics,reports and figures generated;manual visual review pending.',flush=True)


def render(cfg,summary,tests,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases={(params['rank'],params['learning_rate'],params['weight_decay'],std):name for name,params,std in variants(cfg)}
    ranks=cfg['grid']['rank'];rates=cfg['grid']['learning_rate'];wds=cfg['grid']['weight_decay'];modes=cfg['target_standardization']
    for metric,label,cmap in [('unseen_rmse','Unseen RMSE (original FOM)','viridis_r'),('unseen_spearman','Unseen Spearman correlation','viridis'),('selected_true_regret','True regret of FM-selected candidate','viridis_r')]:
        finite=[s[metric]['median'] for s in summary.values() if 'median' in s[metric]]
        lo,hi=(min(finite),max(finite)) if finite else (0,1)
        if lo==hi:hi=lo+1e-12
        fig,axes=plt.subplots(len(modes),len(rates),figsize=(3.5*len(rates),3.2*len(modes)),squeeze=False,layout='constrained')
        for row,std in enumerate(modes):
            for col,lr in enumerate(rates):
                matrix=np.array([[summary[cases[(rank,lr,wd,std)]][metric].get('median',np.nan) for wd in wds] for rank in ranks])
                ax=axes[row,col];im=ax.imshow(matrix,cmap=cmap,vmin=lo,vmax=hi,aspect='auto')
                ax.set_xticks(range(len(wds)),[f'{wd:g}' for wd in wds]);ax.set_yticks(range(len(ranks)),ranks)
                ax.set_xlabel('weight decay');ax.set_ylabel(f"rank\n{'standardized' if std else 'raw'}, lr={lr:g}")
                for i,j in itertools.product(range(len(ranks)),range(len(wds))):
                    value=matrix[i,j];normalized=(value-lo)/(hi-lo)
                    # Text contrast follows the actual colormap luminance.
                    rgba=plt.get_cmap(cmap)(normalized if np.isfinite(value) else .5)
                    lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(j,i,f'{value:.3f}' if np.isfinite(value) else 'N/A',ha='center',va='center',fontsize=8,color='black' if lum>.55 else 'white')
        fig.colorbar(im,ax=axes.ravel().tolist(),label=label,shrink=.8)
        # constrained_layout handles shared colorbar;no tight_layout here.
        for fmt in ['png','pdf','svg']:
            (out/fmt).mkdir(exist_ok=True);fig.savefig(out/fmt/f'grid_{metric}.{fmt}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    if False not in modes or True not in modes:return
    ds={t['setting']:t for t in tests if t['kind']=='standardized_minus_raw' and t['metric']=='unseen_rmse'}
    matrices=[np.array([[ds[cases[(rank,lr,wd,True)]].get('median_difference',np.nan) for wd in wds] for rank in ranks]) for lr in rates]
    bound=max([float(np.nanmax(abs(m))) for m in matrices if np.isfinite(m).any()]+[1e-12])
    fig,axes=plt.subplots(1,len(rates),figsize=(3.5*len(rates),3.5),squeeze=False,layout='constrained')
    for ax,lr,matrix in zip(axes[0],rates,matrices):
        im=ax.imshow(matrix,cmap='PuOr_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(len(wds)),[f'{wd:g}' for wd in wds]);ax.set_yticks(range(len(ranks)),ranks)
        ax.set(xlabel='weight decay',ylabel=f'rank\nlr={lr:g}')
        for i,j in itertools.product(range(len(ranks)),range(len(wds))):
            v=matrix[i,j];ax.text(j,i,f'{v:+.3f}' if np.isfinite(v) else 'N/A',ha='center',va='center',fontsize=8,color='white' if abs(v)>.6*bound else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Paired RMSE difference (standardized - raw)',shrink=.8)
    for fmt in ['png','pdf','svg']:
        (out/fmt).mkdir(exist_ok=True);fig.savefig(out/fmt/f'normalization_paired_rmse.{fmt}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def reports(cfg,result,out):
    def fmt(d):return f"{d['median']:.6g} [{d['q25']:.6g},{d['q75']:.6g}]" if 'median' in d else 'undefined'
    lines=['# FM full-factorial and target-standardization study','',f"{result['fit_count']} audited fits,{result['setting_count']} settings;{len(cfg['seeds'])} fixed40 data seeds x{cfg['initialization_repeats']} initializations.",
           'Metrics use initialization medians within each data seed,then median [Q25,Q75] over data seeds. All metrics are in original optical FOM scale.',
           '', '| rank | learning rate | weight decay | standardized | unseen RMSE | Spearman | selected true regret |','|---|---|---|---|---|---|---|']
    for name,params,std in variants(cfg):
        s=result['settings'][name];lines.append(f"|{params['rank']}|{params['learning_rate']:g}|{params['weight_decay']:g}|{std}|"+'|'.join(fmt(s[m]) for m in cfg['primary_metrics'])+'|')
    lines+=['',f"Paired tests: {result['bonferroni_family_size']} Bonferroni-corrected contrasts across3 primary metrics;see json/summary.json.",
            'Figures:three metric heatmaps share a color scale across all eight facets. Every cell is a measured setting median over5 data seeds after aggregating3 initializations. Axes:rank rows,weight-decay columns;learning rate and raw/standardized mode appear in axis labels. All color scales are linear. The fourth figure shows median within-data-seed RMSE differences (standardized minus raw),negative favors standardization. No main titles or bottom annotations.',
            'Every declared fit is retained;no failed runs are excluded. Models,transformations,predictions,hashes and metrics were audited. Manual figure review remains pending.',
            'The4056 unseen candidates are exploratory evaluation,not independent final confirmation. No SA/QAOA/BBO or new optical simulation was run.']
    (out/'md').mkdir(exist_ok=True);(out/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'md/DISCUSSION.md').write_text('# Limits\n\nThis factorial study follows an exploratory OFAT study on the same five datasets. Hyperparameter selection from these metrics requires independent confirmation on new data seeds. Epochs remain120 and optimizer/initialization method remain fixed. Model initialization repeats do not increase the independent data-seed sample size. Raw-scale RMSE,ranking and candidate regret answer different questions. Standardization statistics come only from the40 training targets. Exact surrogate selection is distinct from SA/QAOA sampling and BBO. Equal epoch counts need not mean equally converged models. All failures/undefined correlations are retained. Visual review pending.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);parser.add_argument('--action',choices=['plan','run','analyze'],default='plan');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());{'plan':plan,'run':run,'analyze':summarize}[args.action](cfg)

if __name__=='__main__':main()
