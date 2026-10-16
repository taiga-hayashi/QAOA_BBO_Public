"""Classical penalty sensitivity on previously learned,frozen standardized FMs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
import evaluate_fm_settings as shared
from evaluator import load_problem
from fm_model import TorchFM,model_sha256
from fm_target_scaling import predict_original,qubo_original_scale,fit_target_transform
from qubo import fm_to_qubo,qubo_to_ising_scale
from sa_sampler import sample_sa

CODE=shared.CODE_DIR
DEFAULT_CONFIG=CODE/'config_fixed_fm_penalty.json'


def alpha_name(alpha):return f'alpha{alpha:g}'


def batch_seed(data_seed,init_rep,sample_rep):
    return int(np.random.SeedSequence([int(data_seed),int(init_rep),int(sample_rep),20261016,7]).generate_state(1)[0]%(2**31-1))


def measure(problem,pred,seen,raw):
    chosen,stats=problem.select_unseen_candidate(raw,pred,set(seen))
    unseen=np.setdiff1d(np.arange(len(pred)),seen);minimum=float(min(pred[unseen]));span=float(np.ptp(pred[unseen]))
    return {**stats,'selected_id':chosen,'unseen_fm_minimum':minimum,
            'gap_score':float(pred[chosen]-minimum) if chosen is not None else span+max(1.,span)}


def load_models(cfg):
    p=load_problem(cfg['problem']);out=shared.resolve(cfg['output_dir']);study=shared.resolve(cfg['source_study'])
    initial=json.loads(shared.resolve(cfg['initial_sets_path']).read_text())
    summary=json.loads((study/'json/summary.json').read_text());assert summary['status']=='audited'
    expected={key:cfg['fm'][key] for key in ['rank','epochs','learning_rate','optimizer','weight_decay']}
    assert cfg['fm']['target_standardization'] is True
    models=[];manifest=[]
    for seed in cfg['seeds']:
        seen=initial['initial_ids'][str(seed)];shared.data_split(seen,len(p.categories))
        for init in range(cfg['initialization_repeats']):
            source=study/f"json/runs/{cfg['source_setting']}_seed{seed}_init{init}.json";r=json.loads(source.read_text())
            assert r['status']=='completed' and r['standardized'] is True and r['fm']==expected
            assert r['data_seed']==seed and r['initialization_rep']==init and r['train_ids']==seen
            assert r['experiment_sha256']==summary['experiment_sha256']
            cp=study/r['checkpoint'];pred_file=study/r['predictions']
            assert shared.sha(cp)==r['checkpoint_sha256'] and shared.sha(pred_file)==r['predictions_sha256']
            checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
            model=TorchFM(p.n_bits,expected['rank']);model.load_state_dict(checkpoint['state_dict']);model.eval()
            assert model_sha256(model)==r['model_sha256']
            transform=checkpoint['target_transform'];assert transform==r['target_transform'] and transform['standardized']
            assert transform['training_count']==40
            _,expected_transform=fit_target_transform([p.evaluate_id(i) for i in seen],True)
            assert transform==expected_transform
            model_pred=predict_original(model,p.patterns,transform)
            np.testing.assert_allclose(model_pred,np.load(pred_file)['predictions'],atol=1e-10,rtol=0)
            qd,q,b=fm_to_qubo(model);q,b=qubo_original_scale(q,b,transform)
            qd={key:float(value)*transform['scale'] for key,value in qd.items()}
            pred=np.einsum('bi,ij,bj->b',p.patterns,q,p.patterns)+b
            np.testing.assert_allclose(pred,model_pred,atol=1e-4,rtol=0)
            S=qubo_to_ising_scale(q);assert np.isfinite(S) and S>0
            key=f'seed{seed}_init{init}';frozen=out/f'data/models/{key}.pt';frozen.parent.mkdir(parents=True,exist_ok=True)
            if frozen.exists():assert shared.sha(frozen)==r['checkpoint_sha256']
            else:shutil.copyfile(cp,frozen)
            info={'id':key,'data_seed':seed,'initialization_rep':init,'model_seed':r['model_seed'],'seen':seen,
                  'model_sha256':r['model_sha256'],'checkpoint':str(frozen.relative_to(out)),
                  'checkpoint_sha256':r['checkpoint_sha256'],'source_run':str(source.relative_to(CODE)),
                  'source_run_sha256':shared.sha(source),'target_transform':transform,'scale_S':S,
                  'qubo_vs_model_max_error':float(max(abs(pred-model_pred)))}
            manifest.append(info);models.append({'info':info,'qd':qd,'q':q,'offset':b,'pred':pred})
    return p,models,manifest


def identity(cfg,p,manifest):
    names=['evaluate_fixed_fm_penalty.py','evaluate_fm_settings.py','fm_model.py','fm_target_scaling.py','qubo.py','sa_sampler.py','evaluator.py','config.py']
    d={'config':cfg,'models':manifest,'dataset_sha256':p.data_sha256,'schema_sha256':shared.sha(p.config_path),
       'sources':{name:shared.sha(CODE/name) for name in names}}
    d['experiment_sha256']=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest();return d


def plan(cfg):
    if len(set(cfg['alphas']))!=len(cfg['alphas']) or any(not np.isfinite(a) or a<0 for a in cfg['alphas']):raise ValueError('Distinct nonnegative finite alphas required')
    if cfg['reference_alpha'] not in cfg['alphas'] or cfg['sampling_repeats']<1:raise ValueError('Invalid reference/repeat count')
    p,models,manifest=load_models(cfg);ident=identity(cfg,p,manifest);out=shared.resolve(cfg['output_dir'])
    path=out/'json/plan.json'
    if path.exists() and json.loads(path.read_text())['identity']!=ident:raise ValueError('Frozen experiment differs;use new output_dir')
    shared.save(path,{'status':'prepared_only','identity':ident,'fixed_models':len(models),
        'batches':len(models)*len(cfg['alphas'])*cfg['sampling_repeats'],
        'outputs':len(models)*len(cfg['alphas'])*cfg['sampling_repeats']*cfg['sa']['num_reads'],
        'sampler_seed_rule':'SeedSequence[data_seed,init_rep,sample_rep,20261016,7];identical seed across alphas',
        'minimum':'exact minimum of extracted raw-scale FM QUBO on unseen feasible candidates;tolerance1e-8',
        'gap':'conditional selected_fm_gap;gap_score uses declared failure score for no-candidate batches',
        'statistics':cfg['statistics'],'scope':cfg['scope']})
    print(f"Frozen{len(models)} models;{len(models)*len(cfg['alphas'])*cfg['sampling_repeats']} SA batches;no retraining.",flush=True)
    return p,models,ident,out


def run(cfg):
    p,models,ident,out=plan(cfg)
    for fm in models:
        info=fm['info'];key=info['id'];path=out/f'json/runs/{key}.json'
        rows=[];raws=[]
        if path.exists():
            old=json.loads(path.read_text());assert old['experiment_sha256']==ident['experiment_sha256']
            if old['status']=='completed':
                assert shared.sha(out/old['raw_file'])==old['raw_sha256'];continue
            shared.save(out/f'json/failures/{key}_{time.time_ns()}.json',old)
        record={'status':'running','model':info,'experiment_sha256':ident['experiment_sha256'],'batches':rows}
        shared.save(path,record);start=time.perf_counter()
        try:
            for alpha in cfg['alphas']:
                for rep in range(cfg['sampling_repeats']):
                    seed=batch_seed(info['data_seed'],info['initialization_rep'],rep)
                    raw,meta=sample_sa(fm['qd'],p.group_sizes,info['scale_S']*alpha,alpha,fm['offset'],cfg['sa'],seed)
                    assert raw.shape==(cfg['sa']['num_reads'],p.n_bits)
                    stats=measure(p,fm['pred'],info['seen'],raw)
                    rows.append({'alpha':alpha,'repeat':rep,'S':info['scale_S'],'sampler':meta,**stats});raws.append(raw)
                    shared.save(path,record)
                print(key,alpha_name(alpha),'completed',flush=True)
            raw_file=out/f'data/raw/{key}.npz';raw_file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(raw_file,raw_bits=np.asarray(raws,dtype=np.int8))
            record.update(status='completed',raw_file=str(raw_file.relative_to(out)),raw_sha256=shared.sha(raw_file))
        except Exception:
            record.update(status='failed',traceback=traceback.format_exc())
            if raws:
                raw_file=out/f'data/raw/{key}_partial_{time.time_ns()}.npz';raw_file.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(raw_file,raw_bits=np.asarray(raws,dtype=np.int8));record['partial_raw_file']=str(raw_file.relative_to(out))
            raise
        finally:
            record['elapsed_seconds']=time.perf_counter()-start;shared.save(path,record)


def summarize(cfg):
    p,models,ident,out=plan(cfg);per_model=[];raw_rows=[];audited=0
    for fm in models:
        info=fm['info'];r=json.loads((out/f"json/runs/{info['id']}.json").read_text())
        assert r['status']=='completed' and r['experiment_sha256']==ident['experiment_sha256'] and r['model']==info
        assert shared.sha(out/r['raw_file'])==r['raw_sha256']
        raw=np.load(out/r['raw_file'])['raw_bits'];assert raw.shape==(len(cfg['alphas'])*cfg['sampling_repeats'],cfg['sa']['num_reads'],p.n_bits)
        assert len(r['batches'])==len(raw)
        for index,(row,bits) in enumerate(zip(r['batches'],raw)):
            alpha=cfg['alphas'][index//cfg['sampling_repeats']];rep=index%cfg['sampling_repeats']
            assert row['alpha']==alpha and row['repeat']==rep and row['S']==info['scale_S']
            sm=row['sampler'];assert sm['penalty_alpha']==alpha and sm['penalty_lambda']==alpha*row['S']
            assert sm['sampler_seed']==batch_seed(info['data_seed'],info['initialization_rep'],rep)
            assert sm['num_reads']==cfg['sa']['num_reads'] and sm['num_sweeps']==cfg['sa']['num_sweeps']
            assert len(sm['beta_range'])==2 and 0<sm['beta_range'][0]<sm['beta_range'][1]
            check=measure(p,fm['pred'],info['seen'],bits)
            for k,v in check.items():assert row[k]==v,(info['id'],index,k)
            raw_rows.append({'data_seed':info['data_seed'],'initialization_rep':info['initialization_rep'],'alpha':alpha,'repeat':rep,
                'lambda':sm['penalty_lambda'],'S':row['S'],'beta_min':sm['beta_range'][0],'beta_max':sm['beta_range'][1],**check})
            audited+=1
        for alpha in cfg['alphas']:
            rr=[row for row in r['batches'] if row['alpha']==alpha];assert len(rr)==cfg['sampling_repeats']
            conditional=[row['selected_fm_gap'] for row in rr if row['selected_fm_gap'] is not None]
            per_model.append({'data_seed':info['data_seed'],'initialization_rep':info['initialization_rep'],'alpha':alpha,
                'capture':float(np.mean([row['fm_minimum_hit'] for row in rr])),
                'feasible':float(np.mean([row['raw_feasible_rate'] for row in rr])),
                'candidate_rate':float(np.mean([row['selected_id'] is not None for row in rr])),
                'gap_score':float(np.median([row['gap_score'] for row in rr])),
                'conditional_gap':float(np.median(conditional)) if conditional else None,
                'candidate_less_batches':sum(row['selected_id'] is None for row in rr)})
    metrics=['capture','feasible','candidate_rate','gap_score','conditional_gap'];summary={}
    for alpha in cfg['alphas']:
        key=alpha_name(alpha);summary[key]={}
        for metric in metrics:
            per_seed={}
            for seed in cfg['seeds']:
                values=[row[metric] for row in per_model if row['alpha']==alpha and row['data_seed']==seed]
                assert len(values)==cfg['initialization_repeats']
                per_seed[str(seed)]=float(np.median(values)) if all(v is not None for v in values) else None
            entry={'per_seed':per_seed}
            if all(v is not None for v in per_seed.values()):
                lo,med,hi=np.percentile(list(per_seed.values()),[25,50,75]);entry.update(median=float(med),q25=float(lo),q75=float(hi))
            else:entry['status']='undefined;candidate failures retained separately'
            summary[key][metric]=entry
    tests=[];correction=(len(cfg['alphas'])-1)*len(cfg['primary_metrics']);reference=alpha_name(cfg['reference_alpha'])
    for alpha in cfg['alphas']:
        if alpha==cfg['reference_alpha']:continue
        for metric in cfg['primary_metrics']:
            diff=np.array(list(summary[alpha_name(alpha)][metric]['per_seed'].values()))-np.array(list(summary[reference][metric]['per_seed'].values()))
            pv=1. if np.all(abs(diff)<1e-12) else float(wilcoxon(diff).pvalue)
            tests.append({'alpha':alpha,'reference_alpha':cfg['reference_alpha'],'metric':metric,'paired_differences':dict(zip(map(str,cfg['seeds']),diff.tolist())),
                          'median_difference':float(np.median(diff)),'p_raw':pv,'p_bonferroni':min(1.,pv*correction)})
    shared.save(out/'json/summary.json',{'status':'audited','experiment_sha256':ident['experiment_sha256'],'fixed_models':len(models),'audited_batches':audited,
        'outputs':audited*cfg['sa']['num_reads'],'per_model':per_model,'per_alpha':summary,'paired_tests':tests,'visual_review':'pending after rendering'})
    (out/'csv').mkdir(exist_ok=True)
    for name,rows in [('batches',raw_rows),('per_model',per_model)]:
        with (out/f'csv/{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render(cfg,summary,per_model,out);reports(cfg,summary,models,audited,out)
    print(f'Audited{audited} SA batches,all raw bits,model/QUBO/scales and metrics;statistics/reports/figures generated;manual visual review pending.',flush=True)


def render(cfg,summary,per_model,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    alphas=np.array(cfg['alphas']);fig,axes=plt.subplots(1,4,figsize=(14,3.7))
    for ax,metric,ylabel in zip(axes,['capture','gap_score','feasible','candidate_rate'],['Unseen FM-minimum capture','FM gap score (failures scored)','Raw One-Hot feasible fraction','Unseen candidate acquisition fraction']):
        entries=[summary[alpha_name(a)][metric] for a in alphas]
        ax.plot(alphas,[d['median'] for d in entries],color='#1f77b4',marker='o')
        ax.fill_between(alphas,[d['q25'] for d in entries],[d['q75'] for d in entries],color='#1f77b4',alpha=.17)
        ax.set(xscale='log',xlabel='Normalized penalty alpha',ylabel=ylabel);ax.grid(alpha=.15)
        if metric!='gap_score':ax.set_ylim(-.02,1.02)
    fig.tight_layout()
    for fmt in ['png','pdf','svg']:
        (out/fmt).mkdir(exist_ok=True);fig.savefig(out/fmt/f'penalty_accuracy.{fmt}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,ylabel in zip(axes,['capture','gap_score'],['Unseen FM-minimum capture','FM gap score (failures scored)']):
        for seed in cfg['seeds']:
            values=[summary[alpha_name(a)][metric]['per_seed'][str(seed)] for a in alphas]
            ax.plot(alphas,values,marker='o',label=f'Seed {seed}',alpha=.75)
        ax.set(xscale='log',xlabel='Normalized penalty alpha',ylabel=ylabel);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout()
    for fmt in ['png','pdf','svg']:fig.savefig(out/fmt/f'penalty_paired_seeds.{fmt}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def reports(cfg,summary,models,audited,out):
    def fmt(d):return f"{d['median']:.6g} [{d['q25']:.6g},{d['q75']:.6g}]" if 'median' in d else 'undefined;failures retained'
    lines=['# Fixed-FM penalty sensitivity under the promoted optical defaults','',
        f"{len(models)} fixed FMs,{audited} audited batches;{cfg['sa']['num_reads']} reads,{cfg['sa']['num_sweeps']} sweeps,{cfg['sampling_repeats']} repeats per model/alpha.",
        'FM:rank4,learning_rate0.1,weight_decay0.1,120epochs,AdamW,training-target standardization. All checkpoints reused from the completed factorial study;no retraining.',
        'Aggregate10 batches within each model,then median over3 model initializations within each data seed;report5-data-seed median [Q25,Q75].',
        '', '| alpha | FM-minimum capture | FM gap score | Raw feasible | Candidate acquisition | Conditional FM gap |','|---|---|---|---|---|---|']
    for alpha in cfg['alphas']:
        values=summary[alpha_name(alpha)];lines.append(f'|{alpha:g}|'+ '|'.join(fmt(values[k]) for k in ['capture','gap_score','feasible','candidate_rate','conditional_gap'])+'|')
    lines+=['','S=max absolute baseFM Ising h/J after restoring original FOM units;lambda=S*alpha. No one-hot penalty enters S.',
        'Each100-output batch is one trial;capture means at least one unseen feasible FM minimizer within tolerance1e-8. Conditional gap is undefined without a candidate;gap_score assigns unseen FM span+max(1,span) in that case. All such failures remain in capture/feasibility/candidate rate and the failure-aware score.',
        '27 paired Wilcoxon contrasts against alpha100 with one Bonferroni family;details in json/summary.json.',
        'Figures:penalty_accuracy shows5-data-seed medians and25–75th percentile bands;penalty_paired_seeds retains every data seed after initialization aggregation. Log alpha axes,linear metric axes. No titles/bottom annotations. Manual visual review pending.',
        'Source/model/raw-bit hashes and every measured metric were audited. No optical true values were used to select a sampled candidate. No BBO or quantum advantage claim.']
    (out/'md').mkdir(exist_ok=True);(out/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'md/DISCUSSION.md').write_text('# Limits\n\nThis is SA sampling quality on fixed FMs,not predictive accuracy or closed-loop BBO. The frozen learning setting was selected exploratorily using the same five datasets. Temperature ranges are chosen automatically by Neal and change with the penalized BQM;this study does not isolate penalty-only effects. Model initialization repeats are not independent data seeds. All candidate failures and every seed are retained. Conditional-gap missingness is distinct from zero gap. Visual review pending.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);parser.add_argument('--action',choices=['plan','run','analyze'],default='plan');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());{'plan':plan,'run':run,'analyze':summarize}[args.action](cfg)

if __name__=='__main__':main()
