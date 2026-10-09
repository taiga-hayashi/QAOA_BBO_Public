"""Fresh SA reads and exact coarse-grained QAOA draws at original fixed angles."""
import sys,json,time,platform,importlib.metadata,itertools,argparse
from pathlib import Path
import numpy as np
import neal
from scipy.stats import wilcoxon
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent;PILOT=ROOT/'fixed_fm_pilot';SHOTS=ROOT/'qaoa_shot_accuracy'
sys.path.insert(0,str(PILOT/'py'))
from run_pilot import PerovskitesEvaluator,score_candidates,build_penalty_bqm,AdaptivePenaltyTracker,sha,save,REPO
P=json.loads((OUT/'json/protocol.json').read_text());METHODS=P['methods']
KEYS=['global_fm_minimum_hit','unseen_fm_minimum_hit','unseen_success','empirical_raw_feasible_rate','selected_fm_gap','selected_true_gap','best_initial_plus_proposals','new_true_optimum_hit']

def context(base):
 bb=PerovskitesEvaluator();cats=[list(c) for c in bb.candidates()];patterns=np.array([bb.encode(c) for c in cats],dtype=np.int8)
 q=np.array(base['qubo']);pred=np.einsum('bi,ij,bj->b',patterns,q,patterns)+base['offset'];truth=np.array([bb.evaluate(c) for c in cats])
 return cats,patterns,q,pred,truth

def score_events(events,base,ctx):
 cats,patterns,q,pred,truth=ctx;events=np.asarray(events,int);assert len(events)>0 and np.all((events>=-1)&(events<192))
 # All invalid states have identical downstream rejection. Zero rows are internal
 # class representatives only; never saved/reported as sampled basis strings.
 bits=np.zeros((len(events),23),np.int8);valid=events>=0;bits[valid]=patterns[events[valid]]
 r=score_candidates(bits,{},q,base['offset'],cats,patterns,base['initial_dataset']['candidate_ids'],truth,{'accepted_candidates':1})
 r.pop('raw_basis_indices');r['candidate_events']=events.tolist();r['allowed_total_bb_budget']=21
 initial=set(base['initial_dataset']['candidate_ids']);unseen=[i for i in range(192) if i not in initial]
 selected=r['accepted_candidates'][0] if r['accepted_count'] else None
 r.update(global_fm_minimum_hit=int(any(abs(pred[i]-pred.min())<1e-10 for i in events if i>=0)),
  unseen_fm_minimum_hit=int(selected is not None and abs(selected['fm_prediction']-pred[unseen].min())<1e-10),
  unseen_success=int(selected is not None),selected_fm_gap=selected['surrogate_gap_to_global_feasible_fm_min'] if selected else None,
  selected_true_gap=selected['hse_gap'] if selected else None,new_true_optimum_hit=int(selected is not None and abs(selected['hse_gap']-truth.min())<1e-10))
 return r

def masses(base,shot,method):
 prob=np.array(base['methods'][method]['feasible_basis_probabilities']);source=shot['qaoa'][method]
 assert np.isfinite(prob).all() and np.all(prob>=0) and abs(prob.sum()-source['exact_feasible_probability'])<1e-10
 assert base['methods'][method]['angles']==source['angles']
 if method=='XY-FMQAOA':assert source['lambda_internal']==0 and abs(prob.sum()-1)<1e-10
 dist=np.r_[prob,max(0.,1-prob.sum())];dist/=dist.sum();return dist

def prevalidate():
 assert P['status']=='ready'
 for source in [PILOT,SHOTS]:assert json.loads((source/'json/artifact_verification.json').read_text())['status']=='passed'
 manifest=json.loads((ROOT/'json/problem_manifest.json').read_text());assert manifest['prevalidation']['legacy_n6_n9_passed'] and manifest['prevalidation']['target_group_sizes_passed']
 checks=0
 for seed in P['seeds']:
  base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());shot=json.loads((SHOTS/f'json/seed_{seed}.json').read_text());assert base['status']==shot['status']=='completed';ctx=context(base)
  assert sha(PILOT/base['checkpoint'])==base['checkpoint_sha256'];assert shot['source_sha256']==sha(PILOT/f'json/seed_{seed}.json')
  lookup={tuple(row):i for i,row in enumerate(ctx[1])}
  for method in METHODS[2:]:
   masses(base,shot,method)
   for n in P['sample_counts']:
    for old in shot['qaoa'][method]['conditions'][str(n)][:3]:
     events=[lookup.get(tuple((int(x)>>i)&1 for i in range(23)),-1) for x in old['raw_basis_indices']]
     new=score_events(events,base,ctx)
     for key in ['empirical_raw_feasible_rate','accepted_candidates','best_initial_plus_proposals','raw_invalid_count']:
      assert new[key]==old[key],(seed,method,n,key)
     checks+=1
 save(OUT/'json/prevalidation.json',{'status':'passed','runner_sha256':sha(Path(__file__)),'coarse_grain_parity_records':checks,'N6_N9_and23_XY_source_checks':'passed'})
 print('prevalidation passed',checks,flush=True)

def run():
 assert json.loads((OUT/'json/prevalidation.json').read_text())['runner_sha256']==sha(Path(__file__))
 save(OUT/'json/environment.json',{'python':sys.version,'platform':platform.platform(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','openqarp','dwave-neal','matplotlib']},'protocol_sha256':sha(OUT/'json/protocol.json'),'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),PILOT/'py/run_pilot.py',REPO/'src/qarp_backend.py',REPO/'intern_0924/src/penalty.py']}})
 for seed in P['seeds']:
  target=OUT/f'json/seed_{seed}.json';assert not target.exists(),'preserve existing raw results'
  base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());shot=json.loads((SHOTS/f'json/seed_{seed}.json').read_text());ctx=context(base);cats,patterns,q,pred,truth=ctx
  record={'seed':seed,'status':'running','initial_sha256':base['initial_sha256'],'model_sha256':base['fixed_model_sha256'],'source_sha256':sha(PILOT/f'json/seed_{seed}.json'),'shot_source_sha256':sha(SHOTS/f'json/seed_{seed}.json'),'initial_true_optimum_present':any(abs(truth[i]-truth.min())<1e-10 for i in base['initial_dataset']['candidate_ids']),'conditions':[]};save(target,record)
  lookup={tuple(row):i for i,row in enumerate(patterns)};qd={(i,j):q[i,j] for i in range(23) for j in range(i,23)}
  for method in METHODS:
   dist=masses(base,shot,method) if 'FMQAOA' in method else None
   if dist is not None:record.setdefault('exact_probabilities',{})[method]={'candidate_masses':dist[:192].tolist(),'invalid_mass':float(dist[-1]),'angles':shot['qaoa'][method]['angles']}
   for n in P['sample_counts']:
    for rep in range(P['repetitions']):
     start=time.perf_counter();batches=[]
     if dist is not None:
      uniforms=np.random.default_rng(np.random.SeedSequence([seed,rep,1009,109])).random(100)
      cdf=np.cumsum(dist);cdf[-1]=1.;events=np.searchsorted(cdf,uniforms[:n],side='right');events[events==192]=-1
     else:
      tracker=AdaptivePenaltyTracker();pool=[]
      for batch,reads in enumerate([3,3,2,2] if n==10 else [25]*4):
       alpha=tracker.get_current_alpha() if method=='Adaptive-FMQA' else 100
       samples=neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qd,[16,3,4],base['base_Ising_scale_S']*alpha,base['offset']),num_reads=reads,num_sweeps=1000,seed=seed*100+rep*100000+batch+n*1000,beta_schedule_type='geometric')
       arr=samples.record.sample[:,[list(samples.variables).index(i) for i in range(23)]];bits=np.repeat(arr,samples.record.num_occurrences,axis=0)
       events_batch=np.array([lookup.get(tuple(row),-1) for row in bits]);rate=float(np.mean(events_batch>=0));nxt=tracker.update(rate) if method=='Adaptive-FMQA' else alpha
       batches.append({'reads':reads,'alpha':alpha,'lambda_internal':base['base_Ising_scale_S']*alpha,'raw_feasible_rate':rate,'next_alpha':nxt});pool.extend(events_batch.tolist())
      events=np.array(pool)
     seconds=time.perf_counter()-start;r=score_events(events,base,ctx);r.update(method=method,samples=n,repetition=rep,sampling_seconds=seconds,batches=batches);record['conditions'].append(r)
    save(target,record);print('finished',seed,method,n,flush=True)
  record['status']='completed';save(target,record)

def stat(vals):
 vals=[v for v in vals if v is not None];return {'median':float(np.median(vals)) if vals else None,'q1':float(np.quantile(vals,.25)) if vals else None,'q3':float(np.quantile(vals,.75)) if vals else None}

def analyze():
 rows=[];verified=0
 for seed in P['seeds']:
  d=json.loads((OUT/f'json/seed_{seed}.json').read_text());base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());ctx=context(base);assert d['status']=='completed' and d['source_sha256']==sha(PILOT/f'json/seed_{seed}.json')
  assert len(d['conditions'])==800
  for method,n in itertools.product(METHODS,P['sample_counts']):
   rs=[r for r in d['conditions'] if r['method']==method and r['samples']==n];assert len(rs)==100 and {r['repetition'] for r in rs}==set(range(100))
   for r in rs:
    assert len(r['candidate_events'])==n;check=score_events(r['candidate_events'],base,ctx)
    for key in KEYS:assert check[key]==r[key]
    verified+=1
   metrics={};defined={}
   for key in KEYS:
    vals=[r[key] for r in rs if r[key] is not None];metrics[key]=float(np.mean(vals)) if vals else None;defined[key]=len(vals)
   rows.append({'seed':seed,'method':method,'samples':n,'metrics':metrics,'defined_repetitions':defined,'initial_true_optimum_present':d['initial_true_optimum_present']})
 groups=[]
 for method,n in itertools.product(METHODS,P['sample_counts']):
  rs=[r for r in rows if r['method']==method and r['samples']==n];groups.append({'method':method,'samples':n,'statistics':{k:stat([r['metrics'][k] for r in rs]) for k in KEYS},'defined_repetitions':{k:sum(r['defined_repetitions'][k] for r in rs) for k in KEYS}})
 tests=[]
 for method,key in itertools.product(METHODS,KEYS[:3]):
  diffs=[next(r['metrics'][key] for r in rows if r['seed']==seed and r['method']==method and r['samples']==100)-next(r['metrics'][key] for r in rows if r['seed']==seed and r['method']==method and r['samples']==10) for seed in P['seeds']]
  p=float(wilcoxon(diffs).pvalue) if np.any(diffs) else 1.;tests.append({'method':method,'metric':key,'contrast':'100_minus10','differences':diffs,'median_difference':float(np.median(diffs)),'p_raw':p,'p_bonferroni':min(1.,p*12)})
 result={'status':'completed','conditions':verified,'seed_rows':rows,'groups':groups,'paired_tests':tests};save(OUT/'json/summary.json',result);save(OUT/'json/artifact_verification.json',{'status':'passed','conditions':verified,'checks':'all model/method/count/rep coverage,source hashes,raw event scoring parity and source coarse-grain equivalence'})
 lines=['# 23変数・固定FMの10/100出力再計算','','初期20件・5FM・各100反復。元の9点選択角度を固定。1件採用。各モデルの反復獲得頻度を集計し、5モデル中央値 [Q1,Q3]を表示。全4000条件。','','| 手法 | 出力数 | 全192候補のFM最小解獲得 | 新規実行可能候補獲得 | 採用真値eV（成功時） |','|---|---:|---:|---:|---:|']
 for g in groups:
  fmt=lambda key: '未定義' if g['statistics'][key]['median'] is None else '{median:.5g} [{q1:.5g},{q3:.5g}]'.format(**g['statistics'][key])
  lines.append(f"| {g['method']} | {g['samples']} | {fmt('global_fm_minimum_hit')} | {fmt('unseen_success')} | {fmt('selected_true_gap')} |")
 lines+=['','獲得率は無効・重複しか得られない試行も含む。真値は候補採用できた試行の条件付き平均をモデル内で求めたもの。標準QAOAの条件付き真値は成功試行が少ないため母数をsummaryに記録。全192候補のFM最小解と未評価候補内のFM最小解を両方保存。真の最適解の新規採用と初期データにある場合を区別。角度再探索・再学習・BBOは未実施。']
 (OUT/'md/RESULTS.md').write_text('\n'.join(lines)+'\n')
 plot(result);print('verified',verified,flush=True)

def plot(d):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 styles=[('blue','o'),('orange','s'),('red','^'),('green','D')]
 panels=[('global_fm_minimum_hit','P(FM minimum acquired)','fm_minimum_capture'),('unseen_success','P(new feasible candidate)','candidate_success'),('selected_fm_gap','Selected FM gap (eV)','selected_fm_gap')]
 def panel(ax,key,label):
  for method,(color,marker) in zip(METHODS,styles):
   g=[next(g for g in d['groups'] if g['method']==method and g['samples']==n)['statistics'][key] for n in P['sample_counts']]
   val=lambda k:np.array([np.nan if s[k] is None else s[k] for s in g])
   ax.plot([10,100],val('median'),color=color,marker=marker,label=method);ax.fill_between([10,100],val('q1'),val('q3'),color=color,alpha=.12)
  ax.set_xscale('log');ax.set_xticks([10,100],['10','100']);ax.set_xlabel('Output samples / shots');ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
  if key.startswith('global') or key=='unseen_success':ax.set_ylim(-.025,1.025)
 def export(fig,name):
  for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{ext}/{name}.{ext}',dpi=250,bbox_inches='tight')
  plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
 for ax,(key,label,name) in zip(axs,panels):panel(ax,key,label)
 export(fig,'sample_counts_overview')
 for key,label,name in panels:
  fig,ax=plt.subplots(figsize=(5.5,4.2),layout='constrained');panel(ax,key,label);export(fig,name)

if __name__=='__main__':
 arg=argparse.ArgumentParser();arg.add_argument('stage',choices=['prevalidate','run','analyze']);args=arg.parse_args();globals()[args.stage]()
