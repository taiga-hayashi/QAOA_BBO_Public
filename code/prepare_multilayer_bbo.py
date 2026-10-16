"""Freeze and validate the user-selected five-seed closed-loop pilot."""
import hashlib
import importlib.metadata
import itertools
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from config import CODE_DIR, load_config
from evaluator import load_problem
from fm_model import model_sha256
from fm_target_scaling import train_scaled_fm, qubo_original_scale
from qubo import fm_to_qubo, qubo_to_ising_scale, build_onehot_penalty_bqm
from qarp_backend import OpenQARPCompactXYQAOA, OpenQARPStandardQAOA
from shared_numerics import _build_qubo_energies

OUT = CODE_DIR / 'results/multilayer_filter_bbo_pilot5'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    for sub in ['json','logs','md','checkpoints']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    if (OUT/'json/protocol.json').exists():raise FileExistsError('Existing frozen protocol; do not overwrite.')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CODE_DIR/'tests'),'-v'],check=True)
    cfg=load_config(['-c',str(CODE_DIR/'config_multilayer_filter.json')])
    cfg['output_dir']='results/multilayer_filter_bbo_pilot5'
    cfg['n_seeds']=5;cfg['seeds']=[42,101,2024,7,19]
    cfg['methods']=['Exact-FM','Adaptive-FMQA','LargePenalty-FMQA','Penalty-FMQAOA','XY-FMQAOA'] + [f'{family}(p={depth},mean,powell49)' for depth in [2,3] for family in ['Penalty-FMQAOA','XY-FMQAOA']]
    cfg['problem'].update(initial_samples=40,bbo_cycles=20)
    cfg['penalty']['adaptive_initial_alpha']=1.
    cfg['qaoa'].update(depth=1,objective='mean',angle_search_mode='powell49',standard_depth=1,
                       standard_objective='mean',standard_angle_search_mode='powell49')
    p=load_problem(cfg['problem']);initials={};models={}
    for seed in cfg['seeds']:
        ids=p.initial_dataset(seed,40);ys=[p.evaluate_id(i) for i in ids]
        initials[str(seed)]={'ids':ids,'values':ys}
        m,transform=train_scaled_fm(p.patterns[ids].astype(np.float32),ys,p.n_bits,cfg['fm'],seed,cfg['fm'].get('target_standardization',False))
        models[str(seed)]=model_sha256(m)
        if seed != 42:continue
        _,q,offset=fm_to_qubo(m);q,offset=qubo_original_scale(q,offset,transform);scale=qubo_to_ising_scale(q);q=q/max(scale,1e-8)
        gamma,beta=.31,.27
        # Independent feasible-basis amplitude update, without QARP mixer matrices.
        labels=np.array(list(itertools.product(range(4),repeat=6)))
        energy=np.einsum('bi,ij,bj->b',p.patterns,q,p.patterns)
        state=np.exp(-1j*gamma*energy)/np.sqrt(4096)
        powers=4**np.arange(5,-1,-1)
        for group in range(6):
            for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                left=np.flatnonzero(labels[:,group]==a)
                right=left+(b-a)*powers[group]
                u,v=state[left].copy(),state[right].copy()
                state[left]=np.cos(beta)*u-1j*np.sin(beta)*v
                state[right]=-1j*np.sin(beta)*u+np.cos(beta)*v
        compact=OpenQARPCompactXYQAOA(q,p.group_sizes)
        probs=compact.feasible_probabilities([gamma],[beta])
        np.testing.assert_allclose(probs,abs(state)**2,atol=1e-10,rtol=0)
        assert abs(probs.sum()-1)<1e-10
        qd={(i,j):q[i,j] for i in range(24) for j in range(i,24)}
        total,off=build_onehot_penalty_bqm(qd,p.group_sizes,5.).to_qubo();mat=np.zeros_like(q)
        for (i,j),v in total.items():mat[min(i,j),max(i,j)]+=v
        t=time.perf_counter();energies=_build_qubo_energies(mat,off,np.dtype(float))
        energy_seconds=time.perf_counter()-t
        ids_check=np.r_[0,p.valid_basis_indices,np.random.default_rng(24).integers(0,1<<24,1000)] .astype(np.uint64)
        bits=((ids_check[:,None]>>np.arange(24,dtype=np.uint64))&1)
        target=np.einsum('bi,ij,bj->b',bits,q,bits)+5*((bits.reshape(-1,6,4).sum(2)-1.)**2).sum(1)
        np.testing.assert_allclose(energies[ids_check.astype(int)],target,atol=1e-10,rtol=0)
        depth_smokes=[]
        for depth in [1,2,3]:
            gs=[gamma]*depth;bs=[beta]*depth
            t=time.perf_counter();sv=OpenQARPStandardQAOA(q,p.group_sizes,5.).statevector(gs,bs)
            smoke_seconds=time.perf_counter()-t
            norm=float(np.vdot(sv,sv).real);assert abs(norm-1)<1e-10
            cp=compact.feasible_probabilities(gs,bs);assert abs(cp.sum()-1)<1e-10
            depth_smokes.append({'depth':depth,'standard_one_angle_seconds':smoke_seconds,'standard_state_norm':norm,'xy_probability_sum':float(cp.sum())})
            if depth<3:del sv
        expectation=0.
        for start in range(0,len(sv),65536):expectation+=float((abs(sv[start:start+65536])**2)@energies[start:start+65536])
        assert np.isfinite(expectation)
        smoke={'standard_N24_one_angle_seconds':smoke_seconds,'energy_construction_seconds':energy_seconds,
               'state_norm':norm,'energy_expectation':expectation,'compact24_reference_max_error':float(max(abs(probs-abs(state)**2))),
               'depth_checks':depth_smokes,'standard_energy_bytes':int(energies.nbytes),'statevector_bytes':int(sv.nbytes)}
        del sv,energies
    save(OUT/'json/run_config.json',cfg)
    per_seed_configs={}
    for seed in cfg['seeds']:
        local=json.loads(json.dumps(cfg));local['seeds']=[seed];local['n_seeds']=1
        local['output_dir']=f'results/multilayer_filter_bbo_pilot5/seeds/seed{seed}'
        path=OUT/f'json/run_config_seed{seed}.json';save(path,local);per_seed_configs[str(seed)]=sha(path)
    sources={str(f.relative_to(CODE_DIR)):sha(f) for f in sorted(CODE_DIR.glob('*.py'))}
    sources.update({str(f.relative_to(CODE_DIR)):sha(f) for f in sorted((CODE_DIR/'tests').glob('*.py'))})
    save(OUT/'json/prevalidation.json',{'status':'passed','tests':'N6/N9/N8 independent full-state circuit/compact/penalty/Ising;FM-QUBO;checkpoint resume;N24 learned-FM smoke',
        'smoke':smoke,'step0_model_sha256':models,'sources':sources})
    packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','dimod','dwave-neal','openqarp','matplotlib']}
    save(OUT/'json/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,
                                    'threads':{'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,'MKL_NUM_THREADS':1},'cpu':'local CPU simulation, no QPU'})
    protocol={'status':'ready','scope':'five-seed exploratory closed-loop BBO;not fixed-FM sampler study',
        'problem_id':p.problem_id,'categorical_variables':6,'onehot_variables':24,'feasible_candidates':4096,
        'per_seed_configs':per_seed_configs,'config_sha256':sha(OUT/'json/run_config.json'),'dataset_sha256':p.data_sha256,'schema_sha256':sha(p.config_path),
        'sources':sources,'initial_data':initials,'step0_model_sha256':models,'methods':cfg['methods'],
        'evaluation':'pinned precomputed optical table;no fresh TMM;lookup count is BB budget;simulation timing is not optical timing',
        'budget':'40 shared initial evaluations +20 proposal attempts;at most60 evaluations per method and seed;no refill/no extrapolation',
        'angles':'X and XY p1/p2/p3,Mean,powell49:9 embedded legacy+16 global+24 Powell;all layer angles optimized,100shots;lambda_XY=0;alpha_X=5',
        'adaptive':'canonical tracker: start1,min.01,max100;rawfeasible<.7:x2,<1:x1.25,otherwise x.85',
        'sa':'100 reads,1000 sweeps,neal geometric auto beta;lambda=S*alpha',
        'truth_isolation':'FM learns only evaluated points;all-truth ranks/regret are post-selection diagnostics,not inputs to sampler',
        'exact_fm':'all4096 FM predictions,lowest unseen;upper reference for surrogate minimization,not true BBO upper bound',
        'statistics':'5-seed median/IQR;all36 paired method differences,Wilcoxon signed-rank with Bonferroni;small-sample exploratory',
        'primary_metric':'final true optical FOM and normalized regret after20 proposal attempts;report actual evaluation counts',
        'checkpoint':'atomic JSON every completed proposal;latest model each method/seed;deterministic reconstruction on restart',
        'failure_policy':'retain partial histories/tracebacks;48hour cap per seed task;dependent audit runs only after all5 successful seed tasks',
        'resources':'one task,one worker,one BLAS/OpenMP thread,nice10;existing paused queue untouched',
        'source_license':'local research only;source/optical table usage terms unresolved;no publication'}
    save(OUT/'json/protocol.json',protocol)
    source_dir=OUT/'data/source_snapshot';source_dir.mkdir(parents=True,exist_ok=True)
    for name in sources:
        dst=source_dir/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(CODE_DIR/name,dst)
    print('READY',OUT,smoke,flush=True)


if __name__=='__main__':main()
