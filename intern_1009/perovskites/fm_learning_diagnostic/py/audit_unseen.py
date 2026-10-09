"""Verify unobserved feature parameters remain at initialization without regularization."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from diagnose import OUT,P,PILOT,context,save
from fm import TorchFM
rows=[]
for seed in P['seeds']:
 base=json.loads((PILOT/f'json/seed_{seed}.json').read_text());cats,x,y,tr,te=context(base);missing=np.flatnonzero(x[tr].sum(axis=0)==0)
 torch.manual_seed(seed);initial=TorchFM(d=23,k=2).state_dict();trained=torch.load(PILOT/base['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
 ve=float(torch.max(torch.abs(initial['V'][missing]-trained['V'][missing])));we=float(torch.max(torch.abs(initial['lin.weight'][:,missing]-trained['lin.weight'][:,missing])))
 assert ve==0 and we==0
 rows.append({'seed':seed,'missing_feature_indices':missing.tolist(),'missing_organic':sorted({c[0] for c in cats}-{cats[i][0] for i in tr}),'V_max_change':ve,'linear_max_change':we})
save(OUT/'json/unseen_parameter_audit.json',{'status':'passed','scope':'original raw rank2 no-regularization FM','rows':rows});print(rows)
