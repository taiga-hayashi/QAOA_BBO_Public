"""Layer-count generalization of experiments/mlfilter/py/evaluator.py.

Optics, FOM and float32 input parsing are unchanged; only the 1..6 layer
limit is lifted and inputs are read from data/inputs/mlfilter/v1-public-code
after checking its manifest. No surrogate or candidate-value table is used.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
INPUT = PROJECT/'data/inputs/mlfilter/v1-public-code'
sys.path.insert(0, str(INPUT/'vendor/site-packages'))
from tmm_fast import coh_tmm

MAX_LAYERS = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def verify_input():
    manifest = json.loads((INPUT/'manifest.json').read_text())
    for item in manifest['files']:
        assert sha(INPUT/item['path']) == item['sha256'], item['path']
    return sha(INPUT/'manifest.json')


class FilterEvaluator:
    materials = ('SiO2','Si3N4','Al2O3','TiO2')

    def __init__(self, layers):
        if isinstance(layers,bool) or not isinstance(layers,(int,np.integer)) or not 1 <= layers <= MAX_LAYERS:
            raise ValueError(f'layers must be an integer from 1 through {MAX_LAYERS}')
        self.layers = int(layers)
        self.input_manifest_sha256 = verify_input()
        ref = INPUT/'reference'
        # Match the public evaluator's float32 input parsing, then promote to float64.
        self.index = np.loadtxt(ref/'dielectric_ref.txt',delimiter=',').astype(np.float32).astype(float)
        self.solar = np.loadtxt(ref/'solar_spectrum.txt',delimiter=',').astype(np.float32).astype(float)
        assert self.index.shape == (4,2201) and self.solar.shape == (2201,)
        assert np.isfinite(self.index).all() and (self.index>0).all()
        assert np.isfinite(self.solar).all() and (self.solar>=0).all() and (self.solar>0).any()
        self.wavelength_nm = np.arange(300,2501,dtype=float)
        self.wavelength = np.linspace(300,2500,2201)*1e-9
        self.ideal = ((self.wavelength_nm>=400)&(self.wavelength_nm<=750)).astype(float)
        self.solar_norm = float(np.sum(self.solar.astype(np.float32)**2))

    def labels(self, values):
        a = np.asarray(values)
        if a.ndim not in [1,2] or a.shape[-1] != self.layers:
            raise ValueError('one material index per layer required')
        if a.dtype.kind not in 'iu' or np.any((a<0)|(a>3)):
            raise ValueError('material indices must be integers 0..3')
        return a.reshape(-1,self.layers).astype(np.int64)

    def encode(self, labels):
        a = self.labels(labels)
        return np.eye(4,dtype=np.int8)[a].reshape(len(a),4*self.layers)

    def decode(self, bits):
        a = np.asarray(bits)
        if a.ndim not in [1,2] or a.shape[-1] != 4*self.layers or not np.isin(a,[0,1]).all():
            raise ValueError('binary one-hot input required')
        a = a.reshape(-1,self.layers,4)
        if not (a.sum(axis=2)==1).all():
            raise ValueError('exactly one material per layer required')
        return a.argmax(axis=2)

    def spectra(self, labels):
        a = self.labels(labels)
        n = np.empty((len(a),self.layers+2,2201),dtype=float)
        n[:,0,:] = 1.4
        n[:,-1,:] = 1.45
        n[:,1:-1,:] = self.index[a]
        thickness = np.full((len(a),self.layers+2),50e-9)
        thickness[:,[0,-1]] = np.inf
        result = coh_tmm('p',n,thickness,np.array([0.]),self.wavelength,device='cpu')
        t,r = result['T'][:,0,:],result['R'][:,0,:]
        if not np.isfinite(t).all() or not np.isfinite(r).all():
            raise ArithmeticError('nonfinite spectrum')
        if max(float(np.max(abs(t+r-1))),float(max(0.,-t.min(),-r.min()))) > 1e-10:
            raise ArithmeticError('lossless power conservation failed')
        return t,r

    def fom(self, transmission):
        t = np.asarray(transmission)
        if t.ndim != 2 or t.shape[1] != 2201 or not np.isfinite(t).all():
            raise ValueError('finite batch of 2201 transmission samples required')
        # Public-code squared weighted residual, not the printed paper Eq.(6).
        return 10*np.sum(((t-self.ideal)*self.solar)**2,axis=1)/self.solar_norm

    def evaluate(self, labels):
        return self.fom(self.spectra(labels)[0])
