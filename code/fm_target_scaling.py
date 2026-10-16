"""Training-only target scaling around the shared FM implementation."""
import numpy as np
import torch
from fm_model import train_fm


def fit_target_transform(y,standardize):
    values=np.asarray(y,dtype=float)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():
        raise ValueError('Finite nonempty training targets required')
    if not isinstance(standardize,(bool,np.bool_)):raise ValueError('standardize must be boolean')
    mean=float(values.mean()) if standardize else 0.
    std=float(values.std(ddof=0)) if standardize else 1.
    constant=bool(standardize and std<=1e-12)
    scale=1. if constant else std
    metadata={'standardized':bool(standardize),'mean':mean,'scale':scale,'constant_training_targets':constant,
              'training_count':len(values),'std_convention':'population ddof=0;unit scale if std<=1e-12'}
    return (values-mean)/scale,metadata


def train_scaled_fm(x,y,d,params,seed,standardize):
    target,transform=fit_target_transform(y,standardize)
    model=train_fm(x,target,d,params,seed)
    return model,transform


def predict_original(model,x,transform):
    with torch.no_grad():prediction=model(torch.tensor(x,dtype=torch.float32)).cpu().numpy().astype(float)
    return prediction*transform['scale']+transform['mean']


def qubo_original_scale(q,offset,transform):
    """Preserve the raw-FOM surrogate if this model is later used in a QUBO."""
    return np.asarray(q)*transform['scale'],float(offset)*transform['scale']+transform['mean']
