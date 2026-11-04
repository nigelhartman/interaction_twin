"""Internal rigidity diagnostics; no metric calibration or pose-accuracy claims."""
import numpy as np

def pixel_intrinsics(normalized, width, height):
    """MoGe uses normalized pixel centres (u+0.5)/W; our image indices start at 0."""
    k=np.diag([width,height,1.])@np.asarray(normalized,dtype=float)
    k[0,2]-=.5;k[1,2]-=.5
    return k

def pair_distances(a,b,valid,ids):
    ids=np.asarray(ids)[np.asarray(valid)[ids]]
    if len(ids)<4:return np.array([]),np.array([])
    da=np.linalg.norm(a[ids,None]-a[None,ids],axis=-1)
    db=np.linalg.norm(b[ids,None]-b[None,ids],axis=-1)
    upper=np.triu(np.ones(da.shape,bool),1)
    minimum=np.percentile(da[upper],25)
    selected=upper&(da>max(minimum,1e-8))&np.isfinite(da)&np.isfinite(db)&(db>1e-8)
    return da[selected],db[selected]

def estimate_depth_scale(reference,current,valid,training_ids):
    a,b=pair_distances(reference,current,valid,training_ids)
    if len(a)<6:raise ValueError('Insufficient training point pairs')
    return float(np.exp(np.median(np.log(a/b))))

def heldout_shape_error(reference,current,valid,heldout_ids):
    a,b=pair_distances(reference,current,valid,heldout_ids)
    if len(a)<6:return None
    return float(np.median(np.abs(b/a-1)))
