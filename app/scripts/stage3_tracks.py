#!/usr/bin/env python3
"""Explicit CPU optical-flow baseline; lift into uncalibrated frame-local depth."""
import argparse,hashlib,json,os,re,sys,time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from geometry import lift_pixels,rigid_fit

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def track_sequence(images,masks,anchor,seed):
    """Track both directions. Once lost, a point stays unknown; no re-identification."""
    total=len(images);n=len(seed)
    uv=np.full((total,n,2),np.nan,dtype=np.float32);valid=np.zeros((total,n),bool);fb=np.full((total,n),np.nan,dtype=np.float32)
    uv[anchor]=seed;valid[anchor]=True;fb[anchor]=0
    for direction in [1,-1]:
        previous=anchor
        for current in range(anchor+direction,total if direction==1 else -1,direction):
            ids=np.flatnonzero(valid[previous])
            if not len(ids):break
            p=uv[previous,ids].reshape(-1,1,2)
            q,s,_=cv2.calcOpticalFlowPyrLK(images[previous],images[current],p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            if q is None:break
            back,bs,_=cv2.calcOpticalFlowPyrLK(images[current],images[previous],q,None,winSize=(21,21),maxLevel=3)
            if back is None:break
            q=q[:,0];errors=np.linalg.norm(back[:,0]-p[:,0],axis=1)
            good=s[:,0].astype(bool)&bs[:,0].astype(bool)&np.isfinite(q).all(axis=1)&(errors<=1.5)
            h,w=masks[current].shape
            good&=(q[:,0]>=0)&(q[:,0]<=w-1)&(q[:,1]>=0)&(q[:,1]<=h-1)
            inside=np.flatnonzero(good);pixel=np.rint(q[inside]).astype(int)
            good[inside]&=masks[current][pixel[:,1],pixel[:,0]]
            accepted=ids[good];uv[current,accepted]=q[good];valid[current,accepted]=True;fb[current,ids]=errors
            previous=current
    return uv,valid,fb

def track_cotracker(out,rows,masks,anchor,seed,device):
    import torch
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('CPU operator fallback must be disabled')
    os.environ['HF_HOME']=str(ROOT/'models/hf-cache');os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    from huggingface_hub import HfApi,hf_hub_download
    from cotracker.predictor import CoTrackerPredictor
    from importlib.metadata import distribution
    checkpoint_dir=ROOT/'models/cotracker3';checkpoint_dir.mkdir(exist_ok=True)
    provenance_path=checkpoint_dir/'provenance.json'
    if not provenance_path.exists():
        revision=HfApi().model_info('facebook/cotracker3',token=False).sha
        path=Path(hf_hub_download('facebook/cotracker3','scaled_offline.pth',revision=revision,local_dir=checkpoint_dir,token=False))
        package=json.loads(distribution('cotracker').read_text('direct_url.json'))
        write(provenance_path,{'repo':'facebook/cotracker3','revision':revision,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'code_commit':package['vcs_info']['commit_id']})
    provenance=read(provenance_path);checkpoint=checkpoint_dir/'scaled_offline.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=provenance['checkpoint_sha256']:raise ValueError('CoTracker checkpoint changed')
    model=CoTrackerPredictor(checkpoint=str(checkpoint)).to(device).eval()
    model.interp_shape=(256,384) # Explicit reduced-resolution local test.
    frames=np.stack([np.array(Image.open(out/r['image']).convert('RGB')) for r in rows])
    video=torch.from_numpy(frames).permute(0,3,1,2)[None].float().to(device)
    queries=torch.from_numpy(np.c_[np.full(len(seed),anchor),seed].astype(np.float32))[None].to(device)
    started=time.monotonic()
    with torch.inference_mode():
        prediction,visibility=model(video,queries=queries,backward_tracking=True)
        if device=='mps':torch.mps.synchronize()
        raw=prediction[0].cpu().numpy();model_visible=visibility[0].cpu().numpy().astype(bool)
    valid=model_visible.copy()
    for j in range(len(rows)):
        q=raw[j];h,w=masks[j].shape
        valid[j]&=np.isfinite(q).all(axis=1)&(q[:,0]>=0)&(q[:,0]<=w-1)&(q[:,1]>=0)&(q[:,1]<=h-1)
        ids=np.flatnonzero(valid[j]);pixel=np.rint(q[ids]).astype(int);valid[j,ids]&=masks[j][pixel[:,1],pixel[:,0]]
    uv=raw.copy();uv[~valid]=np.nan
    details={'model_provenance':provenance,'inference_s':round(time.monotonic()-started,3),'model_input_hw':[256,384],'cpu_operator_fallback_enabled':False,'visibility_policy':'official predictor visibility gate plus image bounds and SAM membership; not measured accuracy'}
    return uv,valid,np.full(valid.shape,np.nan,dtype=np.float32),{'raw_model_uv':raw,'model_visibility':model_visible},details

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='camera-battery-action-5fps');p.add_argument('--run',default='local-battery-v1');p.add_argument('--anchor-sample',type=int,default=31);p.add_argument('--backend',choices=['lk','cotracker'],default='lk');p.add_argument('--device',choices=['mps','cpu'],default='mps');a=p.parse_args()
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*',s) for s in [a.case,a.run]):p.error('Invalid case/run')
    out=ROOT/'data'/a.case/'geometry'/a.run;pack=read(out/'experiment.json');rows=pack['samples'];depth=read(out/'depth/metadata.json')
    if depth['status']!='complete':raise ValueError('Depth is incomplete')
    dest=out/('motion' if a.backend=='lk' else 'motion-cotracker');dest.mkdir(exist_ok=False);started=time.monotonic()
    anchor=next(i for i,r in enumerate(rows) if r['sample_index']==a.anchor_sample)
    images=[cv2.imread(str(out/r['image']),cv2.IMREAD_GRAYSCALE) for r in rows]
    masks=[np.asarray(Image.open(out/r['mask']))>0 for r in rows]
    seed_mask=cv2.erode(masks[anchor].astype(np.uint8),np.ones((5,5),np.uint8))*255
    points=cv2.goodFeaturesToTrack(images[anchor],maxCorners=64,qualityLevel=.003,minDistance=5,mask=seed_mask,blockSize=5)
    if points is None or len(points)<6:raise ValueError('Not enough interior features; no invented point tracks')
    extras={};details={}
    if a.backend=='lk':uv,valid,fb=track_sequence(images,masks,anchor,points[:,0])
    else:uv,valid,fb,extras,details=track_cotracker(out,rows,masks,anchor,points[:,0],a.device)
    xyz=np.full((len(rows),len(points),3),np.nan,dtype=np.float32)
    for j,row in enumerate(rows):
        xyz[j]=lift_pixels(uv[j],np.load(out/'depth'/f"{row['sample_index']:06d}.npy"),pack['camera']['intrinsics'])
    np.savez_compressed(dest/'tracks.npz',uv=uv,valid=valid,xyz_camera=xyz,forward_backward_error_px=fb,sample_indices=np.array([r['sample_index'] for r in rows]),timestamps_s=np.array([r['timestamp_s'] for r in rows]),**extras)
    reference_spread=float(np.percentile(np.linalg.norm(xyz[anchor]-np.median(xyz[anchor],axis=0),axis=1),90)*2)
    diagnostics=[]
    for j,row in enumerate(rows):
        good=valid[j]&np.isfinite(xyz[j]).all(axis=1)
        rec={'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'tracked_points':int(good.sum()),'status':'insufficient_correspondences','pose_exported':False}
        if good.sum()>=6:
            try:
                transform,errors=rigid_fit(xyz[anchor,good],xyz[j,good])
                ratio=float(np.median(errors)/reference_spread)
                rec.update(status='diagnostic_only_low_residual' if ratio<=.1 else 'depth_shape_inconsistent',median_rigid_residual_model_units=float(np.median(errors)),residual_fraction_of_reference_span=ratio,diagnostic_transform=transform.tolist())
            except ValueError:
                rec['status']='degenerate_correspondences'
        diagnostics.append(rec)
    nonanchor=[r for i,r in enumerate(diagnostics) if i!=anchor]
    stable=sum(r['status']=='diagnostic_only_low_residual' for r in nonanchor)
    metadata={'status':'complete','backend':'OpenCV pyramidal Lucas-Kanade','provider_role':'CPU classical baseline, not TAPIP3D or a learned 3D tracker','device':'cpu','opencv':cv2.__version__,'source_sha256':pack['source_sha256'],'parent_mask_run':pack['parent_mask_run'],'track_id':pack['track_id'],'anchor_sample':a.anchor_sample,'seed_count':len(points),'frame_count':len(rows),'wall_time_s':round(time.monotonic()-started,3),'coordinate_frame':'per-frame recording-camera coordinates; filming-camera motion NOT compensated','intrinsics_calibrated':False,'metric_scale_verified':False,'full_component_poses_exported':False,'forward_backward_threshold_px':1.5,'identity_policy':'local feature indices only; no reacquisition after loss','frames':diagnostics,'low_residual_nonanchor_frames':stable,'evaluated_nonanchor_frames':len(nonanchor),'limitations':['Optical-flow consistency and SAM membership are quality gates, not ground-truth correspondence accuracy.','3D tracks inherit independent per-frame depth errors and assumed intrinsics.','Rigid transforms are diagnostic fits only, not accepted alignment or recovered physical motion.','Symmetry, hand overlap, blur and segmentation leakage can produce plausible wrong tracks.']}
    if a.backend=='cotracker':
        metadata.update(backend='Meta CoTracker3 offline',provider_role='local learned 2D point tracker, not TAPIP3D',device=a.device,forward_backward_threshold_px=None,identity_policy='learned feature correspondences, visibility and SAM gated; gaps remain unknown; no physical identity confirmation',**details)
    write(dest/'metadata.json',metadata);print(json.dumps({'backend':metadata['backend'],'frames':len(rows),'seed_count':len(points),'tracked_point_counts':[r['tracked_points'] for r in diagnostics],'low_residual_nonanchor_frames':stable,'wall_time_s':metadata['wall_time_s']},indent=2))

if __name__=='__main__':main()
