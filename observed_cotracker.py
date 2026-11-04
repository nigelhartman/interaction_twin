"""Versioned, mask-constrained CoTracker correspondences for observed SfM.

TRAIN images alone produce geometry tracks. Each held-out image is processed
after geometry fitting in an isolated reference/test pair. Learned tracking
does not establish physical identity, rigidity, calibrated shape or motion.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

import observed_geometry as G


PROTOCOL = {
    'feature_provider': 'cotracker3_mask_core_v2', 'version': 2,
    'maximum_queries': 384, 'model_maximum_side_px': 512,
    'query_core_min_px': 3., 'query_core_radius_fraction': .08,
    'observation_core_min_px': 2., 'observation_core_radius_fraction': .05,
    'query_minimum_texture_std': 4., 'query_minimum_spacing_px': 3.,
    'maximum_cycle_error_source_px': 2.5,
    'minimum_training_observations_per_track': 3,
    'crop_margin_fraction': .20, 'crop_margin_min_px': 16,
    'crop_derived_from_training_masks_only': True,
    'reference_selection': 'maximum_training_mask_core_area_times_log1p_masked_laplacian_variance',
    'loss_policy': 'first failed mask/visibility/cycle gate stops support in each direction; no reacquisition',
    'reverse_cycle_endpoint_policy': 'farthest contiguous forward-supported observation; never a reappearance beyond support loss',
    'geometry_tracking_images': 'training frames only; heldout frames are never in the training tracking tensor',
    'heldout_tracking': 'isolated closest-registered-training-reference plus one heldout frame, after geometry freeze',
    'heldout_point_split': 'unchanged engine sha256 track-ID anchors vs scores',
    'device': 'mps', 'cpu_operator_fallback': False, 'model_downloads': False,
    'learned_outputs': '2D pixel tracks/visibility only; no predicted depth, mesh or old pose',
    'new_api_calls': 0,
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def core_mask(mask, minimum, fraction):
    distance = cv2.distanceTransform((np.asarray(mask)>0).astype(np.uint8), cv2.DIST_L2, 5)
    margin = max(float(minimum), float(distance.max())*fraction)
    return distance >= margin, margin


def automatic_queries(image, mask, maximum=384):
    core, margin = core_mask(mask, PROTOCOL['query_core_min_px'], PROTOCOL['query_core_radius_fraction'])
    if not core.any():
        return np.empty((0,2),np.float32), margin
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = cv2.goodFeaturesToTrack(gray, maxCorners=maximum, qualityLevel=.002, minDistance=3,
                                  mask=core.astype(np.uint8)*255, blockSize=3)
    proposals = [] if found is None else found[:,0].tolist()
    ys,xs = np.where(core)
    spacing = max(3,int(math.ceil(math.sqrt(core.sum()/maximum))))
    proposals += [[float(x),float(y)] for y in range(int(ys.min()),int(ys.max())+1,spacing)
                  for x in range(int(xs.min()),int(xs.max())+1,spacing) if core[y,x]]
    result=[]
    for x,y in proposals:
        ix,iy=int(round(x)),int(round(y))
        if ix<3 or iy<3 or ix>=gray.shape[1]-3 or iy>=gray.shape[0]-3:
            continue
        if gray[iy-3:iy+4,ix-3:ix+4].std()<PROTOCOL['query_minimum_texture_std']:
            continue
        if result and np.min(np.linalg.norm(np.asarray(result)-[x,y],axis=1))<PROTOCOL['query_minimum_spacing_px']:
            continue
        result.append([x,y])
        if len(result)>=maximum:
            break
    return np.asarray(sorted(result,key=lambda p:(p[1],p[0])),np.float32).reshape(-1,2),margin


def training_crop(frames):
    train=[f for f in frames if f['split']=='train']
    if not train:
        raise ValueError('No training masks for a source-pixel crop')
    h,w=train[0]['mask'].shape
    bounds=[]
    for frame in train:
        ys,xs=np.where(frame['mask']>0)
        if len(xs): bounds.append([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
    if not bounds:
        raise ValueError('All training component masks are empty')
    b=np.asarray(bounds);lo=b[:,:2].min(axis=0);hi=b[:,2:].max(axis=0)
    margin=np.maximum(PROTOCOL['crop_margin_min_px'],np.ceil((hi-lo)*PROTOCOL['crop_margin_fraction'])).astype(int)
    lo=np.maximum(0,lo-margin);hi=np.minimum([w,h],hi+margin)
    return tuple(int(v) for v in [*lo,*hi])


def mask_support(uv, visible, frames):
    uv=np.asarray(uv,float);visible=np.asarray(visible,bool)
    if uv.shape[:2]!=visible.shape or uv.shape[0]!=len(frames) or uv.shape[2]!=2:
        raise ValueError('Tracking arrays must be frame-by-point source pixels')
    support=np.zeros_like(visible)
    for i,frame in enumerate(frames):
        core,_=core_mask(frame['mask'],PROTOCOL['observation_core_min_px'],PROTOCOL['observation_core_radius_fraction'])
        h,w=core.shape
        finite=np.isfinite(uv[i]).all(axis=1)
        ids=np.flatnonzero(finite & visible[i] & (uv[i,:,0]>=0)&(uv[i,:,0]<=w-1)&(uv[i,:,1]>=0)&(uv[i,:,1]<=h-1))
        points=np.rint(uv[i,ids]).astype(int)
        support[i,ids]=core[points[:,1],points[:,0]]
    return support


def no_reacquisition(support, anchor):
    valid=np.asarray(support,bool).copy()
    for direction in (-1,1):
        previous=anchor
        for index in range(anchor+direction,len(valid) if direction>0 else -1,direction):
            valid[index]&=valid[previous]
            previous=index
    return valid


def cycle_endpoints(support,anchor):
    contiguous=no_reacquisition(support,anchor)
    endpoints=[]
    for point in range(contiguous.shape[1]):
        positions=np.flatnonzero(contiguous[:,point])
        endpoints.append(int(max(positions,key=lambda i:abs(int(i)-anchor))) if len(positions) else int(anchor))
    return endpoints


class CorrespondenceProvider:
    def __init__(self,root,device='mps'):
        if device!='mps': raise ValueError('CoTracker provider requires explicit MPS; no fallback')
        self.root=Path(root).resolve();self.device=device;self.model=None;self.output=None
        self.tracks={};self.training=[];self.crop=None;self.calls=[];self.heldout=[]
        self.checkpoint=self.root/'models/cotracker3/scaled_offline.pth'
        self.model_provenance=json.loads(self.checkpoint.with_name('provenance.json').read_text())
        if G.digest(self.checkpoint)!=self.model_provenance['checkpoint_sha256']:
            raise ValueError('Existing CoTracker checkpoint identity mismatch')

    def describe(self,frames):
        self.crop=training_crop(frames) if frames else None
        self.heldout_ids={f['id'] for f in frames if f['split']=='heldout'}
        from importlib.util import find_spec
        self.predictor_path=Path(find_spec('cotracker.predictor').origin)
        self.predictor_sha256=G.digest(self.predictor_path)
        return {'protocol':dict(PROTOCOL),'checkpoint_path':str(self.checkpoint.relative_to(self.root)),
                'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
                'checkpoint_sha256':G.digest(self.checkpoint),'model_provenance':self.model_provenance,
                'provider_implementation_sha256':G.digest(__file__),
                'predictor_implementation_sha256':self.predictor_sha256,
                'training_source_frame_ids':[f['id'] for f in frames if f['split']=='train'],
                'heldout_source_frame_ids':[f['id'] for f in frames if f['split']=='heldout'],
                'training_derived_crop_xyxy':self.crop,
                'maximum_model_calls':2+2*sum(f['split']=='heldout' for f in frames),
                'actual_model_calls_record':'tracking.json','manual_queries':False,'heldout_images_used_for_geometry':False}

    def snapshot(self,output):
        self.output=Path(output)
        (self.output/'correspondence-implementation.py').write_bytes(Path(__file__).read_bytes())
        (self.output/'cotracker-predictor.py').write_bytes(self.predictor_path.read_bytes())
        (self.output/'cotracker-model-provenance.json').write_text(json.dumps(self.model_provenance,indent=2)+'\n')
        self._save('prepared_no_inference')

    def _save(self,status):
        if self.output:
            write(self.output/'tracking.json',{'status':status,'feature_provider':PROTOCOL['feature_provider'],
                  'model_provenance':self.model_provenance,'actual_model_calls':len(self.calls),
                  'calls':self.calls,'heldout_tracking':self.heldout,'new_api_calls':0,
                  'learned_3d_or_depth_calls':0,'manual_queries':False})

    def _load_model(self):
        if self.model is not None:return
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
            raise RuntimeError('CPU operator fallback is forbidden')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
        import torch
        from cotracker.predictor import CoTrackerPredictor
        from importlib.metadata import distribution
        package=json.loads(distribution('cotracker').read_text('direct_url.json'))
        if package['vcs_info']['commit_id']!=self.model_provenance['code_commit']:
            raise ValueError('Installed CoTracker code differs from checkpoint provenance')
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback')
        if G.digest(self.checkpoint)!=self.model_provenance['checkpoint_sha256'] or G.digest(self.predictor_path)!=self.predictor_sha256:
            raise ValueError('Frozen model/checkpoint identity changed before inference')
        self.model=CoTrackerPredictor(checkpoint=str(self.checkpoint)).to('mps').eval()

    def _infer(self,frames,queries,scope):
        self._load_model()
        import torch
        x0,y0,x1,y1=self.crop
        images=np.stack([f['image'][y0:y1,x0:x1,::-1] for f in frames]).copy()
        height,width=images.shape[1:3];factor=PROTOCOL['model_maximum_side_px']/max(height,width)
        interp=(max(64,int(round(height*factor/8))*8),max(64,int(round(width*factor/8))*8))
        self.model.interp_shape=interp
        q=np.asarray(queries,np.float32).copy();q[:,1]-=x0;q[:,2]-=y0
        if not np.isfinite(q).all() or (q[:,0]<0).any() or (q[:,0]>=len(frames)).any() or (q[:,1:]<0).any() or (q[:,1]>width-1).any() or (q[:,2]>height-1).any():
            raise ValueError('Tracker queries must be finite and inside the fixed training-derived crop')
        video=torch.from_numpy(images).permute(0,3,1,2)[None].float().to('mps')
        query=torch.from_numpy(q)[None].to('mps')
        call={'scope':scope,'source_frame_ids':[f['id'] for f in frames],
              'host_started_at_utc':datetime.now(timezone.utc).isoformat(),
              'image_sha256':[f['image_sha256'] for f in frames],'queries':len(q),
              'input_hw':[height,width],'model_interp_hw':list(interp),'crop_xyxy':list(self.crop),
              'device':'mps','cpu_operator_fallback':False,'status':'started'}
        self.calls.append(call);self._save('running');started=time.monotonic()
        with torch.inference_mode():
            prediction,visibility=self.model(video,queries=query,backward_tracking=True)
            torch.mps.synchronize()
            uv=prediction[0].cpu().numpy().astype(float);visible=visibility[0].cpu().numpy().astype(bool)
        uv+=np.array([x0,y0]);call.update(status='complete',inference_seconds=time.monotonic()-started)
        self._save('running')
        return uv,visible

    def build_tracks(self,frames,output):
        self.output=Path(output);self.training=[f for f in frames if f['split']=='train']
        for frame in frames:
            frame['xy']=np.empty((0,2),float);frame['feature_filter']={'provider':PROTOCOL['feature_provider'],'raw_mask_unchanged':True}
        if len(self.training)<2:return {},{},[],0
        ranked=[]
        for i,frame in enumerate(self.training):
            core,_=core_mask(frame['mask'],PROTOCOL['query_core_min_px'],PROTOCOL['query_core_radius_fraction'])
            gray=cv2.cvtColor(frame['image'],cv2.COLOR_BGR2GRAY)
            sharp=float(cv2.Laplacian(gray,cv2.CV_64F)[core].var()) if core.any() else 0
            ranked.append((float(core.sum())*math.log1p(sharp),-i,i))
        anchor=max(ranked)[2]
        seeds,margin=automatic_queries(self.training[anchor]['image'],self.training[anchor]['mask'],PROTOCOL['maximum_queries'])
        write(self.output/'automatic-queries.json',{'reference_frame_id':self.training[anchor]['id'],
              'reference_training_position':anchor,'source_pixels':seeds.tolist(),'core_margin_px':margin,
              'manual_coordinates':False,'heldout_images_read':False})
        if len(seeds)<G.PROTOCOL['seed_min_points']:
            self._save('insufficient_automatic_interior_queries');return {},{},[],0
        queries=np.c_[np.full(len(seeds),anchor),seeds]
        raw,visibility=self._infer(self.training,queries,'train_geometry_forward')
        supported=mask_support(raw,visibility,self.training)
        supported&=(raw[:,:,0]>=self.crop[0])&(raw[:,:,0]<self.crop[2])&(raw[:,:,1]>=self.crop[1])&(raw[:,:,1]<self.crop[3])
        reverse_queries=[]
        endpoints=cycle_endpoints(supported,anchor)
        for point in range(len(seeds)):
            end=endpoints[point]
            reverse_queries.append([len(self.training)-1-int(end),*raw[int(end),point]])
        reverse,reverse_visible=self._infer(list(reversed(self.training)),reverse_queries,'train_geometry_cycle_reverse')
        reverse=reverse[::-1];reverse_visible=reverse_visible[::-1]
        cycle=np.linalg.norm(raw-reverse,axis=2)
        accepted=supported & mask_support(reverse,reverse_visible,self.training) & np.isfinite(cycle) & (cycle<=PROTOCOL['maximum_cycle_error_source_px'])
        accepted=no_reacquisition(accepted,anchor)
        np.savez_compressed(self.output/'training-correspondences.npz',source_uv=raw,model_visible=visibility,
                            reverse_source_uv=reverse,reverse_visible=reverse_visible,
                            cycle_error_px=cycle,accepted=accepted,queries_source=queries,
                            reverse_endpoint_training_positions=np.asarray(endpoints))
        for point in range(len(seeds)):
            observations={frame['index']:raw[i,point].copy() for i,frame in enumerate(self.training) if accepted[i,point]}
            if len(observations)>=PROTOCOL['minimum_training_observations_per_track']:
                tid=f'ct_{point:06d}';self.tracks[tid]=observations
        for frame in self.training:
            frame['xy']=np.array([obs[frame['index']] for obs in self.tracks.values() if frame['index'] in obs],float).reshape(-1,2)
        pairs=[]
        for i,a in enumerate(self.training):
            for b in self.training[i+1:]:
                count=sum(a['index'] in obs and b['index'] in obs for obs in self.tracks.values())
                if count:pairs.append({'a':a['index'],'b':b['index'],'accepted_mask_cycle_tracks':count})
        self._save('training_tracks_frozen')
        return self.tracks,{},pairs,0

    def match_heldout(self,frame,training_frames,node_to_track,points):
        if frame['split']!='heldout':raise ValueError('Only held-out observations are allowed here')
        candidates=[]
        for ref in training_frames:
            ids=[tid for tid in points if ref['index'] in self.tracks.get(tid,{})]
            candidates.append((-abs(ref['timestamp_s']-frame['timestamp_s']),len(ids),-ref['index'],ref,ids))
        if not candidates:return {}
        _,_,_,reference,ids=max(candidates,key=lambda row:row[:3])
        if len(ids)<G.PROTOCOL['pnp_min_inliers']:return {}
        seeds=np.array([self.tracks[tid][reference['index']] for tid in ids])
        pair=[reference,frame]
        raw,visible=self._infer(pair,np.c_[np.zeros(len(ids)),seeds],'heldout_isolated_ref_test_forward')
        support=mask_support(raw,visible,pair)
        eligible=support.all(axis=0)&(raw[1,:,0]>=self.crop[0])&(raw[1,:,0]<self.crop[2])&(raw[1,:,1]>=self.crop[1])&(raw[1,:,1]<self.crop[3])
        valid_ids=np.flatnonzero(eligible)
        backward=np.full_like(raw,np.nan);back_visible=np.zeros_like(visible)
        if len(valid_ids):
            back,back_vis=self._infer(pair[::-1],np.c_[np.zeros(len(valid_ids)),raw[1,valid_ids]],'heldout_isolated_test_ref_cycle')
            backward[:,valid_ids]=back;back_visible[:,valid_ids]=back_vis
        cycle=np.linalg.norm(backward[1]-seeds,axis=1)
        accepted=support.all(axis=0)&mask_support(backward,back_visible,pair[::-1]).all(axis=0)&np.isfinite(cycle)&(cycle<=PROTOCOL['maximum_cycle_error_source_px'])
        observed={tid:raw[1,i].copy() for i,tid in enumerate(ids) if accepted[i]}
        frame['xy']=np.array(list(observed.values()),float).reshape(-1,2)
        record={'frame_id':frame['id'],'reference_frame_id':reference['id'],'queries':len(ids),
                'accepted_observations':len(observed),'geometry_frozen_before_tracking':True,
                'model_2d_predictions_not_manual_correspondences':True}
        self.heldout.append(record)
        np.savez_compressed(self.output/('heldout-'+str(frame['index'])+'.npz'),source_uv=raw,
                            visible=visible,reverse_source_uv=backward,reverse_visible=back_visible,
                            cycle_error_px=cycle,accepted=accepted,track_ids=np.asarray(ids))
        self._save('heldout_evaluation')
        return observed

    def finalize(self,output,result):
        self._save('complete')
        audit={'feature_provider':PROTOCOL['feature_provider'],'geometry_used_only_training_images':all(
               not any(frame_id in self.heldout_ids for frame_id in call['source_frame_ids'])
               for call in self.calls if call['scope'].startswith('train_geometry')),
               'actual_model_calls':len(self.calls),'training_model_calls':sum(c['scope'].startswith('train_geometry') for c in self.calls),
               'heldout_model_calls':sum(c['scope'].startswith('heldout_') for c in self.calls),
               'new_api_calls':0,'old_poses_or_generated_geometry_read':False,
               'tracking_json_sha256':G.digest(Path(output)/'tracking.json'),
               'provider_implementation_sha256':G.digest(__file__),'checkpoint_sha256':G.digest(self.checkpoint)}
        write(Path(output)/'correspondence-audit.json',audit)


def run(program_path,track_id,output,*,root,max_views=49,sensitivity=True,max_nfev=70,device='mps',camera_provider=None):
    provider=CorrespondenceProvider(root,device)
    return G.run(program_path,track_id,output,root=root,max_views=max_views,sensitivity=sensitivity,
                 max_nfev=max_nfev,correspondence_provider=provider,camera_provider=camera_provider)
