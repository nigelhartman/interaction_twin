"""CPU image refinement of a frozen CoTracker TRAIN correspondence cache.

Learned initial pixels are refined using actual source images; held-out frames
use independent two-image optical flow after geometry freeze. No model runs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

import observed_geometry as G
import observed_cotracker as C


PROTOCOL={'feature_provider':'cotracker3_cached_lk_core_v1','version':1,
          'method':'cached learned 2D initialization plus source-image bidirectional pyramidal Lucas-Kanade',
          'window_size_px':21,'pyramid_levels':3,'maximum_iterations':30,'termination_epsilon':.01,
          'maximum_source_cycle_error_px':1.5,'minimum_patch_zncc':.70,'minimum_patch_std':4.,
          'patch_radius_px':3,'minimum_training_observations':3,
          'heldout_method':'closest registered TRAIN image to heldout image, optical flow only; no target geometry/pose initialization',
          'no_reacquisition':True,'new_model_inference_calls':0,'new_api_calls':0,
          'camera_intrinsics_calibrated':False,'physical_identity_verified':False}


def refine_pair(reference,target,source_pixels,initial_pixels=None):
    seeds=np.asarray(source_pixels,np.float32).reshape(-1,2)
    initial=seeds.copy() if initial_pixels is None else np.asarray(initial_pixels,np.float32).reshape(-1,2)
    if seeds.shape!=initial.shape:raise ValueError('Source and initial pixel arrays must match')
    output=np.full(seeds.shape,np.nan);accepted=np.zeros(len(seeds),bool)
    cycle=np.full(len(seeds),np.inf);zncc=np.full(len(seeds),np.nan)
    h,w=reference['mask'].shape
    finite=np.isfinite(seeds).all(1)&np.isfinite(initial).all(1)&(seeds[:,0]>=0)&(seeds[:,0]<w)&(seeds[:,1]>=0)&(seeds[:,1]<h)&(initial[:,0]>=0)&(initial[:,0]<w)&(initial[:,1]>=0)&(initial[:,1]<h)
    ids=np.flatnonzero(finite)
    if not len(ids):return output,accepted,cycle,zncc
    gray_a=cv2.cvtColor(reference['image'],cv2.COLOR_BGR2GRAY);gray_b=cv2.cvtColor(target['image'],cv2.COLOR_BGR2GRAY)
    options={'winSize':(PROTOCOL['window_size_px'],)*2,'maxLevel':PROTOCOL['pyramid_levels'],
             'criteria':(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,PROTOCOL['maximum_iterations'],PROTOCOL['termination_epsilon']),
             'flags':cv2.OPTFLOW_USE_INITIAL_FLOW,'minEigThreshold':1e-4}
    forward,valid_forward,_=cv2.calcOpticalFlowPyrLK(gray_a,gray_b,seeds[ids,None,:],initial[ids,None,:].copy(),**options)
    if forward is None or valid_forward is None:return output,accepted,cycle,zncc
    backward,valid_backward,_=cv2.calcOpticalFlowPyrLK(gray_b,gray_a,forward,seeds[ids,None,:].copy(),**options)
    if backward is None or valid_backward is None:return output,accepted,cycle,zncc
    uv=forward[:,0];cycle[ids]=np.linalg.norm(backward[:,0]-seeds[ids],axis=1);output[ids]=uv
    support=C.mask_support(np.array([seeds[ids],uv]),np.stack([valid_backward[:,0]>0,valid_forward[:,0]>0]),[reference,target]).all(0)
    radius=PROTOCOL['patch_radius_px']
    for position,index in enumerate(ids):
        if not support[position] or not np.isfinite(cycle[index]) or cycle[index]>PROTOCOL['maximum_source_cycle_error_px']:continue
        valid=True
        for pixels,frame in [(seeds[index],reference),(uv[position],target)]:
            x,y=np.rint(pixels).astype(int)
            if x<radius or y<radius or x>=w-radius or y>=h-radius or not (frame['mask'][y-radius:y+radius+1,x-radius:x+radius+1]>0).all():
                valid=False;break
        if not valid:continue
        patch_a=cv2.getRectSubPix(gray_a,(2*radius+1,)*2,tuple(map(float,seeds[index]))).astype(float).ravel()
        patch_b=cv2.getRectSubPix(gray_b,(2*radius+1,)*2,tuple(map(float,uv[position]))).astype(float).ravel()
        patch_a-=patch_a.mean();patch_b-=patch_b.mean()
        if min(patch_a.std(),patch_b.std())<PROTOCOL['minimum_patch_std']:continue
        score=float(np.dot(patch_a,patch_b)/(np.linalg.norm(patch_a)*np.linalg.norm(patch_b)))
        zncc[index]=score;accepted[index]=score>=PROTOCOL['minimum_patch_zncc']
    return output,accepted,cycle,zncc


class CachedFlowProvider:
    def __init__(self,root,cache):
        self.root=Path(root).resolve();self.cache=Path(cache).resolve();self.output=None;self.tracks={};self.rows=[]
        if not self.cache.is_relative_to(self.root/'data'):raise ValueError('Only frozen project data caches are supported')
        self.source_protocol=json.loads((self.cache/'protocol.json').read_text())
        self.source_provider=self.source_protocol['correspondence_provider']
        if self.source_provider['protocol']['feature_provider'] not in {'cotracker3_mask_core_v1','cotracker3_mask_core_v2'}:
            raise ValueError('Expected a genuine versioned CoTracker correspondence cache')
        if G.digest(self.cache/'correspondence-implementation.py')!=self.source_provider['provider_implementation_sha256'] or G.digest(self.cache/'cotracker-predictor.py')!=self.source_provider['predictor_implementation_sha256']:
            raise ValueError('Frozen cached provider implementation mismatch')
        source_audit=json.loads((self.cache/'correspondence-audit.json').read_text())
        if not source_audit['geometry_used_only_training_images'] or G.digest(self.cache/'tracking.json')!=source_audit['tracking_json_sha256']:
            raise ValueError('Cached inference scopes or accounting mismatch')
        self.source_paths=[self.cache/name for name in ['protocol.json','correspondence-audit.json','tracking.json','training-correspondences.npz','automatic-queries.json','correspondence-implementation.py','cotracker-predictor.py']]
        self.source_hashes={str(p.relative_to(self.root)):G.digest(p) for p in self.source_paths}

    def describe(self,frames):
        fields=['id','split','image_sha256','mask_sha256']
        actual=[{k:f[k] for k in fields} for f in frames]
        expected=[{k:f[k] for k in fields} for f in self.source_protocol['input_views']]
        if actual!=expected:raise ValueError('Refinement requires identical ordered source frames, masks and frozen splits')
        self.training=[f for f in frames if f['split']=='train']
        return {'protocol':copy.deepcopy(PROTOCOL),'provider_implementation_sha256':G.digest(__file__),
                'checkpoint_sha256':self.source_provider['checkpoint_sha256'],'checkpoint_path':self.source_provider['checkpoint_path'],
                'predictor_implementation_sha256':self.source_provider['predictor_implementation_sha256'],
                'cached_correspondence_files_sha256':self.source_hashes,
                'source_cache':str(self.cache.relative_to(self.root)),
                'training_source_frame_ids':[f['id'] for f in self.training],
                'heldout_source_frame_ids':[f['id'] for f in frames if f['split']=='heldout'],
                'new_model_inference_calls':0,'manual_queries':False,'heldout_images_used_for_geometry':False}

    def snapshot(self,output):
        self.output=Path(output)
        (self.output/'correspondence-implementation.py').write_bytes(Path(__file__).read_bytes())
        (self.output/'cached-cotracker-implementation.py').write_bytes((self.cache/'correspondence-implementation.py').read_bytes())
        (self.output/'cotracker-predictor.py').write_bytes((self.cache/'cotracker-predictor.py').read_bytes())
        self._save('prepared_no_inference')

    def _save(self,status):
        if self.output:
            C.write(self.output/'tracking.json',{'status':status,'feature_provider':PROTOCOL['feature_provider'],
                    'actual_model_calls':0,'new_model_inference_calls':0,'calls':[],'cpu_flow_pairs':self.rows,
                    'cached_learned_2d_initialization':str(self.cache.relative_to(self.root)),'new_api_calls':0})

    def build_tracks(self,frames,output):
        self.output=Path(output)
        for frame in frames:frame['xy']=np.empty((0,2));frame['feature_filter']={'provider':PROTOCOL['feature_provider'],'raw_mask_unchanged':True}
        saved=np.load(self.cache/'training-correspondences.npz')
        anchor=json.loads((self.cache/'automatic-queries.json').read_text())['reference_training_position']
        seeds=saved['queries_source'][:,1:]
        raw=saved['source_uv'];visibility=saved['model_visible']
        if len(self.training)!=len(raw):raise ValueError('Cached training frame count mismatch')
        refined=np.full_like(raw,np.nan);accepted=np.zeros(raw.shape[:2],bool);cycles=np.full(raw.shape[:2],np.inf);scores=np.full(raw.shape[:2],np.nan)
        reference=self.training[anchor]
        for i,frame in enumerate(self.training):
            if i==anchor:
                refined[i]=seeds;accepted[i]=True;cycles[i]=0;scores[i]=1
            else:
                refined[i],accepted[i],cycles[i],scores[i]=refine_pair(reference,frame,seeds,raw[i])
                accepted[i]&=visibility[i]
            self.rows.append({'scope':'train_geometry_cpu_refinement','frame_id':frame['id'],'accepted':int(accepted[i].sum()),'heldout_images_read':False})
        accepted=C.no_reacquisition(accepted,anchor)
        for point in range(len(seeds)):
            observations={frame['index']:refined[i,point].copy() for i,frame in enumerate(self.training) if accepted[i,point]}
            if len(observations)>=PROTOCOL['minimum_training_observations']:self.tracks[f'ct_{point:06d}']=observations
        for frame in self.training:frame['xy']=np.array([o[frame['index']] for o in self.tracks.values() if frame['index'] in o]).reshape(-1,2)
        np.savez_compressed(self.output/'refined-training-correspondences.npz',source_uv=refined,accepted=accepted,cycle_error_px=cycles,zncc=scores,queries_source=saved['queries_source'])
        self._save('training_tracks_frozen')
        return self.tracks,{},[],0

    def match_heldout(self,frame,training_frames,node_to_track,points):
        ranked=[]
        for reference in training_frames:
            ids=[t for t in points if reference['index'] in self.tracks.get(t,{})]
            ranked.append((-abs(reference['timestamp_s']-frame['timestamp_s']),len(ids),reference,ids))
        if not ranked:return {}
        _,_,reference,ids=max(ranked,key=lambda row:row[:2])
        if not ids:return {}
        seeds=np.array([self.tracks[t][reference['index']] for t in ids])
        uv,accepted,cycle,score=refine_pair(reference,frame,seeds)
        self.rows.append({'scope':'heldout_isolated_cpu_flow','frame_id':frame['id'],'reference_frame_id':reference['id'],'accepted':int(accepted.sum()),'geometry_frozen_before_tracking':True})
        observations={tid:uv[i] for i,tid in enumerate(ids) if accepted[i]}
        frame['xy']=np.array(list(observations.values())).reshape(-1,2)
        self._save('heldout_evaluation')
        return observations

    def finalize(self,output,result):
        self._save('complete')
        for path,digest in self.source_hashes.items():
            if G.digest(self.root/path)!=digest:raise ValueError('Frozen correspondence cache changed')
        C.write(Path(output)/'correspondence-audit.json',{'feature_provider':PROTOCOL['feature_provider'],
                'geometry_used_only_training_images':True,'actual_model_calls':0,'training_model_calls':0,'heldout_model_calls':0,
                'provider_implementation_sha256':G.digest(__file__),'checkpoint_sha256':self.source_provider['checkpoint_sha256'],
                'tracking_json_sha256':G.digest(Path(output)/'tracking.json'),'new_api_calls':0,
                'source_cache_files_unchanged':True,'cached_learned_initialization_not_new_inference':True})


def run(program_path,track_id,output,*,root,cache,max_views=49,max_nfev=70,sensitivity=False):
    return G.run(program_path,track_id,output,root=root,max_views=max_views,max_nfev=max_nfev,
                 sensitivity=sensitivity,correspondence_provider=CachedFlowProvider(root,cache))
