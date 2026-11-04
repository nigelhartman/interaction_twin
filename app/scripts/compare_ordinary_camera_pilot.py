#!/usr/bin/env python3
"""Independent, read-only projection audit of the frozen ordinary-video pilot."""
from pathlib import Path
import hashlib
import json
import math
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN='output/automatic-camera-20261101/local-pilot-plan-v1.json'
OUTPUT='data/automatic-camera-20261101/paired-pilot-comparison-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def audit_case(folder):
    protocol=read(folder/'protocol.json');viewer=read(folder/'viewer.json');raw=read(folder/'geometry.json')
    if viewer['input_protocol_sha256']!=sha(folder/'protocol.json') or sha(folder/'implementation.py')!=protocol['implementation_sha256']:
        raise ValueError('Source protocol or engine snapshot identity differs')
    if viewer['prior_geometry_or_depth_used'] is not False or viewer['new_api_calls']!=0:
        raise ValueError('Pilot exceeds its observed local scope')
    cameras={camera['frame_id']:camera for camera in viewer['cameras']}
    poses={int(i):pose for i,pose in raw['training_poses'].items()}
    K=np.array(next(iter(cameras.values()))['K'])
    if not np.isfinite(K).all() or not np.array_equal(K[2],[0.,0.,1.]):raise ValueError('Invalid pilot camera K')
    points={p:np.array(x) for p,x in raw['points'].items()}
    if {p['id'] for p in viewer['points']}!=set(points):raise ValueError('Viewer/raw point identities differ')
    for point in viewer['points']:
        if not np.array_equal(point['xyz'],points[point['id']]):raise ValueError('Viewer/raw points differ')
    errors={'train':{},'reserved_score':{}};raw_train={};max_prediction_difference=0.;prediction_count=0
    reserved_rows=[]
    saved_reserved={row['frame_index']:row for row in viewer['heldout_evaluation']}
    identities={}
    for fi,(frame,frozen) in enumerate(zip(viewer['frames'],protocol['input_views'])):
        if any(frame[k]!=frozen[k] for k in ['id','timestamp_s','image_url','mask_url','split']):
            raise ValueError('Pilot source frame or frozen split differs')
        for kind in ['image','mask']:
            path=ROOT/frozen[kind+'_url'].removeprefix('/files/')
            if path.resolve()!=path or sha(path)!=frozen[kind+'_sha256']:raise ValueError('Pilot source pixels changed')
            identities[str(path.relative_to(ROOT))]=sha(path)
        camera=cameras.get(frame['id']);frame_errors=[];scored_xy=[]
        if camera:
            R,t=np.array(camera['R']),np.array(camera['t'])
            if not np.array_equal(camera['K'],K) or not np.allclose(R.T@R,np.eye(3),atol=1e-9,rtol=0) or not math.isclose(np.linalg.det(R),1.,abs_tol=1e-9):
                raise ValueError('Pilot contains inconsistent or improper cameras')
            if frame['split']=='train' and (fi not in poses or not np.array_equal(poses[fi]['R'],R) or not np.array_equal(poses[fi]['t'],t)):
                raise ValueError('Viewer/raw TRAIN poses differ')
        for observation in frame['observations']:
            tid=observation['track_id'];role=observation['role'];xy=np.array(observation['xy'])
            if frame['split']=='heldout' and tid in points:
                expected='heldout' if int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)%3==0 else 'pose_anchor'
                if role!=expected:raise ValueError('Reserved anchor/scoring role differs from fixed track hash')
            if tid not in points or camera is None:continue
            prediction=cv2.projectPoints(points[tid][None],cv2.Rodrigues(R)[0],t,K,None)[0].reshape(2)
            depth=float((R@points[tid]+t)[2]);error=float(np.linalg.norm(prediction-xy))
            if observation['predicted_xy'] is None or observation['error_px'] is None:raise ValueError('Missing saved true projection')
            max_prediction_difference=max(max_prediction_difference,float(np.max(np.abs(prediction-observation['predicted_xy']))),abs(error-observation['error_px']))
            prediction_count+=1
            if depth<=0 or not np.isfinite(error):raise ValueError('Unexpected invalid or behind-camera projected observation')
            key=(frame['id'],tid)
            if role=='train':
                errors['train'][key]=error
                raw_train[key]=np.array(raw['tracks'][tid][str(fi)])
                if not np.array_equal(raw_train[key],xy):raise ValueError('TRAIN raw/viewer pixels differ')
            elif role=='heldout':errors['reserved_score'][key]=error;frame_errors.append(error);scored_xy.append(xy)
        if frame['split']=='heldout':
            row=saved_reserved[fi];coverage=None
            if camera:
                mask=cv2.imread(str(ROOT/frozen['mask_url'].removeprefix('/files/')),cv2.IMREAD_GRAYSCALE)>127
                for other in frozen['subtracted_components']:
                    path=ROOT/other['mask_path']
                    if sha(path)!=other['mask_sha256']:raise ValueError('Subtracted component mask changed')
                    mask[cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>127]=False
                yy,xx=np.nonzero(mask)
                area=float((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)) if len(xx) else 1.
                coverage=float(cv2.contourArea(cv2.convexHull(np.array(scored_xy,np.float32))))/max(area,1.) if len(scored_xy)>=3 else 0.
            median=float(np.median(frame_errors)) if frame_errors else None
            p90=float(np.percentile(frame_errors,90)) if frame_errors else None
            passes=bool(len(frame_errors)>=15 and median<=3. and p90<=6. and coverage>=.08)
            if passes!=row['passes'] or bool(camera)!=row['pose_registered']:raise ValueError('Reserved view gate cannot be reproduced')
            for name,value in [('score_median_px',median),('score_p90_px',p90),('score_coverage_fraction',coverage)]:
                if value is not None and not math.isclose(row[name],value,abs_tol=1e-9,rel_tol=0):raise ValueError('Reserved derived metric differs')
            reserved_rows.append({'frame_id':frame['id'],'timestamp_s':frame['timestamp_s'],'registered':bool(camera),
                'scored_observations':len(frame_errors),'median_px':median,'p90_px':p90,'coverage_fraction':coverage,'passes':passes})
    summary={}
    for name,values in errors.items():
        e=list(values.values());summary[name]={'observations':len(e),'median_px':float(np.median(e)) if e else None,
            'p90_px':float(np.percentile(e,90)) if e else None}
    for name,saved in [('median_px','heldout_median_px'),('p90_px','heldout_p90_px')]:
        if not math.isclose(summary['reserved_score'][name],viewer['metrics'][saved],abs_tol=1e-9,rel_tol=0):raise ValueError('Reserved aggregate cannot be reproduced')
    if max_prediction_difference>1e-8:raise ValueError('Saved projections disagree with independent OpenCV projections')
    return {'K':K.tolist(),'status':viewer['status'],'metrics':viewer['metrics'],
        'bundle_adjustment':viewer['bundle_adjustment'],'seed':viewer['seed'],
        'independent_projection_metrics':summary,'independent_reserved_views':reserved_rows,
        'projection_count':prediction_count,'max_saved_projection_difference_px':max_prediction_difference,
        'source_artifacts':identities,'source_frames':protocol['input_views'],
        'raw_tracks':raw['tracks'],'points':points,'train_pixels':raw_train,
        'reserved_errors':errors['reserved_score'],'registered_train_indices':sorted(poses)}


def main():
    output=ROOT/OUTPUT
    if output.exists():raise ValueError('Preserve prior comparisons; output already exists')
    plan=read(ROOT/PLAN);inputs=dict(plan['inputs'])
    for name,expected in inputs.items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen pilot input/code changed')
    folders={case['camera_mode']:ROOT/case['output'] for case in plan['cases']}
    learned=audit_case(folders['moge_prior']);control=audit_case(folders['assumed'])
    same_frames=learned['source_frames']==control['source_frames'];same_tracks=learned['raw_tracks']==control['raw_tracks']
    if not same_frames or not same_tracks:raise ValueError('Pilot source selection or raw correspondences differ')
    common_points=sorted(set(learned['points'])&set(control['points']))
    common_train=set(learned['train_pixels'])&set(control['train_pixels'])
    common_reserved=set(learned['reserved_errors'])&set(control['reserved_errors'])
    if any(not np.array_equal(learned['train_pixels'][key],control['train_pixels'][key]) for key in common_train):raise ValueError('Common TRAIN pixels differ')
    prior=read(folders['moge_prior']/'camera-prior/camera-prior.json')
    native=[]
    w,h=prior['native_size']
    for prediction in prior['predictions']:
        k=np.diag([w,h,1.])@np.array(prediction['normalized_K']);k[0,2]-=.5;k[1,2]-=.5;native.append(k)
    if not np.array_equal(np.median(native,axis=0),learned['K']):raise ValueError('Learned K does not reproduce from exact TRAIN priors')
    if prior['actual_model_calls']!=3 or prior['completed_model_calls']!=3 or any(f['split']!='train' for f in prior['input_views']):raise ValueError('Pilot prior call accounting or TRAIN source scope differs')
    for mode,folder in folders.items():
        for path in folder.rglob('*'):
            if path.is_file():
                if path.resolve()!=path:raise ValueError('Pilot artifact symlink')
                inputs[str(path.relative_to(ROOT))]=sha(path)
    report={'format':'ordinary-video-paired-camera-comparison','version':1,
        'source':read(folders['moge_prior']/'protocol.json')['source'],
        'track_id':'camera_body','plan':{'path':PLAN,'sha256':sha(ROOT/PLAN)},
        'outcome':'coverage_tie_no_quality_gain_established','selected_production_camera':None,
        'summary':'Three automatic TRAIN-image lens estimates and a fresh reconstruction work on ordinary footage. This pilot ties the assumed-camera coverage and does not improve matched reserved projection metrics.',
        'matched_support':{'exact_source_views_and_splits_equal':same_frames,'selected_views':len(learned['source_frames']),
            'raw_correspondence_tracks_identical':same_tracks,'raw_feature_tracks':len(learned['raw_tracks']),
            'raw_TRAIN_observations':sum(len(x) for x in learned['raw_tracks'].values()),
            'retained_point_ids_identical':set(learned['points'])==set(control['points']),
            'common_retained_points':len(common_points),'common_retained_TRAIN_observations':len(common_train),
            'common_registered_TRAIN_indices':learned['registered_train_indices'],
            'registered_TRAIN_indices_identical':learned['registered_train_indices']==control['registered_train_indices'],
            'same_seed_pair':all(learned['seed'][key]==control['seed'][key] for key in ['a','b']),
            'common_reserved_scored_observations':len(common_reserved),
            'reserved_scored_support_identical':set(learned['reserved_errors'])==set(control['reserved_errors'])},
        'learned_camera_inputs':prior['input_views'],'per_TRAIN_image_native_K':np.array(native).tolist(),
        'cases':{mode:{k:row[k] for k in ['K','status','metrics','bundle_adjustment','seed',
            'independent_projection_metrics','independent_reserved_views','projection_count','max_saved_projection_difference_px']}
            for mode,row in [('moge_prior',learned),('assumed',control)]},
        'source_frames':learned['source_frames'],'new_api_calls':0,'new_model_calls_for_this_comparison':0,
        'actual_pilot_local_camera_model_calls':3,'physical_accuracy_verified':False,'physical_scale_known':False,
        'camera_calibrated':False,'production_camera_changed':False,'input_artifact_sha256':inputs,
        'limitations':['Only one existing ordinary-video camera-body clip was tested; this is not a broad benchmark.',
            'The frozen plan is diagnostic; reserved metrics never select or tune production intrinsics.',
            'Three learned estimates agreeing does not establish the true lens or physical shape.',
            'Only two of seven reserved views pass; source gaps and unregistered views remain explicit.',
            'Raw SfM, masks and rigid-component assumptions remain fallible; metric scale, hidden shape and assembly remain unverified.',
            'This paired 35-view run is distinct from the earlier 49-view cached-prior hypothesis comparison.']}
    for name,expected in inputs.items():
        if sha(ROOT/name)!=expected:raise ValueError('A frozen input changed during independent comparison')
    output.mkdir(parents=True)
    write(output/'comparison.json',report)
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'path':str((output/'comparison.json').relative_to(ROOT)),'sha256':sha(output/'comparison.json'),
        'outcome':report['outcome'],'matched_support':report['matched_support']}))


if __name__=='__main__':main()
