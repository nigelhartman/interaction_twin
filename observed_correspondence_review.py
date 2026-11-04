"""Read-only evidence audit and capture guidance for saved observed geometry.

Checks stored source-pixel correspondences and projections without running the
tracker, estimating a new camera, or using inferred mesh/depth geometry.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(xyz, camera):
    camera_xyz=np.asarray(camera['R'],float)@np.asarray(xyz,float)+np.asarray(camera['t'],float)
    q=np.asarray(camera['K'],float)@camera_xyz
    return q[:2]/q[2],float(camera_xyz[2])


def source_scopes(calls, frames):
    train={f['id']:f for f in frames if f['split']=='train'}
    heldout={f['id']:f for f in frames if f['split']=='heldout'}
    failures=[]
    for i,call in enumerate(calls):
        ids=call.get('source_frame_ids',[])
        if call.get('scope','').startswith('train_geometry'):
            valid=bool(ids) and all(frame in train for frame in ids)
        elif call.get('scope','').startswith('heldout_'):
            valid=len(ids)==2 and sum(frame in heldout for frame in ids)==1 and sum(frame in train for frame in ids)==1
        else:
            valid=False
        if not valid:failures.append({'call':i,'reason':'invalid_source_scope'})
        if call.get('image_sha256')!=[({**train,**heldout}.get(frame) or {}).get('image_sha256') for frame in ids]:
            failures.append({'call':i,'reason':'source_image_hash_binding_mismatch'})
        if call.get('status')!='complete':failures.append({'call':i,'reason':'incomplete_model_call'})
    return failures


def capture_guidance(metrics):
    rows=[]
    if metrics.get('feature_tracks',0)<25:
        rows.append({'code':'insufficient_persistent_correspondence','priority':1,
            'evidence':{'feature_tracks':metrics.get('feature_tracks',0)},
            'suggestion':'Record a longer close-up with the same rigid component visible continuously, move slowly, keep fingers away, and use diffuse light. Glossy reflections can move independently of the surface.',
            'automatic_repair_established':False})
    if not metrics.get('recovered_points',0):
        rows.append({'code':'no_observed_3d','priority':1,
            'suggestion':'Preserve the empty result. Capture several visibly different viewing directions while the rigid part remains unchanged; do not treat a model-generated preview as recovered geometry.',
            'automatic_repair_established':False})
    else:
        ratio=metrics.get('registered_train_views',0)/max(1,metrics.get('training_views',0))
        if ratio<.8:rows.append({'code':'partial_temporal_support','priority':2,'registered_train_fraction':ratio,
            'suggestion':'Inspect the unsupported time intervals and recapture them with overlap before changing the component state or obscuring it.'})
        fraction=metrics.get('heldout_views_passing',0)/max(1,metrics.get('heldout_views',0))
        if fraction<.8:rows.append({'code':'partial_heldout_support','priority':1,'passing_heldout_fraction':fraction,
            'suggestion':'The independent projection checks cover only part of this clip. Obtain more unobstructed, textured observations across the missing views.'})
    if not metrics.get('focal_sensitivity_stable',False):
        rows.append({'code':'intrinsics_unverified','priority':1,
            'suggestion':'Use recorded or independently calibrated intrinsics with digital stabilization/zoom disabled. A low image reprojection error alone does not establish correct 3D shape.',
            'calibration_performed_by_this_review':False})
    rows.append({'code':'metric_assembly_unresolved','priority':2,
        'suggestion':'Independent rigid-part reconstructions have separate scales and origins. A shared scale/camera reconstruction and simultaneous visible observations are required before fitting an assembly, joint, or robot contact.',
        'physical_contact_or_assembly_established':False})
    return rows


def review(directory, *, root):
    root=Path(root).resolve();directory=Path(directory).resolve()
    if not directory.is_relative_to(root/'data'):raise ValueError('Only saved project data results can be reviewed')
    files={name:directory/name for name in ['protocol.json','viewer.json','geometry.json','tracking.json','correspondence-audit.json','correspondence-implementation.py','implementation.py','cotracker-predictor.py']}
    values={name:json.loads(path.read_text()) for name,path in files.items() if name.endswith('.json')}
    policy=values['protocol.json'];view=values['viewer.json'];geometry=values['geometry.json'];tracking=values['tracking.json'];audit=values['correspondence-audit.json']
    provider=policy['correspondence_provider'];identifier=provider['protocol']['feature_provider']
    if identifier not in {'cotracker3_mask_core_v1','cotracker3_mask_core_v2','cotracker3_cached_lk_core_v1'}:
        raise ValueError('Unsupported versioned correspondence result')
    checks={};errors=[]
    def check(name,passed,detail=None):
        checks[name]=bool(passed)
        if not passed:errors.append({'check':name,'detail':detail})
    checksums={str(p.relative_to(root)):digest(p) for p in files.values()}
    check('geometry_implementation_snapshot',digest(files['implementation.py'])==policy['implementation_sha256'])
    check('provider_implementation_snapshot',digest(files['correspondence-implementation.py'])==provider['provider_implementation_sha256'])
    check('predictor_implementation_snapshot',digest(files['cotracker-predictor.py'])==provider['predictor_implementation_sha256'])
    check('tracking_record_hash',digest(files['tracking.json'])==audit['tracking_json_sha256'])
    check('provider_snapshot_matches_final_audit',provider['provider_implementation_sha256']==audit['provider_implementation_sha256'])
    check('checkpoint_hash_matches_audit',provider['checkpoint_sha256']==audit['checkpoint_sha256'])
    check('no_manual_or_prior_geometry',not any(policy.get(k) for k in ['manual_coordinates','generated_meshes_read','depth_maps_read','old_poses_read']))
    frames=policy['input_views'];train=[f for f in frames if f['split']=='train'];test=[f for f in frames if f['split']=='heldout']
    check('disjoint_train_test_ids',not ({f['id'] for f in train}&{f['id'] for f in test}))
    source_errors=[]
    for frame in frames:
        for kind in ['image','mask']:
            path=(root/frame[kind+'_url'].removeprefix('/files/')).resolve()
            if not path.is_relative_to(root/'data') or not path.is_file() or digest(path)!=frame[kind+'_sha256']:
                source_errors.append({'frame_id':frame['id'],'kind':kind})
    check('source_frame_and_mask_hashes',not source_errors,source_errors)
    scopes=source_scopes(tracking['calls'],frames)
    check('actual_model_call_scopes',not scopes,scopes)
    check('actual_model_call_count',len(tracking['calls'])==tracking['actual_model_calls']==audit['actual_model_calls'])
    check('training_poses_only_train_frames',all(frames[int(i)]['split']=='train' for i in geometry['training_poses']))
    npzname='refined-training-correspondences.npz' if identifier=='cotracker3_cached_lk_core_v1' else 'training-correspondences.npz'
    array_path=directory/npzname;uv_count=0;uv_errors=[]
    if array_path.exists():
        checksums[str(array_path.relative_to(root))]=digest(array_path)
        with np.load(array_path) as arrays:
            train_position={i:j for j,i in enumerate(i for i,f in enumerate(frames) if f['split']=='train')}
            for tid,observations in geometry['tracks'].items():
                query=int(tid.removeprefix('ct_'))
                for index,uv in observations.items():
                    i=int(index);j=train_position.get(i)
                    if j is None or query>=arrays['accepted'].shape[1] or not arrays['accepted'][j,query] or not np.array_equal(arrays['source_uv'][j,query],np.asarray(uv)):
                        uv_errors.append({'track_id':tid,'frame_index':i})
                    uv_count+=1
    else:
        check('missing_saved_correspondences_only_when_no_tracks',not geometry['tracks'])
    check('geometry_tracks_equal_saved_source_pixels',not uv_errors,uv_errors[:20])
    cameras={c['frame_id']:c for c in view['cameras']};projection_count=0;max_uv_delta=0.;max_error_delta=0.;scored_errors=[];roles_ok=True;behind=0;unreconstructed=0
    for frame in view['frames']:
        if frame['id'] not in cameras:continue
        camera=cameras[frame['id']];anchors=set();scores=set()
        for observation in frame['observations']:
            tid=observation['track_id']
            if tid not in geometry['points']:
                unreconstructed+=1
                check('unreconstructed_track_has_no_predicted_3d_'+str(unreconstructed),observation['predicted_xy'] is None and observation['error_px'] is None)
                continue
            uv,z=project(geometry['points'][tid],camera)
            error=float(np.linalg.norm(uv-np.asarray(observation['xy'])))
            max_uv_delta=max(max_uv_delta,float(np.linalg.norm(uv-np.asarray(observation['predicted_xy']))))
            max_error_delta=max(max_error_delta,abs(error-observation['error_px']))
            projection_count+=1
            if frame['split']=='heldout':
                expected='heldout' if int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)%3==0 else 'pose_anchor'
                roles_ok &= expected==observation['role']
                if observation['role']=='pose_anchor':anchors.add(tid)
                elif observation['role']=='heldout':
                    scores.add(tid);scored_errors.append(error if z>0 else float('inf'));behind+=int(z<=0)
        roles_ok &= not (anchors&scores)
    check('independent_stored_projection_math',max_uv_delta<1e-6 and max_error_delta<1e-6)
    check('disjoint_hashed_pose_anchors_and_scores',roles_ok)
    check('heldout_score_count_recomputed',len(scored_errors)==view['metrics']['heldout_scored_observations'])
    finite_errors=[e for e in scored_errors if np.isfinite(e)]
    median=float(np.median(finite_errors)) if finite_errors else None
    p90=float(np.percentile(finite_errors,90)) if finite_errors else None
    for key,number in [('heldout_finite_error_median_px',median),('heldout_finite_error_p90_px',p90)]:
        saved=view['metrics'][key]
        check(key+'_recomputed',(saved is None and number is None) or (saved is not None and number is not None and abs(saved-number)<1e-7))
    return {'format':'interaction-twin-observed-correspondence-review','version':1,'passed':all(checks.values()),
        'checks':checks,'errors':errors,'input_sha256':checksums,'result_directory':str(directory.relative_to(root)),
        'component_id':policy['track_id'],'feature_provider':identifier,'status':view['status'],'metrics':view['metrics'],
        'counts':{'training_source_pixel_observations_checked':uv_count,'stored_3d_projections_recomputed':projection_count,
                  'disjoint_heldout_scored_observations':len(scored_errors),'scored_behind_camera':behind,
                  'actual_new_model_calls':tracking['actual_model_calls'],'unreconstructed_source_tracks_not_claimed_as_3d':unreconstructed},
        'maximum_saved_projection_delta_px':max_uv_delta,'maximum_saved_error_delta_px':max_error_delta,
        'capture_guidance':capture_guidance(view['metrics']),
        'limitations':['Integrity and source-pixel projection checks do not establish physical identity, metric scale, calibrated intrinsics, complete shape, independent rigidity, assembly or simulator readiness.',
                       'The tracker uses learned 2D correspondences. This review does not run or validate a model against labelled ground truth.',
                       'Held-out poses use separate hashed anchor points; scored points are disjoint. Different correspondence providers can score different points.'],
        'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
        'new_api_calls':0,'new_model_inference_calls':0,'manual_geometry_or_poses':False}


def run(directories,output,*,root):
    root=Path(root).resolve();output=Path(output).resolve()
    if not output.is_relative_to(root/'data') or output.exists():raise ValueError('Require a fresh data review directory')
    output.mkdir(parents=True)
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    try:reports=[review(directory,root=root) for directory in directories]
    except Exception as error:
        failure={'format':'interaction-twin-observed-correspondence-review-failure','implementation_sha256':digest(__file__),
                 'error_type':type(error).__name__,'message':str(error),'new_api_calls':0,'new_model_inference_calls':0}
        (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise
    report={'format':'interaction-twin-observed-correspondence-reviews','version':1,
        'implementation_sha256':digest(__file__),'passed':all(r['passed'] for r in reports),'results':reports}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
