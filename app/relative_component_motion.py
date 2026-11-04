"""Conditional relative-rotation diagnostics without inventing assembly scale.

Matching simultaneous cameras can cancel common camera rotation. Independent
SfM length gauges still prevent relative translations, joint pivots and metric
assembly. Rotation estimates also inherit segmentation/intrinsic uncertainty.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import observed_geometry as G


def valid_rotation(value):
    R=np.asarray(value,float)
    return bool(R.shape==(3,3) and np.isfinite(R).all() and np.linalg.norm(R@R.T-np.eye(3))<1e-5 and abs(np.linalg.det(R)-1)<1e-5)


def angle_degrees(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


def compare_views(first,second,source_a,source_b):
    same_source=source_a.get('sha256')==source_b.get('sha256') and bool(source_a.get('sha256'))
    result={'format':'interaction-twin-relative-motion-diagnostic','version':1,
            'component_ids':[first['id'],second['id']], 'same_source_video':same_source,
            'relative_metric_translation_observable':False,'relative_translation':None,
            'common_physical_scale_known':False,'physical_assembly_established':False,
            'joint_axis_established':False,'joint_pivot':None,'insertion_or_contact_established':False,
            'rotation_estimates_are_conditional':True,'rows':[],
            'limitations':['Independent part SfM reconstructions have independent length gauges and origins; relative translations, pivots and metric assembly are unavailable.',
                           'At identical source PTS, R_camera_from_base.T @ R_camera_from_part cancels common camera rotation only conditional on valid rigid-part poses and a shared camera convention.',
                           'Changing either native orientation gauge changes the relative matrix; change-angle magnitudes are invariant to constant native orientation gauges.',
                           'Camera intrinsics, segmentation, physical identity and reconstructed orientation remain unverified; conditional angular change is not a verified hinge or robot trajectory.',
                           'Individual component orientation changes without simultaneous base poses remain confounded by camera motion. No interpolation or missing pose is invented.']}
    if not same_source:
        result.update(status='different_source_videos_not_comparable',simultaneous_pose_pairs=0)
        return result
    def poses(view):
        frame_by_id={str(f['id']):f for f in view['frames']}
        values={}
        for camera in view['cameras']:
            frame=frame_by_id.get(str(camera['frame_id']))
            if not frame or not valid_rotation(camera['R']):continue
            values[round(float(frame['timestamp_s']),7)]=(camera,frame)
        return values
    pa,pb=poses(first),poses(second)
    times=sorted(set(pa)|set(pb));reference=None
    for time in times:
        row={'timestamp_s':time,'relative_translation':None,'conditional_relative_rotation_change_deg':None}
        if time not in pa or time not in pb:
            row['status']='missing_simultaneous_component_pose'
        else:
            ca,fa=pa[time];cb,fb=pb[time]
            if fa['width']!=fb['width'] or fa['height']!=fb['height'] or not np.allclose(ca['K'],cb['K'],atol=1e-8,rtol=0):
                row['status']='camera_intrinsics_or_pixel_convention_mismatch'
            else:
                relative=np.array(ca['R']).T@np.array(cb['R'])
                if reference is None:reference=relative
                row.update(status='conditional_relative_rotation_only',
                           conditional_relative_rotation_change_deg=angle_degrees(relative@reference.T),
                           reference_time_s=next(r['timestamp_s'] for r in result['rows'] if r.get('status')=='conditional_relative_rotation_only') if result['rows'] and any(r.get('status')=='conditional_relative_rotation_only' for r in result['rows']) else time,
                           camera_motion_cancellation='same-frame relative rotations, conditional on both pose estimates',
                           component_pose_splits=[ca.get('split'),cb.get('split')])
        result['rows'].append(row)
    matched=[r for r in result['rows'] if r['status']=='conditional_relative_rotation_only']
    result.update(status='conditional_relative_rotation_candidate_translation_unobservable' if matched else 'no_comparable_simultaneous_poses',
                  simultaneous_pose_pairs=len(matched),
                  maximum_conditional_rotation_change_deg=max((r['conditional_relative_rotation_change_deg'] for r in matched),default=None))
    return result


def run(first_directory,second_directory,output,*,root):
    root=Path(root).resolve();first_directory=Path(first_directory).resolve();second_directory=Path(second_directory).resolve();output=Path(output).resolve()
    if any(not p.is_relative_to(root/'data') for p in [first_directory,second_directory,output]) or output.exists():
        raise ValueError('Require existing data inputs and a fresh data output directory')
    output.mkdir(parents=True)
    inputs=[directory/name for directory in [first_directory,second_directory] for name in ['viewer.json','protocol.json']]
    hashes={str(p.relative_to(root)):G.digest(p) for p in inputs}
    first,second=[json.loads((d/'viewer.json').read_text()) for d in [first_directory,second_directory]]
    sources=[json.loads((d/'protocol.json').read_text())['source'] for d in [first_directory,second_directory]]
    protocol={'input_sha256':hashes,'implementation_sha256':G.digest(__file__),
              'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
              'manual_geometry_or_poses':False,'new_api_calls':0,'new_model_inference_calls':0,
              'method':'same-source exact-PTS conditional relative rotation; independent translations remain absent'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    report=compare_views(first,second,*sources)
    (output/'relative-motion.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
