#!/usr/bin/env python3
"""Freeze real tracking experiment metrics for the app; never runs inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/observed-cotracker-experiment-20261101'
NAMES=['camera-body-35-sift-baseline-v1','camera-body-35-v1',
       'jabra-lid-35-sift-baseline-v1','jabra-lid-35-v1','jabra-lid-35-v2',
       'jabra-lid-35-cpu-refinement-v1','jabra-base-49-fresh-masks-v3',
       'jabra-lid-49-fresh-masks-v3','jabra-lid-49-cpu-refinement-v1']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def url(path):return '/files/'+str(Path(path).relative_to(ROOT))


def build(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'data'):raise ValueError('Require a fresh data comparison directory')
    output.mkdir(parents=True)
    rows=[];protocols={}
    for name in NAMES:
        directory=BASE/name
        protocol=json.loads((directory/'protocol.json').read_text());view=json.loads((directory/'viewer.json').read_text());protocols[name]=protocol
        provider=protocol.get('correspondence_provider',{}).get('protocol',{}).get('feature_provider','sift')
        tracking_path=directory/'tracking.json'
        tracking=json.loads(tracking_path.read_text()) if tracking_path.exists() else None
        rows.append({'id':view['id'],'component_id':protocol['track_id'],
            'group':'camera_same_views' if name.startswith('camera-') else 'jabra_parts',
            'method':provider,'viewer_url':url(directory/'viewer.json'),'viewer_sha256':sha(directory/'viewer.json'),
            'geometry_url':url(directory/'geometry.json'),'geometry_sha256':sha(directory/'geometry.json'),
            'protocol_url':url(directory/'protocol.json'),'protocol_sha256':sha(directory/'protocol.json'),
            'source_program_url':'/files/'+protocol['program_path'],'source_program_sha256':protocol['program_sha256'],
            'source_video':protocol['source'],'status':view['status'],'reason':view['reason'],'metrics':view['metrics'],
            'actual_model_calls':tracking['actual_model_calls'] if tracking else 0,
            'actual_inference_seconds':sum(c.get('inference_seconds',0) for c in tracking['calls']) if tracking else 0,
            'physical_identity_verified':False,'metric_scale_known':False,'intrinsics_calibrated':False,
            'physical_assembly_established':False,'complete_object_established':False,
            'automatic_registered_twin_reuse_supported':name.startswith('camera-'),
            'reuse_binding_note':'Existing published source binding is supported.' if name.startswith('camera-') else 'Standalone four-part segmentation diagnostic; no registered semantic publication binding. Do not retarget to older component IDs.',
            'dense_surface_established_by_this_run':False,'new_api_calls':0})
    first=protocols[NAMES[0]];second=protocols[NAMES[1]]
    same=first['input_views']==second['input_views'] and first['program_sha256']==second['program_sha256'] and first['track_id']==second['track_id']
    if not same:raise ValueError('Camera comparison source/split/mask binding mismatch')
    relative_path=BASE/'jabra-base-lid-relative-v1/relative-motion.json'
    relative=json.loads(relative_path.read_text())
    review_path=BASE/'independent-review-v2/report.json'
    audit=json.loads(review_path.read_text())
    report={'format':'interaction-twin-observed-tracking-comparison','version':1,
        'client_task_date':'2026-10-31','host_timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'implementation_sha256':sha(__file__),'attempts':rows,
        'matched_camera_comparison':{
            'first_id':NAMES[0],'second_id':NAMES[1],
            'exact_same_ordered_source_views_masks_and_splits':same,
            'same_assumed_focal_px':rows[0]['metrics']['focal_px']==rows[1]['metrics']['focal_px'],
            'maximum_ba_evaluations_per_run':70,'heldout_frames_never_used_to_fit_geometry':True,
            'result':'CoTracker supplies fewer, more persistent tracks and covers more TRAIN/held-out frames in this clip. Physical shape accuracy is not measured.',
            'caveats':['Different correspondence providers score different 3D point sets; median/p90 errors are conditional and not an accuracy ranking against ground truth.',
                'SIFT retains more sparse 3D points (443 vs 93); CoTracker registers more TRAIN views (20/28 vs 8/28) and more held-out views pass (5/7 vs 2/7).',
                'Camera focal sensitivity was not run for this matched comparison. Intrinsics, scale and complete shape remain unverified.',
                'The v1 CoTracker camera output keeps its original frozen provider implementation; a later contiguous cycle-endpoint correction is separately versioned as v2.']},
        'moving_component_conclusion':{
            'base_id':'jabra-base-49-fresh-masks-v3','base_sparse_points':113,
            'base_train_registered':'18/39','base_heldout_passing':'3/10',
            'lid_geometry_recovered':False,'lid_supported_3d_points':0,
            'mask_version_comparison_controlled':False,
            'caveats':['Fresh v2 SAM masks/audits and the 49-view plan differ from the earlier 35-view attempts; changes cannot be attributed to a single method.',
                'Mask physical identity and rigidity remain unverified. Model audit verdicts are fallible and not labelled accuracy.',
                'The CPU image-refinement attempts use cached learned initialization but make zero new model calls. They did not recover usable lid geometry.']},
        'relative_motion':{'report_url':url(relative_path),'report_sha256':sha(relative_path),
            'status':relative['status'],'simultaneous_pose_pairs':relative['simultaneous_pose_pairs'],
            'relative_translation':None,'common_physical_scale_known':False,'joint_axis_established':False,
            'physical_assembly_established':False,'insertion_or_contact_established':False},
        'independent_evidence_audit':{'report_url':url(review_path),'report_sha256':sha(review_path),'passed':audit['passed'],
            'results_checked':len(audit['results']),
            'training_source_pixel_observations_checked':sum(r['counts']['training_source_pixel_observations_checked'] for r in audit['results']),
            'stored_3d_projections_recomputed':sum(r['counts']['stored_3d_projections_recomputed'] for r in audit['results']),
            'disjoint_heldout_scored_observations':sum(r['counts']['disjoint_heldout_scored_observations'] for r in audit['results'])},
        'regression_test_command':'.venv-sam/bin/python -m unittest tests.test_observed_correspondence_review tests.test_observed_track_refinement tests.test_relative_component_motion tests.test_observed_segmentation_input tests.test_observed_cotracker tests.test_observed_geometry tests.test_observed_geometry_review',
        'new_paid_calls_by_this_task':0,'manual_seeds_geometry_or_poses':False,'new_checkpoints_downloaded':False}
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    report=build(ROOT/sys.argv[1])
    print(json.dumps({'attempts':len(report['attempts']),'matched_source_comparison':True,'relative_pose_pairs':report['relative_motion']['simultaneous_pose_pairs'],'independent_audit':report['independent_evidence_audit']}))
