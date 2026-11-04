#!/usr/bin/env python3
"""Summarize preserved genuine motion runs; no inference or output mutation."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import interval_motion as M


def artifact(path):
    p=Path(path).resolve()
    if not (p.is_relative_to(ROOT/'data') or p.is_relative_to(ROOT/'output')) or not p.is_file():
        raise ValueError('Expected an existing project data/report artifact')
    return {'path':str(p.relative_to(ROOT)),'sha256':M.sha(p)}


def quant(values):
    return {'count':len(values),'median_px':float(np.median(values)) if values else None,
        'p90_px':float(np.percentile(values,90)) if values else None}


def summarize():
    base=ROOT/'data/interval-motion-20261031';initial=[];density=[];sam=[];failures=[]
    for name in ['jabra-case-base-3fps-v3','jabra-case-lid-3fps-v1','jabra-earbud-a-3fps-v1']:
        p=base/name/'interval-motion.json';a=M.read(p);calls=M.read(p.parent/'inference-calls.json')
        initial.append({'result':artifact(p),'status':a['status'],'component':a['track_id'],
            'planned_frames':a['frame_count'],'supported_nonreference_frames':a['supported_nonreference_frames'],
            'reference_attempts':a['reference_frames'],'comparison':{k:v for k,v in a['comparison'].items() if k!='exact_pts_rows'},
            'wall_seconds':a['wall_seconds'],'actual_cotracker_calls':calls['actual_model_calls']})
    for name in ['density-lid-comparison-v1','density-earbud-a-episode-02-comparison-v1','density-earbud-a-episode-01-comparison-v1']:
        p=base/name/'density-experiment.json';g=M.read(p);variants=[]
        common={r['sample_index'] for r in g['comparison']['exact_common_pts_rows'] if not r['reference']}
        for item in g['variants']:
            v=M.read(M.file_path(ROOT,item['path']));e=v['episodes'][0];rows=[r for r in e['frames'] if not r['reference']]
            variants.append({**item,'planned_nonreference_frames':len(rows),
                'reserved_evaluation_count':sum(r['evaluation_timestamp_reserved_before_reference_selection'] for r in rows),
                'reserved_evaluation_supported':sum(r['evaluation_timestamp_reserved_before_reference_selection'] and r['status']==M.SUPPORTED for r in rows),
                'common_train_fit_residuals':quant([x for r in rows if r['sample_index'] in common for x in r.get('training_residual_px',[])]),
                'all_train_fit_residuals':quant([x for r in rows for x in r.get('training_residual_px',[])]),
                'invalid_source_rows':[{'sample_index':r['sample_index'],'timestamp_s':r['timestamp_s'],'status':r['status'],'reasons':r['reasons']} for r in rows if r['status']!=M.SUPPORTED],
                'supported_native_samples':[{'sample_index':r['sample_index'],'timestamp_s':r['timestamp_s']} for r in rows if r['status']==M.SUPPORTED]})
        density.append({'result':artifact(p),'status':g['status'],'comparison':g['comparison'],'variants':variants,
            'point_identity_proof':g['point_identity_proof'],'actual_cotracker_calls':g['actual_cotracker_model_calls'],'wall_seconds':g['wall_seconds']})
    for name in ['jabra-case-base-3fps-v1','jabra-case-base-3fps-v2']:
        p=base/name/'interval-motion.json';a=M.read(p);calls=p.parent/'inference-calls.json'
        failures.append({'result':artifact(p),'status':a['status'],'error':a.get('error'),
            'model_calls':M.read(calls)['actual_model_calls'] if calls.exists() else 0})
    for p in [base/'density-lid-input-12fps-v1/dense-input.json',base/'density-earbud-a-input-family-v1/episode-02/dense-input.json',base/'density-earbud-a-input-family-v1/episode-01/dense-input.json']:
        a=M.read(p);sam.append({'input':artifact(p),'frames':a['frame_count'],'wall_seconds':a['wall_seconds'],
            'model_runs':1,'mask_prompt_calls':1,'propagation_passes':2,'checkpoint_sha256':a['model']['checkpoint_sha256']})
    reports=ROOT/'output/interval-motion-20261031'
    audit_names=['actual-interval-independent-audit-v1.json','density-input-independent-audit-v1.json','density-lid-pair-independent-audit-v1.json',
        'density-earbud-pairs-independent-audit-v1.json']
    return {'format':'short-interval-motion-technical-delivery','version':1,'host_generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'client_task_start_date':'2026-10-31','status':'complete','new_api_calls_by_this_agent':0,
        'manual_geometry_points_boxes_or_poses':False,'physical_pose_accuracy_measured':False,
        'common_assembly_or_joint_established':False,'original_programs_assets_and_results_overwritten':False,
        'initial_3fps_results':initial,'density_results':density,'preserved_failed_attempts':failures,'sam2_preparations':sam,
        'own_actual_cotracker_calls':sum(a['actual_cotracker_calls'] for a in initial+density),'own_actual_sam2_model_runs':len(sam),
        'new_tracker_parameters_tuned_on_heldout_outcomes':False,
        'critical_review_rounds':[
            'Saved project URL resolution initially failed before inference; strict data URL resolution/traversal tests fixed it, preserving the failure.',
            'Exact-native reference search was too slow. A declared full-triangle SEARCH-only proxy was prospectively checked on fixed TRAIN references and fixtures. Final reference/motion gates still use exact rendering.',
            'Source scope semantics originally confused outside-mask objects with mask contents. Root preserved original judgments and issued a separately frozen inside-mask audit before motion outcomes.',
            'Independent references improved base/lid conditional coverage but worsened the earbud versus baseline. The stronger original earbud evidence stays available.',
            'Density tests freeze common images/masks, automatic reference, native geometry, K, query/ray identities, crop, point splits and gates. All three pairs tie on common non-reference support.',
            'An extra dense earbud mask had uncertain scope. Unchanged rules split its context before inference, exposing a gap rather than bridging it. High cadence did not rescue that difficult episode.',
            'A copied historical old-initialization flag in the lid fit receipt is ambiguous. A hash/numeric clarification preserves raw files and confirms intentional inherited automatic reference; future receipts make reuse explicit.'],
        'independent_audits':[artifact(reports/name) for name in audit_names if (reports/name).is_file()],
        'reference_provenance_clarification':artifact(reports/'density-lid-reference-provenance-clarification.json'),
        'training_failure_diagnostic':artifact(reports/'earbud-training-diagnostics-v1.json'),
        'verification':{'command':'.venv-sam/bin/python -B -m unittest tests.test_interval_motion_density tests.test_interval_motion_dense_input tests.test_interval_motion tests.test_interval_motion_search','tests_passed':27},
        'recommended_next_action':{'action':'Capture a short well-lit, slow sequence with clear separate-component views, a textured stationary background, little hand occlusion, and calibrated or preserved intrinsics when available.',
            'benefit':'Tests mesh/camera/reference and correspondence bottlenecks; more meaningful than changing gates on current outcomes.',
            'additional_capture_needed':True,'api_needed':'Optional hosted masks/scope; local preparation needs no API.',
            'gpu_needed':'Installed local MPS supports bounded checks; SAM3D/large reconstruction is a separate later GPU experiment.'},
        'next_priorities':[
            {'priority':1,'task':'Bind calibrated intrinsics and an actual measured dimension when available.','benefit':'Reduce camera/shape/scale ambiguity before common assembly.','capture':'Recommended','api':'Not inherently','gpu':'Not inherently'},
            {'priority':2,'task':'Compare observed multi-view component geometry with generated native references.','benefit':'Address glossy/symmetric/wrong-shape reference failure instead of increasing fit coverage alone.','capture':'Yes','api':'Optional','gpu':'Useful for larger reconstruction models'},
            {'priority':3,'task':'Retain stronger original earbud baseline and unsupported alternatives in the viewer.','benefit':'Prevent continuity or quality claims beyond saved evidence.','capture':'No','api':'No','gpu':'No'},
            {'priority':4,'task':'Recover a shared scene frame from independent simultaneous observations.','benefit':'Separate camera motion from component motion and independent gauges before proposing joints.','capture':'Likely','api':'Not inherently','gpu':'Not inherently'},
            {'priority':5,'task':'Validate common-scale collisions, articulation/contact and simulator execution separately.','benefit':'Turn descriptive assets into tested simulator inputs.','capture':'Likely','api':'Optional','gpu':'Simulator-dependent'}]}


if __name__=='__main__':
    report=summarize();out=ROOT/'output/interval-motion-20261031/technical-delivery-v1.json'
    if out.exists():raise ValueError('Preserve the existing delivery; choose a new version for a new report')
    M.write(out,report)
    print(json.dumps({'path':str(out.relative_to(ROOT)),'cotracker_calls':report['own_actual_cotracker_calls'],'sam2_runs':report['own_actual_sam2_model_runs']}))
