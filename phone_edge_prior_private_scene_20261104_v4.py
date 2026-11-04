"""Freeze a private, explicitly typed edge-prior scene with rejection barriers.

No further optimization. Only exact saved candidate poses are evaluated. The
failed fresh test stays failed; it adds an unknown motion barrier, never a pass.
This new source reconstruction cannot impersonate a native-SAM derivative.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=ROOT/'data/geometry-quality-20261104/phone-multiview-edge-prior-v4/private-scene'
REPORT=ROOT/'output/submission-polish-20261104-v1/phone-edge-prior-v4'
ASSET=OUTPUT.parent/'asset.glb'


def build():
    OUTPUT.mkdir(exist_ok=False)
    original=G.OUTPUT/'full-scene-v1/scene.json';old=json.loads(original.read_text());scene=deepcopy(old)
    previous=next(c for c in old['components']if c['id']=='phone_blue');part=next(c for c in scene['components']if c['id']=='phone_blue')
    gauge=previous['native_uniform_scale'];oldmesh=load_native_glb(ROOT/previous['asset']['path']);mesh=load_native_glb(ASSET)
    k=np.asarray(scene['camera']['K']);sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    fitted=json.loads((REPORT/'source-training-unit-poses.json').read_text())
    fresh=json.loads((REPORT/'fresh-tests-v1/report.json').read_text())
    plan={'format':'interaction-twin-private-phone-edge-scene-freeze-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'previous_scene':P.ref(original),'asset':P.ref(ASSET),'source_training_poses':P.ref(REPORT/'source-training-unit-poses.json'),
        'fresh_test_report':P.ref(REPORT/'fresh-tests-v1/report.json'),
        'per_source_acceptance':{'minimum_IoU':.65,'maximum_previous_prism_IoU_regression':.05,
            'maximum_RGB_MAE_increase_byte_values':1.,'RGB_tolerance_reason':'one byte display/raster quantization tolerance, explicitly not physical validation'},
        'forced_visual_conflict_unknown_source_indices':[39],
        'baseline_unknowns_and_previous_barriers_preserved':True,'failed_fresh_predictions_add_unknown_motion_barriers':True,
        'further_pose_optimization':False,'public_or_default_eligibility':False,'new_API_or_GPU_calls':0}
    P.save(OUTPUT/'selection-plan.json',plan)
    audit=[];rows=[]
    for before in previous['poses']:
        row=deepcopy(before)
        if before['status']=='unknown':rows.append(row);continue
        index=before['source_sample_index']
        if index is None:raise ValueError('Supported synthetic source row not allowed')
        rgb,mask,sref=G.frame(index);target=cv2.resize(mask.astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0;srgb=cv2.resize(rgb,(216,288),interpolation=cv2.INTER_AREA)
        m=np.asarray(before['camera_from_native_mesh']);unit=m.copy();unit[:3,:]/=gauge
        def measure(model,pose):
            render=render_native_mesh(model,sk,pose[:3,:3],pose[:3,3],216,288)
            inter=render['mask']&target;iou=float(inter.sum()/max(1,(render['mask']|target).sum()))
            pixels=cv2.erode(inter.astype(np.uint8),np.ones((3,3),np.uint8))>0
            error=float(np.abs(render['rgb'].astype(float)-srgb.astype(float))[pixels].mean()) if pixels.any() else 255.
            return iou,error
        oldiou,olderror=measure(oldmesh,unit)
        choices=[('same_saved_pose_new_shape',unit)]
        if str(index)in fitted:choices.append(('saved_source_training_orientation_candidate',np.asarray(fitted[str(index)])))
        evaluated=[]
        for method,pose in choices:
            iou,error=measure(mesh,pose)
            eligible=iou>=.65 and iou>=oldiou-.05 and error<=olderror+1 and index!=39
            evaluated.append({'method':method,'matrix':pose.tolist(),'IoU':iou,'RGB_MAE':error,'eligible':eligible,
                'score':error/255+.25*(1-iou)})
        eligible=[r for r in evaluated if r['eligible']]
        if eligible:
            chosen=min(eligible,key=lambda r:r['score']);pm=np.asarray(chosen['matrix']);pm[:3,:]*=gauge
            row.update(camera_from_native_mesh=pm.tolist(),geometry_backend='automatic_source_silhouette_2_5D_prism_multiview_edge_prior',
                method=chosen['method'],full_triangle_raw_mask_iou=chosen['IoU'],source_RGB_MAE=chosen['RGB_MAE'],
                accepted_motion_evidence=False,reason='private_source_geometry_with_source_projection_and_appearance_gates')
        else:
            row.update(status='unknown',camera_from_native_mesh=None,method='private_edge_prior_quality_withholding',
                reason='Muse_source_appearance_conflict_kept_unknown'if index==39 else'new_shape_or_appearance_regressed_no_pose_fallback',
                accepted_motion_evidence=False)
        audit.append({'source_index':index,'source':sref,'previous_prism_IoU':oldiou,'previous_RGB_MAE':olderror,
            'saved_candidates':evaluated,'status':row['status']})
        rows.append(row)
    failedbarriers=[]
    for test in fresh['results']:
        if test['passes_unrelaxed_gate']:continue
        barrier={'timestamp_s':test['timestamp_s'],'source_sample_index':None,'status':'unknown','camera_from_native_mesh':None,
            'method':'fresh_prediction_failure_motion_withholding','reason':'fresh_mask_withheld_RGB_prediction_failed_preserved_gate',
            'accepted_motion_evidence':False,'fresh_test_id':test['test_id'],'failed_candidate_IoU':test['new_edge_prior_IoU']}
        if all(abs(r['timestamp_s']-barrier['timestamp_s'])>1e-6 for r in rows):rows.append(barrier);failedbarriers.append(barrier)
    rows.sort(key=lambda r:r['timestamp_s'])
    metadata={'format':'interaction-twin-source-derived-asset-metadata','version':2,
        'backend':'automatic_source_silhouette_2_5D_prism_multiview_edge_prior','component_id':'phone_blue',
        'asset':P.ref(ASSET),'source_prism':previous['asset'],'source_derivation':P.ref(ASSET.parent/'derivation.json'),
        'original_SAM3D_preserved':previous['original_native_asset'],'source':scene['source'],
        'metric_scale_verified':False,'physical_accuracy_verified':False,'simulation_ready':False,'public_default_eligible':False}
    P.save(OUTPUT/'metadata.json',metadata)
    part.update(asset=P.ref(ASSET),native_metadata=P.ref(OUTPUT/'metadata.json'),geometry_backend=metadata['backend'],poses=rows)
    scene.update(created_utc=datetime.now(timezone.utc).isoformat(),input_signature=P.ref(OUTPUT/'selection-plan.json')['sha256'],
        scene_backend='private_source_2_5D_edge_prior_and_unchanged_SAM_connector',fitter_version='saved_pose_quality_gates_no_new_fit',
        private_research_comparison=True,automatic_publication=False,physical_accuracy_verified=False,simulation_ready=False)
    scene['inputs']=[*scene['inputs'],P.ref(OUTPUT/'selection-plan.json'),P.ref(ASSET.parent/'derivation.json'),P.ref(REPORT/'fresh-tests-v1/report.json')]
    source_rows=[r for r in rows if r['source_sample_index']is not None]
    coverage={'id':'phone_blue','source_samples':47,'supported':sum(r['status']=='estimated'for r in source_rows),
        'unknown':sum(r['status']=='unknown'for r in source_rows),'unknown_barrier_rows':sum(r['source_sample_index']is None for r in rows),
        'source39_withheld':next(r for r in source_rows if r['source_sample_index']==39)['status']=='unknown',
        'original_unknowns_preserved':True,'fresh_failed_tests_preserved':sum(not r['passes_unrelaxed_gate']for r in fresh['results'])}
    scene['diagnostics']=[coverage,*[d for d in scene['diagnostics']if d['id']!='phone_blue']]
    scene['limitations']=[*scene['limitations'],'Fresh independent source-frame test at10.128322s failed and remains failed. A new unknown playback barrier preserves that failure.',
        'This private edge-prior candidate has no public/default eligibility and does not implement a native-SAM geometry derivative contract.']
    P.save(OUTPUT/'source-quality-audit.json',audit);P.save(OUTPUT/'scene.json',scene)
    declaration={'format':'interaction-twin-declared-source-reconstruction-derivative','version':1,
        'backend':metadata['backend'],'source':scene['source'],'evidence_program':scene['original_program'],
        'private_scene':P.ref(OUTPUT/'scene.json'),'original_SAM_GLBS_retained':True,
        'original_native_GLB':previous['original_native_asset'],
        'construction_input_source_prism':previous['asset'],'new_asset':P.ref(ASSET),
        'source_reconstruction_derivation':P.ref(G.OUTPUT/'report.json'),
        'source_derived_edge_prior':P.ref(ASSET.parent/'derivation.json'),
        'source_quality_plan':P.ref(OUTPUT/'selection-plan.json'),'source_quality_audit':P.ref(OUTPUT/'source-quality-audit.json'),
        'fresh_tests':P.ref(REPORT/'fresh-tests-v1/report.json'),'development_tests':P.ref(REPORT/'development-results.json'),
        'connector_projection_diagnostic':P.ref(REPORT/'connector-projection-diagnostic.json'),
        'native_SAM_derivative_contract_compatible':False,
        'native_contract_incompatibility_reason':'Source prism uses new source-SAM silhouette topology, not indexed native triangles; it must not impersonate a native affine/subset derivative.',
        'coverage':coverage,'joints':[],'port_contact_frame':None,'grasp_pose':None,'insertion_depth':None,
        'shape_or_pose_authored_by_agent':False,'automatic_source_prior_used':True,
        'physical_accuracy_verified':False,'metric_scale_verified':False,'contact_verified':False,'simulation_ready':False,
        'public_default_eligible':False,'automatic_publication':False,'published':False}
    P.save(OUTPUT/'declared-source-reconstruction.json',declaration)
    report={'format':'interaction-twin-private-phone-edge-scene-completion','version':1,
        'scene':P.ref(OUTPUT/'scene.json'),'declaration':P.ref(OUTPUT/'declared-source-reconstruction.json'),
        'plan':P.ref(OUTPUT/'selection-plan.json'),'quality_audit':P.ref(OUTPUT/'source-quality-audit.json'),
        'coverage':coverage,'new_failure_barriers':failedbarriers,'original_scene':P.ref(original),
        'public_default_eligible':False,'new_API_or_GPU_calls':0,'automatic_publication':False}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'coverage':coverage}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
