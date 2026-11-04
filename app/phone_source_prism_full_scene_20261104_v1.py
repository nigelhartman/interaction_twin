"""Build a private full-timeline source-prism scene without publishing it.

All baseline unknown rows remain unknown. Successful automatic RGB/PnP rows
are distinct from automatic mask-fit training rows. Rotation discontinuities
receive explicit unknown barriers instead of misleading temporal interpolation.
"""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
import phone_source_prism_rgb_tracking_20261104_v1 as T
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=G.OUTPUT/'full-scene-v1'


def build():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False)
    baseline=json.loads(E.BASE.read_text());scene=deepcopy(baseline)
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text())
    evidence={e['source_index']:e for e in program['evidence']}
    part=next(c for c in scene['components']if c['id']=='phone_blue')
    prior={p['source_sample_index']:deepcopy(p)for p in part['poses']}
    geometry=json.loads((G.OUTPUT/'report.json').read_text())
    privatefits=json.loads((G.OUTPUT/'evaluation-v1/fitted-training-poses.json').read_text())
    fits={r['source']['source_index']:r for r in privatefits}
    masked=json.loads((G.OUTPUT/'RGB-tracking-evaluation-v1/predictions-before-target-mask-reads.json').read_text())
    tracked={r['target_index']:r for r in masked if r.get('target_index')is not None}
    k=np.asarray(scene['camera']['K']);smallk=k.copy();smallk[0]*=.3;smallk[1]*=.3
    smallk[0,2]+=(.3-1)/2;smallk[1,2]+=(.3-1)/2
    mesh=load_native_glb(G.OUTPUT/'asset.glb');original=load_native_glb(ROOT/part['asset']['path'])
    gauge=part['native_uniform_scale'];oldunknown=[i for i,p in prior.items()if p['status']=='unknown']
    plan={'format':'interaction-twin-private-source-prism-full-scene-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'baseline':P.ref(E.BASE),'source_program':P.ref(P.CASE/'flow-procedure-1/interaction.json'),
        'geometry':P.ref(G.OUTPUT/'report.json'),'asset':P.ref(G.OUTPUT/'asset.glb'),
        'baseline_unknown_source_indices_preserved':oldunknown,'mask_withheld_RGB_tests_fixed':[5,26],
        'mask_fit_allowed_on_other_estimated_training_source_rows':True,
        'minimum_full_triangle_mask_IoU':.65,'maximum_training_baseline_regression':.05,
        'maximum_continuous_rotation_step_degrees':50.,'maximum_interpolation_gap_s':.8,
        'relative_scale_gauge':gauge,'relative_scale_gauge_origin':'unchanged_baseline_display_scale_not_metric_measurement',
        'other_component_assets_and_poses_unchanged':True,'automatic_publication':False,
        'new_API_or_GPU_calls':0,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    rows=[];audit=[];previous=None;previous_index=None;previous_rgb=None;previous_mask=None;unitposes={}
    def score(which,matrix,mask,scale=1.):
        r=matrix[:3,:3]/scale;t=matrix[:3,3]
        pred=render_native_mesh(which,smallk,r,t,216,288,scale=scale)['mask']
        target=cv2.resize(mask.astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        return float((pred&target).sum()/max(1,(pred|target).sum()))
    for index in range(1,48):
        originalrow=prior[index]
        row={'timestamp_s':originalrow['timestamp_s'],'source_sample_index':index,'status':'unknown',
            'camera_from_native_mesh':None,'accepted_motion_evidence':False,'method':'unknown',
            'reason':'baseline_unknown_barrier_preserved'if index in oldunknown else'new_source_derived_pose_not_supported'}
        if index in oldunknown:
            row['original_unknown_reason']=originalrow['reason'];rows.append(row)
            audit.append({'sample_index':index,'status':'unknown','reason':row['reason']})
            previous=None;previous_index=None;continue
        rgb,mask,source=G.frame(index)
        oldmatrix=np.asarray(originalrow['camera_from_native_mesh'])
        oldscale=float(np.cbrt(np.linalg.det(oldmatrix[:3,:3])))
        baselineiou=score(original,oldmatrix,mask,oldscale)
        options=[]
        if index in tracked:
            m=np.asarray(tracked[index]['matrix']);options.append({'matrix':m,'kind':'RGB_tracking_target_mask_withheld',
                'fit':tracked[index],'iou':score(mesh,m,mask),'reference_source_index':4 if index==5 else 25})
        elif previous is not None:
            tracking=T.track(previous_rgb,rgb,previous_mask,previous,k,geometry['estimated_edge_thickness_over_length'])
            if tracking['status']=='estimated':
                m=np.asarray(tracking['matrix']);options.append({'matrix':m,'kind':'automatic_RGB_PnP_tracking',
                    'fit':tracking,'iou':score(mesh,m,mask),'reference_source_index':previous_index})
        valid=[o for o in options if o['iou']>=.65 and o['iou']>=baselineiou-.05]
        if not valid and index not in [5,26]:
            fitted=fits.get(index)
            if fitted is None:fitted=E.fit(mesh,k,rgb,mask,previous[:3,:3]if previous is not None else None)
            m=np.asarray(fitted['matrix']);value=score(mesh,m,mask)
            option={'matrix':m,'kind':'automatic_source_mask_and_RGB_fit_training',
                'fit':fitted,'iou':value,'reference_source_index':None}
            options.append(option)
            if value>=.65 and value>=baselineiou-.05:valid.append(option)
        if valid:
            selected=max(valid,key=lambda o:o['iou'])
            unit=selected['matrix'];unitposes[index]=unit.tolist()
            matrix=unit.copy();matrix[:3,:]*=gauge
            row.update(status='estimated',camera_from_native_mesh=matrix.tolist(),method=selected['kind'],
                reason='automatic_source_derived_display_estimate_not_validated_6DoF',source_mask=source['mask'],
                native_uniform_scale=gauge,full_triangle_raw_mask_iou=selected['iou'],
                fitting_target_mask_used=selected['kind']=='automatic_source_mask_and_RGB_fit_training',
                reference_source_index=selected['reference_source_index'])
            previous=unit;previous_index=index;previous_rgb=rgb;previous_mask=mask
            audit.append({'sample_index':index,'status':'estimated','method':selected['kind'],
                'baseline_IoU':baselineiou,'candidate_IoU':selected['iou'],'source':source,'method_diagnostics':selected['fit']})
        else:
            row['reason']='private_source_derived_pose_fails_unrelaxed_mask_gate'
            previous=None;previous_index=None
            audit.append({'sample_index':index,'status':'unknown','baseline_IoU':baselineiou,
                'attempted_fits':[{'method':o['kind'],'full_triangle_IoU':o['iou'],'details':o['fit']}for o in options]})
        rows.append(row)
        print('Phone private full source',index,row['status'],row.get('method'),flush=True)
    barriers=[];withbarriers=[]
    for row in rows:
        if withbarriers:
            before=withbarriers[-1]
            if before['status']=='estimated'and row['status']=='estimated':
                ra=np.asarray(before['camera_from_native_mesh'])[:3,:3]/gauge
                rb=np.asarray(row['camera_from_native_mesh'])[:3,:3]/gauge
                angle=float(Rotation.from_matrix(rb@ra.T).magnitude()*180/np.pi)
                if angle>50:
                    barrier={'timestamp_s':(before['timestamp_s']+row['timestamp_s'])/2,
                        'source_sample_index':None,'status':'unknown','camera_from_native_mesh':None,
                        'reason':'unreliable_large_rotation_step_no_temporal_interpolation',
                        'method':'automatic_source_pose_discontinuity_withholding','accepted_motion_evidence':False,
                        'neighbor_source_indices':[before['source_sample_index'],row['source_sample_index']],
                        'rotation_difference_degrees':angle}
                    barriers.append(barrier);withbarriers.append(barrier)
        withbarriers.append(row)
    # Retain the common renderer data contract without pretending the new
    # component was generated by SAM3D. Metadata identifies its actual backend.
    metadata={'format':'interaction-twin-source-derived-asset-metadata','version':1,
        'backend':'automatic_source_silhouette_2_5D_prism_not_SAM3D','component_id':'phone_blue',
        'asset':P.ref(G.OUTPUT/'asset.glb'),'construction':P.ref(G.OUTPUT/'report.json'),
        'original_SAM3D_preserved':part['asset'],'source':scene['source'],'physical_accuracy_verified':False,
        'metric_scale_verified':False,'simulation_ready':False,'published':False}
    P.save(OUTPUT/'metadata.json',metadata)
    part.update(asset=P.ref(G.OUTPUT/'asset.glb'),native_metadata=P.ref(OUTPUT/'metadata.json'),
        geometry_backend=metadata['backend'],native_uniform_scale=gauge,
        original_native_asset=deepcopy(metadata['original_SAM3D_preserved']),poses=withbarriers,
        sampled_vertices_for_estimation_only=min(2000,len(mesh.vertices)),
        full_mesh_geometry_modified=True,source_anchor={'sample_index':9,'timestamp_s':evidence[9]['timestamp_s'],
            'source_case':P.CASE.name,'time_domain':'source_video_presentation_seconds','geometry_source_indices':[4,9,18]},
        depth_gauge={'status':'unverified_source_prism_display_gauge','scale':gauge,'metric_scale_verified':False})
    part.pop('predicted_SAM_instance_scale',None)
    scene.update(created_utc=datetime.now(timezone.utc).isoformat(),
        input_signature=P.ref(OUTPUT/'selection-plan.json')['sha256'],
        fitter_version='private_source_prism_RGB_tracking_and_mask_fit_v1',
        learned_tracker_run=False,old_backend_geometry_or_poses_read=True,
        scene_backend='mixed_source_derived_prism_and_unchanged_SAM3D_cable',
        private_research_comparison=True,automatic_publication=False,
        model_or_api_calls=0,accepted_motion_poses=0,physical_accuracy_verified=False,simulation_ready=False)
    scene['inputs']=[*scene['inputs'],P.ref(OUTPUT/'selection-plan.json'),P.ref(G.OUTPUT/'report.json')]
    scene['limitations']=[*scene['limitations'],*geometry['limitations'],
        'Phone is explicitly a source-derived2.5D prism, not regenerated SAM3D; the cable asset and poses are unchanged.',
        'A constant inherited render gauge preserves projection; it does not establish relative metric scale or connector contact.',
        'Baseline unknown poses and new failed fits remain unknown. Extra midpoint barriers prevent interpolation across rotation jumps.']
    scene['evaluation_protocol']={'format':'interaction-twin-private-source-prism-evaluation-protocol','version':1,
        'mask_withheld_RGB_tracking_source_indices':[5,26],'source_mask_fit_training_indices':[r['sample_index']for r in audit if r.get('method')=='automatic_source_mask_and_RGB_fit_training'],
        'prior_blind_prediction_failure_preserved':P.ref(G.OUTPUT/'evaluation-v1/report.json'),
        'RGB_tracking_mask_withheld_test':P.ref(G.OUTPUT/'RGB-tracking-evaluation-v1/report.json'),
        'not_independent_unseen_RGB_validation':True,'physical_accuracy_verified':False}
    scene['diagnostics']=[{'id':'phone_blue','estimated':sum(r['status']=='estimated'for r in rows),
        'unknown':sum(r['status']=='unknown'for r in rows),'original_source_samples':47,
        'baseline_unknowns_preserved':len(oldunknown),'inserted_unknown_rotation_barriers':len(barriers),
        'interpolated_short_missing':0,'manual_poses_used':False},
        *[d for d in scene['diagnostics']if d['id']!='phone_blue']]
    scene['model_metadata_paths']=[P.ref(OUTPUT/'metadata.json')['path'],
        *[c['native_metadata']['path']for c in scene['components']if c['id']!='phone_blue']]
    P.save(OUTPUT/'audit.json',audit);P.save(OUTPUT/'unit-gauge-poses.json',unitposes)
    P.save(OUTPUT/'scene.json',scene)
    sidecar={'format':'interaction-twin-source-derived-object-program-sidecar','version':1,
        'source':scene['source'],'base_object_program':scene['original_program'],
        'private_scene':P.ref(OUTPUT/'scene.json'),'component_id':'phone_blue','geometry_backend':metadata['backend'],
        'asset':part['asset'],'metadata':part['native_metadata'],'construction_report':P.ref(G.OUTPUT/'report.json'),
        'mechanism_type':'rigid_phone_body','joints':[],'phone_body_deformation_established':False,
        'source_derived_shape':{'outline':'automatic_convex_SAM_mask_outline','broad_faces':'flat_surface_prior',
            'width_over_length':geometry['source_aspect_width_over_length'],
            'thickness_over_length_estimate':geometry['estimated_edge_thickness_over_length'],
            'thickness_is_metric_or_verified':False,'camera_bump_or_port_depth_recovered':False},
        'interfaces':{'USB_C_port':{'status':'source_visible_semantic_proposal_not_metric_contact_frame',
            'frame_transform':None,'insertion_depth':None,'grasp_pose':None,'compatibility_tested':False}},
        'motion':{'source_samples':47,'supported':scene['diagnostics'][0]['estimated'],
            'unknown':scene['diagnostics'][0]['unknown'],'inserted_unknown_barriers':len(barriers),
            'unknown_source_indices':[r['source_sample_index']for r in rows if r['status']=='unknown'],
            'mask_withheld_tracking_samples':[5,26],'accepted_observed_6DoF':0},
        'validation':scene['evaluation_protocol'],'assembly_contact_validated':False,
        'physical_grasping_or_insertion':False,'automatic_publication':False,'published':False}
    P.save(OUTPUT/'descriptive-sidecar.json',sidecar)
    report={'format':'interaction-twin-private-phone-full-scene-report','version':1,
        'plan':P.ref(OUTPUT/'selection-plan.json'),'scene':P.ref(OUTPUT/'scene.json'),
        'audit':P.ref(OUTPUT/'audit.json'),'sidecar':P.ref(OUTPUT/'descriptive-sidecar.json'),
        'coverage':scene['diagnostics'][0],'unknown_rotation_barriers':barriers,
        'unchanged_other_component_ids':[c['id']for c in scene['components']if c['id']!='phone_blue'],
        'all_baseline_unknowns_preserved':all(next(r for r in rows if r['source_sample_index']==i)['status']=='unknown'for i in oldunknown),
        'elapsed_seconds':time.monotonic()-started,'API_or_GPU_calls':0,'automatic_publication':False}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'coverage':report['coverage']}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
