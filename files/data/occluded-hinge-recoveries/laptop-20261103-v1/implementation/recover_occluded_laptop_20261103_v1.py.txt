#!/usr/bin/env python3
"""Build a new private source-state-gated laptop assembly; never publish it."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import occluded_hinge_recovery_v1 as R
import sam3d_common_scene as S
import sam3d_mechanisms as M
import object_program as O
import mechanism_pipeline as P
from native_appearance import load_native_glb

SCENE = 'data/native-orientation-diagnostics/laptop-20261103-v1-attempt2/appearance-refit-v1/constrained/scene.json'
SCENE_SHA = '4069bcb58593272d0903b6c725ad7ab62a710d3d8683cd5ed133469ee187fd51'
COMMON = 'data/mechanism-scenes/1e7771cdbc91174f8e57900b42b1b95789556255aaa73911cc09909f50db8422/private-common/scene.json'
PACKET = 'data/source-observation-refinements/laptop-coverage-20261103-v3'
OUTPUT = 'data/occluded-hinge-recoveries/laptop-20261103-v1'


def write(path, value):
    R.require(not path.exists(), 'Preserve earlier recovery attempts')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def ref(path):
    return S.ref(path, ROOT)


def checked(reference):
    path = ROOT/reference['path']
    R.require(ref(path) == reference, 'Exact recovery source changed')
    return path


def hull_score(vertices, matrix, k, target):
    uv, valid = M.project(vertices, matrix, k)
    if not valid.all(): return 0.
    return S.silhouette_score(uv,target)[1]


def build(output=OUTPUT):
    start=time.monotonic(); folder=ROOT/output
    R.require(not folder.exists(), 'Fresh immutable recovery directory required')
    scene_ref=ref(ROOT/SCENE);R.require(scene_ref['sha256']==SCENE_SHA,'Exact approved baseline scene required')
    baseline=json.loads(checked(scene_ref).read_text());common_ref=ref(ROOT/COMMON);common=json.loads(checked(common_ref).read_text())
    evidence=json.loads(checked(baseline['original_program']).read_text());by_evidence={e['source_index']:e for e in evidence['evidence']}
    packet=ROOT/PACKET;result_ref=ref(packet/'attempt/model-result.json');result=json.loads(checked(result_ref).read_text())
    proposal_ref=ref(packet/'attempt/proposals.json');proposal=O.load_proposals(proposal_ref['path'],baseline['original_program']['path'],root=ROOT)
    receipt_ref=ref(packet/'attempt/request/meta-request.json');receipt=json.loads(checked(receipt_ref).read_text())
    R.require(receipt.get('status')=='complete_accounted','Exact newly accounted source-state request required')
    source_rows=result['frames']
    R.require([r['sample_index'] for r in source_rows]==[1,16,26,27,30,54,55,66],'Exact predeclared source observations required')
    parts={c['id']:c for c in baseline['components']};meshes={c:load_native_glb(checked(part['asset']))for c,part in parts.items()}
    R.require(set(parts)=={'laptop_base','laptop_lid'},'Two exact native panels required')
    old_joint=baseline['mechanics']['joints'][0];closed=R.closure_geometry(meshes['laptop_base'].vertices,meshes['laptop_lid'].vertices,old_joint)
    rel=np.asarray(closed['relative_matrix']);q_closed=closed['angle_rad'];k=np.asarray(baseline['camera']['K'])
    features={};masks={};image_refs={};mask_refs={}
    for e in evidence['evidence']:
        image_ref={'path':e['frame_path'],'sha256':e['frame_sha256'],'bytes':(ROOT/e['frame_path']).stat().st_size}
        image=np.asarray(Image.open(checked(image_ref)).convert('RGB'));observation=next(o for o in e['observations']if o['track_id']=='laptop_lid')
        mask_ref={'path':observation['mask_path'],'sha256':observation['mask_sha256'],'bytes':(ROOT/observation['mask_path']).stat().st_size}
        mask=np.asarray(Image.open(checked(mask_ref)))>0
        R.require(observation['gate_status']=='candidate_geometric_gate_only' and not str(observation.get('audit_status','')).startswith('reject_'),'Recovery must not bypass a rejected visible mask')
        features[e['source_index']]=R.template_features(image,mask);masks[e['source_index']]=mask
        image_refs[e['source_index']]=image_ref;mask_refs[e['source_index']]=mask_ref
    anchors={}
    for row in source_rows:
        index=row['sample_index'];R.require(row['timestamp_s']==by_evidence[index]['timestamp_s'],'Muse source PTS changed')
        if row['laptop_state']!='closed' or row['state_confidence']<.7 or row['visible_panel_identity_confidence']<.7:continue
        panel=row['visible_laptop_panel']
        expected='exterior' if panel=='laptop_lid' else 'underside' if panel=='laptop_base' else None
        field='lid_visible_face'if panel=='laptop_lid'else'base_visible_face'
        if panel in parts and row[field]==expected:anchors[index]={'panel':panel,'face':expected,'source_observation':row}
    R.require(any(a['panel']=='laptop_lid'for a in anchors.values()),'At least one newly observed closed exterior is required')
    source_observations={row['sample_index']:row for row in source_rows}
    negative_state_checks=[]
    for row in source_rows:
        if row['laptop_state'] not in {'open','partially_open'} or row['state_confidence']<.7:continue
        matches=[]
        for anchor_id in anchors:
            support=R.template_support(features[anchor_id],features[row['sample_index']])
            matches.append({'anchor_sample_index':anchor_id,**support})
        negative_state_checks.append({'source_sample_index':row['sample_index'],'source_state':row['laptop_state'],
                                      'any_closed_template_match':any(m['accepted']for m in matches),'matches':matches,
                                      'explicit_open_state_always_overrides_template':True})
    normals={}
    for identifier,mesh in meshes.items():
        normal=np.linalg.svd(mesh.vertices-mesh.vertices.mean(0),full_matrices=False)[2][-1]
        # Anchor44 is the already supported generation/view of both inside faces.
        anchor=np.asarray(parts[identifier]['poses'][43]['camera_from_native_mesh'])
        normal*=np.sign((anchor[:3,:3]@normal)@(-anchor[:3,3]))
        normals[identifier]=normal
    rows=[];recovered={}
    parent_rows={r['source_sample_index']:r for r in parts['laptop_base']['poses']}
    common_rows={c['id']:{r['source_sample_index']:r for r in c['poses']}for c in common['components']}
    for e in evidence['evidence']:
        index=e['source_index'];audit={'source_sample_index':index,'timestamp_s':e['timestamp_s'],'original_status':parent_rows[index]['status'],'retained':False}
        if parent_rows[index]['status']=='estimated':
            audit.update(reason='original_open_source_pose_preserved',retained=True,newly_recovered=False);rows.append(audit);continue
        known=source_observations.get(index)
        if known and known['laptop_state']!='closed' and known['state_confidence']>=.7:
            audit['reason']='direct_Muse_nonclosed_state_forbids_closed_recovery';rows.append(audit);continue
        candidates=[]
        for anchor_id,anchor in anchors.items():
            support=R.template_support(features[anchor_id],features[index],same_frame=anchor_id==index)
            if support['accepted']:candidates.append((anchor_id,anchor,support))
        if not candidates:
            audit['reason']='no_direct_closed_observation_or_distributed_matching_exterior';rows.append(audit);continue
        candidates.sort(key=lambda item:(item[0]!=index,-(item[2].get('inliers')or 0),abs(item[0]-index)))
        anchor_id,anchor,support=candidates[0];panel=anchor['panel'];prior=common_rows[panel][index]['camera_from_native_mesh']
        if prior is None:
            available=[r for r in common_rows[panel].values()if r['status']=='estimated']
            if not available:
                audit['reason']='no_prior_for_long_edge_ambiguity';rows.append(audit);continue
            prior=min(available,key=lambda r:abs(r['source_sample_index']-index))['camera_from_native_mesh']
        try:
            fitted=R.panel_pose(meshes[panel].vertices,masks[index],baseline['camera'],parts[panel]['native_uniform_scale'],prior,normals[panel])
            observed_matrix=np.asarray(fitted['camera_from_native_mesh'])
            parent=observed_matrix@np.linalg.inv(rel)if panel=='laptop_lid'else observed_matrix
            child=parent@rel
            scores={'visible_panel':hull_score(meshes[panel].vertices,observed_matrix,k,S.mask_shape(masks[index]))}
            uvp,vp=M.project(meshes['laptop_base'].vertices,parent,k);uvc,vc=M.project(meshes['laptop_lid'].vertices,child,k)
            scores['union']=S.silhouette_score(np.r_[uvp,uvc],S.mask_shape(masks[index]))[1]if vp.all()and vc.all()else 0.
            R.require(scores['visible_panel']>=R.POLICY['minimum_convex_visible_panel_iou']and scores['union']>=R.POLICY['minimum_convex_union_iou'],'Closed assembly source silhouette gate failed')
            recovered[index]={'parent':parent,'child':child,'panel':panel,'mask_ref':mask_refs[index]}
            audit.update(retained=True,newly_recovered=True,reason='closed_visible_panel_drives_fixed_connected_assembly',visible_panel=panel,
                         hidden_panel='laptop_base'if panel=='laptop_lid'else'laptop_lid',hidden_panel_pose_observed=False,
                         source_state_anchor_index=anchor_id,source_state_support=support,panel_fit=fitted,convex_mask_iou=scores,
                         observed_mask_ref=mask_refs[index],source_rgb_ref=image_refs[index])
        except ValueError as error:
            audit.update(reason='withheld_closed_recovery_gate',diagnostic=str(error),source_state_anchor_index=anchor_id,source_state_support=support)
        rows.append(audit)
    R.require(len(recovered)>=8,'Too few gated source-visible closed recoveries')
    folder.mkdir(parents=True)
    snapshots=[]
    for name in ['occluded_hinge_recovery_v1.py','scripts/recover_occluded_laptop_20261103_v1.py']:
        destination=folder/'implementation'/(Path(name).name+'.txt');destination.parent.mkdir(exist_ok=True);destination.write_bytes((ROOT/name).read_bytes());snapshots.append(ref(destination))
    audit={'format':'interaction-twin-occluded-hinge-recovery-audit','version':1,'baseline_scene':scene_ref,'original_common_scene':common_ref,
           'source_observation_result':result_ref,'source_observation_receipt':receipt_ref,'source_proposal':proposal_ref,'policy':R.POLICY,
           'closed_geometry':closed,'source_closed_anchors':{str(i):a for i,a in anchors.items()},'rows':rows,'implementation_snapshots':snapshots,
           'explicit_nonclosed_source_checks':negative_state_checks,
           'original_supported_samples':sum(r['status']=='estimated'for r in parent_rows.values()),'newly_recovered_closed_samples':len(recovered),
           'retained_supported_samples':sum(r['retained']for r in rows),'unknown_samples':sum(not r['retained']for r in rows),
           'mask_metrics_are_conditional_fit_not_heldout_accuracy':True,'geometry_changed':False,'manual_geometry_or_poses_used':False,
           'new_api_calls':0,'new_gpu_calls':0,'published':False,'physical_accuracy_verified':False,'wall_seconds':time.monotonic()-start}
    write(folder/'audit.json',audit);audit_ref=ref(folder/'audit.json')
    research=copy.deepcopy(baseline);research.pop('mechanics');research['fitter_version']='source_closed_panel_occluded_parent_recovery_v1'
    for part in research['components']:
        for row in part['poses']:
            if row['source_sample_index']in recovered:
                solved=recovered[row['source_sample_index']];visible=part['id']==solved['panel'];matrix=solved['parent'if part['id']=='laptop_base'else'child']
                row.update(status='estimated',camera_from_native_mesh=matrix.tolist(),reason='source_visible_closed_panel_estimate'if visible else'occluded_part_estimated_only_via_fixed_joint_and_source_closed_state',
                           method='automatic_source_mask_quadrilateral_PnP_fixed_joint_FK',source_mask=solved['mask_ref'],source_visible_panel=solved['panel'],
                           hidden_part_pose_observed=False if not visible else None,accepted_motion_evidence=False)
    inputs={r['path']:r for r in [*research['inputs'],scene_ref,common_ref,result_ref,receipt_ref,proposal_ref,audit_ref,*snapshots,*image_refs.values(),*mask_refs.values()]}
    research['inputs']=[inputs[p]for p in sorted(inputs)]
    research['research_derivation']={'kind':'source_observed_closed_panel_drives_occluded_fixed_hinge_assembly','baseline_scene':scene_ref,'recovery_audit':audit_ref,
                                    'new_source_observations':result_ref,'implementation_snapshots':snapshots,'original_assets_changed':False,'manual_geometry_or_poses_used':False}
    research['input_signature']=hashlib.sha256(json.dumps(research['research_derivation'],sort_keys=True).encode()).hexdigest()
    research['limitations']+=['Hidden panels in source-closed configurations are estimates through the retained joint, not independently observed poses.',
                            'Generated reverse-side textures can disagree with real laptop exterior; no new texture was invented or painted.',
                            'Closed geometry is a thin native-panel estimate; contact, interpenetration and physical opening angle remain unverified.']
    write(folder/'source-scene.json',research)
    scene=copy.deepcopy(research);mechanics=copy.deepcopy(baseline['mechanics']);joint=mechanics['joints'][0]
    old_axis=np.asarray(joint['axis']);rotation=np.eye(4);rotation[:3,:3]=M.axis_rotation(old_axis,q_closed)
    joint['joint_from_child_zero']=(rotation@np.asarray(joint['joint_from_child_zero'])).tolist();joint['axis']=(-old_axis).tolist()
    semantic=next(j for j in proposal['joints']if j['id']==joint['id']);joint['evidence_ids']=semantic['evidence_ids'];joint['hinge_hints']=semantic['hinge_hints']
    for sample in joint['samples']:
        index=sample['source_sample_index']
        if index in recovered:sample.update(status='estimated',q_rad=0.,reason='source_closed_state_and_native_thin_panel_zero',accepted_motion_evidence=False)
        elif sample['status']=='estimated':sample['q_rad']=q_closed-sample['q_rad']
    zero_index=next(i for i in sorted(anchors)if i in recovered)
    joint['zero_reference']={'source_sample_index':zero_index,'timestamp_s':by_evidence[zero_index]['timestamp_s'],'state':'estimated_closed'}
    joint['limits_rad']=[min(r['q_rad']for r in joint['samples']if r['status']=='estimated'),max(r['q_rad']for r in joint['samples']if r['status']=='estimated')]
    joint['axis_sign_flipped_for_positive_median_cited_open']=True
    samples={r['source_sample_index']:r for r in joint['samples']};comparisons=[]
    for hint in semantic['state_hints']:
        sample=samples.get(hint['source_sample_index'])
        if hint['state']in {'closed','open'}and sample and sample['status']=='estimated':
            comparisons.append({'source_sample_index':sample['source_sample_index'],'timestamp_s':sample['timestamp_s'],'cited_state':hint['state'],'q_rad':sample['q_rad'],'distance_from_closed_reference_rad':abs(sample['q_rad'])})
    quality=joint['angle_quality'];opened=[r for r in comparisons if r['cited_state']=='open'];consistent=bool(opened and all(r['distance_from_closed_reference_rad']>math.pi/12 for r in opened))
    quality.update(state_comparisons=comparisons,angle_visual_consistency=consistent,angle_interpretation='estimated_opening_configuration_not_physically_calibrated'if consistent else'model_configuration_parameter_not_verified_opening_angle',
                   quality_flags=['native_closure_contact_and_interpenetration_unverified','hidden_part_pose_estimated_from_visible_closed_panel'])
    joint['fit_diagnostics']={'method':'retained_exact_native_seam_anchors_and_open_source_matrices_plus_new_gated_closed_panel_recovery',
                              'original_native_anchor_fit_scene':scene_ref,'recovery_audit':audit_ref,'prior_open_source_matrices_unchanged':True,
                              'angle_visual_consistency':consistent,'physical_joint_recovery_verified':False,'source_alignment_verified':False,'structural_connectivity_verified':True}
    joint.pop('shared_parent_refinement',None)
    joint['closed_panel_recovery']={'audit':audit_ref,'original_configuration_parameter_at_closed':q_closed,'geometry':closed,'hidden_part_pose_observed':False}
    mechanics.update(source_scene=ref(folder/'source-scene.json'),proposal=proposal_ref,wall_seconds=time.monotonic()-start)
    scene['mechanics']=mechanics
    write(folder/'mechanics.json',mechanics);write(folder/'scene.json',scene)
    exported=P.export_scene(ref(folder/'scene.json'),root=ROOT,publish=False)
    result={'status':'new_private_source_gated_occluded_hinge_candidate','scene_ref':ref(folder/'scene.json'),'object_program_ref':exported['object_program'],'audit_ref':audit_ref,
            'original_supported_samples':audit['original_supported_samples'],'retained_supported_samples':audit['retained_supported_samples'],'unknown_samples':audit['unknown_samples'],
            'new_api_calls':0,'new_gpu_calls':0,'automatic_approval':False,'automatic_publication':False}
    write(folder/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=OUTPUT);args=parser.parse_args()
    print(json.dumps(build(args.output)),flush=True)
