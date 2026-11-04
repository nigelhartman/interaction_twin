"""Source-bracketed removable earbud seating; fixed case-relative attachment.

The cap orientation is selected by a real source/render Muse review among24
proper rotations. Only automatically mask-contained samples get the temporary
attachment. A native gauge is derived from the case rim support plane, and only
actually visible RGB paints the cap. Original assets/unknowns remain intact.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import charger_tracking_recovery_20261103_v1 as C
import native_assembly_refinement_v1 as N
import observed_surface_projection_v1 as P
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mask
from declared_geometry_view_v1 import project_joint,CLAIMS

ROOT=Path(__file__).resolve().parent
BASE='data/assembly-refinements/jabra-source-refit-20261103-v2/scene.json'
PARENT='data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json'
OBJECT='data/object-programs/auto-06f94599100e4d661ca4ca4bd97fc82bd37f259060767ec78c3a1d988ca12b67/object-program.json'
ORIENTATION='data/geometry-quality-20261104/jabra-earbud-orientation-v1'
OUTPUT='data/geometry-quality-20261104/jabra-temporary-seating-v1'
HELDOUT=[75,77]


def build():
    folder=ROOT/OUTPUT;folder.mkdir(parents=True,exist_ok=False);old=C.read(ROOT/BASE);scene=copy.deepcopy(old);base=C.read(ROOT/OBJECT);program=C.read(ROOT/old['original_program']['path']);plan=C.read(ROOT/ORIENTATION/'plan.json');result=C.read(ROOT/ORIENTATION/'attempt/result.json');review=C.ref(ROOT/ORIENTATION/'attempt/summary.json')
    C.require(result['selected_id']!='none','Muse did not find a supported cap orientation');selected=next(c for c in plan['context']['candidates']if c['id']==result['selected_id']);reference=plan['context']['source_samples'][len(plan['context']['source_samples'])//2]['source_sample_index'];C.require(reference==76 and reference not in HELDOUT,'Predeclared single seating witness and disjoint global-fit holdouts required')
    oldparts={p['id']:p for p in old['components']};parts={p['id']:p for p in scene['components']};parent=parts['case_base'];child=parts['earbud_a'];cm=load_native_glb(ROOT/child['asset']['path']);observations={e['source_index']:e for e in program['evidence']};contained=[];coverage=[]
    for row in child['poses']:
        if row['status']!='estimated':continue
        e=observations[row['source_sample_index']];o={r['track_id']:r for r in e['observations']};a=np.asarray(Image.open(ROOT/o['earbud_a']['mask_path']))>0;b=np.asarray(Image.open(ROOT/o['case_base']['mask_path']))>0;contours,_=cv2.findContours(b.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);hull=np.zeros_like(b,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.concatenate(contours)),1);ratio=float(np.count_nonzero(a&(hull>0))/max(1,a.sum()));coverage.append({'sample_index':row['source_sample_index'],'source_mask_inside_case_hull_fraction':ratio})
        if ratio>=.9:contained.append(row['source_sample_index'])
    C.require(contained==[75,76,77,78],'Automatic temporary-containment source interval differs; do not assume a different interaction')
    witness=np.asarray(next(r['matrix']for r in selected['source_pose_estimates']if r['source_sample_index']==reference));parent_witness=np.asarray(next(r['camera_from_native_mesh']for r in parent['poses']if r['source_sample_index']==reference));prior=C.read('data/assembly-refinements/jabra-rim-fit-20261103-final-v1/report.json')['joint']['original']['cavity_mating_prior']['parent'];normal=parent_witness[:3,:3]@np.asarray(prior['normal']);normal/=np.linalg.norm(normal);rim=(parent_witness@np.r_[prior['rim_center'],1.])[:3];world=cm.vertices@witness[:3,:3].T+witness[:3,3];cap=float(np.quantile(world@normal,.99));target=float(rim@normal);scale=target/cap;C.require(.75<=scale<=1.5,'Seating native gauge exceeds bounded inferred calibration')
    affine=np.eye(4);affine[:3,:3]*=scale;raw,encoding=N.rewrite_positions((ROOT/child['asset']['path']).read_bytes(),cm.vertices*scale);geometry=folder/'earbud-a-gauge-derived.glb';geometry.write_bytes(raw);witness[:3,3]*=scale;relative=np.linalg.inv(parent_witness)@witness
    selection={'format':'interaction-twin-source-temporary-containment-plan','version':1,'baseline_scene_ref':C.ref(BASE),'parent_derived_assembly_ref':C.ref(PARENT),'visual_review_ref':review,
        'selected_automatic_orientation_id':result['selected_id'],'reference_sample_index':reference,'heldout_source_sample_indices':HELDOUT,'containment_rule':'earbud_visible_mask_inside_parent_convex_source_mask>=0.9','contained_source_samples':contained,'coverage':coverage,
        'parent_from_derived_earbud_native':relative.tolist(),'constant_native_camera_gauge':scale,'plane_rule':'observed-cap99percent-support flush with automatic parent cavity-rim plane at witness','target_plane_offset':target,
        'cap_plane_offset_before':cap,'cap_plane_offset_after':cap*scale,'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,
        'heldout_scope':'Global gauge/temporary-attachment parameters exclude target masks at75/77; original parent poses and fallible MuseRGBreview already sampled those views. Not an independent pose benchmark.'}
    C.write_new(folder/'selection-plan.json',selection)
    parent_rows={r['source_sample_index']:r for r in parent['poses']}
    for row in child['poses']:
        if row['status']!='estimated':continue
        index=row['source_sample_index'];matrix=np.asarray(row['camera_from_native_mesh']);matrix[:3,3]*=scale
        if index in contained:
            C.require(parent_rows[index]['status']=='estimated','Temporary attachment parent unavailable');matrix=np.asarray(parent_rows[index]['camera_from_native_mesh'])@relative;row['method']='source_mask_contained_removable_case_relative_FK';row['reason']='Muse_cap_orientation_and_automatic_rim_gauge_fixed_parent_relative_attachment'
        else:row['method']='declared_constant_camera_gauge_source_projection_identity';row['reason']='same_rigid_earbud_native_scale_gauge_all_known_source_states'
        row['camera_from_native_mesh']=matrix.tolist();row['accepted_motion_evidence']=False
    source=observations[reference];sourceobs=next(r for r in source['observations']if r['track_id']=='earbud_a');rgbref=C.ref(source['frame_path']);maskref=C.ref(sourceobs['mask_path']);appearance=P.bake(C.ref(geometry),rgbref,maskref,scene['camera'],witness.tolist(),OUTPUT+'/observed-cap',root=ROOT);child['asset']=appearance['asset_ref']
    originals={p['component_id']:p for p in base['component_assets']};assets=[];bindings=[]
    for part in scene['components']:
        identifier=part['id'];native=originals[identifier];changed=identifier=='earbud_a';inherited=identifier=='case_lid';origin='automatic_geometry_operation_chain'if changed else'inherited_approved_geometry_derivative'if inherited else'unchanged_SAM_3D_inference';operations=[{'kind':'affine_positions','matrix':affine.tolist()},appearance['operation']]if changed else[]
        assets.append({'component_id':identifier,'original_asset':native['native_glb'],'input_asset':oldparts[identifier]['asset'],'derived_asset':part['asset'],'geometry_origin':origin,'operations':operations})
        bindings.append({'component_id':identifier,'original_native_glb':native['native_glb'],'native_glb':part['asset'],'native_metadata':part['native_metadata'],'native_uniform_scale':part['native_uniform_scale'],'geometry_origin':origin})
    report={'format':'interaction-twin-geometry-derivative-report','version':1,'base_object_program_ref':C.ref(OBJECT),'baseline_scene_ref':C.ref(BASE),'source_sha256':scene['source']['sha256'],
        'parent_derived_assembly_ref':C.ref(PARENT),'assets':assets,'kinematic_refinement':True,'temporary_attachments':[{'parent_id':'case_base','child_id':'earbud_a','source_sample_indices':contained,'parent_from_child':relative.tolist(),'physical_contact_verified':False}],
        'selection_plan_ref':C.ref(folder/'selection-plan.json'),'source_orientation_review_ref':review,'encoding_proof':encoding,'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False}
    C.write_new(folder/'geometry-report.json',report);meshes={p['id']:load_native_glb(ROOT/p['asset']['path'])for p in scene['components']};oldmeshes={p['id']:load_native_glb(ROOT/p['asset']['path'])for p in old['components']};h=288;w=162;k=np.asarray(scene['camera']['K']).copy();sx=w/scene['source']['width'];sy=h/scene['source']['height'];k[0]*=sx;k[1]*=sy;k[0,2]+=(sx-1)/2;k[1,2]+=(sy-1)/2;rows=[]
    for index in HELDOUT:
        source=observations[index];targets=[];maskrefs=[]
        for o in source['observations']:
            if o['track_id']in parts and o['gate_status']=='candidate_geometric_gate_only':
                mask=np.asarray(Image.open(ROOT/o['mask_path']))>0;targets.append(cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0);maskrefs.append({'component_id':o['track_id'],'artifact':C.ref(o['mask_path'])})
        targetmask=np.logical_or.reduce(targets);scores=[]
        for doc,models in[(old,oldmeshes),(scene,meshes)]:
            union=np.zeros((h,w),bool)
            for part in doc['components']:
                row=next(r for r in part['poses']if r['source_sample_index']==index)
                if row['camera_from_native_mesh']is None:continue
                r,t,s=M.rotation_pose(row['camera_from_native_mesh']);union|=render_native_mask(models[part['id']],k,r,t,w,h,scale=s)['mask']
            scores.append(float(np.count_nonzero(targetmask&union)/max(1,np.count_nonzero(targetmask|union))))
        rows.append({'sample_index':index,'role':'heldout','raster':{'width':w,'height':h,'K':k.tolist()},'source_rgb_ref':C.ref(source['frame_path']),'source_mask_refs':maskrefs,'before_union_iou':scores[0],'after_union_iou':scores[1],'retained':scores[1]>=.65 and scores[1]>=scores[0]-.05})
    eligible=all(r['retained']for r in rows);summary={'default_eligible':eligible,'geometric_gates_passed':True,'source_pixel_gates_passed':eligible,'physical_accuracy_verified':False}
    audit={'format':'interaction-twin-geometry-source-audit','version':1,'baseline_scene_ref':C.ref(BASE),'geometry_report_ref':C.ref(folder/'geometry-report.json'),'source_sha256':scene['source']['sha256'],'summary':summary,'validation_samples':rows,'visual_review_refs':[review],
        'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,'same_mask_or_prior_model_pose_disclosure':selection['heldout_scope'],'actual_earbud_seating_or_contact_verified':False}
    C.write_new(folder/'source-audit.json',audit);scene['declared_geometry_derivative']={'geometry_report_ref':C.ref(folder/'geometry-report.json'),'source_alignment_audit_ref':C.ref(folder/'source-audit.json')};scene['temporary_attachments']=report['temporary_attachments'];C.write_new(folder/'scene.json',scene)
    document={'format':'interaction-twin-declared-geometry-derivative','version':1,'claims':CLAIMS,'source':scene['source'],'evidence_program':scene['original_program'],'base_object_program_ref':C.ref(OBJECT),'parent_derived_assembly_ref':C.ref(PARENT),'baseline_scene_ref':C.ref(BASE),'scene_ref':C.ref(folder/'scene.json'),
        'refinement_report_ref':C.ref(folder/'geometry-report.json'),'geometry_report_ref':C.ref(folder/'geometry-report.json'),'source_alignment_audit_ref':C.ref(folder/'source-audit.json'),'source_audit_summary':summary,'component_assets':bindings,
        'joints':[project_joint(j)for j in scene['mechanics']['joints']],'removable_relationships':base['removable_relationships'],'unresolved_connections':base['unresolved_connections'],'visual_review_refs':[review],
        'temporary_attachments':report['temporary_attachments'],'physical_accuracy_verified':False,'simulation_ready':False,'limitations':['Case-relative earbud attachment is a source-bracketed estimated state, not verified insertion/contact. Native unseen appearance remains as fallback. Other earbud depth and grasp remain unknown.']}
    C.write_new(folder/'derived-assembly.json',document);return {'document':C.ref(folder/'derived-assembly.json'),'scene':C.ref(folder/'scene.json'),'scale':scale,'summary':summary,'validation':rows,'painted_appearance':appearance}


if __name__=='__main__':print(json.dumps(build()),flush=True)
