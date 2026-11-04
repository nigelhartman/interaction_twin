#!/usr/bin/env python3
"""Source-reviewed panel calibration and all-source pose refit, private only."""
from __future__ import annotations
import argparse,copy,json,hashlib,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import planar_assembly_calibration_v1 as C
import native_assembly_refinement_v1 as N
import occluded_hinge_recovery_v1 as R
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mesh
import declared_geometry_view_v1 as V

BASE='data/object-programs/auto-96949ee5d2b737b57417c7454d3775b5b69617c2cbe3aedfe1f1e150da002a25/object-program.json'
FOLDER='data/laptop-mating-20261104-v1/calibrated-v1'
REVIEW='data/laptop-mating-20261104-v1/critic-before-v1/attempt/summary.json'


def ref(path):return N.ref(path,ROOT)
def read(pointer):return json.loads(N.checked(pointer,ROOT))
def write(path,doc):N.write(path,doc)
def iou(a,b):return float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))


def build(width_selection=None):
    started=time.monotonic();out=ROOT/FOLDER;R.require(not out.exists(),'Fresh calibration attempt required');out.mkdir(parents=True)
    base_ref=ref(ROOT/BASE);base=read(base_ref);old_ref=base['constrained_scene'];old=read(old_ref);source=read(old['original_program']);parts={c['id']:c for c in old['components']};joint=old['mechanics']['joints'][0]
    review_ref=ref(ROOT/REVIEW);review=read(review_ref);criticism=read(review['model_result']);R.require(review['verdict']=='needs_repair'and any(i['kind']=='relative_scale'and i['severity']=='high'for i in criticism['issues']),'New actual source/render relative-scale finding required')
    meshes={c:load_native_glb(N.checked(p['asset'],ROOT))for c,p in parts.items()};pv=meshes['laptop_base'].vertices;cv=meshes['laptop_lid'].vertices
    normals={c:C.normal_from_supported_interior(meshes[c].vertices,np.asarray(p['poses'][43]['camera_from_native_mesh']))for c,p in parts.items()}
    width_ref=ref(ROOT/width_selection)if width_selection else None
    width_quantiles=read(width_ref)['selected_quantiles']if width_ref else None
    child,a,axis,b,calibration=C.calibrate(pv,cv,joint,normals['laptop_base'],normals['laptop_lid'],width_quantiles=width_quantiles)
    raw,encoding=N.rewrite_positions(N.checked(parts['laptop_lid']['asset'],ROOT),child);asset=out/'laptop-lid-calibrated.glb';asset.write_bytes(raw);derived_ref=ref(asset);derived_mesh=load_native_glb(raw)
    child=derived_mesh.vertices
    geometry_report={'format':V.REPORT,'version':1,'base_object_program_ref':base_ref,'baseline_scene_ref':old_ref,'source_sha256':old['source']['sha256'],
        'assets':[{'component_id':cid,'original_asset':p['asset'],'derived_asset':derived_ref if cid=='laptop_lid'else p['asset'],
                   'geometry_origin':'automatic_affine_calibration'if cid=='laptop_lid'else'unchanged_SAM_3D_inference',
                   'operations':[{'kind':'affine_positions','matrix':calibration['affine_matrix']}]if cid=='laptop_lid'else [],'appearance_preserved':True}for cid,p in parts.items()],
        'calibration':calibration,'position_encoding':encoding,'kinematic_refinement':True,'source_constraint_reviews':[review_ref],
        'source_rule':'Muse observed flush laptop panel width, matching closed depth and rear hinge; no authored metric dimensions',
        'geometry_train_source_indices':[1,27,44],'geometry_heldout_source_indices':[16,26,30,54,66],
        'heldout_scope':'global_geometry_frozen_before_target_frame_pose_fit; conditional_pose_fit_not_independent_pose_accuracy',
        'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,'new_model_calls':0}
    if width_ref:geometry_report['source_width_selection_ref']=width_ref
    write(out/'geometry-report.json',geometry_report);report_ref=ref(out/'geometry-report.json')
    masks={};mask_refs={};source_refs={};features={};sample_rows={};snapshot=out/'implementation';snapshot.mkdir()
    for name in ['planar_assembly_calibration_v1.py','scripts/calibrate_laptop_mating_20261104_v1.py']:(snapshot/(Path(name).name+'.txt')).write_bytes((ROOT/name).read_bytes())
    write(out/'selection-plan.json',{'geometry_report_ref':report_ref,'policy':C.POLICY,'training_samples':[1,27,44],'heldout_samples':[16,26,30,54,66],
        'source_or_native_geometry_parameters_frozen_before_heldout_pixels':True,'implementation':[ref(p)for p in snapshot.iterdir()]})
    for e in source['evidence']:
        i=e['source_index'];source_refs[i]={'path':e['frame_path'],'sha256':e['frame_sha256'],'bytes':(ROOT/e['frame_path']).stat().st_size};rgb=np.asarray(Image.open(ROOT/e['frame_path']).convert('RGB'))
        oldrows={cid:p['poses'][i-1]for cid,p in parts.items()};available=all(r['status']=='estimated'for r in oldrows.values());sample_rows[i]={'source_sample_index':i,'timestamp_s':e['timestamp_s'],'baseline_supported':available,'retained':False}
        if not available:continue
        closed=abs(joint['samples'][i-1]['q_rad'])<1e-8
        if closed:
            visible=oldrows['laptop_base'].get('source_visible_panel','laptop_lid');pointer=oldrows[visible].get('source_mask');R.require(pointer,'Closed visible mask provenance required');array=np.asarray(Image.open(ROOT/pointer['path']))>0
            masks[i]={'laptop_base':array if visible=='laptop_base'else np.zeros_like(array),'laptop_lid':array if visible=='laptop_lid'else np.zeros_like(array)};mask_refs[i]={visible:pointer}
        else:
            masks[i]={};mask_refs[i]={}
            for cid in parts:
                pointer=oldrows[cid].get('source_mask');R.require(pointer,'Visible open panel mask provenance required');masks[i][cid]=np.asarray(Image.open(ROOT/pointer['path']))>0;mask_refs[i][cid]=pointer
            cleaned=masks[i]['laptop_lid']&~masks[i]['laptop_base']
            if np.count_nonzero(cleaned)>=1000:
                path=out/'source-visible-masks'/f'laptop-lid-{i:06d}.png';path.parent.mkdir(exist_ok=True);Image.fromarray(cleaned.astype(np.uint8)*255).save(path)
                masks[i]['laptop_lid']=cleaned;mask_refs[i]['laptop_lid']=ref(path)
            visible='laptop_base'
        features[i]=R.template_features(rgb,masks[i]['laptop_base']if not closed else masks[i][visible])
        sample_rows[i].update(closed=closed,visible_panel=visible,source_mask_refs=[{'component_id':c,'artifact':p}for c,p in mask_refs[i].items()],source_rgb_ref=source_refs[i])
    refquad=R.mask_quad(masks[44]['laptop_base']);fitted={}
    for i,row in sample_rows.items():
        if not row['baseline_supported']:continue
        prior=np.asarray(parts['laptop_base']['poses'][i-1]['camera_from_native_mesh']);quad=None
        if not row['closed']and i!=44:
            support=R.template_support(features[44],features[i]);row['source_quad_support']=support
            if support['accepted']:
                quad=cv2.perspectiveTransform(refquad[None].astype(float),np.asarray(support['homography']))[0]
                row['source_quad_method']='source44_foreground_homography_extrapolates_full_base_contour'
        try:
            fit=C.fit_sample(pv,child,masks[i]['laptop_base'],masks[i]['laptop_lid'],old['camera'],a,axis,b,prior,normals['laptop_base'],normals['laptop_lid'],closed=row['closed'],visible_panel=row['visible_panel'],quad=quad)
            keep=fit['scores']['union']>=C.POLICY['minimum_union_iou']and (row['closed']or min(fit['scores'].get('parent',0),fit['scores'].get('child',0))>=C.POLICY['minimum_visible_part_iou'])
            row.update(retained=keep,convex_fit_scores=fit['scores'],q_rad=fit['q'],source_quad=fit['source_quad'],fit_evaluations=fit['fit_evaluations'])
            if keep:fitted[i]=fit
            else:row['withheld_reason']='new_source_fit_failed_prespecified_silhouette_gate'
        except ValueError as exc:row['withheld_reason']=str(exc)
        print(json.dumps({'source':i,'retained':row['retained'],'fit':row.get('convex_fit_scores'),'q':row.get('q_rad')}),flush=True)
    scene=copy.deepcopy(old);newparts={c['id']:c for c in scene['components']};newparts['laptop_lid']['asset']=derived_ref
    nj=scene['mechanics']['joints'][0];nj['parent_from_joint']=a.tolist();nj['axis']=axis.tolist();nj['joint_from_child_zero']=b.tolist();nj['parent_anchor_native']=calibration['pivot_parent_native'];nj['child_anchor_native']=calibration['pivot_child_derived_native'];nj['anchor_kind']='source_reviewed_rear_support_plane_midthickness_intersection';nj.pop('parent_anchor_vertex_id',None);nj.pop('child_anchor_vertex_id',None)
    for i,row in sample_rows.items():
        for cid,key in [('laptop_base','parent'),('laptop_lid','child')]:
            pose=newparts[cid]['poses'][i-1]
            if i in fitted:
                pose.update(camera_from_native_mesh=fitted[i][key].tolist(),status='estimated',reason='global_source_reviewed_panel_calibration_and_rear_joint_source_refit',method='full_mesh_affine_calibration_source_quad_and_fixed_rear_hinge')
                if cid in mask_refs[i]:pose['source_mask']=mask_refs[i][cid]
            else:pose.update(camera_from_native_mesh=None,status='unknown',reason=row.get('withheld_reason','baseline_unknown_preserved'))
        sample=nj['samples'][i-1];sample.update(status='estimated'if i in fitted else'unknown',q_rad=fitted[i]['q']if i in fitted else None)
    angles=[s['q_rad']for s in nj['samples']if s['status']=='estimated'];nj['limits_rad']=[min(angles),max(angles)];nj['fit_diagnostics']={'method':'global_source_reviewed_panel_geometry_then_source_pose_and_angle_refit','geometry_report_ref':report_ref,'physical_joint_recovery_verified':False}
    # Keep angle meaning honest after the new source-conditioned curve.
    by_index={s['source_sample_index']:s for s in nj['samples']};proposal=read(base['proposal']);semantic=next(j for j in proposal['joints']if j['id']==nj['id']);comparisons=[]
    for hint in semantic['state_hints']:
        q=by_index[hint['source_sample_index']]
        if hint['state']in {'open','closed'}and q['status']=='estimated':comparisons.append({'source_sample_index':q['source_sample_index'],'timestamp_s':q['timestamp_s'],'cited_state':hint['state'],'q_rad':q['q_rad'],'distance_from_closed_reference_rad':abs(q['q_rad'])})
    nj['angle_quality']['state_comparisons']=comparisons;opened=[s for s in comparisons if s['cited_state']=='open'];consistent=bool(opened and all(abs(s['q_rad'])>np.pi/12 for s in opened));nj['angle_quality'].update(angle_visual_consistency=consistent,angle_interpretation='estimated_opening_configuration_not_physically_calibrated'if consistent else'model_configuration_parameter_not_verified_opening_angle')
    # Full triangle measurements on the predeclared train and geometry-heldout
    # views. Pose fitting is conditional on target masks and is labelled as such.
    validation=[];tiles=[];w,h=241,320;k=np.asarray(old['camera']['K']).copy();k[0]*=w/722;k[1]*=h/960;k[0,2]+=(w/722-1)/2;k[1,2]+=(h/960-1)/2
    for i in [1,16,26,27,30,44,54,66]:
        rgb=np.full((h,w,3),245,np.uint8);z=np.full((h,w),np.inf);baseline_mask=np.zeros((h,w),bool)
        for cid,p in parts.items():
            mat=p['poses'][i-1]['camera_from_native_mesh']
            if mat is not None:
                rot,t,scale=M.rotation_pose(mat);im=render_native_mesh(meshes[cid],k,rot,t,w,h,scale=scale);baseline_mask|=im['mask']
            if i in fitted:
                mat=fitted[i]['parent'if cid=='laptop_base'else'child'];rot,t,scale=M.rotation_pose(mat);im=render_native_mesh(meshes[cid]if cid=='laptop_base'else derived_mesh,k,rot,t,w,h,scale=scale);closer=im['mask']&(im['depth']<z);rgb[closer]=im['rgb'][closer];z[closer]=im['depth'][closer]
        target=cv2.resize((masks[i]['laptop_base']|masks[i]['laptop_lid']).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        metric={'sample_index':i,'role':'train'if i in [1,27,44]else'heldout','source_rgb_ref':source_refs[i],'source_mask_refs':sample_rows[i]['source_mask_refs'],'retained':i in fitted,
                'before_union_iou':iou(target,baseline_mask),'after_union_iou':iou(target,np.isfinite(z)),'raster':{'width':w,'height':h,'K':k.tolist()},'conditional_pose_fit':True}
        validation.append(metric);tile=Image.new('RGB',(w*2,h+42),'#f7f8fb');tile.paste(Image.open(ROOT/source_refs[i]['path']).resize((w,h)),(0,42));tile.paste(Image.fromarray(rgb),(w,42));d=ImageDraw.Draw(tile);d.text((8,6),f'Source {i} | global panel mating | {metric["after_union_iou"]:.3f}',fill='#171923');d.text((8,23),f'{metric["role"]}: geometry frozen / pose estimated',fill='#3766ed');tiles.append(tile)
    sheet=Image.new('RGB',(w*2*4,(h+42)*2),'white')
    for n,tile in enumerate(tiles):sheet.paste(tile,((n%4)*w*2,(n//4)*(h+42)))
    sheet.save(out/'source-calibrated-proof.jpg')
    held=[r for r in validation if r['role']=='heldout'];pixel_pass=all(r['retained']and r['after_union_iou']>=.65 and r['after_union_iou']>=r['before_union_iou']-.05 for r in held)
    summary={'default_eligible':pixel_pass,'geometric_gates_passed':True,'source_pixel_gates_passed':pixel_pass,'supported_samples':len(fitted),'unknown_samples':len(sample_rows)-len(fitted),
             'heldout_sample_count':len(held),'minimum_heldout_union_iou':min(r['after_union_iou']for r in held),'physical_accuracy_verified':False}
    audit={'format':V.AUDIT,'version':1,'baseline_scene_ref':old_ref,'geometry_report_ref':report_ref,'source_sha256':old['source']['sha256'],'policy':{**C.POLICY,'unknown_policy':'preserve_or_withhold_baseline_unknowns'},
           'rows':list(sample_rows.values()),'validation_samples':validation,'summary':summary,'visual_review_refs':[review_ref],
           'source_proof':ref(out/'source-calibrated-proof.jpg'),'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,'new_api_calls':0,'new_gpu_calls':0}
    write(out/'source-audit.json',audit);audit_ref=ref(out/'source-audit.json');scene['declared_geometry_derivative']={'geometry_report_ref':report_ref,'source_alignment_audit_ref':audit_ref}
    write(out/'scene.json',scene);scene_ref=ref(out/'scene.json')
    doc={'format':V.FORMAT,'version':1,'base_object_program_ref':base_ref,'evidence_program':base['evidence_program'],'baseline_scene_ref':old_ref,'source':old['source'],'scene_ref':scene_ref,'refinement_report_ref':report_ref,'source_alignment_audit_ref':audit_ref,
         'component_assets':[{'component_id':cid,'native_glb':newparts[cid]['asset'],'original_native_glb':p['asset'],'native_metadata':p['native_metadata'],'native_uniform_scale':p['native_uniform_scale'],'geometry_origin':'automatic_affine_calibration'if cid=='laptop_lid'else'unchanged_SAM_3D_inference'}for cid,p in parts.items()],
         'joints':[V.project_joint(nj)],'removable_relationships':base['removable_relationships'],'unresolved_connections':base['unresolved_connections'],'source_audit_summary':summary,'visual_review_refs':[review_ref],
         'claims':V.CLAIMS,'physical_accuracy_verified':False,'simulation_ready':False,'limitations':['Relative panel scale and rear joint are source-conditioned estimates, not metric CAD.',
             'Native exterior texture may remain wrong; original assets and native program are preserved.','Masks and pose fitting are conditional image diagnostics; contact and robot execution are not validated.']}
    write(out/'declared-geometry.json',doc)
    result={'status':'calibrated_private_candidate'if pixel_pass else'calibrated_private_candidate_failed_heldout_gate','document_ref':ref(out/'declared-geometry.json'),'scene_ref':scene_ref,'geometry_report_ref':report_ref,'source_audit_ref':audit_ref,'summary':summary,'new_api_calls':0,'new_gpu_calls':0,'wall_seconds':time.monotonic()-started}
    write(out/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=FOLDER);parser.add_argument('--width-selection');args=parser.parse_args();FOLDER=args.output;print(json.dumps(build(args.width_selection)),flush=True)
