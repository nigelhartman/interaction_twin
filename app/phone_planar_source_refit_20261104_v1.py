"""Automatic private source refit of a declared compact planar phone subset.

Two predeclared RGB/mask targets are not read until their poses have been
predicted from neighboring fitting rows. Native original unknowns stay unknown.
This evaluates image fit, not physical geometry or complete recovered 6DoF.
"""
from __future__ import annotations
import copy,json,math,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation,Slerp
import charger_tracking_recovery_20261103_v1 as C
import native_orientation_diagnostic_v1 as D
import sam3d_common_scene as S
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mask

ROOT=Path(__file__).resolve().parent
BASE='data/mechanism-scenes/648d3924ab106192ea1cdad94c6cdf0b2d44d5c61bba3090cd5dbb1fcb00f298/constrained/scene.json'
CANDIDATE='data/geometry-quality-20261104/phone-planar-candidates-v5/planar_surface_2/report.json'
PROGRAM='data/flow-capture-phone-20261103-v1/flow-procedure-1/interaction.json'
OUTPUT='data/geometry-quality-20261104/phone-source-refit-v1'
HELDOUT=[5,26]


def interpolate(a,b,f):
    ra,ta,sa=M.rotation_pose(a);rb,tb,sb=M.rotation_pose(b);C.require(np.isclose(sa,sb),'Same native scale required')
    m=np.eye(4);m[:3,:3]=sa*Slerp([0.,1.],Rotation.from_matrix([ra,rb]))(f).as_matrix();m[:3,3]=ta*(1-f)+tb*f;return m


def build():
    started=time.monotonic();folder=ROOT/OUTPUT;folder.mkdir(parents=True,exist_ok=False);baseline=C.read(ROOT/BASE);scene=copy.deepcopy(baseline);program=C.read(ROOT/PROGRAM);chosen=C.read(ROOT/CANDIDATE)
    C.require(C.ref(chosen['original_asset']['path'])==chosen['original_asset']and C.ref(chosen['derived_asset']['path'])==chosen['derived_asset'],'Exact native and automatic subset required')
    mesh=load_native_glb(ROOT/chosen['derived_asset']['path']);old_mesh=load_native_glb(ROOT/chosen['original_asset']['path']);component=next(c for c in scene['components']if c['id']=='phone_blue');old=copy.deepcopy(component)
    component['asset']=chosen['derived_asset'];evidence={e['source_index']:e for e in program['evidence']};rows={r['source_sample_index']:r for r in component['poses']};oldrows={r['source_sample_index']:r for r in old['poses']}
    model=S.load_model(ROOT/component['native_metadata']['path'],scene['source']['sha256'],ROOT);initial_rotation=model['layout'].rotation_cv_from_glb;scale=component['native_uniform_scale'];initial_depth=float(model['layout'].translation_cv[2]*component['depth_gauge']['scale']);k=np.asarray(scene['camera']['K']);h0,w0=scene['source']['height'],scene['source']['width']
    vertices=mesh.vertices[np.linspace(0,len(mesh.vertices)-1,min(2400,len(mesh.vertices))).astype(int)];pca=D.pca_rotations(mesh.vertices);mask_cache={};input_refs=[];diagnostics=[]
    plan={'format':'interaction-twin-phone-refit-selection-plan','version':1,'baseline':C.ref(BASE),'candidate_report':C.ref(CANDIDATE),'original_program':C.ref(PROGRAM),
        'heldout_source_sample_indices':HELDOUT,'target_pixels_loaded_after_prediction':True,'original_unknowns_remain_unknown':True,'native_scale_and_camera_unchanged':True,
        'hypothesis_generation':'automatic local22degree search; if inadequate,24proper native PCA orientations; no manualpose',
        'minimum_heldout_iou':.65,'maximum_heldout_regression':.05,'minimum_full_triangle_train_iou':.35,'maximum_training_regression':.05,'new_api_calls':0,'new_gpu_calls':0}
    C.write_new(folder/'selection-plan.json',plan)
    def mask_at(index,*,evaluation=False):
        C.require(evaluation or index not in HELDOUT,'Heldout mask cannot enter pose fitting')
        if index not in mask_cache:
            obs=next(x for x in evidence[index]['observations']if x['track_id']=='phone_blue');ref=C.ref(obs['mask_path']);C.require(ref['sha256']==obs['mask_sha256'],'Source mask changed');mask_cache[index]=np.asarray(Image.open(ROOT/ref['path']))>0;input_refs.append(ref)
        return mask_cache[index]
    factor=288/h0;w,h=round(w0*factor),288;smallk=k.copy();smallk[0]*=w/w0;smallk[1]*=h/h0;smallk[0,2]+=(w/w0-1)/2;smallk[1,2]+=(h/h0-1)/2
    def raw_iou(target,which,matrix):
        r,t,s=M.rotation_pose(matrix);pred=render_native_mask(which,smallk,r,t,w,h,scale=s)['mask'];target=cv2.resize(target.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        return float(np.count_nonzero(pred&target)/max(1,np.count_nonzero(pred|target)))
    anchor=component['source_anchor']['sample_index'];indices=[r['source_sample_index']for r in component['poses']]
    for sequence in [[i for i in indices if i>=anchor],list(reversed([i for i in indices if i<anchor]))]:
        rotation,depth=initial_rotation,initial_depth
        for index in sequence:
            row=rows[index];original=oldrows[index]
            if original['status']!='estimated' or index in HELDOUT:
                if index in HELDOUT:row.update(status='unknown',camera_from_native_mesh=None,reason='predeclared_phone_mask_holdout_not_fitted')
                continue
            mask=mask_at(index);fit=S.estimate_pose(vertices,initial_rotation,scale,k,mask,depth,previous_rotation=rotation);options=[fit]
            if fit['convex_silhouette_iou']<.70 or not .4*initial_depth<=fit['depth']<=2.5*initial_depth:
                for hypothesis in pca:
                    r=initial_rotation@np.asarray(hypothesis['rotation']);candidate=S.estimate_pose(vertices,r,scale,k,mask,initial_depth,previous_rotation=None,search=False);options.append(candidate)
            options=[f for f in options if .4*initial_depth<=f['depth']<=2.5*initial_depth and f['convex_silhouette_iou']>=.35]
            for f in options:f['regularized_objective']=1-f['convex_silhouette_iou']+.02*Rotation.from_matrix(f['rotation']@rotation.T).magnitude()+.04*abs(math.log(f['depth']/initial_depth))
            options=sorted(options,key=lambda f:f['regularized_objective'])[:3];baseline_iou=raw_iou(mask,old_mesh,original['camera_from_native_mesh']);scored=[(raw_iou(mask,mesh,f['matrix']),f)for f in options]
            best=max(scored,key=lambda p:p[0]-.005*Rotation.from_matrix(p[1]['rotation']@rotation.T).magnitude())if scored else None
            if best is not None and best[0]>=.35 and best[0]>=baseline_iou-.05:
                value,fit=best;row.update(status='estimated',camera_from_native_mesh=fit['matrix'].tolist(),method='declared_planar_subset_automatic_mask_refit',reason='automatic_source_mask_fit_of_declared_triangle_subset',convex_silhouette_fit_iou=fit['convex_silhouette_iou'],full_triangle_raw_mask_iou=value,accepted_motion_evidence=False);rotation,depth=fit['rotation'],fit['depth']
                retained=True
            else:
                # Geometry changed: blindly retaining an old pose could turn an
                # old mask fit into an unsupported display of the new mesh.
                row.update(status='unknown',camera_from_native_mesh=None,reason='derived_phone_geometry_fit_failed_no_pose_fallback');value=None;retained=False
            diagnostics.append({'sample_index':index,'role':'fitting','baseline_raw_full_triangle_iou':baseline_iou,'candidate_raw_full_triangle_iou':value,'retained':retained})
    predictions=[]
    for index in HELDOUT:
        row=rows[index];left=[r for i,r in rows.items()if i<index and r['status']=='estimated'];right=[r for i,r in rows.items()if i>index and r['status']=='estimated'];C.require(left and right,'Heldout prediction needs both neighbors')
        a,b=left[-1],right[0];gap=b['timestamp_s']-a['timestamp_s'];C.require(0<gap<=1.6,'Bounded heldout temporal gap required');matrix=interpolate(a['camera_from_native_mesh'],b['camera_from_native_mesh'],(row['timestamp_s']-a['timestamp_s'])/gap)
        predictions.append({'sample_index':index,'matrix':matrix.tolist(),'source_neighbor_indices':[a['source_sample_index'],b['source_sample_index']],'gap_s':gap})
    C.write_new(folder/'heldout-predictions-before-target-read.json',predictions)
    for prediction in predictions:
        index=prediction['sample_index'];mask=mask_at(index,evaluation=True);matrix=prediction['matrix'];before=raw_iou(mask,old_mesh,oldrows[index]['camera_from_native_mesh']);after=raw_iou(mask,mesh,matrix);passed=after>=.65 and after>=before-.05
        diagnostics.append({'sample_index':index,'role':'heldout_mask_prediction','baseline_raw_full_triangle_iou':before,'candidate_raw_full_triangle_iou':after,'retained':passed,'source_neighbor_indices':prediction['source_neighbor_indices']})
        if passed:rows[index].update(status='estimated',camera_from_native_mesh=matrix,method='bounded_heldout_neighbor_prediction_no_target_refit',reason='neighbor_prediction_passed_predeclared_heldout_mask_gate',accepted_motion_evidence=False)
    held=[r for r in diagnostics if r['role']=='heldout_mask_prediction'];eligible=all(r['retained']for r in held)
    scene['research_derivation']={'kind':'declared_planar_compact_source_refit','candidate_report':C.ref(CANDIDATE),'selection_plan':C.ref(folder/'selection-plan.json'),'heldout_predictions':C.ref(folder/'heldout-predictions-before-target-read.json'),'physical_accuracy_verified':False}
    scene['inputs']=[*scene['inputs'],C.ref(BASE),C.ref(CANDIDATE),*input_refs];scene['fitter_version']='declared_phone_compact_subset_refit_v1';scene['limitations']+=['Phone is a declared automatic source-tested triangle subset. Open trimming boundaries and uncertain generated appearance remain; metric shape and physical contact unverified.']
    C.write_new(folder/'scene.json',scene);report={'format':'interaction-twin-phone-planar-refit-audit','version':1,'selection_plan':C.ref(folder/'selection-plan.json'),'baseline_scene':C.ref(BASE),'candidate_scene':C.ref(folder/'scene.json'),'candidate_report':C.ref(CANDIDATE),'validation_samples':diagnostics,'heldout_passed':eligible,
        'baseline_supported':sum(r['status']=='estimated'for r in old['poses']),'candidate_supported':sum(r['status']=='estimated'for r in component['poses']),'native_original_unknowns_filled':False,'physical_accuracy_verified':False,'automatic_publication':False,'wall_seconds':time.monotonic()-started}
    C.write_new(folder/'report.json',report);return {'report':C.ref(folder/'report.json'),'eligible_by_heldout_mask_gate':eligible,'supported':report['candidate_supported'],'heldout':held}


if __name__=='__main__':print(json.dumps(build()),flush=True)
