"""Automatic semantic contact-plane gauge proposal; all projections preserved.

One constant positive scale changes the declared adapter geometry and its camera
translation together. No pose is authored and the native original stays intact.
Contact support planes are automatic geometric proxies, not insertion/physics.
"""
import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import charger_tracking_recovery_20261103_v1 as C
import native_assembly_refinement_v1 as N
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mask

ROOT=Path(__file__).resolve().parent
BASE='data/tracking-recoveries/charger-20261103-v1/mask-refit-v1/constrained/scene.json'
BASE_PROGRAM='data/object-programs/auto-36df04c484f3b2c641895f533cc3e020671680e8b4b6d345b7b0617c42b0c40e/object-program.json'
OUTPUT='data/geometry-quality-20261104/charger-contact-gauge-v1'
TRAIN=[46,54,66]
HELDOUT=[48,60]


def scaled_pose(matrix,scale):
    m=np.asarray(matrix,float).copy();m[:3,3]*=scale;return m


def build():
    folder=ROOT/OUTPUT;folder.mkdir(parents=True,exist_ok=False);scene=C.read(ROOT/BASE);old=copy.deepcopy(scene);parts={p['id']:p for p in scene['components']};parent=parts['wall_outlet'];child=parts['adapter_late'];pm=load_native_glb(ROOT/parent['asset']['path']);cm=load_native_glb(ROOT/child['asset']['path']);pv=pm.vertices[np.unique(pm.faces)];cv=cm.vertices[np.unique(cm.faces)]
    plan={'format':'interaction-twin-source-contact-gauge-plan','version':1,'baseline_scene_ref':C.ref(BASE),'base_object_program_ref':C.ref(BASE_PROGRAM),
        'training_source_samples':TRAIN,'heldout_source_samples':HELDOUT,'constraint':'Automatic outlet thin-axis front90percent support plane and adapter back1percent support plane; robust constant camera-origin gauge. Semantic at_outlet from actual earlier Muse source46/54/66.',
        'source_localization_result':C.ref('data/tracking-recoveries/charger-20261103-v1/localization-attempt/result.json'),'minimum_scale':.8,'maximum_scale':1.6,'maximum_training_scale_relative_spread':.08,
        'all_source_projections_preserved':True,'native_uniform_scale_unchanged':True,'manual_pose_or_geometry_coordinates':False,'physical_contact_verified':False}
    C.write_new(folder/'plan.json',plan);_,axes=np.linalg.eigh(np.cov(pv.T));native_normal=axes[:,0];rows=[]
    for i in TRAIN:
        p=np.asarray(next(r['camera_from_native_mesh']for r in parent['poses']if r['source_sample_index']==i));c=np.asarray(next(r['camera_from_native_mesh']for r in child['poses']if r['source_sample_index']==i));wp=pv@p[:3,:3].T+p[:3,3];wc=cv@c[:3,:3].T+c[:3,3];normal=p[:3,:3]@native_normal;normal/=np.linalg.norm(normal)
        if normal@(-wp.mean(0))<0:normal=-normal
        front=float(np.quantile(wp@normal,.90));back=float(np.quantile(wc@normal,.01));C.require(abs(back)>1e-8 and front/back>0,'Finite positive contact gauge required');rows.append({'sample_index':i,'normal_camera':normal.tolist(),'parent_front_plane_offset':front,'child_back_plane_offset_before':back,'proposed_scale':front/back,'parent_extent':float(np.linalg.norm(np.ptp(wp,axis=0)))})
    values=[r['proposed_scale']for r in rows];scale=float(np.median(values));spread=(max(values)-min(values))/scale;C.require(.8<=scale<=1.6 and spread<=.08,'Source contact gauges disagree or exceed bound')
    raw=(ROOT/child['asset']['path']).read_bytes();derived,encoding=N.rewrite_positions(raw,cm.vertices*scale);path=folder/'adapter-derived.glb';path.write_bytes(derived);new_mesh=load_native_glb(derived)
    original_asset=child['asset'];child['asset']=C.ref(path)
    for row in child['poses']:
        if row['status']=='estimated':row['camera_from_native_mesh']=scaled_pose(row['camera_from_native_mesh'],scale).tolist();row['method']='declared_semantic_contact_constant_camera_gauge';row['reason']='source_projection_preserving_gauge_from_automatic_outlet_contact_support';row['accepted_motion_evidence']=False
    for row in rows:
        row['back_plane_offset_after']=row['child_back_plane_offset_before']*scale;row['absolute_plane_residual_before']=abs(row['child_back_plane_offset_before']-row['parent_front_plane_offset']);row['absolute_plane_residual_after']=abs(row['back_plane_offset_after']-row['parent_front_plane_offset'])
    # Geometry operation is a declared uniform affine transform; camera stays
    # unchanged and every known source estimate retains exactly the same rays.
    matrix=np.eye(4);matrix[:3,:3]*=scale;assets=[]
    base=C.read(ROOT/BASE_PROGRAM);originals={p['component_id']:p for p in base['component_assets']}
    for part in scene['components']:
        changed=part['id']=='adapter_late';assets.append({'component_id':part['id'],'original_asset':originals[part['id']]['native_glb'],'derived_asset':part['asset'],
            'geometry_origin':'automatic_affine_calibration'if changed else'unchanged_SAM_3D_inference','operations':[{'kind':'affine_positions','matrix':matrix.tolist()}]if changed else[]})
    report={'format':'interaction-twin-geometry-derivative-report','version':1,'base_object_program_ref':C.ref(BASE_PROGRAM),'baseline_scene_ref':C.ref(BASE),'source_sha256':scene['source']['sha256'],
        'assets':assets,'constant_camera_gauge':scale,'training_gauge_relative_spread':spread,'contact_plane_proxies':rows,'encoding_proof':encoding,
        'automatic_geometry_constraint':'semantic at_outlet source evidence and geometric support-plane proxies; not measured plug insertion','manual_geometry_or_poses_used':False,'physical_accuracy_verified':False}
    C.write_new(folder/'geometry-report.json',report);validation=[];h0,w0=scene['source']['height'],scene['source']['width'];h=288;w=round(w0*h/h0);k=np.asarray(scene['camera']['K']).copy();k[0]*=w/w0;k[1]*=h/h0;k[0,2]+=(w/w0-1)/2;k[1,2]+=(h/h0-1)/2;oldchild=next(p for p in old['components']if p['id']=='adapter_late')
    for i in HELDOUT:
        before=next(r for r in oldchild['poses']if r['source_sample_index']==i);after=next(r for r in child['poses']if r['source_sample_index']==i);mask=np.asarray(Image.open(ROOT/before['source_mask']['path']))>0;mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0;metrics=[];renders=[]
        for mesh,row in[(cm,before),(new_mesh,after)]:
            r,t,s=M.rotation_pose(row['camera_from_native_mesh']);render=render_native_mask(mesh,k,r,t,w,h,scale=s)['mask'];renders.append(render);metrics.append(float(np.count_nonzero(render&mask)/max(1,np.count_nonzero(render|mask))))
        validation.append({'sample_index':i,'role':'heldout','before_full_triangle_union_iou':metrics[0],'after_full_triangle_union_iou':metrics[1],
            'changed_raster_pixels':int(np.count_nonzero(renders[0]^renders[1])),'target_mask':before['source_mask'],'physical_accuracy_verified':False})
    passed=all(r['after_full_triangle_union_iou']>=.65 and r['after_full_triangle_union_iou']>=r['before_full_triangle_union_iou']-.05 for r in validation)
    scene['research_derivation']={'kind':'declared_projection_preserving_contact_gauge','geometry_report_ref':C.ref(folder/'geometry-report.json'),'plan':C.ref(folder/'plan.json'),'contact_verified':False}
    C.write_new(folder/'scene.json',scene);audit={'format':'interaction-twin-geometry-source-audit','version':1,'baseline_scene_ref':C.ref(BASE),'geometry_report_ref':C.ref(folder/'geometry-report.json'),'source_sha256':scene['source']['sha256'],
        'summary':{'default_eligible':passed,'geometric_gates_passed':True,'source_pixel_gates_passed':passed,'physical_accuracy_verified':False},'validation_samples':validation,
        'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,'camera_gauge_projection_identity_proof':True,'contact_scope':'Support-plane residual reduced, actual surfaces/prongs/seat unknown',
        'median_plane_residual_before':float(np.median([r['absolute_plane_residual_before']for r in rows])),'median_plane_residual_after':float(np.median([r['absolute_plane_residual_after']for r in rows]))}
    C.write_new(folder/'source-audit.json',audit);return {'scene':C.ref(folder/'scene.json'),'report':C.ref(folder/'geometry-report.json'),'audit':C.ref(folder/'source-audit.json'),'scale':scale,'validation':validation,'eligible':passed}


if __name__=='__main__':print(json.dumps(build()),flush=True)
