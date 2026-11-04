"""Private fresh-pose and withheld-frame evaluation of the source prism.

Target masks at source5/26 and the previously failed local midpoint4 never enter
this fitter. All are previously viewed development data, not unseen benchmark
ground truth. No relaxed threshold, copied old pose, or publication occurs.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import sam3d_common_scene as S
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=G.OUTPUT/'evaluation-v1'
BASE=ROOT/'data/mechanism-scenes/648d3924ab106192ea1cdad94c6cdf0b2d44d5c61bba3090cd5dbb1fcb00f298/constrained/scene.json'
TRAIN=[4,6,25,27,37,38]


def interpolate(a,b,f):
    m=np.eye(4)
    m[:3,:3]=Slerp([0,1],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))(f).as_matrix()
    m[:3,3]=a[:3,3]*(1-f)+b[:3,3]*f
    return m


def fit(mesh,k,rgb,mask,previous=None):
    target=S.mask_shape(mask)
    yy,xx=np.nonzero(mask)
    _,vectors=np.linalg.eigh(np.cov(np.column_stack([xx,yy]).T))
    long=vectors[:,-1]
    if long[np.argmax(abs(long))]<0:long=-long
    roll=math.degrees(math.atan2(long[1],long[0])-math.pi/2)
    # Geometry determines canonical axes; image PCA determines in-plane roll.
    # Both front/back and all signed tilts are searched, never authored poses.
    vertices=mesh.vertices[np.linspace(0,len(mesh.vertices)-1,min(2000,len(mesh.vertices))).astype(int)]
    z0=float(k[0,0]/max(12,np.ptp(np.column_stack([xx,yy])@long)))
    options=[]
    for flip in [0,180]:
        for yaw in [-80,-60,-40,-20,0,20,40,60,80]:
            for pitch in [-70,-45,-20,0,20,45,70]:
                r=Rotation.from_euler('z',roll,degrees=True).as_matrix()@Rotation.from_euler('y',yaw+flip,degrees=True).as_matrix()@Rotation.from_euler('x',pitch,degrees=True).as_matrix()
                p=S.projected_pose(vertices,r,1,k,target,z0)
                if p is None:continue
                t,uv,z=p
                _,iou=S.silhouette_score(uv,target)
                options.append({'rotation':r,'translation':t,'depth':z,'convex_iou':iou})
    options.sort(key=lambda x:-x['convex_iou'])
    # Fine local search remains mask-only; no withheld target is read.
    refined=[]
    for item in options[:16]:
        for delta in [8,3,1]:
            choices=[item]
            for axis in np.eye(3):
                for sign in [-1,1]:
                    r=Rotation.from_rotvec(axis*math.radians(delta*sign)).as_matrix()@item['rotation']
                    p=S.projected_pose(vertices,r,1,k,target,item['depth'])
                    if p is not None:
                        t,uv,z=p;_,iou=S.silhouette_score(uv,target)
                        choices.append({'rotation':r,'translation':t,'depth':z,'convex_iou':iou})
            item=max(choices,key=lambda x:x['convex_iou'])
        refined.append(item)
    # Shape and source RGB jointly break symmetry without manual face choices.
    h0,w0=mask.shape;h=288;w=round(w0*h/h0);smallk=k.copy()
    smallk[0]*=w/w0;smallk[1]*=h/h0
    smallk[0,2]+=(w/w0-1)/2;smallk[1,2]+=(h/h0-1)/2
    smallmask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    smallrgb=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    for item in refined:
        rend=render_native_mesh(mesh,smallk,item['rotation'],item['translation'],w,h)
        pred=rend['mask'];inter=pred&smallmask
        item['full_triangle_iou']=float(inter.sum()/max(1,(pred|smallmask).sum()))
        eroded=cv2.erode(inter.astype(np.uint8),np.ones((3,3),np.uint8))>0
        item['observed_RGB_MAE']=float(np.abs(rend['rgb'].astype(float)-smallrgb.astype(float))[eroded].mean()) if eroded.any() else 255.
        continuity=float(Rotation.from_matrix(item['rotation']@previous.T).magnitude()) if previous is not None else 0.
        item['selection_score']=item['full_triangle_iou']-.12*item['observed_RGB_MAE']/255-.01*continuity
    best=max(refined,key=lambda x:x['selection_score'])
    matrix=np.eye(4);matrix[:3,:3]=best['rotation'];matrix[:3,3]=best['translation']
    return {'matrix':matrix.tolist(),'full_triangle_IoU':best['full_triangle_iou'],
        'fitting_RGB_mean_absolute_error':best['observed_RGB_MAE'],
        'convex_search_IoU_diagnostic':best['convex_iou'],'selected_by':'full_mesh_IoU_source_RGB_and_small_continuity_penalty'}


def build():
    started=time.monotonic();OUTPUT.mkdir(exist_ok=False)
    base=json.loads(BASE.read_text());k=np.asarray(base['camera']['K'])
    component=next(c for c in base['components']if c['id']=='phone_blue')
    prior={p['source_sample_index']:p for p in component['poses']}
    asset=G.OUTPUT/'asset.glb';mesh=load_native_glb(asset)
    midpoint=ROOT/'output/assembly-quality-20261104/phone-midpoint-evaluation-v2/attempt/test-4/observation.json'
    # Read metadata only here; target RGB/mask files are delayed until predictions
    # are durably written. The old test result remains failed and preserved.
    meta=json.loads(midpoint.read_text())
    plan={'format':'interaction-twin-phone-source-prism-withheld-evaluation-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'asset':P.ref(asset),'baseline':P.ref(BASE),'construction':P.ref(G.OUTPUT/'report.json'),
        'training_source_indices':TRAIN,'withheld_source_indices':[5,26],
        'withheld_midpoint':P.ref(midpoint),'withheld_pixels_used_for_pose_fitting':False,
        'all_targets_already_seen_as_development_data_in_earlier_experiments':True,
        'new_camera_or_physical_calibration':False,'arbitrary_canonical_scale':1.,
        'minimum_target_IoU':.65,'maximum_baseline_regression':.05,
        'full_triangle_rendering':True,'source_occlusion_mask_not_applied_to_render':True,
        'automatic_publication':False,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    poses={};training=[];previous=None
    for index in TRAIN:
        rgb,mask,source=G.frame(index)
        if index in [4,25,37]:previous=None
        result=fit(mesh,k,rgb,mask,previous)
        previous=np.asarray(result['matrix'])[:3,:3]
        poses[index]=np.asarray(result['matrix'])
        training.append({'source':source,**result})
        print('Private source-prism fitted',index,round(result['full_triangle_IoU'],4),flush=True)
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text())
    ev={e['source_index']:e for e in program['evidence']}
    predictions=[]
    for index,left,right in [(5,4,6),(26,25,27)]:
        fraction=(ev[index]['timestamp_s']-ev[left]['timestamp_s'])/(ev[right]['timestamp_s']-ev[left]['timestamp_s'])
        predictions.append({'id':f'source-{index}','sample_index':index,'timestamp_s':ev[index]['timestamp_s'],
            'neighbors':[left,right],'matrix':interpolate(poses[left],poses[right],fraction).tolist()})
    fraction=(meta['timestamp_s']-ev[37]['timestamp_s'])/(ev[38]['timestamp_s']-ev[37]['timestamp_s'])
    assert 0<fraction<1
    predictions.append({'id':'prior-midpoint-4','timestamp_s':meta['timestamp_s'],'neighbors':[37,38],
        'matrix':interpolate(poses[37],poses[38],fraction).tolist()})
    P.save(OUTPUT/'fitted-training-poses.json',training)
    P.save(OUTPUT/'predictions-before-target-pixels.json',predictions)
    original=load_native_glb(ROOT/component['asset']['path'])
    h0,w0=base['source']['height'],base['source']['width'];h=288;w=round(w0*h/h0)
    sk=k.copy();sk[0]*=w/w0;sk[1]*=h/h0;sk[0,2]+=(w/w0-1)/2;sk[1,2]+=(h/h0-1)/2
    scores=[];sheet=Image.new('RGB',(1080,3*360),'#f6f7fb');draw=ImageDraw.Draw(sheet)
    for rownumber,pred in enumerate(predictions):
        if pred.get('sample_index')is not None:
            rgb,mask,source=G.frame(pred['sample_index']);oldmatrix=np.asarray(prior[pred['sample_index']]['camera_from_native_mesh'])
        else:
            rgb=np.asarray(Image.open(ROOT/meta['rgb']['path']).convert('RGB'))
            mask=np.asarray(Image.open(ROOT/meta['mask']['path']))>0
            source={'rgb':meta['rgb'],'mask':meta['mask'],'origin':'prior_local_SAM2_model_observation_not_ground_truth'}
            oldmatrix=np.asarray(meta['baseline_matrix'])
        target=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        m=np.asarray(pred['matrix']);cand=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],w,h)
        scale=float(np.cbrt(np.linalg.det(oldmatrix[:3,:3])))
        before=render_native_mesh(original,sk,oldmatrix[:3,:3]/scale,oldmatrix[:3,3],w,h,scale=scale)
        def iou(binary):return float((binary&target).sum()/max(1,(binary|target).sum()))
        baseline=iou(before['mask']);after=iou(cand['mask'])
        passed=after>=.65 and after>=baseline-.05
        scores.append({**pred,'source':source,'baseline_full_triangle_IoU':baseline,
            'candidate_full_triangle_IoU':after,'passes_unrelaxed_gate':passed})
        items=[('Actual source + SAM observation',cv2.resize(rgb,(w,h)),target),
               (f'Original SAM3D | IoU {baseline:.3f}',before['rgb'],before['mask']),
               (f'Source-derived prism | IoU {after:.3f}',cand['rgb'],cand['mask'])]
        union=np.logical_or.reduce([r[2]for r in items]);ys,xs=np.nonzero(union)
        crop=[max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(w,int(xs.max())+13),min(h,int(ys.max())+13)]
        for col,(label,pixels,binary)in enumerate(items):
            if col:
                pixels=pixels.copy();pixels[~binary]=[238,241,247]
            part=Image.fromarray(pixels).crop(crop);part.thumbnail((340,300))
            sheet.paste(part,(col*360+(360-part.width)//2,rownumber*360+48))
            draw.text((col*360+8,rownumber*360+8),pred['id'],fill='#121824')
            draw.text((col*360+8,rownumber*360+25),label,fill='#121824')
    sheet.save(OUTPUT/'withheld-comparison.png')
    report={'format':'interaction-twin-phone-source-prism-withheld-evaluation','version':1,
        'selection_plan':P.ref(OUTPUT/'selection-plan.json'),'training':P.ref(OUTPUT/'fitted-training-poses.json'),
        'predictions_frozen_before_target_pixel_reads':P.ref(OUTPUT/'predictions-before-target-pixels.json'),
        'comparison':P.ref(OUTPUT/'withheld-comparison.png'),'test_results':scores,
        'all_unrelaxed_gates_pass':all(r['passes_unrelaxed_gate']for r in scores),
        'public_promotion_authorized':False,'pose_accuracy_or_physical_fit_verified':False,
        'new_API_or_GPU_calls':0,'elapsed_seconds':time.monotonic()-started,
        'limitations':['Targets are previously seen development evidence, not a new benchmark.',
            'Passing 2D projection gates does not prove hidden geometry, unique rotations, metric thickness or insertion fit.',
            'No existing failed candidate result is changed or waived. This hypothesis uses a separately labelled source-derived backend.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'all_unrelaxed_gates_pass':report['all_unrelaxed_gates_pass'],
        'test_results':[{k:r[k]for k in ['id','baseline_full_triangle_IoU','candidate_full_triangle_IoU','passes_unrelaxed_gate']}for r in scores]}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
