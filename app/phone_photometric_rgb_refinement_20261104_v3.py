"""Private direct RGB alignment without target-mask fitting.

Dense real reference pixels are lifted by automatic ray/prism intersections.
Robust colour alignment optimizes a rigid camera-relative pose using target RGB
only. Existing four targets remain development cases, never fresh heldout tests.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
import phone_source_prism_ray_tracking_20261104_v2 as R
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=ROOT/'output/submission-polish-20261104-v1/phone-photometric-v3'


def track(reference,target,mask,matrix,k,planes):
    # Deterministic dense grid plus gradient points; no authored features.
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY).astype(float)
    gradient=np.hypot(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3))
    yy,xx=np.nonzero(mask);points=np.column_stack([xx,yy]).astype(float)
    regular=np.flatnonzero((xx%3==0)&(yy%3==0))
    strongest=np.argsort(-gradient[yy,xx],kind='stable')[:1200]
    regular=regular[np.linspace(0,len(regular)-1,min(1200,len(regular))).astype(int)]
    selected=np.unique(np.r_[regular,strongest]);points=points[selected]
    modelpoints,hits=R.model_hits(points,matrix,k,planes)
    points=points[hits];modelpoints=modelpoints[hits]
    if len(points)<200:return {'status':'unknown','reason':'fewer_than200_actual_reference_surface_samples','samples':len(points)}
    locations=np.rint(points).astype(int)
    colors=reference[locations[:,1],locations[:,0]].astype(float)/255
    weights=.5+np.clip(gradient[locations[:,1],locations[:,0]]/80,0,2)
    targetfloat=target.astype(np.float32)/255
    r0=matrix[:3,:3];t0=matrix[:3,3];z0=t0[2]
    def transform(p):
        r=Rotation.from_rotvec(p[:3]).as_matrix()@r0
        t=t0+np.array([p[3]*z0,p[4]*z0,z0*(np.exp(p[5])-1)])
        return r,t
    def residual(p):
        r,t=transform(p);cam=modelpoints@r.T+t;uvw=cam@k.T;uv=uvw[:,:2]/uvw[:,2:]
        pix=cv2.remap(targetfloat,uv[:,0].astype(np.float32).reshape(-1,1),uv[:,1].astype(np.float32).reshape(-1,1),
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(1,1,1)).reshape(-1,3)
        # Bounded exposure adjustment cannot explain arbitrary new object colour.
        difference=pix-colors;offset=np.clip(np.median(difference,axis=0),-.06,.06)
        error=(difference-offset)*np.sqrt(weights[:,None])
        return error.ravel()
    step=np.array([.001,.001,.001,.0005,.0005,.001])
    def jacobian(p):
        result=[]
        for i in range(6):
            a=p.copy();b=p.copy();a[i]+=step[i];b[i]-=step[i]
            result.append((residual(a)-residual(b))/(2*step[i]))
        return np.column_stack(result)
    initial=np.zeros(6);initialerror=float(np.mean(np.abs(residual(initial))))
    solved=least_squares(residual,initial,jac=jacobian,bounds=([-1.,-1.,-1.,-.22,-.22,-.35],[1.,1.,1.,.22,.22,.35]),
        loss='soft_l1',f_scale=.04,max_nfev=100,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    r,t=transform(solved.x);m=np.eye(4);m[:3,:3]=r;m[:3,3]=t
    final=float(np.mean(np.abs(residual(solved.x))))
    angular=float(Rotation.from_matrix(r@r0.T).magnitude()*180/np.pi)
    cam=modelpoints@r.T+t;uvw=cam@k.T;uv=uvw[:,:2]/uvw[:,2:]
    onscreen=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<target.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<target.shape[0]-1)
    accepted=final<=.15 and onscreen.mean()>=.9 and angular<=90
    return {'status':'estimated'if accepted else'unknown','matrix':m.tolist()if accepted else None,
        'actual_reference_surface_samples':len(points),'initial_robust_RGB_MAE':initialerror,
        'final_robust_RGB_MAE':final,'on_screen_surface_fraction':float(onscreen.mean()),
        'rotation_update_degrees':angular,'function_evaluations':solved.nfev,
        'target_mask_used':False,'method':'direct_real_RGB_surface_sample_alignment_with_automatic_ray_prism_hits',
        'confidence_for_reference_selection':float(onscreen.mean()/(.001+final))if accepted else None}


def build():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    olddir=G.OUTPUT/'new-disjoint-tests-v1'
    observations=json.loads((olddir/'attempt/new-observations.json').read_text())
    unit=json.loads((G.OUTPUT/'full-scene-v1/unit-gauge-poses.json').read_text())
    scene=json.loads(E.BASE.read_text());k=np.asarray(scene['camera']['K'])
    mesh=load_native_glb(G.OUTPUT/'asset.glb');planes=np.unique(np.round(ConvexHull(mesh.vertices).equations,10),axis=0)
    plan={'format':'interaction-twin-private-dense-photometric-RGB-development-plan','version':3,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'candidate':P.ref(G.OUTPUT/'asset.glb'),'source_reference_poses':P.ref(G.OUTPUT/'full-scene-v1/unit-gauge-poses.json'),
        'existing_target_metadata':P.ref(olddir/'attempt/new-observations.json'),
        'maximum_target_frames':4,'maximum_references_each':2,'maximum_optimization_evaluations_each':100,
        'target_RGB_is_tracking_input':True,'target_masks_not_used_for_fitting_or_reference_selection':True,
        'previously_seen_development_frames_not_fresh_heldout':True,
        'confidence_gates':{'maximum_final_RGB_MAE':.15,'minimum_onscreen_surface_fraction':.9,'maximum_rotation_step_degrees':90},
        'unchanged_mask_validation_gates':{'minimum_IoU':.65,'maximum_baseline_regression':.05},
        'new_API_or_GPU_calls':0,'geometry_or_publication_changed':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    predictions=[]
    for item in observations:
        target=np.asarray(Image.open(ROOT/item['target_RGB']['path']).convert('RGB'));options=[]
        for index in [item['reference_source_index'],item['right_source_index']]:
            if str(index)not in unit:continue
            reference,mask,sref=G.frame(index)
            result=track(reference,target,mask,np.asarray(unit[str(index)]),k,planes)
            options.append({'source_reference_index':index,'source':sref,**result})
        known=[o for o in options if o['status']=='estimated']
        selected=max(known,key=lambda o:o['confidence_for_reference_selection'])if known else None
        predictions.append({'test_id':item['test_id'],'timestamp_s':item['timestamp_s'],'target_RGB':item['target_RGB'],
            'options':options,'selected_prediction':selected,'target_mask_used':False})
        print('Dense real RGB development prediction frozen',item['test_id'],selected['source_reference_index']if selected else'unknown',flush=True)
    P.save(OUTPUT/'predictions-before-target-mask-evaluation.json',predictions)
    oldreport=json.loads((olddir/'report.json').read_text());old={r['test_id']:r for r in oldreport['results']}
    sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    results=[]
    for item,pred in zip(observations,predictions):
        target=cv2.resize((np.asarray(Image.open(ROOT/item['target_mask']['path']))>0).astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        selected=pred['selected_prediction'];value=None
        if selected is not None:
            m=np.asarray(selected['matrix']);render=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],216,288)
            value=float((render['mask']&target).sum()/max(1,(render['mask']|target).sum()))
        baseline=old[item['test_id']]['baseline_full_triangle_IoU']
        results.append({'test_id':item['test_id'],'baseline_IoU':baseline,'first_pass_IoU':old[item['test_id']]['candidate_full_triangle_IoU'],
            'dense_RGB_IoU':value,'passes_unrelaxed_mask_gate':value is not None and value>=.65 and value>=baseline-.05,
            'selected_reference_index':selected['source_reference_index']if selected else None,'target_mask_fitted':False})
    report={'format':'interaction-twin-private-dense-RGB-development-report','version':3,
        'plan':P.ref(OUTPUT/'selection-plan.json'),'predictions_before_target_mask_evaluation':P.ref(OUTPUT/'predictions-before-target-mask-evaluation.json'),
        'results':results,'passing_mask_gates':sum(r['passes_unrelaxed_mask_gate']for r in results),
        'fresh_heldout_validation':False,'geometry_publication_or_viewer_changed':False,'new_API_or_GPU_calls':0,
        'limitations':['These are previously tested development targets; adaptation results are not independent new validation.',
            'Dense photometric samples are correlated and reflections can violate rigidity. Their count is not independent feature evidence.',
            'Only target RGB enters fitting; saved SAM target masks are opened afterward for unchanged validation gates.',
            'The2.5D shape, depth gauge, camera and connector interface remain physically unverified.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'passing_mask_gates':report['passing_mask_gates'],'results':results}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
