"""One bounded private development iteration: real prism ray hits + two RGB refs.

No target SAM mask enters tracking or reference selection. The four targets have
already been tested and seen, so this is explicitly NOT a fresh heldout test.
The earlier1/4 result, all scenes, the v1 viewer and its server remain untouched.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=G.OUTPUT/'ray-tracking-development-v2'


def model_hits(source,matrix,k,planes):
    rotation=matrix[:3,:3];translation=matrix[:3,3]
    origin=-rotation.T@translation
    directions=(np.column_stack([source,np.ones(len(source))])@np.linalg.inv(k).T)@rotation
    numerator=-(origin@planes[:,:3].T+planes[:,3])
    denominator=directions@planes[:,:3].T
    with np.errstate(divide='ignore',invalid='ignore'):
        t=numerator/denominator
    lower=np.max(np.where(denominator < -1e-10,t,-np.inf),axis=1)
    upper=np.min(np.where(denominator > 1e-10,t,np.inf),axis=1)
    parallel_outside=np.any((np.abs(denominator)<1e-10)&(numerator[None,:]<0),axis=1)
    good=(lower>0)&(upper>=lower-1e-7)&np.isfinite(lower)&~parallel_outside
    return origin+directions*lower[:,None],good


def track(reference,target,mask,matrix,k,planes):
    a=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    inner=cv2.erode(mask.astype(np.uint8)*255,np.ones((3,3),np.uint8))
    features=cv2.goodFeaturesToTrack(a,maxCorners=400,qualityLevel=.004,minDistance=3,mask=inner,blockSize=3)
    source=[];destination=[];methods=[]
    lkcount=0
    if features is not None and len(features):
        nextpoints,valid,_=cv2.calcOpticalFlowPyrLK(a,b,features,None,winSize=(25,25),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        back,validback,_=cv2.calcOpticalFlowPyrLK(b,a,nextpoints,None,winSize=(25,25),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        error=np.linalg.norm((back-features).reshape(-1,2),axis=1)
        keep=(valid.ravel()>0)&(validback.ravel()>0)&(error<1.5)
        source.extend(features.reshape(-1,2)[keep]);destination.extend(nextpoints.reshape(-1,2)[keep])
        methods.extend(['LK_forward_backward']*int(keep.sum()));lkcount=int(keep.sum())
    # Descriptor matching adds distinct actual image features, not more copies
    # of a small LK consensus. Target features use RGB only, with no SAM filter.
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.01)
    ka,da=sift.detectAndCompute(a,inner);kb,db=sift.detectAndCompute(b,None)
    siftcount=0
    if da is not None and db is not None and len(db)>=2:
        for m,n in cv2.BFMatcher().knnMatch(da,db,k=2):
            if m.distance>=.7*n.distance:continue
            first=np.asarray(ka[m.queryIdx].pt);second=np.asarray(kb[m.trainIdx].pt)
            if source and np.min(np.linalg.norm(np.asarray(source)-first,axis=1))<3:continue
            source.append(first);destination.append(second);methods.append('SIFT_ratio_0.7');siftcount+=1
    summary={'LK_consistent_features':lkcount,'distinct_SIFT_matches':siftcount,
        'source_RGB_used':True,'target_RGB_used':True,'target_SAM_mask_used':False}
    if len(source)<12:return {'status':'unknown','reason':'fewer_than12_distinct_RGB_tracks',**summary}
    source=np.asarray(source);destination=np.asarray(destination)
    modelpoints,hits=model_hits(source,matrix,k,planes)
    source=source[hits];destination=destination[hits];modelpoints=modelpoints[hits]
    if len(source)<12:return {'status':'unknown','reason':'fewer_than12_visible_full_prism_ray_hits','ray_hits':len(source),**summary}
    coverage=cv2.contourArea(cv2.convexHull(source.astype(np.float32)))/max(1,int(mask.sum()))
    if coverage<.15:return {'status':'unknown','reason':'insufficient_reference_feature_coverage','coverage':float(coverage),**summary}
    rvec=cv2.Rodrigues(matrix[:3,:3])[0];tvec=matrix[:3,3].reshape(3,1).copy()
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(modelpoints.astype(np.float64),destination.astype(np.float64),
        k,None,rvec,tvec,useExtrinsicGuess=True,iterationsCount=500,reprojectionError=2.5,
        confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers)<12:
        return {'status':'unknown','reason':'fewer_than12_full_prism_PnP_inliers',
            'ray_hits':len(source),'PnP_inliers':len(inliers)if inliers is not None else 0,**summary}
    ids=inliers.ravel();r=cv2.Rodrigues(rvec)[0]
    projected=cv2.projectPoints(modelpoints[ids],rvec,tvec,k,None)[0].reshape(-1,2)
    error=np.linalg.norm(projected-destination[ids],axis=1)
    visible=(modelpoints[ids]@r.T+tvec.ravel())[:,2]>0
    angle=float(Rotation.from_matrix(r@matrix[:3,:3].T).magnitude()*180/np.pi)
    if not visible.all()or np.median(error)>2.5 or angle>90:
        return {'status':'unknown','reason':'PnP_cheirality_error_or_large_rotation_rejected',
            'PnP_inliers':len(ids),'rotation_step_degrees':angle,**summary}
    result=np.eye(4);result[:3,:3]=r;result[:3,3]=tvec.ravel()
    # This confidence ordering is frozen before any target mask is opened.
    confidence=float(len(ids)*min(coverage,1)/(1+np.median(error)))
    return {'status':'estimated','matrix':result.tolist(),'PnP_inliers':len(ids),
        'reference_mask_coverage':float(coverage),'median_reprojection_error_px':float(np.median(error)),
        'confidence_for_reference_selection':confidence,'rotation_step_degrees':angle,
        'reference_feature_pixels':source[ids].tolist(),'target_feature_pixels':destination[ids].tolist(),
        'model_ray_hit_points':modelpoints[ids].tolist(),'method':'distinct_RGB_features_full_convex_prism_ray_intersections_and_PnP',**summary}


def build():
    OUTPUT.mkdir(exist_ok=False)
    first=G.OUTPUT/'new-disjoint-tests-v1'
    prior=json.loads((first/'attempt/new-observations.json').read_text())
    unit=json.loads((G.OUTPUT/'full-scene-v1/unit-gauge-poses.json').read_text())
    base=json.loads(E.BASE.read_text());k=np.asarray(base['camera']['K'])
    oldpart=next(c for c in base['components']if c['id']=='phone_blue')
    mesh=load_native_glb(G.OUTPUT/'asset.glb');original=load_native_glb(ROOT/oldpart['asset']['path'])
    planes=np.unique(np.round(ConvexHull(mesh.vertices).equations,10),axis=0)
    plan={'format':'interaction-twin-private-ray-prism-RGB-development-plan','version':2,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'candidate':P.ref(G.OUTPUT/'asset.glb'),'preserved_first_pass_result':P.ref(first/'report.json'),
        'full_scene_unit_poses':P.ref(G.OUTPUT/'full-scene-v1/unit-gauge-poses.json'),
        'test_observation_metadata':P.ref(first/'attempt/new-observations.json'),
        'maximum_target_frames':4,'maximum_references_per_target':2,'one_bounded_iteration':True,
        'reference_policy':'earlier_and_later_existing_source_RGB;choose_by_PnP_confidence_before_target_mask_read',
        'geometry_correspondences':'nearest_positive_ray_hit_on_actual_generated_convex_prism;front_back_and_side_planes',
        'minimum_distinct_PnP_inliers':12,'minimum_reference_feature_coverage':.15,
        'maximum_forward_backward_error_px':1.5,'SIFT_ratio_threshold':.7,
        'maximum_reprojection_error_px':2.5,'maximum_rotation_step_degrees':90,
        'unchanged_validation_gates':{'minimum_IoU':.65,'maximum_baseline_regression':.05},
        'target_RGB_used_for_tracking':True,'target_SAM_masks_not_used_for_fitting_or_reference_selection':True,
        'targets_previously_seen_and_tested':True,'fresh_heldout_or_independent_test':False,
        'geometry_not_changed':True,'automatic_publication':False,'original_viewer_or_server_modified':False,
        'new_API_GPU_or_segmentation_inference_calls':0,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    predictions=[]
    for item in prior:
        rgb=np.asarray(Image.open(ROOT/item['target_RGB']['path']).convert('RGB'))
        options=[]
        for index in [item['reference_source_index'],item['right_source_index']]:
            if str(index)not in unit:
                options.append({'reference_source_index':index,'status':'unknown','reason':'unknown_reference_pose'});continue
            reference,mask,sref=G.frame(index)
            result=track(reference,rgb,mask,np.asarray(unit[str(index)]),k,planes)
            options.append({'reference_source_index':index,'reference':sref,**result})
        known=[r for r in options if r['status']=='estimated']
        selected=max(known,key=lambda r:r['confidence_for_reference_selection'])if known else None
        predictions.append({'test_id':item['test_id'],'timestamp_s':item['timestamp_s'],'target_RGB':item['target_RGB'],
            'reference_options':options,'selected_prediction':selected,'selection_used_target_mask':False})
    P.save(OUTPUT/'predictions-before-target-mask-evaluation.json',predictions)
    sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    scores=[];sheet=Image.new('RGB',(1080,1440),'#f7f8fb');draw=ImageDraw.Draw(sheet)
    oldreport=json.loads((first/'report.json').read_text());oldresults={r['test_id']:r for r in oldreport['results']}
    for rownumber,(item,pred)in enumerate(zip(prior,predictions)):
        target=cv2.resize((np.asarray(Image.open(ROOT/item['target_mask']['path']))>0).astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        selected=pred['selected_prediction'];value=None;render=None
        if selected is not None:
            m=np.asarray(selected['matrix']);render=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],216,288)
            value=float((render['mask']&target).sum()/max(1,(render['mask']|target).sum()))
        old=oldresults[item['test_id']];baseline=old['baseline_full_triangle_IoU']
        passed=value is not None and value>=.65 and value>=baseline-.05
        scores.append({'test_id':item['test_id'],'timestamp_s':item['timestamp_s'],
            'baseline_full_triangle_IoU':baseline,'first_pass_candidate_IoU':old['candidate_full_triangle_IoU'],
            'ray_prism_candidate_IoU':value,'selected_reference_source_index':selected['reference_source_index']if selected else None,
            'PnP_inliers':selected['PnP_inliers']if selected else None,'passes_unchanged_mask_gate':passed,
            'target_mask_used_for_fitting':False,'evaluation_scope':'development_after_first_test_not_fresh_heldout'})
        rgb=cv2.resize(np.asarray(Image.open(ROOT/item['target_RGB']['path']).convert('RGB')),(216,288))
        entries=[('Actual RGB + saved SAM2',rgb,target)]
        if render is not None:entries.append((f'Ray-prism tracking | IoU {value:.3f}',render['rgb'],render['mask']))
        union=np.logical_or.reduce([e[2]for e in entries]);ys,xs=np.nonzero(union)
        crop=[max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(216,int(xs.max())+13),min(288,int(ys.max())+13)]
        for col,(label,pixels,binary)in enumerate(entries):
            if col:pixels=pixels.copy();pixels[~binary]=[238,241,247]
            part=Image.fromarray(pixels).crop(crop);part.thumbnail((340,300));sheet.paste(part,(col*360+(360-part.width)//2,rownumber*360+48))
            draw.text((col*360+8,rownumber*360+8),f'Development target{item["test_id"]}',fill='#121824')
            draw.text((col*360+8,rownumber*360+25),label,fill='#121824')
        if selected is None:draw.text((380,rownumber*360+70),'Unknown: insufficient RGB/PnP confidence',fill='#52617c')
    sheet.save(OUTPUT/'comparison.png')
    report={'format':'interaction-twin-private-ray-prism-RGB-development-report','version':2,
        'selection_plan':P.ref(OUTPUT/'selection-plan.json'),'predictions_before_target_mask_evaluation':P.ref(OUTPUT/'predictions-before-target-mask-evaluation.json'),
        'comparison':P.ref(OUTPUT/'comparison.png'),'results':scores,
        'supported_predictions':sum(r['ray_prism_candidate_IoU']is not None for r in scores),
        'passing_mask_gates':sum(r['passes_unchanged_mask_gate']for r in scores),
        'first_pass_supported_predictions':sum(r['candidate_full_triangle_IoU']is not None for r in oldresults.values()),
        'first_pass_result_preserved':P.ref(first/'report.json'),'fresh_heldout_validation':False,
        'automatic_publication':False,'viewer_or_server_changed':False,'new_API_GPU_or_segmentation_calls':0,
        'limitations':['These four frames were already tested; v2 is a development iteration, not independent new validation.',
            'Full prism ray intersections improve feature geometry but still rely on a simplified2.5D shape and uncertain reference poses.',
            'Reference choice uses only RGB/PnP evidence; target mask IoU is evaluated afterward and cannot choose a better-looking failed hypothesis.',
            'Neither feature consensus nor2D mask overlap certifies thickness, metric pose, grasping, contact or insertion.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'supported_predictions':report['supported_predictions'],
        'passing_mask_gates':report['passing_mask_gates'],'results':scores}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
