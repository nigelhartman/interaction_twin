"""Private RGB-tracked pose test with target SAM masks withheld.

Unlike blind temporal interpolation, this protocol permits the target RGB as a
tracking input. Reference model points come from automatic reference-plane ray
hits and existing private source-prism fits, never authored pose coordinates.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=G.OUTPUT/'RGB-tracking-evaluation-v1'
BASE=ROOT/'data/mechanism-scenes/648d3924ab106192ea1cdad94c6cdf0b2d44d5c61bba3090cd5dbb1fcb00f298/constrained/scene.json'


def track(reference,target,mask,matrix,k,thickness):
    a=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    inner=cv2.erode(mask.astype(np.uint8)*255,np.ones((3,3),np.uint8))
    points=cv2.goodFeaturesToTrack(a,maxCorners=400,qualityLevel=.004,minDistance=3,mask=inner,blockSize=3)
    if points is None or len(points)<12:
        return {'status':'unknown','reason':'fewer_than12_automatic_reference_features'}
    nextpoints,valid,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(25,25),maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    back,validback,_=cv2.calcOpticalFlowPyrLK(b,a,nextpoints,None,winSize=(25,25),maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    error=np.linalg.norm((back-points).reshape(-1,2),axis=1)
    keep=(valid.ravel()>0)&(validback.ravel()>0)&(error<1.5)
    source=points.reshape(-1,2)[keep];destination=nextpoints.reshape(-1,2)[keep]
    if len(source)<12:
        return {'status':'unknown','reason':'fewer_than12_forward_backward_consistent_tracks','tracks':len(source)}
    homography,inliers=cv2.findHomography(source,destination,cv2.USAC_MAGSAC,2.5,maxIters=4000,confidence=.999)
    if inliers is None or int(inliers.sum())<12:
        return {'status':'unknown','reason':'fewer_than12_planar_consensus_tracks','tracks':len(source),
            'inliers':int(inliers.sum()) if inliers is not None else 0}
    source=source[inliers.ravel()>0];destination=destination[inliers.ravel()>0]
    rotation=matrix[:3,:3];translation=matrix[:3,3]
    ray_origin=-rotation.T@translation
    ray_direction=(np.column_stack([source,np.ones(len(source))])@np.linalg.inv(k).T)@rotation
    # The nearer broad plane is selected from the fitted rotation, not a manual
    # front/back label. The2.5D flat-surface prior remains explicit.
    plane=-np.sign(rotation[2,2])*thickness/2
    distance=(plane-ray_origin[2])/ray_direction[:,2]
    modelpoints=ray_origin+ray_direction*distance[:,None]
    good=(distance>0)&np.isfinite(modelpoints).all(axis=1)
    modelpoints=modelpoints[good];destination=destination[good];source=source[good]
    if len(source)<12:
        return {'status':'unknown','reason':'insufficient_positive_reference_plane_hits'}
    coverage=cv2.contourArea(cv2.convexHull(source))/max(1,int(mask.sum()))
    if coverage<.15:
        return {'status':'unknown','reason':'feature_tracks_cover_less_than15percent_of_reference_mask','coverage':float(coverage)}
    rvec=cv2.Rodrigues(rotation)[0];tvec=translation.reshape(3,1).copy()
    ok,rvec,tvec,pnpinliers=cv2.solvePnPRansac(modelpoints.astype(np.float64),destination.astype(np.float64),
        k,None,rvec,tvec,useExtrinsicGuess=True,iterationsCount=500,reprojectionError=2.5,
        confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or pnpinliers is None or len(pnpinliers)<12:
        return {'status':'unknown','reason':'insufficient_PnP_consensus','feature_tracks':len(source)}
    ids=pnpinliers.ravel();newr=cv2.Rodrigues(rvec)[0]
    projected=cv2.projectPoints(modelpoints[ids],rvec,tvec,k,None)[0].reshape(-1,2)
    reproj=np.linalg.norm(projected-destination[ids],axis=1)
    fitted=np.eye(4);fitted[:3,:3]=newr;fitted[:3,3]=tvec.ravel()
    return {'status':'estimated','matrix':fitted.tolist(),'automatic_features':len(points),
        'forward_backward_consistent_tracks':int(keep.sum()),'planar_inliers':len(source),'PnP_inliers':len(ids),
        'reference_mask_coverage':float(coverage),'median_reprojection_error_px':float(np.median(reproj)),
        'reference_feature_pixels':source[ids].tolist(),'target_feature_pixels':destination[ids].tolist(),
        'source_RGB_used_as_input':True,'target_SAM_mask_used_for_fitting':False,
        'method':'automatic_ShiTomasi_LK_forward_backward_planar_RANSAC_then_PnP_from_reference_plane_ray_hits'}


def build():
    OUTPUT.mkdir(exist_ok=False)
    base=json.loads(BASE.read_text());k=np.asarray(base['camera']['K'])
    component=next(c for c in base['components']if c['id']=='phone_blue')
    oldposes={r['source_sample_index']:r for r in component['poses']}
    original=load_native_glb(ROOT/component['asset']['path']);mesh=load_native_glb(G.OUTPUT/'asset.glb')
    geom=json.loads((G.OUTPUT/'report.json').read_text())
    training=json.loads((G.OUTPUT/'evaluation-v1/fitted-training-poses.json').read_text())
    poses={r['source']['source_index']:np.asarray(r['matrix'])for r in training}
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text());ev={r['source_index']:r for r in program['evidence']}
    midpath=ROOT/'output/assembly-quality-20261104/phone-midpoint-evaluation-v2/attempt/test-4/observation.json'
    midpoint=json.loads(midpath.read_text())
    plan={'format':'interaction-twin-phone-RGB-tracked-target-mask-withheld-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'candidate':P.ref(G.OUTPUT/'asset.glb'),'geometry':P.ref(G.OUTPUT/'report.json'),
        'reference_training':P.ref(G.OUTPUT/'evaluation-v1/fitted-training-poses.json'),
        'cases':[{'id':'source-5','reference_index':4,'target_index':5},
                 {'id':'source-26','reference_index':25,'target_index':26},
                 {'id':'prior-midpoint-4','reference_index':37,'target_metadata':P.ref(midpath)}],
        'target_RGB_allowed_as_tracking_input':True,'target_mask_not_read_until_all_predictions_saved':True,
        'this_is_not_blind_temporal_interpolation_or_unseen_RGB_test':True,
        'minimum_target_IoU':.65,'maximum_baseline_regression':.05,
        'minimum_RANSAC_PnP_inliers':12,'minimum_reference_mask_feature_coverage':.15,
        'maximum_forward_backward_error_px':1.5,'maximum_RANSAC_reprojection_error_px':2.5,
        'prior_failed_blind_temporal_evaluation_preserved':P.ref(G.OUTPUT/'evaluation-v1/report.json'),
        'automatic_publication':False,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',plan)
    predictions=[]
    for case in plan['cases']:
        reference,mask,sref=G.frame(case['reference_index'])
        if case.get('target_index')is not None:
            e=ev[case['target_index']];tref=P.ref(ROOT/e['frame_path'])
        else:tref=midpoint['rgb']
        target=np.asarray(Image.open(ROOT/tref['path']).convert('RGB'))
        result=track(reference,target,mask,poses[case['reference_index']],k,
            geom['estimated_edge_thickness_over_length'])
        predictions.append({**case,'reference':sref,'target_RGB':tref,**result})
    P.save(OUTPUT/'predictions-before-target-mask-reads.json',predictions)
    h=288;w=216;sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    scores=[];sheet=Image.new('RGB',(1080,1080),'#f7f8fb');draw=ImageDraw.Draw(sheet)
    for number,row in enumerate(predictions):
        target=np.asarray(Image.open(ROOT/row['target_RGB']['path']).convert('RGB'))
        if row.get('target_index')is not None:
            obs=next(o for o in ev[row['target_index']]['observations']if o['track_id']=='phone_blue')
            targetmask=P.ref(ROOT/obs['mask_path']);beforematrix=np.asarray(oldposes[row['target_index']]['camera_from_native_mesh'])
        else:targetmask=midpoint['mask'];beforematrix=np.asarray(midpoint['baseline_matrix'])
        mask=cv2.resize((np.asarray(Image.open(ROOT/targetmask['path']))>0).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        scale=float(np.cbrt(np.linalg.det(beforematrix[:3,:3])))
        before=render_native_mesh(original,sk,beforematrix[:3,:3]/scale,beforematrix[:3,3],w,h,scale=scale)
        def iou(binary):return float((binary&mask).sum()/max(1,(binary|mask).sum()))
        beforeiou=iou(before['mask']);candidateiou=None;render=None
        if row['status']=='estimated':
            m=np.asarray(row['matrix']);render=render_native_mesh(mesh,sk,m[:3,:3],m[:3,3],w,h)
            candidateiou=iou(render['mask'])
        passed=candidateiou is not None and candidateiou>=.65 and candidateiou>=beforeiou-.05
        scores.append({'id':row['id'],'status':row['status'],'target_mask':targetmask,
            'baseline_full_triangle_IoU':beforeiou,'candidate_full_triangle_IoU':candidateiou,
            'passes_unrelaxed_mask_gate':passed,'target_mask_fitted':False})
        items=[('Actual RGB + withheld SAM',cv2.resize(target,(w,h)),mask),
               (f'Original SAM3D | IoU {beforeiou:.3f}',before['rgb'],before['mask'])]
        if render is not None:items.append((f'Source prism RGB-PnP | IoU {candidateiou:.3f}',render['rgb'],render['mask']))
        union=np.logical_or.reduce([a[2]for a in items]);ys,xs=np.nonzero(union)
        crop=[max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(w,int(xs.max())+13),min(h,int(ys.max())+13)]
        for col,(label,pixels,binary)in enumerate(items):
            if col:pixels=pixels.copy();pixels[~binary]=[238,241,247]
            part=Image.fromarray(pixels).crop(crop);part.thumbnail((340,300))
            sheet.paste(part,(col*360+(360-part.width)//2,number*360+48))
            draw.text((col*360+8,number*360+8),row['id'],fill='#121824');draw.text((col*360+8,number*360+25),label,fill='#121824')
    sheet.save(OUTPUT/'comparison.png')
    report={'format':'interaction-twin-phone-RGB-tracked-mask-withheld-evaluation','version':1,
        'selection_plan':P.ref(OUTPUT/'selection-plan.json'),'predictions_frozen_before_target_mask_reads':P.ref(OUTPUT/'predictions-before-target-mask-reads.json'),
        'comparison':P.ref(OUTPUT/'comparison.png'),'test_results':scores,
        'all_unrelaxed_mask_gates_pass':all(r['passes_unrelaxed_mask_gate']for r in scores),
        'metric_geometry_or_pose_accuracy_verified':False,'physical_fit_contact_or_insertion_verified':False,
        'automatic_publication':False,'new_API_or_GPU_calls':0,
        'limitations':['Target RGB is deliberately used for automatic tracking; this is not independent unseen-image pose prediction.',
            'The withheld SAM outputs are fallible model observations, not ground truth.',
            'A mask-gate pass cannot waive earlier failed blind temporal interpolation or validate actual phone thickness.',
            'Feature matches on glossy reflections can violate rigid-scene assumptions; confidence and image coverage gates apply.',
            'The source-derived model remains a simple private2.5D hypothesis, not a repaired SAM3D output.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'all_unrelaxed_mask_gates_pass':report['all_unrelaxed_mask_gates_pass'],'test_results':scores}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
