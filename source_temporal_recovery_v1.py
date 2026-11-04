"""Additive CPU source-conditioned temporal pose hypotheses.

Original scene poses and unknown barriers are never edited. Automatic foreground
features are attached to the existing native surface at a known source pose,
tracked through real video RGB in both time directions, then fitted with PnP.
Hidden rigid descendants may follow an unchanged, independently saved joint only
when both neighbouring saved joint states agree. This is estimated replay, not
physical pose/contact validation or an observation of hidden geometry.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

import sam3d_mechanisms as M
from native_appearance import load_native_glb, render_native_mask

ROOT = Path(__file__).resolve().parent
POLICY = {
    'fps': 6., 'maximum_tracking_dimension': 480,
    'maximum_anchor_gap_s': 4., 'maximum_viewer_interpolation_gap_s': .22,
    'maximum_corners': 450, 'minimum_PnP_inliers': 12,
    'minimum_PnP_inlier_fraction': .45, 'minimum_reference_feature_coverage': .07,
    'maximum_forward_backward_error_px': 1.25, 'maximum_PnP_reprojection_px': 2.,
    'maximum_median_PnP_reprojection_px': 1.1,
    'maximum_bidirectional_rotation_rad': math.radians(22),
    'maximum_bidirectional_center_extent_fraction': .18,
    'maximum_bidirectional_depth_ratio': 1.35,
    'maximum_one_sided_distance_s': 1.0,
    'maximum_constant_joint_state_difference_rad': math.radians(3),
    'maximum_frame_rotation_rad': math.radians(35),
    'minimum_target_mask_iou': .48, 'minimum_target_mask_coverage': .62,
    'minimum_dense_photometric_correlation': .93, 'maximum_dense_warp_cycle_px': 1.25,
    'dense_photometric_fallback_enabled': False,
    'new_API_calls': 0, 'new_GPU_calls': 0,
    'original_unknown_poses_or_barriers_changed': False,
    'physical_accuracy_verified': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path, root=ROOT):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read_checked(reference, root=ROOT):
    path = Path(root) / reference['path']
    require(path.resolve() == path and Path(root) in path.parents and not any(p.is_symlink() for p in [path, *path.parents]), 'Ordinary contained evidence required')
    require(ref(path, root) == reference, 'Temporal source reference changed')
    return json.loads(path.read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def matrix_parts(matrix):
    matrix = np.asarray(matrix, float)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0,0,0,1]), 'Finite affine pose required')
    scale = float(np.cbrt(np.linalg.det(matrix[:3,:3])))
    require(scale > 0 and np.allclose((matrix[:3,:3]/scale).T @ (matrix[:3,:3]/scale), np.eye(3), atol=2e-5), 'Positive uniform-scale native pose required')
    return matrix[:3,:3]/scale, matrix[:3,3], scale


def blend(a, b, alpha):
    ra, ta, sa = matrix_parts(a); rb, tb, sb = matrix_parts(b)
    out = np.eye(4)
    out[:3,:3] = Slerp([0.,1.], Rotation.from_matrix([ra,rb]))([float(alpha)]).as_matrix()[0] * math.exp((1-alpha)*math.log(sa)+alpha*math.log(sb))
    out[:3,3] = (1-alpha)*ta+alpha*tb
    return out


def project(points, matrix, k):
    p = np.asarray(points) @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]
    h = p @ np.asarray(k).T
    return h[:,:2]/h[:,2,None]


@dataclass
class Frame:
    timestamp_s: float
    gray: np.ndarray
    evidence: dict
    original_index: int | None = None


def video_frames(program, evidence, camera, *, root=ROOT, policy=POLICY):
    """Read bounded real codec frames; original source JPEG rows stay exact."""
    root = Path(root); source = program['source']; source_ref = ref(root/source['path'], root)
    require(source_ref['sha256'] == source['sha256'], 'Source video hash changed')
    width, height = int(camera['width']), int(camera['height'])
    factor = min(1., policy['maximum_tracking_dimension']/max(width,height))
    size = (round(width*factor), round(height*factor))
    k = np.asarray(camera['K'], float).copy()
    for axis, scale in enumerate([size[0]/width, size[1]/height]):
        k[axis] *= scale; k[axis,2] += (scale-1)/2
    capture = cv2.VideoCapture(str(root/source['path']))
    require(capture.isOpened(), 'Source video cannot be decoded')
    frames=[]; index=0; next_time=0.; cadence=1/policy['fps']
    try:
        while True:
            ok, bgr = capture.read()
            if not ok: break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            if timestamp+1e-6 >= next_time:
                resized = cv2.resize(bgr,size,interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
                frames.append(Frame(timestamp,gray,{'video_ref':source_ref,'decoded_frame_index':index,
                    'timestamp_s':timestamp,'decoded_gray_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),
                    'tracking_width':size[0],'tracking_height':size[1]}))
                next_time += cadence
            index += 1
            require(index <= 20000, 'Bounded source video decode exceeded')
    finally: capture.release()
    for row in evidence:
        raw_ref=ref(root/row['frame_path'],root)
        require(raw_ref['sha256']==row['frame_sha256'], 'Saved source RGB changed')
        rgb=np.asarray(Image.open(root/row['frame_path']).convert('RGB'))
        gray=cv2.cvtColor(cv2.resize(rgb,size,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
        frames.append(Frame(float(row['timestamp_s']),gray,{'image_ref':raw_ref,'timestamp_s':float(row['timestamp_s']),
            'decoded_gray_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),'tracking_width':size[0],'tracking_height':size[1]},int(row['source_index'])))
    # Original observations win a duplicate timestamp. Their index is retained,
    # but the new prediction is never written back into that original ledger.
    result={round(f.timestamp_s,6):f for f in frames}
    return [result[t] for t in sorted(result)], k, size, source_ref


def seed_surface(frame, component, pose, mesh, k, size, *, root=ROOT, policy=POLICY):
    rotation,translation,scale=matrix_parts(pose['camera_from_native_mesh'])
    rendered=render_native_mask(mesh,k,rotation,translation,*size,scale=scale)
    foreground=rendered['mask'].copy()
    reference=pose.get('source_mask')
    if reference:
        require(ref(Path(root)/reference['path'],root)==reference,'Reference source mask changed')
        mask=np.asarray(Image.open(Path(root)/reference['path']))>0
        foreground &= cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    foreground=cv2.erode(foreground.astype(np.uint8),np.ones((5,5),np.uint8))>0
    corners=cv2.goodFeaturesToTrack(frame.gray,maxCorners=policy['maximum_corners'],qualityLevel=.003,
        minDistance=3,mask=foreground.astype(np.uint8)*255,blockSize=3)
    if corners is None or len(corners)<policy['minimum_PnP_inliers']:
        return {'status':'withheld','reason':'insufficient_foreground_reference_features','feature_count':0 if corners is None else len(corners)}
    uv=corners[:,0];xy=np.rint(uv).astype(int);depth=rendered['depth'][xy[:,1],xy[:,0]]
    good=np.isfinite(depth)&(depth>0);uv=uv[good];depth=depth[good]
    camera_points=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T)*depth[:,None]
    native=(camera_points-translation)@rotation/scale
    # Thin native parts provide a measured dominant-plane prior, rather than
    # interpreting tiny generated surface bumps as real rigid feature depth.
    center=native.mean(0);basis=np.linalg.svd(native-center,full_matrices=False)[2]
    spread=np.std((native-center)@basis.T,axis=0)
    plane=None
    if spread[-1] < .10*spread[1]:
        plane={'point':center,'normal':basis[-1]}
        rays=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T)@rotation/scale
        origin=-translation@rotation/scale
        distance=((center-origin)@basis[-1])/(rays@basis[-1])
        if np.isfinite(distance).all()and(distance>0).all():native=origin+rays*distance[:,None]
    coverage=float(cv2.contourArea(cv2.convexHull(uv))/max(1,int(foreground.sum())))if len(uv)>=3 else 0.
    if len(uv)<policy['minimum_PnP_inliers'] or coverage<policy['minimum_reference_feature_coverage']:
        return {'status':'withheld','reason':'insufficient_reference_surface_feature_coverage','feature_count':len(uv),'coverage':coverage}
    return {'status':'estimated','native':native,'uv':uv.astype(np.float32),'reference_mask_area':int(foreground.sum()),
        'reference_coverage':coverage,'reference_mask_ref':reference,'matrix':np.asarray(pose['camera_from_native_mesh'],float),
        'reference_gray':frame.gray,'foreground':foreground,'rendered_depth':rendered['depth'],'plane':plane}


def direct_source_features(seed,target,k,*,policy=POLICY):
    """Independent source descriptor re-identification, never target-pose hints."""
    if seed['status']!='estimated':return {'status':'withheld','reason':'no_source_seed'}
    sift=cv2.SIFT_create(nfeatures=2200)
    if 'sift' not in seed:
        seed['sift']=sift.detectAndCompute(seed['reference_gray'],seed['foreground'].astype(np.uint8)*255)
    ka,da=seed['sift'];kb,db=sift.detectAndCompute(target.gray,None)
    if da is None or db is None:return {'status':'withheld','reason':'insufficient_direct_source_descriptors'}
    matcher=cv2.BFMatcher();ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2)
    reverse={m.queryIdx:m.trainIdx for pair in ba if len(pair)==2 for m,n in [pair]if m.distance<.74*n.distance}
    matches=[m for pair in ab if len(pair)==2 for m,n in [pair]if m.distance<.74*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    if len(matches)<policy['minimum_PnP_inliers']:return {'status':'withheld','reason':'insufficient_mutual_source_descriptor_matches','matches':len(matches)}
    source=np.float32([ka[m.queryIdx].pt for m in matches]);destination=np.float32([kb[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(0)
    homography,inliers=cv2.findHomography(source,destination,cv2.USAC_MAGSAC,2.,maxIters=2000,confidence=.999)
    if homography is None or inliers is None or int(inliers.sum())<policy['minimum_PnP_inliers']:
        return {'status':'withheld','reason':'insufficient_source_descriptor_planar_consensus','matches':len(matches)}
    keep=inliers[:,0]>0;source=source[keep];destination=destination[keep]
    rotation,translation,scale=matrix_parts(seed['matrix']);rays=(np.c_[source,np.ones(len(source))]@np.linalg.inv(k).T)@rotation/scale
    origin=-translation@rotation/scale
    if seed['plane']is not None:
        plane=seed['plane'];distance=((plane['point']-origin)@plane['normal'])/(rays@plane['normal'])
        native=origin+rays*distance[:,None]
    else:
        xy=np.rint(source).astype(int);depth=seed['rendered_depth'][xy[:,1],xy[:,0]]
        native=((np.c_[source,np.ones(len(source))]@np.linalg.inv(k).T)*depth[:,None]-translation)@rotation/scale
    good=np.isfinite(native).all(1);native=native[good];source=source[good];destination=destination[good]
    coverage=float(cv2.contourArea(cv2.convexHull(source))/max(1,seed['reference_mask_area']))if len(source)>=3 else 0.
    if len(source)<policy['minimum_PnP_inliers']or coverage<policy['minimum_reference_feature_coverage']:
        return {'status':'withheld','reason':'insufficient_direct_feature_spatial_coverage','matches':len(source),'coverage':coverage}
    ok,rvec,tvec,kept=cv2.solvePnPRansac(np.ascontiguousarray(native*scale,np.float64),np.ascontiguousarray(destination,np.float64),k,None,
        cv2.Rodrigues(rotation)[0],translation[:,None].copy(),useExtrinsicGuess=True,iterationsCount=500,
        reprojectionError=policy['maximum_PnP_reprojection_px'],confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or kept is None or len(kept)<policy['minimum_PnP_inliers']or len(kept)/len(source)<policy['minimum_PnP_inlier_fraction']:
        return {'status':'withheld','reason':'insufficient_direct_feature_PnP_consensus','matches':len(source)}
    fitted=np.eye(4);fitted[:3,:3]=cv2.Rodrigues(rvec)[0]*scale;fitted[:3,3]=tvec[:,0];ids=kept[:,0]
    errors=np.linalg.norm(project(native[ids],fitted,k)-destination[ids],axis=1)
    if np.median(errors)>policy['maximum_median_PnP_reprojection_px']or fitted[2,3]<=0:
        return {'status':'withheld','reason':'direct_source_feature_reprojection_or_depth_failed'}
    chosen=ids[np.linspace(0,len(ids)-1,min(20,len(ids)),dtype=int)]
    return {'status':'estimated','method':'mutual_source_SIFT_MAGSAC_native_surface_PnP',
        'camera_from_native_mesh':fitted.tolist(),'PnP_inliers':len(ids),'surviving_features':len(source),
        'inlier_fraction':float(len(ids)/len(source)),'median_reprojection_px':float(np.median(errors)),
        'feature_coverage':coverage,'native_points':native[chosen].tolist(),'target_pixels':destination[chosen].tolist(),
        'reference_feature_coverage':seed['reference_coverage'],'reference_surface_model':'native_dominant_plane_ray_hits'if seed['plane']is not None else'native_triangle_depth_ray_hits'}


def dense_source_alignment(seed,target,k,prior,*,policy=POLICY):
    """Image-verified planar hypothesis; neighbours initialize, RGB decides.

    Grid/feature points transformed by ECC are NOT independent feature matches.
    Dense correlation and reverse-warp consistency are recorded as their own
    evidence class; PnP only converts that measured projective warp to a pose.
    """
    if seed['status']!='estimated'or seed['plane']is None:return {'status':'withheld','reason':'no_native_dominant_plane_for_dense_alignment'}
    uv=seed['uv'];native=seed['native'];destination=project(native,prior,k)
    initial,_=cv2.findHomography(uv,destination,0)
    if initial is None:return {'status':'withheld','reason':'dense_alignment_initial_projection_unavailable'}
    size=(target.gray.shape[1],target.gray.shape[0]);initial=np.asarray(initial/initial[2,2],np.float32)
    targetmask=cv2.warpPerspective(seed['foreground'].astype(np.uint8)*255,initial,size)
    targetmask=cv2.dilate(targetmask,np.ones((15,15),np.uint8))
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,.00003)
    try:
        cc,warp=cv2.findTransformECC(seed['reference_gray'],target.gray,initial.copy(),cv2.MOTION_HOMOGRAPHY,criteria,targetmask,5)
        backcc,reverse=cv2.findTransformECC(target.gray,seed['reference_gray'],np.asarray(np.linalg.inv(warp),np.float32),cv2.MOTION_HOMOGRAPHY,criteria,seed['foreground'].astype(np.uint8)*255,5)
    except cv2.error:return {'status':'withheld','reason':'dense_photometric_alignment_not_converged'}
    mapped=cv2.perspectiveTransform(uv[:,None,:],warp)[:,0]
    cycled=cv2.perspectiveTransform(mapped[:,None,:],reverse)[:,0]
    cycle=np.linalg.norm(cycled-uv,axis=1)
    if min(cc,backcc)<policy['minimum_dense_photometric_correlation']or float(np.median(cycle))>policy['maximum_dense_warp_cycle_px']:
        return {'status':'withheld','reason':'dense_RGB_correlation_or_reverse_warp_failed','dense_correlation':float(cc),'reverse_dense_correlation':float(backcc),'median_warp_cycle_px':float(np.median(cycle))}
    rotation,translation,scale=matrix_parts(prior)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(np.ascontiguousarray(native*scale,np.float64),np.ascontiguousarray(mapped,np.float64),k,None,
        cv2.Rodrigues(rotation)[0],translation[:,None].copy(),useExtrinsicGuess=True,iterationsCount=400,
        reprojectionError=policy['maximum_PnP_reprojection_px'],confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers)<policy['minimum_PnP_inliers']:
        return {'status':'withheld','reason':'dense_warp_cannot_support_native_plane_PnP'}
    fitted=np.eye(4);fitted[:3,:3]=cv2.Rodrigues(rvec)[0]*scale;fitted[:3,3]=tvec[:,0];ids=inliers[:,0]
    errors=np.linalg.norm(project(native[ids],fitted,k)-mapped[ids],axis=1)
    normal=seed['plane']['normal'];before=float((rotation@normal)@(-translation));after=float((fitted[:3,:3]/scale@normal)@(-fitted[:3,3]))
    if np.median(errors)>policy['maximum_median_PnP_reprojection_px']or before*after<=0 or not .65<fitted[2,3]/prior[2,3]<1.55:
        return {'status':'withheld','reason':'dense_warp_native_pose_reprojection_facing_or_depth_failed'}
    selected=ids[np.linspace(0,len(ids)-1,min(20,len(ids)),dtype=int)]
    return {'status':'estimated','method':'dense_bidirectional_ECC_native_plane_pose_hypothesis',
        'camera_from_native_mesh':fitted.tolist(),'PnP_inliers':len(ids),'surviving_features':len(native),
        'inlier_fraction':float(len(ids)/len(native)),'median_reprojection_px':float(np.median(errors)),
        'feature_coverage':seed['reference_coverage'],'reference_feature_coverage':seed['reference_coverage'],
        'native_points':native[selected].tolist(),'target_pixels':mapped[selected].tolist(),
        'dense_correlation':float(cc),'reverse_dense_correlation':float(backcc),'median_warp_cycle_px':float(np.median(cycle)),
        'source_to_target_homography':warp.tolist(),'target_to_source_homography':reverse.tolist(),
        'PnP_grid_points_are_independent_feature_matches':False,'dense_photometric_pixels':int(seed['foreground'].sum()),
        'reference_surface_model':'native_dominant_plane_ray_hits','neighbour_prior_used_only_for_initialization':True}


def track_direction(frames, anchor, seed, k, *, policy=POLICY):
    """Track fixed source-surface points without reacquisition after loss."""
    if seed['status']!='estimated': return {}
    points=seed['uv'].copy();native=seed['native'].copy();previous=anchor.gray
    alive=np.ones(len(points),bool);matrix=seed['matrix'].copy();output={}
    previous_time=anchor.timestamp_s
    for frame in frames:
        ids=np.flatnonzero(alive)
        if len(ids)<policy['minimum_PnP_inliers']:
            output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'insufficient_surviving_surface_tracks'};continue
        source=points[ids,None,:].copy()
        options={'winSize':(25,25),'maxLevel':4,'criteria':(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01),'minEigThreshold':1e-4}
        forward,valid,_=cv2.calcOpticalFlowPyrLK(previous,frame.gray,source,None,**options)
        if forward is None or valid is None:
            alive[:]=False;output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'flow_unavailable'};continue
        backward,backvalid,_=cv2.calcOpticalFlowPyrLK(frame.gray,previous,forward,None,**options)
        if backward is None or backvalid is None:
            alive[:]=False;output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'reverse_flow_unavailable'};continue
        cycle=np.linalg.norm((backward-source)[:,0],axis=1)
        tracked=forward[:,0];height,width=frame.gray.shape
        keep=(valid[:,0]>0)&(backvalid[:,0]>0)&(cycle<=policy['maximum_forward_backward_error_px'])&np.isfinite(tracked).all(1)
        keep&=(tracked[:,0]>=2)&(tracked[:,0]<width-2)&(tracked[:,1]>=2)&(tracked[:,1]<height-2)
        alive[ids[~keep]]=False;kept=ids[keep];points[kept]=tracked[keep]
        previous=frame.gray;previous_time=frame.timestamp_s
        if len(kept)<policy['minimum_PnP_inliers']:
            output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'too_few_bidirectionally_consistent_tracks','surviving_features':len(kept)};continue
        rotation,translation,scale=matrix_parts(matrix);rvec=cv2.Rodrigues(rotation)[0];tvec=translation[:,None].copy()
        cv2.setRNGSeed(0)
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(np.ascontiguousarray(native[kept]*scale,np.float64),
            np.ascontiguousarray(points[kept],np.float64),k,None,rvec,tvec,useExtrinsicGuess=True,
            iterationsCount=350,reprojectionError=policy['maximum_PnP_reprojection_px'],confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers)<policy['minimum_PnP_inliers'] or len(inliers)/len(kept)<policy['minimum_PnP_inlier_fraction']:
            output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'weak_source_surface_PnP_consensus','surviving_features':len(kept),'inliers':0 if inliers is None else len(inliers)};continue
        fitted=np.eye(4);fitted[:3,:3]=cv2.Rodrigues(rvec)[0]*scale;fitted[:3,3]=tvec[:,0]
        selected=kept[inliers[:,0]];reprojection=np.linalg.norm(project(native[selected],fitted,k)-points[selected],axis=1)
        angle=float(Rotation.from_matrix(fitted[:3,:3]/scale@rotation.T).magnitude())
        if float(np.median(reprojection))>policy['maximum_median_PnP_reprojection_px'] or angle>policy['maximum_frame_rotation_rad'] or not .65< fitted[2,3]/matrix[2,3] <1.55:
            output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'unstable_temporal_PnP_step','median_reprojection_px':float(np.median(reprojection)),'step_rotation_rad':angle};continue
        coverage=float(cv2.contourArea(cv2.convexHull(points[selected]))/max(1.,seed['reference_mask_area']*(seed['matrix'][2,3]/fitted[2,3])**2))
        if coverage<policy['minimum_reference_feature_coverage']:
            output[round(frame.timestamp_s,6)]={'status':'withheld','reason':'spatially_concentrated_surviving_tracks','coverage':coverage};continue
        matrix=fitted
        # Twenty actual correspondences allow independent numerical reprojection
        # checks; all surviving features participate in fitting.
        proof_ids=selected[np.linspace(0,len(selected)-1,min(20,len(selected)),dtype=int)]
        output[round(frame.timestamp_s,6)]={'status':'estimated','method':'forward_backward_LK_native_surface_RANSAC_PnP','camera_from_native_mesh':fitted.tolist(),
            'PnP_inliers':len(selected),'surviving_features':len(kept),'inlier_fraction':float(len(selected)/len(kept)),
            'median_reprojection_px':float(np.median(reprojection)),'maximum_selected_cycle_px':float(np.max(cycle[np.isin(ids,selected)])),
            'feature_coverage':coverage,'native_points':native[proof_ids].tolist(),'target_pixels':points[proof_ids].tolist(),
            'reference_feature_coverage':seed['reference_coverage']}
    return output


def anchor_record(component, pose):
    return {'component_id':component['id'],'source_sample_index':pose['source_sample_index'],
        'timestamp_s':pose['timestamp_s'],'camera_from_native_mesh':pose['camera_from_native_mesh'],
        'source_mask_ref':pose.get('source_mask')}


def agree(first, second, vertices, k, *, policy=POLICY):
    ra,ta,sa=matrix_parts(first);rb,tb,sb=matrix_parts(second)
    center=np.mean(vertices,axis=0,keepdims=True);a=project(center,first,k)[0];b=project(center,second,k)[0]
    extent=max(1.,float(np.linalg.norm(np.ptp(project(vertices,first,k),axis=0))))
    rotation=float(Rotation.from_matrix(ra@rb.T).magnitude());distance=float(np.linalg.norm(a-b)/extent)
    ratio=float(max(ta[2]/tb[2],tb[2]/ta[2]))if min(ta[2],tb[2])>0 else math.inf
    return {'accepted':rotation<=policy['maximum_bidirectional_rotation_rad'] and distance<=policy['maximum_bidirectional_center_extent_fraction'] and ratio<=policy['maximum_bidirectional_depth_ratio'],
        'rotation_rad':rotation,'center_extent_fraction':distance,'depth_ratio':ratio}


def bounded_episodes(poses, max_gap):
    known=[(i,p)for i,p in enumerate(poses)if p['status']=='estimated' and p.get('camera_from_native_mesh')is not None]
    episodes=[]
    for (a,left),(b,right) in zip(known,known[1:]):
        if b>a+1:
            episodes.append({'left':left,'right':right,'original_unknown_indices':[p['source_sample_index']for p in poses[a+1:b]],
                'eligible':right['timestamp_s']-left['timestamp_s']<=max_gap,
                'reason':None if right['timestamp_s']-left['timestamp_s']<=max_gap else 'anchor_gap_exceeds_bounded_source_tracking_policy'})
    return episodes


def source_mask_check(prediction,frame,program_evidence,parts,meshes,k,size,*,root=ROOT,policy=POLICY):
    """Existing target SAM is read only after source-RGB motion is frozen."""
    observations=[o for o in program_evidence['observations']if o['track_id']in prediction['component_poses']and o.get('mask_path')]
    if not observations:return {'status':'unavailable','reason':'no_existing_target_component_mask','target_mask_used_for_fitting':False}
    target=np.zeros((size[1],size[0]),bool);references=[]
    for observation in observations:
        pointer=ref(Path(root)/observation['mask_path'],root)
        require(pointer['sha256']==observation['mask_sha256'],'Target evaluation mask changed')
        references.append(pointer);mask=np.asarray(Image.open(Path(root)/pointer['path']))>0
        target|=cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    if not target.any():return {'status':'unavailable','reason':'empty_existing_target_component_mask','mask_refs':references,'target_mask_used_for_fitting':False}
    rendered=np.zeros_like(target)
    for cid,matrix in prediction['component_poses'].items():
        if cid not in meshes:meshes[cid]=load_native_glb(Path(root)/parts[cid]['asset']['path'])
        rotation,translation,scale=matrix_parts(matrix)
        rendered|=render_native_mask(meshes[cid],k,rotation,translation,*size,scale=scale)['mask']
    overlap=int((target&rendered).sum());iou=overlap/max(1,int((target|rendered).sum()));coverage=overlap/int(target.sum())
    return {'status':'passed'if iou>=policy['minimum_target_mask_iou']and coverage>=policy['minimum_target_mask_coverage']else'failed',
        'mask_refs':references,'raw_full_triangle_union_iou':float(iou),'source_foreground_coverage':float(coverage),
        'target_mask_used_for_fitting':False,'SAM_mask_is_ground_truth':False,'full_native_triangle_render':True}


def build_case(twin_id, output, *, root=ROOT, only_episode=None, policy=None):
    root=Path(root).resolve();out=Path(output);out=out if out.is_absolute()else root/out
    require(not out.exists(),'New temporal output directory required');out.mkdir(parents=True)
    policy={**POLICY,**(policy or {})};start=time.monotonic();implementation_ref=ref(Path(__file__),root)
    catalog_ref=ref(root/'showcase/public/library-catalog.json',root);catalog=read_checked(catalog_ref,root)
    record=next(r for r in catalog['records']if r['id']==twin_id)
    program=read_checked(record['interaction'],root)
    registry=json.loads((root/'showcase/public/aligned-scenes.json').read_text())
    entry=next(e for e in registry['scenes']if e['source_sha256']==record['source']['sha256'])
    scene_ref=entry['manifest'];scene=read_checked(scene_ref,root)
    require(scene['source']['sha256']==program['source']['sha256'],'Scene/program source mismatch')
    frames,k,size,video_ref=video_frames(program,program['evidence'],scene['camera'],root=root,policy=policy)
    by_index={f.original_index:f for f in frames if f.original_index is not None}
    parts={c['id']:c for c in scene['components']};joints=scene.get('mechanics',{}).get('joints',[])
    evidence_by_index={e['source_index']:e for e in program['evidence']}
    def anchored(component,pose):
        result=anchor_record(component,pose);e=evidence_by_index[pose['source_sample_index']]
        result['source_RGB_ref']=ref(root/e['frame_path'],root)
        return result
    excluded=[]
    if record.get('derived_assembly'):
        declared=read_checked(record['derived_assembly'],root)
        if declared.get('nonrigid_observations_ref'):
            nonrigid=read_checked(declared['nonrigid_observations_ref'],root)
            require(nonrigid['evidence_program']==record['interaction']and nonrigid['source']['sha256']==program['source']['sha256'],'Nonrigid exclusion source mismatch')
            excluded.append({'component_id':nonrigid['component_id'],'reason':'exact_declared_nonrigid_source_component','reference':declared['nonrigid_observations_ref']})
    children={j['child_id']for j in joints};groups=[]
    for component in scene['components']:
        if component['id']in children or component['id']in {e['component_id']for e in excluded}:continue
        related=[j for j in joints if j['parent_id']==component['id']]
        groups.append((component,related))
    meshes={};seed_cache={};proofs=[];groups_output=[]
    def mesh(component_id):
        if component_id not in meshes:meshes[component_id]=load_native_glb(root/parts[component_id]['asset']['path'])
        return meshes[component_id]
    def get_seed(component_id,pose):
        key=(component_id,pose['source_sample_index'])
        if key not in seed_cache:seed_cache[key]=seed_surface(by_index[pose['source_sample_index']],parts[component_id],pose,mesh(component_id),k,size,root=root,policy=policy)
        return seed_cache[key]
    for parent,related in groups:
        episodes=bounded_episodes(parent['poses'],policy['maximum_anchor_gap_s']);gout={'root_component_id':parent['id'],'component_ids':[parent['id']]+[j['child_id']for j in related], 'episodes':[]}
        for number,episode in enumerate(episodes):
            left,right=episode['left'],episode['right']
            if only_episode is not None and left['source_sample_index']!=only_episode:continue
            eid=f'{parent["id"]}-{left["source_sample_index"]}-{right["source_sample_index"]}'
            row={'id':eid,'anchor_before':anchored(parent,left),'anchor_after':anchored(parent,right),
                'original_unknown_indices':episode['original_unknown_indices'],'status':'withheld','reason':episode['reason'],'predictions':[],'constant_joint_states':[]}
            if not episode['eligible']:gout['episodes'].append(row);continue
            # An attached descendant is inferred only from the same saved joint
            # state on BOTH sides. A changing/unknown angle cannot be fabricated.
            joint_states=[]
            for joint in related:
                samples={p['source_sample_index']:p for p in joint['samples']};a=samples[left['source_sample_index']];b=samples[right['source_sample_index']]
                if a['status']!='estimated' or b['status']!='estimated' or abs(a['q_rad']-b['q_rad'])>policy['maximum_constant_joint_state_difference_rad']:
                    row['reason']='changing_or_unknown_joint_state_requires_independent_two_part_motion_evidence';break
                joint_states.append({'id':joint['id'],'q_rad':float((a['q_rad']+b['q_rad'])/2),'method':'matching_neighboring_saved_joint_states_only','hidden_part_observed':False})
            else:
                row['constant_joint_states']=joint_states
                drivers=[]
                for anchor in [left,right]:
                    options=[parent['id']]+[j['child_id']for j in related]
                    preferred=anchor.get('source_visible_panel')
                    if preferred in options:options=[preferred]+[c for c in options if c!=preferred]
                    choices=[]
                    for cid in options:
                        cp=next(p for p in parts[cid]['poses']if p['source_sample_index']==anchor['source_sample_index'])
                        if cp['status']!='estimated':continue
                        seed=get_seed(cid,cp)
                        if seed['status']=='estimated':choices.append((cid,cp,seed))
                        if choices and cid==preferred:break
                    if not choices:drivers=[];row['reason']='no_reference_foreground_native_surface_tracks';break
                    drivers.append(max(choices,key=lambda x:len(x[2]['uv'])))
                if len(drivers)==2:
                    targets=[f for f in frames if left['timestamp_s']<f.timestamp_s<right['timestamp_s']]
                    before=track_direction(targets,by_index[left['source_sample_index']],drivers[0][2],k,policy=policy)
                    after=track_direction(list(reversed(targets)),by_index[right['source_sample_index']],drivers[1][2],k,policy=policy)
                    # Descriptor re-identification is evaluated against the
                    # original anchor, so a lost LK chain cannot silently attach
                    # itself to a hand or a newly visible opposite surface.
                    for frame in targets:
                        for results,driver in [(before,drivers[0]),(after,drivers[1])]:
                            direct=direct_source_features(driver[2],frame,k,policy=policy)
                            if direct['status']=='estimated':results[round(frame.timestamp_s,6)]=direct
                            elif policy['dense_photometric_fallback_enabled']and results.get(round(frame.timestamp_s,6),{}).get('status')!='estimated'and drivers[0][0]==drivers[1][0]:
                                alpha=(frame.timestamp_s-left['timestamp_s'])/(right['timestamp_s']-left['timestamp_s'])
                                prior=blend(drivers[0][1]['camera_from_native_mesh'],drivers[1][1]['camera_from_native_mesh'],alpha)
                                dense=dense_source_alignment(driver[2],frame,k,prior,policy=policy)
                                if dense['status']=='estimated':results[round(frame.timestamp_s,6)]=dense
                    row['driver_before']=anchored(parts[drivers[0][0]],drivers[0][1]);row['driver_after']=anchored(parts[drivers[1][0]],drivers[1][1])
                    row['view_change_break']=drivers[0][0]!=drivers[1][0]
                    def to_root(matrix,cid):
                        if cid==parent['id']:return np.asarray(matrix)
                        joint=next(j for j in related if j['child_id']==cid);q=next(s['q_rad']for s in joint_states if s['id']==joint['id'])
                        relative=M.forward(np.eye(4),joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero'])
                        return np.asarray(matrix)@np.linalg.inv(relative)
                    for frame in targets:
                        key=round(frame.timestamp_s,6);a=before.get(key,{'status':'withheld'});b=after.get(key,{'status':'withheld'})
                        predicted={'timestamp_s':frame.timestamp_s,'source_sample_index':frame.original_index,'source_RGB':frame.evidence,
                            'status':'withheld','reason':'no_consistent_source_conditioned_motion','component_poses':{},'joint_states':[],
                            'original_unknown_row_preserved':True,'method':'source_native_depth_ShiTomasi_bidirectional_LK_then_RANSAC_PnP'}
                        candidates=[]
                        for result,driver in [(a,drivers[0]),(b,drivers[1])]:
                            if result['status']=='estimated':candidates.append(to_root(result['camera_from_native_mesh'],driver[0]))
                        comparison=None;matrix=None
                        if a['status']==b['status']=='estimated'and not row['view_change_break']:
                            comparison=agree(candidates[0],candidates[1],mesh(parent['id']).vertices,k,policy=policy)
                            if comparison['accepted']:
                                alpha=(frame.timestamp_s-left['timestamp_s'])/(right['timestamp_s']-left['timestamp_s'])
                                matrix=blend(candidates[0],candidates[1],alpha)
                            else:predicted['reason']='past_future_native_pose_disagreement_or_view_change'
                        elif len(candidates)>=1:
                            use_past=a['status']=='estimated'and(b['status']!='estimated'or frame.timestamp_s-left['timestamp_s']<=right['timestamp_s']-frame.timestamp_s)
                            distance=frame.timestamp_s-left['timestamp_s']if use_past else right['timestamp_s']-frame.timestamp_s
                            if distance<=policy['maximum_one_sided_distance_s']:
                                chosen=a if use_past else b;driver=drivers[0]if use_past else drivers[1]
                                matrix=to_root(chosen['camera_from_native_mesh'],driver[0]);predicted['reason']='one_sided_source_tracks_near_saved_anchor'
                                predicted['view_segment_id']=eid+(':before'if use_past else':after')
                        if matrix is not None:
                            predicted.update(status='estimated',reason=None,component_poses={parent['id']:matrix.tolist()},joint_states=copy.deepcopy(joint_states),
                                support='past_and_future_RGB_agree'if a['status']==b['status']=='estimated'and not row['view_change_break']else'one_sided_near_anchor_RGB')
                            predicted.setdefault('view_segment_id',eid+':same_source_view')
                            for joint in related:
                                q=next(s['q_rad']for s in joint_states if s['id']==joint['id'])
                                predicted['component_poses'][joint['child_id']]=M.forward(matrix,joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero']).tolist()
                            if frame.original_index is not None:
                                check=source_mask_check(predicted,frame,evidence_by_index[frame.original_index],parts,meshes,k,size,root=root,policy=policy)
                                predicted['independent_target_mask_check']=check
                                if check['status']=='failed':predicted.update(status='withheld',reason='independent_existing_target_mask_disagrees',component_poses={},joint_states=[])
                        proof_id=f'{eid}@{frame.timestamp_s:.6f}'
                        proofs.append({'id':proof_id,'timestamp_s':frame.timestamp_s,'past':a,'future':b,'agreement':comparison,
                            'driver_before':drivers[0][0],'driver_after':drivers[1][0],'target_RGB':frame.evidence})
                        predicted['flow_proof_id']=proof_id;row['predictions'].append(predicted)
                    row['status']='estimated'if any(p['status']=='estimated'for p in row['predictions'])else'withheld'
                    row['reason']=None if row['status']=='estimated'else'all_source_tracks_failed_or_disagreed'
            gout['episodes'].append(row)
        groups_output.append(gout)
    write_new(out/'flow-proofs.json',{'format':'interaction-twin-temporal-source-flow-proofs','version':1,'rows':proofs})
    result={'format':'interaction-twin-source-conditioned-motion-hypotheses','version':1,'twin_id':twin_id,
        'source_ref':video_ref,'evidence_program_ref':record['interaction'],'baseline_scene_ref':scene_ref,
        'catalog_ref':catalog_ref,'implementation_ref':implementation_ref,'nonroutable_fingerprints':['catalog_ref','implementation_ref'],'policy':policy,
        'camera':{'K':k.tolist(),'width':size[0],'height':size[1],'intrinsics_calibrated':False},
        'flow_proofs_ref':ref(out/'flow-proofs.json',root),'groups':groups_output,'excluded_components':excluded,
        'source_unknowns_preserved':True,'native_geometry_modified':False,'physical_accuracy_verified':False,
        'simulation_ready':False,'new_API_calls':0,'new_GPU_calls':0,'wall_seconds':time.monotonic()-start,
        'limitations':['Existing native geometry and camera are uncertain; RGB PnP is source-conditioned prediction, not independently measured physical motion.',
            'Matching past/future joint states constrain hidden rigid descendants; they do not observe hidden parts or establish contact.',
            'View changes, low texture, hand occlusion and conflicting past/future pose hypotheses remain withheld.',
            'Original unknown poses and timing barriers remain unchanged in the baseline; only this explicitly estimated layer crosses selected source-supported gaps.']}
    require(ref(Path(__file__),root)==implementation_ref,'Executing temporal helper changed during run; preserve this diagnostic and start a new output')
    write_new(out/'hypotheses.json',result)
    return {'hypotheses':ref(out/'hypotheses.json',root),'flow_proofs':result['flow_proofs_ref'],
        'estimated':sum(p['status']=='estimated'for g in groups_output for e in g['episodes']for p in e['predictions']),
        'withheld':sum(p['status']=='withheld'for g in groups_output for e in g['episodes']for p in e['predictions']),
        'wall_seconds':result['wall_seconds']}
