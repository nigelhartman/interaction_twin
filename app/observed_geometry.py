"""Experimental object-centred SfM from source pixels and saved SAM masks.

No generated meshes, depth maps or prior poses. The default SIFT path has no
model inference; optional versioned correspondence providers record their own
learned 2D tracking calls and source scopes. All camera poses map an independently
reconstructed rigid part into each view. Source-frame splits and point roles are
deterministic and fixed before fitting.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


PROTOCOL = {
    'version': 'observed-sparse-geometry-v5', 'random_seed': 3102026,
    'max_views': 45, 'max_features_per_view': 1200,
    'mask_erosion_px': 3, 'descriptor_boundary_radius_keypoint_sizes': 2.0,
    'feature_core_fraction_of_max_inscribed_mask_radius': .15,
    'sift_ratio': .75, 'fundamental_ransac_px': 1.5,
    'minimum_training_observations_per_track': 2,
    'essential_ransac_px': 1.5, 'point_reprojection_max_px': 4.,
    'point_min_angle_deg': 1., 'seed_median_angle_min_deg': 2.,
    'seed_min_accepted_coverage_fraction': .12,
    'homography_dominance_fraction': .85,
    'linearity_ratio_min': .04,
    'sensitivity_min_common_reference_fraction': .5,
    'sensitivity_min_common_candidate_fraction': .5,
    'seed_min_points': 25, 'pnp_min_inliers': 12,
    'pnp_inlier_fraction_min': .5, 'pnp_ransac_px': 3.,
    'heldout_frame_rule': 'selected_view_position_mod_5_equals_2',
    'test_point_rule': 'sha256_track_id_mod_3_equals_0',
    'heldout_min_score_points_per_frame': 15, 'heldout_min_frames': 3,
    'heldout_median_px_max': 3., 'heldout_p90_px_max': 6.,
    'main_focal_factor_of_max_image_dimension': 1.,
    'sensitivity_focal_factors': [.8, 1.25],
    'sensitivity_seed_policy': 'same_training_seed_pair_as_main_reconstruction',
    'seed_rank_policy': 'prefer_non_homography_dominant_accepted_inliers_then_point_count_times_parallax',
    'focal_calibrated': False, 'physical_scale_known': False,
    'training_observations_only_for_geometry_and_seed_selection': True,
    'heldout_pose_anchors_disjoint_from_scored_tracks': True,
    'test_views_never_bundle_adjusted': True,
    'prior_geometry_or_depth_used': False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_file(root, name, expected=None):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root / 'data') or not path.is_file():
        raise ValueError('Only existing experiment data files are accepted')
    if expected is not None and digest(path) != expected:
        raise ValueError('Saved source artifact hash mismatch')
    return path


def camera_matrix(width, height, factor=1.):
    f = max(width, height) * factor
    return np.array([[f, 0., width/2], [0., f, height/2], [0., 0., 1.]])


def learned_camera_assumption(provider, frames, output, root, declaration, source_binding):
    """Validate one source-bound TRAIN-only camera result before reconstruction.

    This boundary accepts intrinsics only. It never accepts the provider's depth,
    points or poses, and it cannot replace the K of an already reconstructed model.
    """
    training=[]
    for frame in frames:
        if frame['split']!='train':
            continue
        image=frame['image'].copy();image.flags.writeable=False
        training.append({**{k:frame[k] for k in ['id','timestamp_s','image_url','image_sha256','split']},
                         'image':image})
    if not training:
        raise ValueError('A learned camera requires saved TRAIN source images')
    original_bindings={frame['id']:{k:frame[k] for k in ['id','timestamp_s','image_url','image_sha256','split']}
                       for frame in training}
    K,metadata=provider.estimate(training,output)
    K=np.asarray(K,dtype=float)
    if (K.shape!=(3,3) or not np.isfinite(K).all() or not np.array_equal(K[2],[0.,0.,1.])
            or K[0,0]<=0 or K[1,1]<=0 or abs(K[0,1])>1e-10 or abs(K[1,0])>1e-10):
        raise ValueError('Camera provider must return a finite positive zero-skew full pinhole K')
    if not isinstance(metadata,dict) or not np.array_equal(np.asarray(metadata.get('K')),K):
        raise ValueError('Camera result metadata does not bind its exact K')
    expected=output/'camera-prior'/'camera-prior.json'
    report_name=metadata.get('report_path')
    if not isinstance(report_name,str) or root/report_name!=expected or expected.resolve()!=expected or not expected.is_file():
        raise ValueError('Camera provider report must be the confined completed source-bound receipt')
    report_sha=digest(expected)
    report=json.loads(expected.read_text())
    width,height=frames[0]['image'].shape[1],frames[0]['image'].shape[0]
    if (report.get('format')!='ordinary-video-camera-prior' or report.get('version')!=1
            or report.get('status')!='unvalidated_learned_camera_assumption'
            or metadata.get('kind')!=report['status'] or report.get('native_size')!=[width,height]
            or not np.array_equal(np.asarray(report.get('K')),K)
            or report.get('policy')!=declaration or metadata.get('policy')!=declaration):
        raise ValueError('Camera receipt dimensions, policy, status or intrinsics do not match')
    for flag in ['camera_calibrated','physical_scale_known','physical_accuracy_verified',
                 'prior_geometry_or_poses_used','learned_depth_or_point_maps_used']:
        if report.get(flag) is not False:
            raise ValueError('Camera receipt exceeds its unverified intrinsics-only scope: '+flag)
    if report.get('new_api_calls')!=0 or report.get('geometry_must_be_reconstructed_with_this_assumption') is not True:
        raise ValueError('Camera provider must be local and precede all geometry')
    inputs=report.get('input_views')
    if not isinstance(inputs,list) or not inputs or metadata.get('input_views')!=inputs:
        raise ValueError('Camera receipt must bind its exact TRAIN image inputs')
    minimum,maximum=declaration.get('minimum_train_images'),declaration.get('maximum_train_images')
    if (type(minimum) is not int or type(maximum) is not int or not 2<=minimum<=maximum<=3
            or not minimum<=len(inputs)<=maximum):
        raise ValueError('Camera consistency requires two or three predeclared TRAIN images')
    if (type(report.get('actual_model_calls')) is not int or type(report.get('completed_model_calls')) is not int
            or report['actual_model_calls']!=len(inputs) or report['completed_model_calls']!=len(inputs)
            or report.get('input_integrity',{}).get('unchanged') is not True):
        raise ValueError('Camera receipt lacks exact successful call counts or unchanged inputs')
    known=original_bindings;seen=set()
    for row in inputs:
        if not isinstance(row,dict) or row.get('id') not in known or row['id'] in seen:
            raise ValueError('Camera receipt has an unknown or duplicate TRAIN source')
        seen.add(row['id']);original=known[row['id']]
        if any(row.get(k)!=original[k] for k in ['id','timestamp_s','image_url','image_sha256','split']):
            raise ValueError('Camera receipt does not match an exact original TRAIN frame')
        safe_file(root,original['image_url'].removeprefix('/files/'),original['image_sha256'])
    if len({row['timestamp_s'] for row in inputs})<minimum or len({row['image_url'] for row in inputs})!=len(inputs):
        raise ValueError('Camera consistency requires distinct TRAIN source times and images')
    if digest(expected)!=report_sha:
        raise ValueError('Camera receipt changed during validation')
    result={'kind':report['status'],'K':K.tolist(),'native_size':[width,height],
            'policy':declaration,'input_views':inputs,'source_binding':source_binding,
            'report_path':report_name,'report_sha256':report_sha,
            'report':{'path':report_name,'sha256':report_sha},
            'camera_calibrated':False,'physical_scale_known':False,'physical_accuracy_verified':False,
            'prior_geometry_or_poses_used':False,'learned_depth_or_point_maps_used':False,
            'new_api_calls':0,'geometry_reconstructed_with_this_assumption':True}
    result_path=output/'camera-assumption.json'
    with result_path.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return K,{**result,'result_record':{'path':str(result_path.relative_to(root)),'sha256':digest(result_path)}}


def project(points, pose, K):
    xyz = np.asarray(points).reshape(-1, 3) @ pose[0].T + pose[1].reshape(1, 3)
    q = xyz @ K.T
    denom = q[:, 2:3].copy()
    denom[np.abs(denom) < 1e-12] = 1e-12
    return q[:, :2] / denom, xyz[:, 2]


def triangulate(a, b, pose_a, pose_b, K):
    p0 = K @ np.column_stack(pose_a)
    p1 = K @ np.column_stack(pose_b)
    homogeneous = cv2.triangulatePoints(p0, p1, np.asarray(a).T, np.asarray(b).T).T
    w = homogeneous[:, 3]
    xyz = np.full((len(w), 3), np.nan)
    finite = np.abs(w) > 1e-10
    xyz[finite] = homogeneous[finite, :3] / w[finite, None]
    c0 = -pose_a[0].T @ pose_a[1]
    c1 = -pose_b[0].T @ pose_b[1]
    v0, v1 = xyz - c0, xyz - c1
    denominator = np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)
    cosine = np.sum(v0*v1, axis=1) / np.maximum(denominator, 1e-20)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    uv0, z0 = project(xyz, pose_a, K)
    uv1, z1 = project(xyz, pose_b, K)
    error = np.maximum(np.linalg.norm(uv0-a, axis=1), np.linalg.norm(uv1-b, axis=1))
    valid = finite & np.isfinite(xyz).all(axis=1) & (z0>0) & (z1>0)
    return xyz, angle, error, valid


def select_views(program, track_id, root, max_views=None):
    """Read only matching original pixels/masks, never the asset/pose fields."""
    rows, excluded = [], []
    for evidence in program['evidence']:
        obs = next((o for o in evidence['observations'] if o['track_id']==track_id), None)
        if obs is None:
            continue
        frame_path = safe_file(root, evidence['frame_path'], evidence['frame_sha256'])
        mask_path = safe_file(root, obs['mask_path'], obs['mask_sha256'])
        if obs.get('audit_status') in {'reject_wrong_or_lost_mask', 'uncertain'}:
            excluded.append({'id': evidence['id'], 'reason': obs.get('audit_status')})
            continue
        image = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2]!=mask.shape:
            raise ValueError('Source/mask image dimensions mismatch')
        mask = np.uint8(mask>127)*255
        subtraction = []
        for other in evidence['observations']:
            if other['track_id']==track_id or other.get('audit_status') in {'reject_wrong_or_lost_mask','uncertain'}:
                continue
            path = safe_file(root, other['mask_path'], other['mask_sha256'])
            other_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if other_mask is None or other_mask.shape!=mask.shape:
                raise ValueError('Other component mask dimensions mismatch')
            mask[other_mask>127]=0
            subtraction.append({'track_id':other['track_id'],'mask_path':other['mask_path'],
                                'mask_sha256':other['mask_sha256']})
        if np.count_nonzero(mask)<150:
            excluded.append({'id':evidence['id'],'reason':'insufficient_visible_component_mask'})
            continue
        rows.append({'id':evidence['id'],'source_index':evidence['source_index'],
                     'timestamp_s':evidence['timestamp_s'], 'image':image,'mask':mask,
                     'image_url':'/files/'+evidence['frame_path'], 'mask_url':'/files/'+obs['mask_path'],
                     'image_sha256':evidence['frame_sha256'],'mask_sha256':obs['mask_sha256'],
                     'audit_status':obs.get('audit_status'),'subtracted_components':subtraction})
    cap = max_views or PROTOCOL['max_views']
    if len(rows)>cap:
        positions=np.unique(np.rint(np.linspace(0,len(rows)-1,cap)).astype(int))
        rows=[rows[i] for i in positions]
    for i, row in enumerate(rows):
        row['index']=i
        row['split']='heldout' if i%5==2 else 'train'
    return rows, excluded


def extract_features(frame):
    gray=cv2.cvtColor(frame['image'],cv2.COLOR_BGR2GRAY)
    mask=frame['mask']
    eroded=cv2.erode(mask,np.ones((7,7),np.uint8))
    detector=cv2.SIFT_create(nfeatures=PROTOCOL['max_features_per_view'],contrastThreshold=.012,edgeThreshold=12)
    kp, des=detector.detectAndCompute(gray,eroded)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    max_radius=min(float(distance.max()),min(gray.shape)/2)
    core_margin=max(3.,PROTOCOL['feature_core_fraction_of_max_inscribed_mask_radius']*max_radius)
    core_rejected=0
    selected=[]; occupied=set()
    for i in sorted(range(len(kp)),key=lambda j:-kp[j].response):
        x,y=kp[i].pt; xx,yy=int(round(x)),int(round(y))
        cell=(int(x/2),int(y/2))
        if cell in occupied or not (0<=xx<gray.shape[1] and 0<=yy<gray.shape[0]):
            continue
        if distance[yy,xx] < max(3.,kp[i].size*PROTOCOL['descriptor_boundary_radius_keypoint_sizes']):
            continue
        if distance[yy,xx]<core_margin:
            core_rejected+=1
            continue
        selected.append(i); occupied.add(cell)
    frame['xy']=np.array([kp[i].pt for i in selected],dtype=np.float64).reshape(-1,2)
    frame['des']=des[selected].copy() if selected else np.empty((0,128),np.float32)
    frame['feature_filter']={'max_inscribed_mask_radius_px':max_radius,'core_margin_px':core_margin,
                             'descriptor_candidates_rejected_by_additional_core_guard':core_rejected,
                             'kept_unique_features':len(selected)}


def mutual_matches(a, b):
    if len(a['des'])<2 or len(b['des'])<2:
        return []
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    ratio=PROTOCOL['sift_ratio']
    forward={x.queryIdx:(x.trainIdx,float(x.distance)) for row in matcher.knnMatch(a['des'],b['des'],k=2)
             if len(row)==2 for x,y in [row] if x.distance<ratio*y.distance}
    backward={x.queryIdx:x.trainIdx for row in matcher.knnMatch(b['des'],a['des'],k=2)
              if len(row)==2 for x,y in [row] if x.distance<ratio*y.distance}
    return [(i,j,d) for i,(j,d) in forward.items() if backward.get(j)==i]


class TrackUnion:
    def __init__(self):
        self.parent={}; self.members={}

    def find(self,node):
        if node not in self.parent:
            self.parent[node]=node; self.members[node]={node[0]:node}
        if self.parent[node]!=node:
            self.parent[node]=self.find(self.parent[node])
        return self.parent[node]

    def join(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb: return True
        if set(self.members[ra]) & set(self.members[rb]): return False
        if len(self.members[ra])<len(self.members[rb]): ra,rb=rb,ra
        self.parent[rb]=ra; self.members[ra].update(self.members.pop(rb)); return True

    def tracks(self,min_length=2):
        rows=[sorted(m.values()) for m in self.members.values() if len(m)>=min_length]
        return sorted(rows,key=lambda r:r[0])


def build_tracks(frames):
    train=[f for f in frames if f['split']=='train']
    links=[]; pair_logs=[]
    for ai,a in enumerate(train):
        for b in train[ai+1:]:
            matched=mutual_matches(a,b)
            record={'a':a['index'],'b':b['index'],'raw_mutual_matches':len(matched),'fundamental_inliers':0}
            if len(matched)>=8:
                pa=np.array([a['xy'][i] for i,j,d in matched]); pb=np.array([b['xy'][j] for i,j,d in matched])
                cv2.setRNGSeed(PROTOCOL['random_seed'])
                F,mask=cv2.findFundamentalMat(pa,pb,cv2.FM_RANSAC,PROTOCOL['fundamental_ransac_px'],.999,5000)
                if F is not None and F.shape==(3,3) and mask is not None:
                    keep=mask.ravel().astype(bool); record['fundamental_inliers']=int(keep.sum())
                    for (i,j,d),ok in zip(matched,keep):
                        if ok: links.append((d,(a['index'],i),(b['index'],j)))
            pair_logs.append(record)
    union=TrackUnion(); conflicts=0
    for _,a,b in sorted(links):
        if not union.join(a,b): conflicts+=1
    tracks=union.tracks()
    packed={}; node_to_track={}
    for i,nodes in enumerate(tracks):
        tid=f'point_{i:06d}'
        packed[tid]={fi:frames[fi]['xy'][ki].copy() for fi,ki in nodes}
        for node in nodes: node_to_track[node]=tid
    return packed,node_to_track,pair_logs,conflicts


def spread_fraction(xy,frame):
    if len(xy)<3: return 0.
    hull=cv2.convexHull(np.asarray(xy,np.float32))
    area=float(cv2.contourArea(hull))
    yy,xx=np.nonzero(frame['mask'])
    boxarea=float((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)) if len(xx) else 1.
    return area/max(boxarea,1.)


def find_seed(tracks,frames,K,seed_pair=None):
    train=[f['index'] for f in frames if f['split']=='train']
    candidates=[]
    for pos,a in enumerate(train):
        for b in train[pos+1:]:
            if seed_pair is not None and (a,b)!=tuple(sorted(seed_pair)):
                continue
            tids=[t for t,obs in tracks.items() if a in obs and b in obs]
            if len(tids)<PROTOCOL['seed_min_points']: continue
            p0=np.array([tracks[t][a] for t in tids]); p1=np.array([tracks[t][b] for t in tids])
            # A seed must span actual source pixels, not one tiny feature cluster.
            coverage=min(spread_fraction(p0,frames[a]),spread_fraction(p1,frames[b]))
            if coverage<PROTOCOL['seed_min_accepted_coverage_fraction']: continue
            cv2.setRNGSeed(PROTOCOL['random_seed'])
            E, emask=cv2.findEssentialMat(p0,p1,K,method=cv2.RANSAC,prob=.999,threshold=PROTOCOL['essential_ransac_px'])
            if E is None or E.shape!=(3,3): continue
            try:
                _,R,t,pmask=cv2.recoverPose(E,p0,p1,K,mask=emask.copy())
            except cv2.error: continue
            poses=((np.eye(3),np.zeros(3)),(R,t.ravel()))
            xyz,angle,error,positive=triangulate(p0,p1,*poses,K)
            keep=positive & pmask.ravel().astype(bool) & (error<=PROTOCOL['point_reprojection_max_px']) & (angle>=PROTOCOL['point_min_angle_deg'])
            H,hmask=cv2.findHomography(p0,p1,cv2.RANSAC,2.)
            raw_homography=float(hmask.sum()/len(tids)) if hmask is not None else 0.
            # Planarity is measured on points which actually enter the seed.
            # Rejected bad matches must not dilute a planar accepted subset.
            homography=1.
            if int(keep.sum())>=4:
                cv2.setRNGSeed(PROTOCOL['random_seed'])
                accepted_h,accepted_hmask=cv2.findHomography(p0[keep],p1[keep],cv2.RANSAC,2.)
                if accepted_hmask is not None:
                    homography=float(accepted_hmask.sum()/keep.sum())
            median_angle=float(np.median(angle[keep])) if keep.any() else 0.
            accepted_coverage=min(spread_fraction(p0[keep],frames[a]),spread_fraction(p1[keep],frames[b]))
            row={'a':a,'b':b,'matched_tracks':len(tids),'accepted_points':int(keep.sum()),
                 'median_angle_deg':median_angle,'positive_depth_fraction':float(positive.mean()),
                 'homography_fraction':homography,'coverage_fraction':accepted_coverage,
                 'all_matched_tracks_homography_fraction':raw_homography,
                 'matched_coverage_fraction':coverage,
                 'homography_dominant':homography>=PROTOCOL['homography_dominance_fraction']}
            valid=(int(keep.sum())>=PROTOCOL['seed_min_points']
                   and median_angle>=PROTOCOL['seed_median_angle_min_deg']
                   and positive.mean()>=.8
                   and accepted_coverage>=PROTOCOL['seed_min_accepted_coverage_fraction'])
            row['passes_seed_gate']=bool(valid)
            candidates.append((row,tids,xyz,keep,poses))
    passing=[c for c in candidates if c[0]['passes_seed_gate']]
    # Main choice uses TRAIN only. Never select a model by held-out residual.
    passing.sort(key=lambda c:(not c[0]['homography_dominant'],c[0]['accepted_points']*min(c[0]['median_angle_deg'],15.),c[0]['coverage_fraction']),reverse=True)
    logs=sorted([c[0] for c in candidates],key=lambda c:(c['passes_seed_gate'],c['accepted_points']),reverse=True)
    if not passing: return None,logs
    row,tids,xyz,keep,poses=passing[0]
    points={t:xyz[i] for i,t in enumerate(tids) if keep[i]}
    return {'seed':row,'poses':{row['a']:poses[0],row['b']:poses[1]},'points':points},logs


def solve_pose(points,observations,K):
    tids=sorted(set(points)&set(observations))
    if len(tids)<PROTOCOL['pnp_min_inliers']: return None
    xyz=np.array([points[t] for t in tids],np.float64)
    uv=np.array([observations[t] for t in tids],np.float64)
    cv2.setRNGSeed(PROTOCOL['random_seed'])
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1500,
                                         reprojectionError=PROTOCOL['pnp_ransac_px'],confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<PROTOCOL['pnp_min_inliers'] or len(inliers)/len(tids)<PROTOCOL['pnp_inlier_fraction_min']:
        return None
    index=inliers.ravel()
    rvec,tvec=cv2.solvePnP(xyz[index],uv[index],K,None,rvec,tvec,True,flags=cv2.SOLVEPNP_ITERATIVE)[1:]
    R=cv2.Rodrigues(rvec)[0]; pose=(R,tvec.ravel())
    _,z=project(xyz[index],pose,K)
    if (z>0).mean()<.9: return None
    return pose,[tids[i] for i in index]


def expand_model(model,tracks,frames,K):
    poses,points=model['poses'],model['points']
    for _ in range(5):
        changed=False
        pending=[]
        for f in frames:
            fi=f['index']
            if f['split']!='train' or fi in poses: continue
            obs={t:o[fi] for t,o in tracks.items() if fi in o}
            pending.append((len(set(obs)&set(points)),fi,obs))
        for _,fi,obs in sorted(pending,reverse=True):
            found=solve_pose(points,obs,K)
            if found is not None and spread_fraction(np.array([obs[t] for t in found[1]]),frames[fi])>=.08:
                poses[fi]=found[0]; changed=True
        for tid,obs in tracks.items():
            if tid in points: continue
            available=sorted(set(obs)&set(poses))
            options=[]
            for ii,a in enumerate(available):
                for b in available[ii+1:]:
                    xyz,angle,error,valid=triangulate(np.array([obs[a]]),np.array([obs[b]]),poses[a],poses[b],K)
                    if valid[0] and angle[0]>=PROTOCOL['point_min_angle_deg'] and error[0]<=PROTOCOL['point_reprojection_max_px']:
                        options.append((angle[0],xyz[0]))
            if not options: continue
            xyz=max(options,key=lambda q:q[0])[1]
            errors=[]; positive=0
            for fi in available:
                uv,z=project(xyz[None],poses[fi],K)
                errors.append(float(np.linalg.norm(uv[0]-obs[fi]))); positive+=int(z[0]>0)
            if positive==len(available) and np.median(errors)<=3. and np.percentile(errors,90)<=6.:
                points[tid]=xyz; changed=True
        if not changed: break
    return model


def bundle_adjust(model,tracks,K,max_nfev=70):
    """Train-only BA; seed poses fixed, including initial unit baseline."""
    poses,points=model['poses'],model['points']
    fixed={model['seed']['a'],model['seed']['b']}
    free=sorted(set(poses)-fixed); tids=sorted(points); pi={t:i for i,t in enumerate(tids)}
    ci={c:i for i,c in enumerate(free)}
    obs=[(fi,t,uv) for t in tids for fi,uv in tracks[t].items() if fi in poses]
    camera_ids=sorted(poses)
    camera_indices={fi:i for i,fi in enumerate(camera_ids)}
    observation_cameras=np.array([camera_indices[fi] for fi,t,uv in obs])
    observation_points=np.array([pi[t] for fi,t,uv in obs])
    observation_pixels=np.array([uv for fi,t,uv in obs])
    initial=np.r_[np.concatenate([np.r_[cv2.Rodrigues(poses[c][0])[0].ravel(),poses[c][1]] for c in free]) if free else np.empty(0),np.array([points[t] for t in tids]).ravel()]
    offset=len(free)*6
    def decode(x):
        p={c:poses[c] for c in fixed}
        for c in free:
            q=x[ci[c]*6:ci[c]*6+6];p[c]=(cv2.Rodrigues(q[:3])[0],q[3:6])
        xyz=x[offset:].reshape(-1,3)
        return p,xyz
    def residual(x):
        p,xyz=decode(x)
        rotations=np.array([p[fi][0] for fi in camera_ids])
        translations=np.array([p[fi][1] for fi in camera_ids])
        transformed=np.einsum('nij,nj->ni',rotations[observation_cameras],xyz[observation_points])+translations[observation_cameras]
        projected=transformed@K.T
        denominator=projected[:,2:3].copy()
        denominator[np.abs(denominator)<1e-12]=1e-12
        return (projected[:,:2]/denominator-observation_pixels).ravel()
    sparsity=lil_matrix((len(obs)*2,len(initial)),dtype=int)
    for i,(fi,t,_) in enumerate(obs):
        if fi in ci: sparsity[2*i:2*i+2,ci[fi]*6:ci[fi]*6+6]=1
        sparsity[2*i:2*i+2,offset+pi[t]*3:offset+pi[t]*3+3]=1
    before=residual(initial).reshape(-1,2)
    fit=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1.5,
                      x_scale='jac',max_nfev=max_nfev,ftol=1e-5,xtol=1e-6,gtol=1e-6)
    newposes,newxyz=decode(fit.x)
    model['poses']=newposes; model['points']={t:newxyz[i] for i,t in enumerate(tids)}
    model['bundle_adjustment']={'training_observations':len(obs),'free_cameras':len(free),'fixed_seed_poses':sorted(fixed),
                               'training_median_before_px':float(np.median(np.linalg.norm(before,axis=1))),
                               'training_median_after_px':float(np.median(np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1))),
                               'optimizer_success':bool(fit.success),'evaluations':int(fit.nfev),'test_observations_used':0}
    # Remove unsupported points rather than letting a low training average hide them.
    rejected=[]; rejected_low_angle=0; retained_angles={}
    for tid,xyz in model['points'].items():
        errors=[]; depths=[]
        for fi,uv in tracks[tid].items():
            if fi not in newposes: continue
            pred,z=project(xyz[None],newposes[fi],K); errors.append(float(np.linalg.norm(pred[0]-uv)));depths.append(z[0])
        angle=maximum_track_angle(xyz,tracks[tid],newposes)
        low_angle=angle<PROTOCOL['point_min_angle_deg']
        if min(depths)<=0 or np.median(errors)>3. or np.percentile(errors,90)>6. or low_angle:
            rejected.append(tid)
            rejected_low_angle+=int(low_angle)
        else:
            retained_angles[tid]=angle
    for tid in rejected: del model['points'][tid]
    model['bundle_adjustment']['rejected_points_after_fit']=len(rejected)
    model['bundle_adjustment']['rejected_low_parallax_points_after_fit']=rejected_low_angle
    model['bundle_adjustment']['retained_point_max_triangulation_angle_median_deg']=(
        float(np.median(list(retained_angles.values()))) if retained_angles else None)
    model['point_max_triangulation_angles_deg']=retained_angles
    return model


def maximum_track_angle(xyz,observations,poses):
    """Largest final ray angle over TRAIN poses which actually observed a point."""
    rays=[]
    for fi in sorted(set(observations)&set(poses)):
        R,t=poses[fi]
        ray=np.asarray(xyz)-(-R.T@t)
        length=float(np.linalg.norm(ray))
        if not np.isfinite(length) or length<=1e-12:
            continue
        rays.append(ray/length)
    if len(rays)<2:
        return 0.
    rays=np.asarray(rays)
    cosine=float(np.min(np.clip(rays@rays.T,-1.,1.)))
    return float(np.degrees(np.arccos(cosine)))


def score_track(tid):
    return int(hashlib.sha256(tid.encode()).hexdigest()[:8],16)%3==0


def match_heldout(frame,training_frames,node_to_track,points):
    votes={}
    for train in training_frames:
        for i,j,distance in mutual_matches(frame,train):
            tid=node_to_track.get((train['index'],j))
            if tid not in points: continue
            votes.setdefault((i,tid),[]).append(distance)
    proposals=sorted([(len(distances),-float(np.median(distances)),i,tid) for (i,tid),distances in votes.items()],reverse=True)
    used_features=set(); used_tracks=set(); result={}
    for count,negdistance,i,tid in proposals:
        if i in used_features or tid in used_tracks: continue
        # At least two train images must independently agree on this descriptor identity.
        if count<2: continue
        used_features.add(i); used_tracks.add(tid);result[tid]=frame['xy'][i].copy()
    return result


def evaluate_heldout(model,tracks,frames,node_to_track,K,observation_provider=None):
    train=[f for f in frames if f['index'] in model['poses']]
    rows=[]; predictions={}; all_errors=[]
    for f in frames:
        if f['split']!='heldout': continue
        obs=(observation_provider(f,train,node_to_track,model['points']) if observation_provider is not None
             else match_heldout(f,train,node_to_track,model['points']))
        anchors={t:uv for t,uv in obs.items() if not score_track(t)}
        scored={t:uv for t,uv in obs.items() if score_track(t)}
        row={'frame_index':f['index'],'timestamp_s':f['timestamp_s'],'matched_3d_tracks':len(obs),
             'pose_anchor_count':len(anchors),'scored_track_count':len(scored),'pose_registered':False,'passes':False,
             'scored_behind_camera_points':0,'scored_nonfinite_error_points':0}
        found=solve_pose(model['points'],anchors,K)
        record={'observations':obs,'roles':{t:'heldout' if score_track(t) else 'pose_anchor' for t in obs},'pose':None}
        if found is not None:
            pose,inliers=found; record['pose']=pose
            scored_tids=sorted(scored)
            errors=np.array([])
            if scored_tids:
                predicted,z=project(np.array([model['points'][t] for t in scored_tids]),pose,K)
                errors=np.linalg.norm(predicted-np.array([scored[t] for t in scored_tids]),axis=1)
                behind=(z<=0)|~np.isfinite(z)
                errors[behind]=np.inf
                row['scored_behind_camera_points']=int(behind.sum())
                row['scored_nonfinite_error_points']=int((~np.isfinite(errors)).sum())
            row.update(pose_registered=True,pose_anchor_inliers=len(inliers),
                       score_median_px=float(np.median(errors)) if len(errors) and np.isfinite(errors).all() else None,
                       score_p90_px=float(np.percentile(errors,90)) if len(errors) and np.isfinite(errors).all() else None,
                       score_coverage_fraction=spread_fraction(np.array(list(scored.values())),f) if scored else 0.)
            row['passes']=bool(len(scored)>=PROTOCOL['heldout_min_score_points_per_frame'] and row['score_median_px'] is not None
                               and row['score_median_px']<=PROTOCOL['heldout_median_px_max'] and row['score_p90_px']<=PROTOCOL['heldout_p90_px_max']
                               and row['score_coverage_fraction']>=.08)
            # Preserve invalid observations in aggregate accounting. Dropping them
            # would make a failed view appear to have excellent pooled accuracy.
            all_errors.extend(float(x) for x in errors)
        predictions[f['index']]=record;rows.append(row)
    return rows,predictions,all_errors


def shape_metrics(points):
    xyz=np.array(list(points.values()))
    if len(xyz)<4: return {'planarity_ratio':None,'linearity_ratio':None}
    singular=np.linalg.svd(xyz-xyz.mean(0),compute_uv=False)
    return {'planarity_ratio':float(singular[2]/max(singular[0],1e-12)),
            'linearity_ratio':float(singular[1]/max(singular[0],1e-12))}


def reconstruct(tracks,frames,K,max_nfev=70,seed_pair=None):
    model,logs=find_seed(tracks,frames,K,seed_pair=seed_pair)
    if model is None: return None,logs
    model=expand_model(model,tracks,frames,K)
    model=bundle_adjust(model,tracks,K,max_nfev)
    return model,logs


def similarity_difference(reference,candidate):
    shared=sorted(set(reference)&set(candidate))
    coverage={'common_points':len(shared),
              'common_reference_fraction':len(shared)/max(len(reference),1),
              'common_candidate_fraction':len(shared)/max(len(candidate),1)}
    if len(shared)<12: return {**coverage,'normalized_rms':None}
    a=np.array([reference[t] for t in shared]);b=np.array([candidate[t] for t in shared])
    aa=a-a.mean(0);bb=b-b.mean(0)
    U,s,Vt=np.linalg.svd(bb.T@aa)
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(U@Vt)
    rotation=U@correction@Vt
    scale=float(np.sum(s*np.diag(correction))/max(np.sum(bb*bb),1e-12))
    fitted=scale*bb@rotation
    radius=float(np.sqrt(np.mean(np.sum(aa*aa,axis=1))))
    rms=float(np.sqrt(np.mean(np.sum((fitted-aa)**2,axis=1))))
    return {**coverage,'normalized_rms':rms/max(radius,1e-12),
            'aligned_scale':scale,'reflection_allowed':False}


def write_ply(path,points):
    lines=['ply','format ascii 1.0',f'element vertex {len(points)}','property float x','property float y','property float z',
           'property uchar red','property uchar green','property uchar blue','end_header']
    for p in points:
        lines.append(' '.join([*(format(float(x),'.9g') for x in p['xyz']),*(str(int(x)) for x in p['rgb'])]))
    Path(path).write_text('\n'.join(lines)+'\n')


def run(program_path,track_id,output,*,root,max_views=None,sensitivity=True,max_nfev=70,correspondence_provider=None,camera_provider=None):
    root=Path(root).resolve();program_path=Path(program_path).resolve();output=Path(output).resolve()
    if camera_provider is not None and sensitivity:
        raise ValueError('Learned camera reconstruction requires sensitivity=False; old assumed-K sensitivity is a separate experiment')
    if not output.is_relative_to(root/'data') or output.exists():
        raise ValueError('Use a fresh output directory inside project data')
    output.mkdir(parents=True)
    program=json.loads(program_path.read_text())
    track=next(t for t in program['tracklets'] if t['id']==track_id)
    cv2.setNumThreads(2);cv2.setRNGSeed(PROTOCOL['random_seed'])
    frames,excluded=select_views(program,track_id,root,max_views)
    protocol=dict(PROTOCOL,max_views=max_views or PROTOCOL['max_views'],max_ba_evaluations=max_nfev,
                  sensitivity_enabled=sensitivity)
    policy={'protocol':protocol,'program_path':str(program_path.relative_to(root)),'program_sha256':digest(program_path),
            'implementation_sha256':digest(__file__),'track_id':track_id,'source':program['source'],
            'input_views':[{k:f[k] for k in ['id','source_index','timestamp_s','image_url','mask_url','image_sha256','mask_sha256','audit_status','subtracted_components','split']} for f in frames],
            'excluded_views':excluded,'new_api_calls':0,'learned_inference_calls':0,'manual_coordinates':False,
            'generated_meshes_read':False,'depth_maps_read':False,'old_poses_read':False}
    if correspondence_provider is not None:
        policy['correspondence_provider']=correspondence_provider.describe(frames)
        policy['learned_inference_calls']=None
        policy['learned_inference_accounting']='tracking.json records actual learned 2D correspondence calls; no learned shape/depth'
    camera_declaration=None
    if camera_provider is not None:
        camera_declaration=json.loads(json.dumps(camera_provider.describe(),allow_nan=False))
        if not isinstance(camera_declaration,dict):
            raise ValueError('Camera provider must declare its frozen policy before inference')
        for flag in ['heldout_images_used','depth_or_point_maps_used','camera_calibrated',
                     'physical_scale_known','physical_accuracy_verified','downloads_allowed']:
            if camera_declaration.get(flag) is not False:
                raise ValueError('Camera provider declaration exceeds its local TRAIN-only scope: '+flag)
        if camera_declaration.get('new_api_calls')!=0:
            raise ValueError('Camera provider may not make paid API calls')
        policy['camera_provider']={'declaration':camera_declaration,
            'result_record_path':str((output/'camera-assumption.json').relative_to(root)),
            'estimation_precedes_geometry':True,'heldout_images_supplied_to_provider':False}
        policy['learned_camera_prior_used']=True
        policy['learned_inference_calls']=None
        policy['learned_inference_accounting']='camera-prior/camera-prior.json records actual camera calls; tracking.json separately records optional correspondence calls'
    (output/'protocol.json').write_text(json.dumps(policy,indent=2)+'\n')
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output/'implementation.py').chmod(0o444)
    if correspondence_provider is not None:
        correspondence_provider.snapshot(output)
    # Source dimensions and the TRAIN camera receipt are fixed before any new
    # features, tracks, geometry, BA or held-out pose estimates are constructed.
    for f in frames:
        if f['image'].shape[:2]!=frames[0]['image'].shape[:2]: raise ValueError('Mixed source dimensions are unsupported')
    width,height=(frames[0]['image'].shape[1],frames[0]['image'].shape[0]) if frames else (0,0)
    K=camera_matrix(width,height)
    camera_assumption=None
    if camera_provider is not None:
        K,camera_assumption=learned_camera_assumption(camera_provider,frames,output,root,camera_declaration,
            {'case_id':program.get('case_id'),'source':program['source'],'track_id':track_id,'program_path':policy['program_path'],
             'program_sha256':policy['program_sha256']})
    print(json.dumps({'phase':'features','track':track_id,'selected_views':len(frames)}),flush=True)
    if correspondence_provider is None:
        for f in frames: extract_features(f)
    tracks,node_to_track,pairs,conflicts=(build_tracks(frames) if correspondence_provider is None
                                       else correspondence_provider.build_tracks(frames,output))
    print(json.dumps({'phase':'tracks','track_count':len(tracks),'pair_count':len(pairs),'conflicts':conflicts}),flush=True)
    model,seed_logs=reconstruct(tracks,frames,K,max_nfev) if len(frames)>=3 else (None,[])
    heldout=[];predictions={};errors=[]; sensitivity_rows=[]
    if model:
        print(json.dumps({'phase':'main_geometry','points':len(model['points']),'train_views':len(model['poses'])}),flush=True)
        heldout,predictions,errors=evaluate_heldout(model,tracks,frames,node_to_track,K,
            observation_provider=correspondence_provider.match_heldout if correspondence_provider is not None else None)
        if sensitivity:
            for factor in PROTOCOL['sensitivity_focal_factors']:
                print(json.dumps({'phase':'sensitivity','focal_factor':factor}),flush=True)
                seed_pair=(model['seed']['a'],model['seed']['b'])
                alternative,_=reconstruct(tracks,frames,camera_matrix(width,height,factor),max_nfev,seed_pair=seed_pair)
                row={'focal_factor':factor,'recovered':alternative is not None,'requested_training_seed_pair':list(seed_pair)}
                if alternative:
                    row.update(similarity_difference(model['points'],alternative['points']))
                    row['points']=len(alternative['points']);row['seed']=alternative['seed']
                sensitivity_rows.append(row)
    points=[]; cameras=[]; frame_rows=[]
    shapes=shape_metrics(model['points']) if model else shape_metrics({})
    scored_pass=sum(int(h['passes']) for h in heldout)
    sensitivity_stable=bool(sensitivity_rows) and all(
        r.get('normalized_rms') is not None and r['normalized_rms']<=.15
        and r.get('common_reference_fraction',0)>=PROTOCOL['sensitivity_min_common_reference_fraction']
        and r.get('common_candidate_fraction',0)>=PROTOCOL['sensitivity_min_common_candidate_fraction']
        for r in sensitivity_rows)
    if model is None:
        status='insufficient_correspondence' if not seed_logs else 'planar_or_low_parallax_ambiguous'
        reason='No training-only two-view initialization passed correspondence, cheirality and parallax gates.'
    elif shapes['linearity_ratio'] is not None and shapes['linearity_ratio']<PROTOCOL['linearity_ratio_min']:
        status='linear_or_degenerate_sparse_candidate'
        reason='Recovered points are predominantly collinear; a two-dimensional surface or object volume is not established.'
    elif scored_pass<PROTOCOL['heldout_min_frames']:
        status='sparse_multiview_candidate'
        reason='Sparse geometry recovered, but too few independently scored held-out views passed to validate it.'
    elif not sensitivity_stable:
        status='intrinsics_sensitive_sparse_candidate'
        reason='Held-out projections pass, but recovered shape is sensitive to assumed camera intrinsics or lacks comparable sensitivity geometry.'
    elif shapes['planarity_ratio'] is not None and shapes['planarity_ratio']<.04:
        status='validated_sparse_surface_patch'
        reason='Independent held-out projections support a predominantly planar observed patch; thickness and a complete object are not established.'
    elif model['seed'].get('homography_dominant',False):
        status='homography_dominant_sparse_candidate'
        reason='Held-out projections pass, but a homography explains most seed correspondences; non-planar shape remains ambiguous.'
    else:
        status='validated_sparse_multiview_candidate'
        reason='Sparse observed geometry passes provisional held-out and focal-sensitivity gates; this is not a calibrated scan or complete mesh.'
    if model:
        # A passing aggregate fit does not validate every point. Preserve which
        # points were actually independently scored, excluding all pose anchors.
        scored_support={tid:[] for tid in model['points']}
        passing_frames={row['frame_index'] for row in heldout if row['passes']}
        for fi,record in predictions.items():
            if fi not in passing_frames or record['pose'] is None:
                continue
            for tid,xy in record['observations'].items():
                if tid not in model['points'] or record['roles'].get(tid)!='heldout':
                    continue
                uv,z=project(model['points'][tid][None],record['pose'],K)
                error=float(np.linalg.norm(uv[0]-xy))
                if z[0]>0 and np.isfinite(error) and error<=PROTOCOL['heldout_median_px_max']:
                    scored_support[tid].append({'frame_id':frames[fi]['id'],
                                                'timestamp_s':frames[fi]['timestamp_s'],
                                                'error_px':error})
        for tid,xyz in model['points'].items():
            fi=next(fi for fi in sorted(tracks[tid]) if fi in model['poses'])
            u,v=np.rint(tracks[tid][fi]).astype(int);im=frames[fi]['image']
            rgb=im[np.clip(v,0,im.shape[0]-1),np.clip(u,0,im.shape[1]-1),::-1].tolist()
            points.append({'id':tid,'xyz':xyz.tolist(),'rgb':rgb,
                           'validated':False,
                           'validation_scope':'no_individual_point_ground_truth_available',
                           'heldout_scored_projection_support':scored_support[tid],
                           'heldout_scored_projection_supported_in_at_least_two_views':len(scored_support[tid])>=2,
                           'max_training_triangulation_angle_deg':model.get('point_max_triangulation_angles_deg',{}).get(tid)})
        for fi,pose in sorted(model['poses'].items()):
            cameras.append({'frame_id':frames[fi]['id'],'R':pose[0].tolist(),'t':pose[1].tolist(),'K':K.tolist(),'split':'train'})
    for f in frames:
        fi=f['index']; pose=model['poses'].get(fi) if model else None;observations=[]
        if f['split']=='train':
            obs={t:o[fi] for t,o in tracks.items() if fi in o}
            roles={t:'train' for t in obs}
        else:
            info=predictions.get(fi,{});obs=info.get('observations',{});roles=info.get('roles',{});pose=info.get('pose')
            if pose is not None:
                cameras.append({'frame_id':f['id'],'R':pose[0].tolist(),'t':pose[1].tolist(),'K':K.tolist(),'split':'heldout_anchor_pose'})
        for tid,xy in obs.items():
            predicted=None;error=None
            if pose is not None and model and tid in model['points']:
                uv,z=project(model['points'][tid][None],pose,K)
                predicted=uv[0].tolist();error=float(np.linalg.norm(uv[0]-xy))
            observations.append({'track_id':tid,'xy':xy.tolist(),'predicted_xy':predicted,'role':roles[tid],'error_px':error})
        # Unregistered views still expose actual extracted keypoints, never fabricated 3D.
        if not observations:
            observations=[{'track_id':f'feature_{j:04d}','xy':xy.tolist(),'predicted_xy':None,'role':f['split'],'error_px':None} for j,xy in enumerate(f['xy'])]
        frame_rows.append({'id':f['id'],'timestamp_s':f['timestamp_s'],'image_url':f['image_url'],'mask_url':f['mask_url'],
                           'width':width,'height':height,'registered':pose is not None,'split':f['split'],
                           'feature_count':len(f['xy']),'feature_filter':f['feature_filter'],
                           'audit_status':f['audit_status'],'observations':observations})
    finite_errors=[x for x in errors if np.isfinite(x)]
    invalid_error_count=len(errors)-len(finite_errors)
    metrics={'selected_views':len(frames),'training_views':sum(f['split']=='train' for f in frames),
             'heldout_views':sum(f['split']=='heldout' for f in frames),'feature_tracks':len(tracks),
             'recovered_points':len(points),'registered_train_views':len(model['poses']) if model else 0,
             'heldout_views_passing':scored_pass,
             'aggregate_sparse_projection_gates_passed':status in {'validated_sparse_surface_patch','validated_sparse_multiview_candidate'},
             'points_independently_scored_in_at_least_two_heldout_views':sum(
                 int(p['heldout_scored_projection_supported_in_at_least_two_views']) for p in points),
             'heldout_scored_observations':len(errors),
             'heldout_invalid_or_behind_camera_observations':invalid_error_count,
             'heldout_median_px':float(np.median(errors)) if errors and not invalid_error_count else None,
             'heldout_p90_px':float(np.percentile(errors,90)) if errors and not invalid_error_count else None,
             'heldout_finite_error_median_px':float(np.median(finite_errors)) if finite_errors else None,
             'heldout_finite_error_p90_px':float(np.percentile(finite_errors,90)) if finite_errors else None,
             'final_training_point_max_angle_median_deg':model['bundle_adjustment'].get('retained_point_max_triangulation_angle_median_deg') if model else None,
             'initialization_homography_dominant':model['seed'].get('homography_dominant',False) if model else None,
             'focal_sensitivity_stable':sensitivity_stable,'focal_px':float(K[0,0]),**shapes}
    limitations=['This is a sparse observed-geometry feasibility experiment, not a complete mesh.',
                 'Rigid component identity and SAM mask correctness remain fallible model assumptions.',
                 'Camera intrinsics are assumed, not calibrated; original source coordinates are preserved.',
                 'Each component has independent arbitrary scale and coordinates; no metric assembly or insertion is established.',
                 'Hidden surfaces, cavities, collision geometry and contact are not reconstructed.',
                 'Held-out views use disjoint pose-anchor tracks; scored observations never enter geometry, seed/focal selection or BA.',
                 'Training reprojection fit and test projections are internal consistency evidence, not ground-truth shape accuracy.']
    if model and model['seed'].get('homography_dominant',False):
        limitations.append('A homography explains most seed correspondences; the initialization may be planar or geometrically degenerate even when projections fit.')
    if shapes['linearity_ratio'] is not None and shapes['linearity_ratio']<PROTOCOL['linearity_ratio_min']:
        limitations.append('The recovered point distribution is predominantly collinear and does not establish a surface.')
    result={'format':'observed-geometry-sparse-experiment','version':'0.1','id':output.name,'title':track['label']+' · source-pixel geometry',
            'status':status,'reason':reason,'metrics':metrics,'frames':frame_rows,'points':points,'cameras':cameras,'limitations':limitations,
            'seed':model['seed'] if model else None,'bundle_adjustment':model.get('bundle_adjustment') if model else None,
            'heldout_evaluation':heldout,'focal_sensitivity':sensitivity_rows,'seed_diagnostics':seed_logs,
            'pair_diagnostics':pairs,'track_merge_conflicts_rejected':conflicts,'new_api_calls':0,
            'prior_geometry_or_depth_used':False,'input_protocol_sha256':digest(output/'protocol.json')}
    if camera_assumption is not None:
        result['camera_assumption']=camera_assumption
        result['learned_camera_prior_used']=True
        result['limitations'].append('Camera intrinsics are a learned TRAIN-image assumption with zero distortion, not measured calibration; no learned depth, points or poses enter reconstruction.')
    (output/'viewer.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_ply(output/'points.ply',points)
    # Save raw source point tracks and resulting geometry for independent rechecks.
    raw={'tracks':{t:{str(fi):uv.tolist() for fi,uv in obs.items()} for t,obs in tracks.items()},
         'points':{t:xyz.tolist() for t,xyz in model['points'].items()} if model else {},
         'training_poses':{str(fi):{'R':p[0].tolist(),'t':p[1].tolist()} for fi,p in model['poses'].items()} if model else {}}
    (output/'geometry.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
    if correspondence_provider is not None:
        correspondence_provider.finalize(output,result)
    print(json.dumps({'phase':'complete','status':status,'metrics':metrics}),flush=True)
    return result
