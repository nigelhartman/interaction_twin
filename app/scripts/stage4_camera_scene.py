#!/usr/bin/env python3
"""Disclosed video-proportioned camera shell and image-evidenced pose candidates.

This is a small local geometry baseline, not a learned complete reconstruction.
All dimensions use the same nominal battery-diameter gauge as the battery pilot.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_render import render_mesh

OUT = ROOT / 'data/camera-battery-full-scene'
SCALE = 0.019257229585835953
DIMENSIONS = np.array([6.6, 4.1, 2.05])
BAY_CENTRES = [-2.25, -1.12]
BAY_RADIUS = .515
BAY_FLOOR_Y = -1.79
SILVER = np.array([184, 197, 200, 255], np.uint8)

# Original 960x540 source pixels.  These are assistant image annotations, not
# generated model results or a human-labelled accuracy set.  Some occluded
# rectangular corners are estimated; each resulting pose remains a candidate.
# Face order is in image quadrilateral order unless the view is upside down.
ANCHORS = [
    (1, 'front', [[371,132],[679,133],[662,295],[380,296]], 'clear front face; rounded corners approximated'),
    (5, 'front', [[291,143],[619,97],[674,181],[275,225]], 'strongly foreshortened; left corner partly hand-occluded'),
    (6, 'back_up', [[167,132],[486,86],[550,235],[131,288]], 'rear display face upside down, partly occluded'),
    (9, 'front', [[154,113],[516,112],[522,314],[146,314]], 'front face, left boundary partly thumb-occluded'),
    (11, 'front', [[170,167],[534,138],[545,365],[176,393]], 'front face'),
    (13, 'front', [[134,227],[431,206],[499,318],[90,354]], 'front face tilted, lower-left edge thumb-occluded'),
    (14, 'bottom_reverse', [[531,249],[99,273],[104,388],[540,355]], 'bottom view reversed; visible closed cover included as surface evidence'),
    (16, 'front', [[217,110],[560,62],[604,275],[248,332]], 'front face'),
    (17, 'front', [[195,171],[541,222],[493,409],[178,352]], 'front face and top edge'),
    (18, 'front', [[67,88],[343,89],[271,369],[30,326]], 'front face narrow angle; left edge partly occluded'),
    (21, 'back', [[166,-8],[550,-45],[596,94],[160,160]], 'motion-blurred rear; upper corners outside frame are estimated'),
    (23, 'bottom', [[189,221],[574,219],[591,325],[191,333]], 'bottom body face, right upper corner under finger'),
    (27, 'front_up', [[169,189],[548,179],[560,402],[196,414]], 'front face upside down; edges partly hand-occluded'),
    (29, 'bottom', [[137,218],[558,214],[571,322],[140,334]], 'bottom face, right side partly hand-occluded'),
    (35, 'bottom', [[164,235],[555,218],[568,323],[169,341]], 'open bay and body edges; cover excluded from corners'),
    (36, 'front_up', [[198,211],[541,218],[538,362],[201,370]], 'front face upside down with open cover above'),
    (38, 'back', [[154,76],[492,15],[553,166],[178,254]], 'rear face; top right partly occluded'),
    (40, 'bottom', [[80,179],[411,57],[448,156],[110,279]], 'withdrawal; full bottom body footprint excluding cover'),
    (44, 'bottom', [[98,177],[431,79],[458,176],[129,282]], 'empty bay; lower left under thumb'),
    (46, 'bottom', [[97,168],[420,82],[442,177],[126,269]], 'empty bay; lower left under thumb'),
    (48, 'bottom', [[24,156],[356,30],[398,130],[57,254]], 'first approach; camera footprint excluding open cover'),
    (49, 'bottom', [[26,114],[377,96],[404,198],[31,219]], 'insertion; upper right under hand, approximate'),
    (52, 'bottom', [[58,198],[401,151],[417,233],[69,294]], 'one visible end; left corners behind thumb, approximate'),
    (55, 'bottom', [[48,199],[369,123],[399,231],[66,306]], 'cover being closed; right corner approximate'),
    (56, 'front_up', [[40,201],[391,191],[392,319],[44,329]], 'front upside down; left corners behind hand'),
    (60, 'front_up', [[149,96],[449,99],[449,328],[145,320]], 'front upside down; right half under thumb'),
    (68, 'front_up', [[187,170],[551,202],[529,374],[196,350]], 'front upside down; approximate partly occluded body edges'),
    (70, 'back', [[236,179],[585,151],[628,376],[251,409]], 'rear face; rounded corners and hand occlusion'),
    (75, 'back', [[234,146],[607,112],[633,352],[262,393]], 'rear face; bottom left partly under thumb'),
]


def corners():
    w, h, d = DIMENSIONS/2
    return np.array([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],
                     [-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]])


FACE_IDS = {'front':[0,1,2,3], 'front_up':[2,3,0,1],
            'back':[5,4,7,6], 'back_up':[7,6,5,4], 'bottom':[6,7,3,2],
            'bottom_reverse':[7,6,2,3]}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def shell():
    """Closed body with two actual open wells; no masking trick creates holes."""
    w, h, d = DIMENSIONS / 2
    outside = np.array([[-w, -d], [w, -d], [w, d], [-w, d]])
    theta = np.arange(40) * 2 * np.pi / 40
    circles = [np.c_[x + BAY_RADIUS*np.cos(theta), BAY_RADIUS*np.sin(theta)] for x in BAY_CENTRES]
    xz = np.concatenate([outside, *circles])
    triangles = Delaunay(xz).simplices
    keep = np.ones(len(triangles), bool)
    for n in range(len(BAY_CENTRES)):
        keep &= ~np.all((triangles >= 4+n*40) & (triangles < 4+(n+1)*40), axis=1)
    triangles = triangles[keep]
    vertices = np.c_[xz[:, 0], np.full(len(xz), h), xz[:, 1]].tolist()
    faces = triangles.tolist()
    colors = [SILVER.tolist() for _ in vertices]
    top_ids = []
    for x, z in outside:
        top_ids.append(len(vertices)); vertices.append([x, -h, z]); colors.append(SILVER.tolist())
    faces += [[top_ids[0], top_ids[2], top_ids[1]], [top_ids[0], top_ids[3], top_ids[2]]]
    for j in range(4):
        k = (j+1) % 4
        faces += [[j, k, top_ids[k]], [j, top_ids[k], top_ids[j]]]
    for n, ring in enumerate(circles):
        ids = np.arange(4+n*40, 4+(n+1)*40)
        floor = []
        for x, z in ring:
            floor.append(len(vertices)); vertices.append([x, BAY_FLOOR_Y, z]); colors.append([46, 55, 62, 255])
        centre = len(vertices)
        vertices.append([BAY_CENTRES[n], BAY_FLOOR_Y, 0]); colors.append([36, 43, 50, 255])
        for j in range(40):
            k = (j+1) % 40
            faces += [[int(ids[j]), floor[j], floor[k]], [int(ids[j]), floor[k], int(ids[k])], [floor[j], centre, floor[k]]]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    # Process merges only duplicate vertices; colors are uniform on the main shell.
    mesh.visual.vertex_colors = np.tile(SILVER, (len(mesh.vertices), 1))
    mesh.visual.vertex_colors[np.isclose(mesh.vertices[:, 1], BAY_FLOOR_Y)] = [76, 87, 91, 255]
    return mesh


def disk(radius, height, centre, color, sections=48):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation(centre)
    mesh.visual.vertex_colors = color
    return mesh


def asset():
    directory = OUT / 'geometry/camera-v1'
    directory.mkdir(parents=True, exist_ok=True)
    body = shell()
    # Lens and rear display are conservative visual cues; dimensions are priors.
    lens_back = disk(1.23, .18, [.72, -.03, -1.115], [95, 106, 113, 255])
    lens_ring = disk(1.08, .12, [.72, -.03, -1.265], [170, 183, 188, 255])
    lens_glass = disk(.88, .025, [.72, -.03, -1.34], [27, 45, 52, 255])
    display = trimesh.creation.box(extents=[3.25, 2.75, .025])
    display.apply_translation([.8, .15, 1.045]); display.visual.vertex_colors = [35, 49, 56, 255]
    mesh = trimesh.util.concatenate([body, lens_back, lens_ring, lens_glass, display])
    for extension in ['glb', 'ply', 'obj']:
        mesh.export(directory / f'camera.{extension}')
    meta = {
        'object_id': 'camera_00001', 'status': 'video_proportioned_shape_prior_candidate',
        'method': 'assistant-reviewed compact-camera proportions; explicit shell, lens and double-AA bay shape priors',
        'source_sha256': '42bc62511fe63ac7cbe6c0e32f017d89b349706be6447c23f41ad9795ad0f43f',
        'asset_units': 'one asset unit equals the fitted battery diameter; arbitrary nominal gauge, not verified metres',
        'asset_scale': SCALE, 'scale_constant_for_all_frames': True,
        'body_dimensions_asset_units': DIMENSIONS.tolist(),
        'body_dimensions_nominal_model_units': (DIMENSIONS*SCALE).tolist(),
        'canonical_axes': '+X camera-body horizontal; +Y toward bottom battery opening; front face at -Z',
        'battery_bay': {'mouth_plane_y': 2.05, 'well_centres_xz': [[x, 0] for x in BAY_CENTRES],
                        'radius': BAY_RADIUS, 'depth': 2.05-BAY_FLOOR_Y,
                        'inward_axis': [0, -1, 0], 'outward_axis': [0, 1, 0],
                        'status': 'inferred hidden circular channels; approximate visible opening; not measured interior'},
        'evidence': ['Source views at 0 s, 13 s, 34 s, 39–43 s, 47–53 s and 69–74 s.',
                     'Two cylindrical cells and double opening constrain approximate bay layout.',
                     'Body dimensions chosen in the existing battery-diameter gauge; no independent metric calibration.'],
        'limitations': ['Hidden walls, well depth, lens details and display are explicit modelling priors.',
                        'Separate compartment cover not included in this rigid body mesh.',
                        'Not SAM 3D, photogrammetry, a complete scan, or physics-verified geometry.',
                        'The shell has real open wells but unobserved internals must not be presented as measured.'],
        'mesh': {'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'body_watertight': bool(body.is_watertight),
                 'all_parts_watertight': bool(mesh.is_watertight), 'sha256': hashlib.sha256((directory/'camera.ply').read_bytes()).hexdigest()},
    }
    write(directory/'metadata.json', meta)
    print(json.dumps({'asset': str(directory), **meta['mesh']}))
    return mesh, meta


def plane_pose(xyz, uv, k):
    """Two planar PnP hypotheses are retained and ranked by image error only."""
    result = cv2.solvePnPGeneric(np.ascontiguousarray(xyz*SCALE), np.ascontiguousarray(uv), k, None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    seeds = list(zip(result[1], result[2]))
    # OpenCV's direct homography initializer supplies an independent seed.  In
    # particular it avoids a poor IPPE local minimum for offset rear planes.
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(xyz*SCALE), np.ascontiguousarray(uv), k, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        seeds.append((rv, tv))
    for rv, tv in seeds:
        rv, tv = cv2.solvePnPRefineLM(np.ascontiguousarray(xyz*SCALE), np.ascontiguousarray(uv), k, None, rv.copy(), tv.copy())
        r = cv2.Rodrigues(rv)[0]; t = tv.ravel()
        if np.any((corners()*SCALE @ r.T + t)[:,2] <= 0):
            continue
        projected = cv2.projectPoints(xyz*SCALE, rv, tv, k, None)[0][:,0]
        errors = np.linalg.norm(projected-uv, axis=1)
        candidates.append((float(np.mean(errors)), r, t, errors))
    if not candidates:
        raise ValueError('No positive-depth planar pose')
    candidates.sort(key=lambda x:x[0])
    return candidates[0], candidates


def corner_projection(xyz, r, t, k):
    p = (xyz*SCALE @ r.T + t) @ k.T
    return p[:,:2]/p[:,2:]


def sift_transfer(source, target, polygon):
    """Track local rigid-face texture with an explicit projective correspondence fit."""
    mask = np.zeros(source.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((11,11),np.uint8))
    detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.02)
    kp1, des1 = detector.detectAndCompute(source, mask)
    kp2, des2 = detector.detectAndCompute(target, None)
    if des1 is None or des2 is None:
        return None, {'reason':'no_sift_features'}
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance < .72*b.distance]
    if len(good)<10:
        return None, {'reason':'fewer_than_ten_ratio_matches','matches':len(good)}
    a = np.float32([kp1[m.queryIdx].pt for m in good]); b = np.float32([kp2[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(20261028)
    h, inliers = cv2.findHomography(a,b,cv2.RANSAC,3.0,maxIters=2000,confidence=.995)
    if h is None or inliers.sum()<10:
        return None, {'reason':'fewer_than_ten_homography_inliers','matches':len(good),'inliers':int(inliers.sum()) if inliers is not None else 0}
    mapped = cv2.perspectiveTransform(np.float64(polygon)[None],h)[0]
    area_before=abs(cv2.contourArea(np.float32(polygon))); area_after=abs(cv2.contourArea(np.float32(mapped)))
    if not .45 < area_after/max(1,area_before) < 2.3 or not np.isfinite(mapped).all():
        return None, {'reason':'implausible_local_homography_area','inliers':int(inliers.sum())}
    prediction = cv2.perspectiveTransform(a[None],h)[0]
    errors = np.linalg.norm(prediction-b,axis=1)[inliers[:,0].astype(bool)]
    return mapped, {'method':'SIFT ratio .72; projective RANSAC 3px on assistant face polygon',
                    'matches':len(good),'inliers':int(inliers.sum()),'median_homography_residual_px':float(np.median(errors)),
                    'limits':'Texture correspondences are conditional on a planar face and may include hand/cover features.'}


def poses(mesh, metadata):
    index = read(OUT/'frames/index.json')
    original = read(ROOT/'data/camera-battery/frames/index.json')['frames']
    pilot = read(ROOT/'data/camera-battery-withdrawal-15fps/alignment/battery-pilot-v2/alignment.json')
    k = np.array(pilot['camera']['intrinsics'])
    point_cloud = corners()
    anchors = []
    for sample, face, uv, note in ANCHORS:
        row = original[sample-1]; uv = np.asarray(uv,np.float64)
        (error,r,t,errs), candidates = plane_pose(point_cloud[FACE_IDS[face]],uv,k)
        anchors.append({'sample_index':sample,'timestamp_s':row['timestamp_s'],'face':face,'uv':uv,
                        'rotation':r,'translation':t,'corner_errors':errs,'note':note,
                        'alternative_errors':[x[0] for x in candidates],
                        'image':cv2.imread(str(ROOT/'data/camera-battery'/row['file']),cv2.IMREAD_GRAYSCALE)})
    times=np.array([a['timestamp_s'] for a in anchors])
    slerp=Slerp(times,Rotation.from_matrix([a['rotation'] for a in anchors]))
    records=[]
    for frame in index['frames']:
        stamp=float(frame['timestamp_s']); nearest=int(np.argmin(abs(times-stamp)));a=anchors[nearest]
        pose_time=float(np.clip(stamp,times[0],times[-1]));r=slerp([pose_time]).as_matrix()[0]
        t=np.array([np.interp(pose_time,times,[a['translation'][j] for a in anchors]) for j in range(3)])
        status='interpolated_candidate';reasons=['No independent 6D ground truth; body proportions and intrinsics uncalibrated.', 'Pose interpolated between assistant-defined image-evidenced keyframes.']
        evidence={'nearest_annotation_timestamp_s':float(times[nearest]),'nearest_annotation_distance_s':float(abs(times[nearest]-stamp))}
        if abs(times[nearest]-stamp)<1e-5:
            r=a['rotation'];t=a['translation'];status='manual_keyframe_candidate'
            evidence.update({'face':a['face'],'corner_reprojection_px':a['corner_errors'].tolist(),'corner_median_px':float(np.median(a['corner_errors'])),'annotation_note':a['note']})
            reasons=[reasons[0],'Assistant-defined four-corner fit; reprojection is fitting evidence, not held-out accuracy.']
        elif abs(times[nearest]-stamp)<1.6:
            path=OUT/frame['file']
            target=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if target is not None:
                mapped,transfer=sift_transfer(a['image'],target,a['uv'])
                evidence['transfer']=transfer
                if mapped is not None:
                    try:
                        (error, rr, tt, residual), alternatives=plane_pose(point_cloud[FACE_IDS[a['face']]],mapped,k)
                        angle=Rotation.from_matrix(rr @ a['rotation'].T).magnitude()
                        if error<6 and angle<.8 and np.linalg.norm(tt-a['translation'])<.12:
                            r,t=rr,tt;status='feature_tracked_candidate'
                            reasons=[reasons[0],'Planar SIFT homography propagated assistant corner geometry; not automatic full-shape reconstruction.']
                            evidence.update({'corner_median_px':float(np.median(residual)),'corner_reprojection_px':residual.tolist(),'face':a['face']})
                    except (ValueError,cv2.error):
                        evidence['transfer']['pose_failure']=True
        if status=='manual_keyframe_candidate' and evidence['corner_median_px']>8:
            reasons.append('Corner reprojection above 8px; weak candidate only.')
            status='unsupported_candidate'
        records.append({'timestamp_s':stamp,'sample_index':frame.get('sample_index',frame.get('index')),
                        'object_id':'camera_00001','rotation':r.tolist(),'translation':t.tolist(),
                        'asset_scale':SCALE,'status':status,'reasons':reasons,'fit_evidence':evidence})
    result={'schema_version':'0.4','object_id':'camera_00001','asset':'geometry/camera-v1/camera.ply',
            'asset_scale':SCALE,'scale_constant_for_all_frames':True,'camera':pilot['camera'],
            'pose_convention':'camera_xyz = asset_scale * canonical_xyz @ rotation.T + translation',
            'status':'camera_relative_approximation_candidates',
            'method':'Assistant corner annotations, planar IPPE/LM, local SIFT homography where supported, otherwise explicitly flagged SLERP/translation interpolation.',
            'limitations':['Geometry, nominal scale, hidden bay interior and camera calibration are unverified.',
                           'Planar face pose has depth/orientation ambiguities; this is not ground-truth 6D tracking.',
                           'Four fitted corners do not provide independent accuracy evidence.',
                           'Interpolated poses are visual continuity candidates and can be wrong during occlusion or fast turns.',
                           'Camera and batteries are fitted separately; no invisible seating, contact constraints or shared physical calibration is asserted.'],
            'summary':{status:sum(f['status']==status for f in records) for status in sorted(set(f['status'] for f in records))},
            'frames':records}
    write(OUT/'camera-poses-v1.json',result)
    annotations=[]
    for a in anchors:
        annotations.append({'source_case':'camera-battery','sample_index':a['sample_index'],'timestamp_s':a['timestamp_s'],
                            'face':a['face'],'corner_ids':FACE_IDS[a['face']], 'corners_xy':a['uv'].tolist(),'note':a['note'],
                            'rotation':a['rotation'].tolist(),'translation':a['translation'].tolist(),
                            'corner_reprojection_px':a['corner_errors'].tolist(),'alternative_planar_mean_errors_px':a['alternative_errors']})
    write(OUT/'geometry/camera-v1/annotations.json',{'origin':'assistant_visual_annotation','human_confirmed':False,'ground_truth':False,'annotations':annotations})
    review_camera(mesh, result, anchors, k)
    metadata['review']={'status':'assistant_reviewed_all_29_source_and_full_projection_keyframes',
                        'contact_sheets':[f'pose-review-{j:02d}.jpg' for j in range(1,5)],
                        'notes':['Visible corners and bay position give useful reference candidates.',
                                 'Full projected body sometimes extends beyond the observed silhouette; the independent camera shape/depth is approximate.',
                                 'No image mask clips away mesh projection errors.',
                                 'Source hands remain foreground in the video; this camera baseline has no new hand-depth ordering.'],
                        'pose_summary':result['summary']}
    write(OUT/'geometry/camera-v1/metadata.json',metadata)
    print(json.dumps({'poses':str(OUT/'camera-poses-v1.json'),'frames':len(records),'summary':result['summary']}))
    return result,anchors


def review_camera(mesh, report, anchors, k):
    """Keep full projections inspectable; never clip camera mesh to SAM masks."""
    for sheet in range((len(anchors)+7)//8):
        canvas=Image.new('RGB',(1280,840),'#eff2f3');draw=ImageDraw.Draw(canvas)
        for j,a in enumerate(anchors[sheet*8:(sheet+1)*8]):
            sample=a['sample_index'];rgb=np.array(Image.open(ROOT/f'data/camera-battery/frames/{sample:06d}.jpg'))
            rendered=render_mesh(mesh.vertices,mesh.faces,k,a['rotation'],a['translation'],960,540,scale=SCALE,vertex_colors=np.array(mesh.visual.vertex_colors))
            overlay=rgb.copy();mask=rendered['mask'];overlay[mask]=(overlay[mask]*.64+rendered['rgb'][mask]*.36).astype(np.uint8)
            for x in BAY_CENTRES:
                theta=np.arange(60)*2*np.pi/60
                v=np.c_[x+BAY_RADIUS*np.cos(theta),np.full(60,DIMENSIONS[1]/2),BAY_RADIUS*np.sin(theta)]
                uv=corner_projection(v,a['rotation'],a['translation'],k)
                cv2.polylines(overlay,[np.rint(uv).astype(np.int32)],True,(255,70,20),2)
            x=(j%2)*640;y=(j//2)*210
            canvas.paste(Image.fromarray(rgb).resize((320,180)),(x,y+30))
            canvas.paste(Image.fromarray(overlay).resize((320,180)),(x+320,y+30))
            draw.text((x+6,y+5),f"{a['timestamp_s']:.3f}s | {a['face']} | fitted corner median {np.median(a['corner_errors']):.1f}px",fill='black')
        canvas.save(OUT/f'geometry/camera-v1/pose-review-{sheet+1:02d}.jpg',quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-only', action='store_true')
    args = parser.parse_args()
    mesh,metadata=asset()
    if not args.asset_only:
        poses(mesh,metadata)


if __name__ == '__main__':
    main()
