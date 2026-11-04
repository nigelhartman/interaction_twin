#!/usr/bin/env python3
"""Full-video battery observations: reused SAM, new dense SAM and fixed-scale poses.

No pose is extended into an unobserved interval. New tracklet poses are explicitly
2D-supported shape-prior candidates, rather than physically measured trajectories.
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
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_fit import solve_pose
from alignment_render import ray_mesh_points, render_mesh
from scripts.stage3_tracks import track_cotracker
from scripts.stage4_align import outline, sample_field, project


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def stats(values):
    a = np.asarray(values)
    return {'median': float(np.median(a)), 'p90': float(np.percentile(a, 90))} if len(a) else {'median': None, 'p90': None}


def cylinder_reference(mask, vertices, k, scale, length):
    """Initialize from visible-mask PCA. Out-of-plane direction remains a prior."""
    y, x = np.where(mask)
    uv = np.c_[x, y].astype(float)
    center = np.median(uv, axis=0)
    _, _, basis = np.linalg.svd(uv - center, full_matrices=False)
    axis2 = basis[0]
    if axis2[0] < 0:
        axis2 *= -1
    local = (uv - center) @ basis.T
    width = np.percentile(local[:, 1], 97) - np.percentile(local[:, 1], 3)
    span = np.percentile(local[:, 0], 99) - np.percentile(local[:, 0], 1)
    z = k[0, 0] * scale / width
    sine = float(np.clip((span - .15 * width) / (length * width), .35, .995))
    axis = np.r_[axis2 * sine, np.sqrt(1 - sine * sine)]
    t = np.array([(center[0] - k[0, 2]) * z / k[0, 0], (center[1] - k[1, 2]) * z / k[1, 1], z])
    front = -t
    front -= axis * np.dot(front, axis)
    front /= np.linalg.norm(front)
    side = np.cross(axis, front)
    rotation = np.stack([front, side, axis], axis=1)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 5)
    initial = np.r_[cv2.Rodrigues(rotation)[0].ravel(), t]
    def residual(p):
        r = cv2.Rodrigues(p[:3])[0]
        contour = outline(vertices, k, r, p[3:], scale)
        # This is a bounded reference silhouette proposal, not independent truth.
        return np.r_[sample_field(distance, contour), (p[:3] - initial[:3]) / .4,
                     (p[5] - initial[5]) / .05]
    lo = initial - np.r_[np.full(3, .55), .035, .035, min(.09, z * .3)]
    hi = initial + np.r_[np.full(3, .55), .035, .035, min(.09, z * .3)]
    fitted = least_squares(residual, initial, bounds=(lo, hi), loss='soft_l1', f_scale=3, diff_step=1e-3, max_nfev=120)
    return cv2.Rodrigues(fitted.x[:3])[0], fitted.x[3:], {'method': 'visible-mask PCA and bounded silhouette fit with circular cylinder prior', 'pca_width_px': float(width), 'pca_length_px': float(span), 'axis_depth_sign_inferred': True, 'metric_depth_measured': False, 'optimizer_success': bool(fitted.success)}


TRACKLETS = [
    {'id': 'battery_track_00002', 'object_id': 'battery_00002', 'case': 'camera-battery-action-5fps', 'run': 'components-v2', 'track': 'out_right', 'anchor': 31},
    {'id': 'battery_track_00003', 'object_id': None, 'case': 'camera-battery-action-5fps', 'run': 'components-v2', 'track': 'table_front', 'anchor': 60},
    {'id': 'battery_track_00004', 'object_id': None, 'case': 'camera-battery-action-5fps', 'run': 'components-v2', 'track': 'table_back', 'anchor': 60},
    {'id': 'battery_track_00007', 'object_id': None, 'case': 'camera-battery-insertion-10fps', 'run': 'battery-v1', 'track': 'in_first', 'anchor': 6},
    {'id': 'battery_track_00008', 'object_id': None, 'case': 'camera-battery-second-insertion-10fps', 'run': 'battery-v1', 'track': 'in_second', 'anchor': 9},
]


def tracklet(spec, out, mesh, pilot, scale, k, device):
    case = ROOT / 'data' / spec['case']
    sam = case / 'segmentation' / spec['run']
    observations = read(sam / 'tracks' / (spec['track'] + '.json'))['frames']
    allrows = read(case / 'frames/index.json')['frames']
    rows = [{**allrows[v['sample_index'] - 1], 'image': allrows[v['sample_index'] - 1]['file']} for v in observations]
    masks = [np.array(Image.open(sam / v['raw_mask_file'])) > 0 for v in observations]
    anchor = next(j for j, r in enumerate(rows) if r['index'] == spec['anchor'])
    dest = out / 'battery-inference-v1' / spec['id']
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / 'tracks.npz'
    if not cache.exists():
        gray = cv2.imread(str(case / rows[anchor]['file']), cv2.IMREAD_GRAYSCALE)
        interior = cv2.erode(masks[anchor].astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
        points = cv2.goodFeaturesToTrack(gray, maxCorners=64, qualityLevel=.002, minDistance=4, mask=interior, blockSize=5)
        if points is None or len(points) < 12:
            y, x = np.where(interior)
            points = np.c_[x, y][np.linspace(0, len(x) - 1, min(64, len(x))).astype(int)][:, None, :].astype(np.float32)
        uv, valid, fb, extra, details = track_cotracker(case, rows, masks, anchor, points[:, 0], device)
        np.savez_compressed(cache, uv=uv, valid=valid, **extra)
        write(dest / 'tracking-metadata.json', {'provider': 'Meta CoTracker3 offline', 'device': device, 'source_case': spec['case'], 'source_track': spec['track'], 'mask_run': spec['run'], 'source_sam_metadata': str((sam / 'metadata.json').relative_to(ROOT)), 'reused_sam': spec['run'] == 'components-v2', 'anchor_sample_index': spec['anchor'], 'seed_count': len(points), 'accuracy_measured': False, **details})
    track = np.load(cache)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if spec['track'] == 'out_right':
        old = pilot['frames'][pilot['reference_sample_index'] - 1]
        rotation = np.array(old['rotation'])
        translation = np.array(old['translation'])
        y, x = np.where(masks[anchor]); target_uv = np.c_[x, y]
        # Both upper end caps are visible; the second body's lower extent is
        # hidden by the hand. Align its visible cap, never the occluded-mask
        # centroid, while retaining the pair's disclosed parallel-axis prior.
        axis = rotation[:2, 2] / np.linalg.norm(rotation[:2, 2])
        side = np.array([-axis[1], axis[0]])
        projected = project(vertices, k, rotation, translation, scale)
        delta = axis * (np.percentile(target_uv @ axis, 1) - np.percentile(projected @ axis, 1))
        delta += side * (np.median(target_uv @ side) - np.median(projected @ side))
        translation[:2] += delta * translation[2] / np.array([k[0, 0], k[1, 1]])
        reference = {'method': 'parallel outward-pair prior; visible upper end and lateral-mask alignment, excluding occluded lower extent', 'metric_depth_measured': False, 'pair_relation_verified': False}
    else:
        rotation, translation, reference = cylinder_reference(masks[anchor], vertices, k, scale, float(mesh.extents[2]))
    canonical, hits = ray_mesh_points(track['raw_model_uv'][anchor], vertices, faces, k, rotation, translation, scale=scale)
    hits &= track['valid'][anchor]
    canonical[~hits] = np.nan
    ids = np.arange(len(canonical))
    train = ids[hits & (ids % 3 != 0)]
    held = ids[hits & (ids % 3 == 0)]
    records = []
    for j, row in enumerate(rows):
        present = train[track['valid'][j, train]]
        base = {'timestamp_s': row['timestamp_s'], 'source_case': spec['case'], 'source_sample_index': row['index'], 'observation_id': spec['id'], 'object_id': spec['object_id'], 'candidate_object_ids': [spec['object_id']] if spec['object_id'] else ['battery_00001', 'battery_00002'], 'asset_scale': scale, 'status': 'unsupported', 'reasons': []}
        fit = solve_pose(canonical[present] * scale, track['uv'][j, present], k, rotation, translation)
        if not fit['success']:
            records.append({**base, 'reasons': [fit['reason']], 'train_visible': len(present)})
            continue
        r, t = fit['rotation'], fit['translation']
        checks = held[track['valid'][j, held]]
        error = np.linalg.norm(project(canonical[checks], k, r, t, scale) - track['uv'][j, checks], axis=1)
        held_stats = stats(error)
        rendered = render_mesh(vertices, faces, k, r, t, 960, 540, scale=scale)
        raw, target = rendered['mask'], masks[j]
        coverage = float((raw & target).sum() / max(1, target.sum()))
        overlap = float((raw & target).sum() / max(1, (raw | target).sum()))
        reasons = []
        if observations[j]['status'] != 'candidate_visible': reasons.append('SAM observation withheld due to occlusion or ambiguity')
        if len(checks) < 5: reasons.append('fewer than five disjoint held-out correspondences')
        if held_stats['median'] is None or held_stats['median'] > 4 or held_stats['p90'] > 10: reasons.append('internal held-out reprojection exceeds 4px median or10pxp90')
        if len(fit['inlier_indices']) < 10: reasons.append('fewer than ten training inliers')
        if coverage < .65: reasons.append('projected mesh covers less than65percent of visible SAM')
        if overlap < .25: reasons.append('full projection IoU below .25; large occlusion/shape disagreement')
        if spec['track'] == 'table_front': reasons.append('front-table reference cropped at bottom edge; full extent/depth not established')
        # Very short blurred second approach is never promoted to a full pose.
        if spec['track'] == 'in_second': reasons.append('second approach too occluded/blurred to establish a full rigid cylinder pose')
        base.update(rotation=r.tolist(), translation=t.tolist(), train_visible=len(present), train_inliers=len(fit['inlier_indices']), heldout_visible=len(checks), heldout_reprojection_px=held_stats, visible_mask_coverage=coverage, full_projection_iou=overlap, depth_supported=False, uncertainty='2D-only shape-prior fit; absolute depth, tilt sign and axial roll unverified', reasons=reasons, status='unsupported' if reasons else ('reference_initialization' if j == anchor else 'supported_candidate'))
        records.append(base)
    np.savez_compressed(dest / 'canonical.npz', points=canonical, train_ids=train, heldout_ids=held, reference_hits=hits)
    result = {**spec, 'observation_id': spec['id'], 'identity_status': 'candidate_continuity_within_tracklet' if spec['object_id'] else 'physical_identity_across_hand_occlusion_unresolved', 'candidate_object_ids': [spec['object_id']] if spec['object_id'] else ['battery_00001', 'battery_00002'], 'reference': reference, 'reference_ray_hit_count': int(hits.sum()), 'reference_train_count': len(train), 'reference_heldout_count': len(held), 'scale_constant': True, 'geometry_is_same_size_cylinder_prior': True, 'frames': records}
    write(dest / 'poses.json', result)
    # Contact sheet preserves raw projections: do not mask away fitting errors.
    tiles = []
    for j in np.linspace(0, len(rows) - 1, min(12, len(rows))).astype(int):
        row, rec = rows[j], records[j]
        rgb = np.array(Image.open(case / row['file']).convert('RGB'))
        if 'rotation' in rec:
            rr = render_mesh(vertices, faces, k, np.array(rec['rotation']), np.array(rec['translation']), 960, 540, scale=scale)
            rgb[rr['mask']] = (rgb[rr['mask']] * .6 + np.array([20, 220, 195]) * .4).astype(np.uint8)
            contours, _ = cv2.findContours(rr['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 210, 40), 2)
        tile = Image.fromarray(rgb).resize((384, 216)); canvas = Image.new('RGB', (384, 245), '#eef4f5'); canvas.paste(tile)
        ImageDraw.Draw(canvas).text((8, 221), f"{row['timestamp_s']:.3f}s {rec['status']}", fill='#12333c')
        tiles.append(canvas)
    sheet = Image.new('RGB', (384 * 3, 245 * int(np.ceil(len(tiles) / 3))), '#eef4f5')
    for j, tile in enumerate(tiles): sheet.paste(tile, (384 * (j % 3), 245 * (j // 3)))
    sheet.save(dest / 'review.jpg', quality=92)
    print(json.dumps({'observation_id': spec['id'], 'frames': len(records), 'supported': sum(r['status'] != 'unsupported' for r in records), 'reference_ray_hits': int(hits.sum())}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    p.add_argument('--only')
    args = p.parse_args()
    pilot_case = ROOT / 'data/camera-battery-withdrawal-15fps'
    pilot = read(pilot_case / 'alignment/battery-pilot-v2/alignment.json')
    out = ROOT / 'data/camera-battery-full-scene'
    out.mkdir(exist_ok=True)
    asset = pilot_case / 'geometry/battery-fitted-v1/battery.ply'
    mesh = trimesh.load(asset, force='mesh')
    k, scale = np.array(pilot['camera']['intrinsics']), pilot['asset_scale']
    tracklets = []
    original = []
    for old in pilot['frames']:
        original.append({**old, 'source_case': pilot_case.name, 'source_sample_index': old['sample_index'], 'observation_id': 'battery_track_00001', 'object_id': 'battery_00001', 'candidate_object_ids': ['battery_00001'], 'source_pilot_status': old['status'], 'status': 'supported_candidate' if old['status'] == 'supported_pilot' else old['status'], 'pose_source': 'preserved_battery-pilot-v2', 'support_type': 'prior depth-gated pilot; internal consistency only'})
    tracklets.append({'observation_id': 'battery_track_00001', 'object_id': 'battery_00001', 'candidate_object_ids': ['battery_00001'], 'source_track': 'out_left', 'frames': original})
    for spec in TRACKLETS:
        if args.only and spec['id'] != args.only:
            prior = out / 'battery-inference-v1' / spec['id'] / 'poses.json'
            if prior.exists(): tracklets.append(read(prior))
            continue
        tracklets.append(tracklet(spec, out, mesh, pilot, scale, k, args.device))
    allframes = [f for t in tracklets for f in t['frames']]
    result = {'schema_version': '0.5', 'status': 'candidate_outputs_with_explicit_unsupported_intervals', 'source_sha256': pilot['source_sha256'], 'camera': {**pilot['camera'], 'width': 960, 'height': 540}, 'asset_path': str(asset.relative_to(ROOT)), 'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(), 'asset_scale': scale, 'scale_constant_for_all_frames': True, 'physical_scale_verified': False, 'pose_convention': pilot['pose_convention'], 'full_video_duration_s': 74.240833, 'unknown_intervals_policy': 'No interpolation across unsupported or unobserved spans; hidden seating is unknown.', 'new_models': 'SAM2.1Tiny MPS dense insertion22+22 samples; MetaCoTracker3 MPS reduced256x384 input', 'geometry_status': 'same video-proportioned cylinder prior reused for both similar batteries; neither a measured scan nor SAM3D output', 'limitations': ['Persistent physical identity is unresolved after hand/table occlusion; later observation IDs retain both candidate batteries.', '2D-only new tracklet fits do not measure true depth, tilt sign, axial roll, camera extrinsics or physical units.', 'Disjoint held-out point IDs from the same CoTracker run measure internal agreement, not independently labelled accuracy.', 'Reference ray assignments depend on an approximate inferred shape and pose.', 'No invented trajectories, fully seated state or unseen full second-insertion pose.'], 'tracklets': tracklets, 'frames': sorted(allframes, key=lambda r: (r['timestamp_s'], r['observation_id']))}
    result['summary'] = {'total_pose_observations': len(allframes), 'supported_or_reference_observations': sum(r['status'] != 'unsupported' for r in allframes), 'unsupported_observations': sum(r['status'] == 'unsupported' for r in allframes), 'dense_insertion_sam_masks': 44, 'full_second_insertion_recovered': False}
    write(out / 'battery-poses-v1.json', result)


if __name__ == '__main__':
    main()
