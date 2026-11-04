#!/usr/bin/env python3
"""Battery-only fixed-scale camera-relative alignment, with preserved fit evidence."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_render import render_mesh, ray_mesh_points
from alignment_fit import solve_pose


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def project(vertices, k, rotation, translation, scale):
    xyz = (vertices * scale) @ rotation.T + translation
    image = xyz @ k.T
    return image[:, :2] / image[:, 2:]


def outline(vertices, k, rotation, translation, scale, count=160):
    hull = cv2.convexHull(project(vertices, k, rotation, translation, scale).astype(np.float32))[:, 0]
    delta = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distance = np.arange(count) * cumulative[-1] / count
    edges = np.minimum(np.searchsorted(cumulative, distance, side='right') - 1, len(hull) - 1)
    return hull[edges] + delta[edges] * ((distance - cumulative[edges]) / lengths[edges])[:, None]


def sample_field(field, uv):
    return cv2.remap(field.astype(np.float32), uv[:, 0].astype(np.float32)[None], uv[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=100).ravel()


def silhouette_metrics(raw, target, occluder):
    visible = raw & ~occluder
    def iou(a, b):
        return float((a & b).sum() / max(1, (a | b).sum()))
    return {
        'full_projection_iou_with_visible_sam': iou(raw, target),
        'occlusion_composite_iou_with_visible_sam': iou(visible, target),
        'visible_target_coverage_by_full_mesh': float((raw & target).sum() / max(1, target.sum())),
        'projection_outside_target_and_foreground_fraction': float((raw & ~target & ~occluder).sum() / max(1, raw.sum())),
        'foreground_occluded_fraction': float((raw & occluder).sum() / max(1, raw.sum())),
    }


def refine_reference(vertices, faces, k, scale, rotation, translation, target, hand):
    """Bounded outline proposal. Never treats a thumb notch as a mesh boundary."""
    boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    distances = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 5)
    hand_margin = cv2.dilate(hand.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(np.float32)
    initial = np.r_[cv2.Rodrigues(rotation)[0].ravel(), translation]
    def residual(p):
        r = cv2.Rodrigues(p[:3])[0]
        uv = outline(vertices, k, r, p[3:], scale)
        free = 1 - np.clip(sample_field(hand_margin, uv), 0, 1)
        return np.r_[sample_field(distances, uv) * free,
                     (p[:3] - initial[:3]) / .12,
                     (p[5] - initial[5]) / .025]
    result = least_squares(residual, initial, bounds=(initial - np.r_[np.full(3, .25), .012, .012, .045], initial + np.r_[np.full(3, .25), .012, .012, .045]), loss='soft_l1', f_scale=2, diff_step=1e-3, max_nfev=100)
    candidate_r = cv2.Rodrigues(result.x[:3])[0]
    raw_initial = render_mesh(vertices, faces, k, rotation, translation, 960, 540, scale=scale)['mask']
    raw_candidate = render_mesh(vertices, faces, k, candidate_r, result.x[3:], 960, 540, scale=scale)['mask']
    before = silhouette_metrics(raw_initial, target, hand)
    after = silhouette_metrics(raw_candidate, target, hand)
    accepted = after['occlusion_composite_iou_with_visible_sam'] > before['occlusion_composite_iou_with_visible_sam'] and after['visible_target_coverage_by_full_mesh'] >= before['visible_target_coverage_by_full_mesh'] - .015
    return (candidate_r, result.x[3:]) if accepted else (rotation, translation), {
        'method': 'bounded robust outline fit outside dilated foreground mask; weak orientation/depth priors',
        'proposal_accepted': bool(accepted), 'optimizer_success': bool(result.success),
        'initial': before, 'proposal': after,
        'initial_rotation': rotation.tolist(), 'initial_translation': translation.tolist(),
        'fitted_rotation': (candidate_r if accepted else rotation).tolist(),
        'fitted_translation': (result.x[3:] if accepted else translation).tolist(),
        'note': 'Reference silhouette is fitting evidence, not independent pose ground truth. Axial roll remains uncertain.'}


def quantiles(values):
    values = np.asarray(values)
    return {'median': float(np.median(values)), 'p90': float(np.percentile(values, 90))} if len(values) else {'median': None, 'p90': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='battery-pilot-v1')
    args = parser.parse_args()
    started = time.time()
    case = ROOT / 'data/camera-battery-withdrawal-15fps'
    geometry = case / 'geometry'
    parent = geometry / 'local-battery-v3'
    asset = geometry / 'battery-fitted-v1'
    hands = case / 'occlusion/hands-v1'
    out = case / 'alignment' / args.run
    if out.exists():
        raise ValueError('Run already exists; preserve evidence and choose a fresh run name')
    pack = read(parent / 'experiment.json')
    metadata = read(asset / 'metadata.json')
    hand_metadata = read(hands / 'metadata.json')
    tracks = np.load(parent / 'motion-cotracker/tracks.npz')
    mesh = trimesh.load(asset / 'battery.ply', force='mesh')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    k = np.array(pack['camera']['intrinsics'])
    scale = float(metadata['estimated_model_units_per_asset_unit'])
    coarse = metadata['reference_camera_transform_for_appearance_only']
    rotation, translation = np.array(coarse['rotation']), np.array(coarse['translation'])
    anchor = 15
    row = pack['samples'][anchor]
    target = np.array(Image.open(parent / row['mask'])) > 0
    hand = np.array(Image.open(hands / 'foreground_masks' / '000016.png')) > 0
    (rotation, translation), reference = refine_reference(vertices, faces, k, scale, rotation, translation, target, hand)
    canonical, ray_valid = ray_mesh_points(tracks['raw_model_uv'][anchor], vertices, faces, k, rotation, translation, scale=scale)
    ray_valid &= tracks['valid'][anchor]
    canonical[~ray_valid] = np.nan
    ids = np.arange(len(canonical))
    train_ids = ids[ray_valid & (ids % 3 != 0)]
    heldout_ids = ids[ray_valid & (ids % 3 == 0)]
    if len(train_ids) < 12 or len(heldout_ids) < 6:
        raise ValueError('Insufficient ray-hit reference correspondences for disjoint fitting and checking')
    out.mkdir(parents=True)
    for directory in ['full_masks', 'visible_masks', 'mesh_rgb', 'depth']:
        (out / directory).mkdir()
    records = []
    rotations, translations = [], []
    for j, row in enumerate(pack['samples']):
        i = row['sample_index']
        present = train_ids[tracks['valid'][j, train_ids]]
        fit = solve_pose(canonical[present] * scale, tracks['uv'][j, present], k, rotation, translation)
        if not fit['success']:
            records.append({'sample_index': i, 'timestamp_s': row['timestamp_s'], 'status': 'unsupported', 'reason': 'pose solver failed', 'train_visible': len(present)})
            rotations.append(np.full((3, 3), np.nan)); translations.append(np.full(3, np.nan))
            continue
        r, t = fit['rotation'], fit['translation']
        rotations.append(r); translations.append(t)
        checks = heldout_ids[tracks['valid'][j, heldout_ids]]
        errors = np.linalg.norm(project(canonical[checks], k, r, t, scale) - tracks['uv'][j, checks], axis=1)
        held = quantiles(errors)
        render = render_mesh(vertices, faces, k, r, t, 960, 540, scale=scale, vertex_colors=np.asarray(mesh.visual.vertex_colors)[:, :3])
        target = np.array(Image.open(parent / row['mask'])) > 0
        hand = np.array(Image.open(hands / 'foreground_masks' / f'{i:06d}.png')) > 0
        metrics = silhouette_metrics(render['mask'], target, hand)
        depth_supported = bool(tracks['xyz_usable'][j].any())
        reasons = []
        if not depth_supported: reasons.append('source depth rejected; motion blur')
        if len(checks) < 6: reasons.append('too few held-out visible correspondences')
        if held['median'] is None or held['median'] > 3 or held['p90'] > 8: reasons.append('held-out reprojection exceeds 3px median or 8px p90')
        if len(fit['inlier_indices']) < 12: reasons.append('fewer than 12 training inliers')
        if metrics['occlusion_composite_iou_with_visible_sam'] < .55: reasons.append('occlusion-composite silhouette agreement below .55')
        status = 'unsupported' if reasons else ('reference_initialization' if j == anchor else 'supported_pilot')
        Image.fromarray(render['mask'].astype(np.uint8) * 255).save(out / 'full_masks' / f'{i:06d}.png')
        Image.fromarray((render['mask'] & ~hand).astype(np.uint8) * 255).save(out / 'visible_masks' / f'{i:06d}.png')
        Image.fromarray(render['rgb']).save(out / 'mesh_rgb' / f'{i:06d}.png')
        np.save(out / 'depth' / f'{i:06d}.npy', render['depth'])
        records.append({'sample_index': i, 'timestamp_s': row['timestamp_s'], 'status': status, 'reasons': reasons,
                        'rotation': r.tolist(), 'translation': t.tolist(), 'asset_scale': scale,
                        'train_visible': len(present), 'train_inliers': len(fit['inlier_indices']),
                        'train_reprojection_px': quantiles(fit['residual_px'][fit['inlier_indices']]),
                        'heldout_visible': len(checks), 'heldout_reprojection_px': held,
                        'depth_supported': depth_supported, **metrics})
    np.savez_compressed(out / 'poses.npz', rotations=np.array(rotations), translations=np.array(translations),
                        asset_scale=np.array(scale), canonical_points=canonical, reference_ray_valid=ray_valid,
                        train_point_ids=train_ids, heldout_point_ids=heldout_ids,
                        sample_indices=tracks['sample_indices'], timestamps_s=tracks['timestamps_s'])
    comparable = [r for r in records if r.get('heldout_reprojection_px', {}).get('median') is not None and r['sample_index'] not in [16, 52]]
    result = {
        'schema_version': '0.4', 'stage': 4, 'scope': 'battery-only alignment pilot',
        'source_sha256': pack['source_sha256'], 'candidate_entity_ids': pack['candidate_entity_ids'],
        'status': 'generated_pending_visual_review', 'source_geometry': '../../geometry/battery-fitted-v1/battery.ply',
        'asset_sha256': hashlib.sha256((asset / 'battery.ply').read_bytes()).hexdigest(),
        'tracks_sha256': hashlib.sha256((parent / 'motion-cotracker/tracks.npz').read_bytes()).hexdigest(),
        'camera': pack['camera'], 'pose_convention': 'camera_xyz = asset_scale * canonical_xyz @ rotation.T + translation',
        'asset_scale': scale, 'scale_constant_for_all_frames': True, 'physical_scale_verified': False,
        'reference_sample_index': 16, 'reference_fit': reference,
        'canonical_correspondence_method': 'nearest positive ray/triangle intersection at fitted reference; misses excluded, never clamped',
        'reference_ray_hits': int(ray_valid.sum()), 'excluded_reference_point_ids': ids[~ray_valid].tolist(),
        'train_point_ids': train_ids.tolist(), 'heldout_point_ids': heldout_ids.tolist(),
        'solver': 'OpenCV EPNP RANSAC (4 px), iterative LM on training inliers; fixed canonical points and scale',
        'support_gate': {'minimum_train_inliers': 12, 'minimum_heldout_visible': 6, 'max_heldout_median_px': 3, 'max_heldout_p90_px': 8, 'minimum_occlusion_composite_iou': .55, 'rejected_source_depth_withheld': True},
        'occlusion': {'run': '../../occlusion/hands-v1', 'method': hand_metadata.get('foreground_mask_meaning'),
                      'metadata_sha256': hashlib.sha256((hands / 'metadata.json').read_bytes()).hexdigest(),
                      'annotations_sha256': hashlib.sha256((hands / 'annotations.json').read_bytes()).hexdigest(),
                      'foreground_masks_sha256': {f'{r["sample_index"]:06d}': hashlib.sha256((hands / 'foreground_masks' / f'{r["sample_index"]:06d}.png').read_bytes()).hexdigest() for r in pack['samples']},
                      'raw_projection_preserved': True, 'target_mask_clipping': False},
        'summary': {'supported_nonreference_frames': sum(r['status'] == 'supported_pilot' for r in records),
                    'unsupported_samples': [r['sample_index'] for r in records if r['status'] == 'unsupported'],
                    'nonreference_nonblurred_median_of_heldout_medians_px': float(np.median([r['heldout_reprojection_px']['median'] for r in comparable])),
                    'median_occlusion_composite_iou': float(np.median([r['occlusion_composite_iou_with_visible_sam'] for r in comparable]))},
        'evaluation_limits': ['Disjoint point IDs come from the same CoTracker run; this is an internal consistency check, not independently labelled accuracy.',
                              'Reference reprojection is tautological after ray assignment and is excluded from aggregate error.',
                              'Cylinder shape, estimated camera and nominal scale can yield plausible projections without correct 6D pose.',
                              'Axial roll, hidden geometry, exact terminals, world-camera extrinsics and physical size remain unverified.',
                              'Foreground masks and segmentation are assistant-reviewed candidates; ordering and mask boundaries remain uncertain.',
                              'Pose candidates for rejected sample 52 are preserved for inspection, never promoted as supported 3D.'],
        'runtime_seconds': time.time() - started, 'frames': records}
    write(out / 'alignment.json', result)
    print(json.dumps({'output': str(out), **result['summary'], 'reference': reference}, indent=2))


if __name__ == '__main__':
    main()
