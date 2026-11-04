"""Diagnostic, source-coloured stereo surfaces from saved train-only SfM.

This module performs no model inference and reads no generated object, old
alignment pose or predicted depth. Its output inherits the input calibration
uncertainty; photometric agreement is not ground-truth shape validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROTOCOL = {
    'version': 'observed-stereo-v1',
    'pair_min_common_points': 40,
    'pair_min_angle_deg': 2.0,
    'pair_max_angle_deg': 15.0,
    'pair_max_rectified_y_p90_px': 2.0,
    'mask_core_fraction': 0.15,
    'mask_core_min_px': 3.0,
    'patch_radius_px': 3,
    'minimum_patch_std': 5.0,
    'pair_min_zncc': 0.80,
    'extra_min_zncc': 0.80,
    'extra_depth_alternative_factors': [0.8, 0.9, 0.95, 1.05, 1.1, 1.2],
    'extra_alternative_min_pixel_shift': 2.0,
    'extra_min_depth_discrimination_margin': 0.025,
    'extra_min_usable_alternatives': 2,
    'minimum_extra_train_views': 2,
    'maximum_extra_train_views_tested': 6,
    'left_right_max_error_px': 1.0,
    'minimum_disparity_px': 1.0,
    'sgbm_block_size': 5,
    'sgbm_uniqueness_ratio': 15,
    'sgbm_disparity_margin_px': 16,
    'candidate_lattice_stride_px': 2,
    'max_dense_candidates': 20000,
    'training_depth_quantiles': [2, 98],
    'training_depth_range_padding_fraction': 0.25,
    'zbuffer_relative_depth_tolerance': 0.02,
    'mesh_max_edge_in_lattice_pixels': 3.5,
    'mesh_max_relative_depth_jump': 0.02,
    'camera_calibrated': False,
    'physical_scale_known': False,
    'heldout_images_read': False,
    'mesh_completion': False,
    'new_api_calls': 0,
    'learned_inference_calls': 0,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(xyz, pose, K):
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    camera = xyz @ pose[0].T + np.asarray(pose[1]).reshape(1, 3)
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def unproject(pixels, z, pose, K):
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    camera = rays * np.asarray(z).reshape(-1, 1)
    return (camera - np.asarray(pose[1]).reshape(1, 3)) @ pose[0]


def mask_core(mask):
    distance = cv2.distanceTransform((np.asarray(mask) > 0).astype(np.uint8), cv2.DIST_L2, 5)
    margin = max(PROTOCOL['mask_core_min_px'], float(distance.max()) * PROTOCOL['mask_core_fraction'])
    return distance >= margin


def mask_at(mask, uv):
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    safe = np.where(np.isfinite(uv), uv, -100)
    xy = np.rint(safe).astype(np.int64)
    h, w = mask.shape
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    accepted = np.zeros(len(uv), bool)
    accepted[inside] = mask[xy[inside, 1], xy[inside, 0]] > 0
    return accepted


def left_right_consistency(left, right, max_error=1.0):
    """Right disparity is x_right - x_left, hence negative for valid positive L."""
    left = np.asarray(left, np.float32)
    right = np.asarray(right, np.float32)
    h, w = left.shape
    yy, xx = np.indices((h, w))
    xr = xx.astype(np.float32) - left
    sampled = cv2.remap(right, xr, yy.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=-1e5)
    return (np.isfinite(left) & np.isfinite(sampled) & (left > 0)
            & (sampled < 0) & (xr >= 0) & (xr < w - 1)
            & (np.abs(left + sampled) <= max_error))


def zncc(a, b, min_std=5.0):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size != b.size or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    a = a - a.mean()
    b = b - b.mean()
    if min(float(a.std()), float(b.std())) < min_std:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sample_patch(image, mask, pixels):
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    h, w = image.shape[:2]
    if (not np.isfinite(pixels).all() or not mask_at(mask, pixels).all()
            or (pixels[:, 0] < 1).any() or (pixels[:, 0] >= w - 2).any()
            or (pixels[:, 1] < 1).any() or (pixels[:, 1] >= h - 2).any()):
        return None
    return cv2.remap(image, pixels[:, 0].astype(np.float32)[None, :],
                     pixels[:, 1].astype(np.float32)[None, :], cv2.INTER_LINEAR)[0]


def plane_patch_points(xyz, reference_pose, K, radius=3):
    """A local reference-frontoparallel patch; it never fills missing surfaces."""
    uv, z = project(np.asarray(xyz)[None], reference_pose, K)
    offsets = np.array([(x, y) for y in range(-radius, radius + 1)
                         for x in range(-radius, radius + 1)], np.float64)
    return unproject(uv + offsets, np.full(len(offsets), z[0]), reference_pose, K)


def photometric_check(xyz, reference, other, K, *, discriminate_depth=False):
    """Check actual source pixels; optional depth competition in another TRAIN view."""
    uv, z = project(np.asarray(xyz)[None], reference['pose'], K)
    ouv, oz = project(np.asarray(xyz)[None], other['pose'], K)
    if z[0] <= 0 or oz[0] <= 0 or not mask_at(reference['core'], uv)[0] or not mask_at(other['core'], ouv)[0]:
        return {'passes': False, 'reason': 'outside_mask_core_or_behind_camera'}
    patch_xyz = plane_patch_points(xyz, reference['pose'], K, PROTOCOL['patch_radius_px'])
    ref_pixels, _ = project(patch_xyz, reference['pose'], K)
    other_pixels, other_z = project(patch_xyz, other['pose'], K)
    a = sample_patch(reference['gray'], reference['core'], ref_pixels)
    b = sample_patch(other['gray'], other['core'], other_pixels) if (other_z > 0).all() else None
    score = zncc(a, b, PROTOCOL['minimum_patch_std']) if a is not None and b is not None else None
    threshold = PROTOCOL['extra_min_zncc'] if discriminate_depth else PROTOCOL['pair_min_zncc']
    if score is None or score < threshold:
        return {'passes': False, 'reason': 'patch_untextured_masked_or_inconsistent', 'zncc': score}
    if not discriminate_depth:
        return {'passes': True, 'reason': 'pair_photo_consistent', 'zncc': score}
    alternatives = []
    for factor in PROTOCOL['extra_depth_alternative_factors']:
        alternative = unproject(uv, np.array([z[0] * factor]), reference['pose'], K)[0]
        auv, az = project(alternative[None], other['pose'], K)
        shift = float(np.linalg.norm(auv[0] - ouv[0]))
        if az[0] <= 0 or shift < PROTOCOL['extra_alternative_min_pixel_shift']:
            continue
        alternative_patch = plane_patch_points(alternative, reference['pose'], K, PROTOCOL['patch_radius_px'])
        pixels, depths = project(alternative_patch, other['pose'], K)
        candidate = sample_patch(other['gray'], other['core'], pixels) if (depths > 0).all() else None
        value = zncc(a, candidate, PROTOCOL['minimum_patch_std']) if candidate is not None else None
        if value is not None:
            alternatives.append({'depth_factor': factor, 'pixel_shift': shift, 'zncc': value})
    if len(alternatives) < PROTOCOL['extra_min_usable_alternatives']:
        return {'passes': False, 'reason': 'insufficient_depth_discrimination', 'zncc': score,
                'alternative_count': len(alternatives)}
    margin = score - max(a['zncc'] for a in alternatives)
    return {'passes': bool(margin >= PROTOCOL['extra_min_depth_discrimination_margin']),
            'reason': 'extra_view_depth_discrimination', 'zncc': score,
            'depth_discrimination_margin': margin, 'alternative_count': len(alternatives)}


def _local(root, url):
    if not url.startswith('/files/data/'):
        raise ValueError('Only cited local data inputs are permitted')
    path = (root / url.removeprefix('/files/')).resolve()
    if not path.is_relative_to(root / 'data'):
        raise ValueError('Input escaped project data')
    return path


def _input_protocol(root, sparse_dir, protocol):
    """Verify either original SfM provenance or its attested focal refinement."""
    if 'source_directory' not in protocol:
        if sha256(sparse_dir / 'implementation.py') != protocol['implementation_sha256']:
            raise ValueError('Saved sparse implementation hash mismatch')
        return protocol
    if protocol.get('protocol', {}).get('version') not in {'observed-shared-focal-v1', 'observed-shared-focal-v2'}:
        raise ValueError('Unsupported calibration provenance contract')
    source = (root / protocol['source_directory']).resolve()
    if not source.is_relative_to(root / 'data'):
        raise ValueError('Calibration source escaped local project data')
    for name, expected in protocol['source_files'].items():
        if name not in {'protocol.json', 'viewer.json', 'geometry.json', 'implementation.py'} or sha256(source / name) != expected:
            raise ValueError('Calibration original sparse artifact mismatch')
    for name, expected in protocol['code_files'].items():
        if name not in {'observed_calibration.py', 'refine_observed_camera.py', 'observed_geometry.py'} or sha256(sparse_dir / name) != expected:
            raise ValueError('Calibration frozen code snapshot mismatch')
    for flag in ['read_generated_mesh', 'read_predicted_depth', 'read_manual_pose']:
        if protocol.get(flag) is not False:
            raise ValueError('Calibration forbidden input exclusion is not established')
    original = json.loads((source / 'protocol.json').read_text())
    if protocol['inputs'] != original['input_views']:
        raise ValueError('Calibration inputs differ from source pixels and train/held-out split')
    program_path = (root / protocol['source_program']).resolve()
    if not program_path.is_relative_to(root / 'data') or sha256(program_path) != protocol['source_program_sha256']:
        raise ValueError('Calibration source program hash mismatch')
    return _input_protocol(root, source, original)


def load_sparse(root, sparse_dir):
    sparse_dir = Path(sparse_dir).resolve()
    if not sparse_dir.is_relative_to(root / 'data'):
        raise ValueError('Sparse input must be inside project data')
    protocol = json.loads((sparse_dir / 'protocol.json').read_text())
    viewer = json.loads((sparse_dir / 'viewer.json').read_text())
    geometry = json.loads((sparse_dir / 'geometry.json').read_text())
    if sha256(sparse_dir / 'protocol.json') != viewer['input_protocol_sha256']:
        raise ValueError('Sparse viewer/protocol hash mismatch')
    original_protocol = _input_protocol(root, sparse_dir, protocol)
    for flag in ['manual_coordinates', 'generated_meshes_read', 'depth_maps_read', 'old_poses_read']:
        if original_protocol.get(flag) is not False:
            raise ValueError('Sparse input cannot establish forbidden-input exclusion: ' + flag)
    if protocol.get('prior_geometry_or_depth_used') or viewer.get('prior_geometry_or_depth_used'):
        raise ValueError('Prior geometry or depth is forbidden')
    viewer_points = {p['id']: np.array(p['xyz']) for p in viewer['points']}
    if set(viewer_points) != set(geometry['points']) or any(
            not np.array_equal(viewer_points[t], np.array(x)) for t, x in geometry['points'].items()):
        raise ValueError('Saved geometry differs from sparse review points')
    frames = []
    manifest = {r['id']: r for r in original_protocol['input_views']}
    for index, f in enumerate(viewer['frames']):
        if str(index) not in geometry['training_poses']:
            continue
        row = manifest[f['id']]
        if f['split'] != 'train' or row['split'] != 'train':
            raise ValueError('Training pose unexpectedly cites held-out data')
        im = _local(root, f['image_url'])
        mask_path = _local(root, f['mask_url'])
        if sha256(im) != row['image_sha256'] or sha256(mask_path) != row['mask_sha256']:
            raise ValueError('Source image/mask hash mismatch')
        image = cv2.imread(str(im), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or mask.shape != image.shape[:2]:
            raise ValueError('Unreadable or mismatched source image/mask')
        saved_pose = geometry['training_poses'][str(index)]
        pose = (np.array(saved_pose['R'], np.float64), np.array(saved_pose['t'], np.float64))
        camera = next(c for c in viewer['cameras'] if c['frame_id'] == f['id'] and c['split'] == 'train')
        if (pose[0].shape != (3, 3) or pose[1].shape != (3,) or not np.isfinite(pose[0]).all()
                or not np.isfinite(pose[1]).all() or not np.allclose(pose[0] @ pose[0].T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[0]), 1., atol=1e-6)
                or not np.array_equal(pose[0], camera['R']) or not np.array_equal(pose[1], camera['t'])):
            raise ValueError('Invalid or inconsistent sparse training pose')
        frames.append(dict(f, index=index, pose=pose, image=image, gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                           mask=mask > 0, core=mask_core(mask), source_row=row))
    Ks = [np.array(c['K'], np.float64) for c in viewer['cameras'] if c['split'] == 'train']
    if not Ks or not all(np.allclose(Ks[0], k) for k in Ks):
        raise ValueError('Shared training camera matrix is required')
    # Preserve the input protocol separately in the output hash manifest; only
    # source/video metadata are inherited from the verified original contract.
    normalized = dict(protocol, source=original_protocol['source'])
    return frames, geometry, viewer, normalized, Ks[0]


def rectify_pair(left, right, K):
    relative_R = right['pose'][0] @ left['pose'][0].T
    relative_t = right['pose'][1] - relative_R @ left['pose'][1]
    h, w = left['gray'].shape
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, np.zeros(5), K, np.zeros(5), (w, h), relative_R, relative_t.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    if abs(P2[1, 3]) > abs(P2[0, 3]) or P2[0, 3] >= -1e-6:
        raise ValueError('Pair must have positive horizontal disparity after rectification')
    out = {'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q, 'roi1': roi1, 'roi2': roi2}
    for side, f, R, P in [('left', left, R1, P1), ('right', right, R2, P2)]:
        maps = cv2.initUndistortRectifyMap(K, None, R, P, (w, h), cv2.CV_32FC1)
        out[side + '_gray'] = cv2.remap(f['gray'], *maps, cv2.INTER_LINEAR)
        out[side + '_core'] = cv2.remap(f['core'].astype(np.uint8), *maps, cv2.INTER_NEAREST) > 0
        out[side + '_image'] = cv2.remap(f['image'], *maps, cv2.INTER_LINEAR)
    return out


def select_pair(frames, geometry, K):
    rows = []
    points = geometry['points']
    tracks = geometry['tracks']
    candidates = []
    for ai, left in enumerate(frames):
        for right in frames[ai + 1:]:
            common = [t for t in points if str(left['index']) in tracks[t] and str(right['index']) in tracks[t]]
            if len(common) < PROTOCOL['pair_min_common_points']:
                continue
            for a, b in [(left, right), (right, left)]:
                try:
                    rect = rectify_pair(a, b, K)
                except (ValueError, cv2.error):
                    continue
                pa = np.array([tracks[t][str(a['index'])] for t in common], np.float64)
                pb = np.array([tracks[t][str(b['index'])] for t in common], np.float64)
                ra = cv2.undistortPoints(pa[:, None, :], K, None, R=rect['R1'], P=rect['P1'])[:, 0]
                rb = cv2.undistortPoints(pb[:, None, :], K, None, R=rect['R2'], P=rect['P2'])[:, 0]
                xyz = np.array([points[t] for t in common])
                ca = -a['pose'][0].T @ a['pose'][1]
                cb = -b['pose'][0].T @ b['pose'][1]
                va, vb = xyz - ca, xyz - cb
                angle = np.degrees(np.arccos(np.clip(np.sum(va * vb, axis=1) /
                                     (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)), -1, 1)))
                median_angle = float(np.median(angle))
                y_p90 = float(np.percentile(np.abs(ra[:, 1] - rb[:, 1]), 90))
                disparity = ra[:, 0] - rb[:, 0]
                keep = (disparity > PROTOCOL['minimum_disparity_px']) & mask_at(rect['left_core'], ra) & mask_at(rect['right_core'], rb)
                row = {'left_frame': a['id'], 'right_frame': b['id'], 'left_index': a['index'],
                       'right_index': b['index'], 'common_points': len(common), 'core_common_points': int(keep.sum()),
                       'median_angle_deg': median_angle, 'rectified_y_p90_px': y_p90,
                       'positive_disparity_fraction': float((disparity > 0).mean()),
                       'passes': bool(PROTOCOL['pair_min_angle_deg'] <= median_angle <= PROTOCOL['pair_max_angle_deg']
                                      and y_p90 <= PROTOCOL['pair_max_rectified_y_p90_px']
                                      and keep.sum() >= PROTOCOL['pair_min_common_points'])}
                rows.append(row)
                if row['passes']:
                    # A single pair is selected before any dense matching or result inspection.
                    score = int(keep.sum()) * min(median_angle, 8.0) / (1 + y_p90)
                    candidates.append((score, row, a, b, rect, disparity[keep]))
                break
    candidates.sort(key=lambda r: (-r[0], r[1]['left_index'], r[1]['right_index']))
    return (candidates[0] if candidates else None), rows


def stereo_disparities(left, right, sparse_disparities):
    values = np.asarray(sparse_disparities)
    minimum = max(0, int(math.floor(float(np.percentile(values, 2)))) - PROTOCOL['sgbm_disparity_margin_px'])
    maximum = int(math.ceil(float(np.percentile(values, 98)))) + PROTOCOL['sgbm_disparity_margin_px']
    count = max(16, int(math.ceil((maximum - minimum + 1) / 16)) * 16)
    if count + minimum >= left.shape[1] - 8:
        raise ValueError('Stereo disparity search exceeds image width')
    block = PROTOCOL['sgbm_block_size']
    common = dict(numDisparities=count, blockSize=block, P1=8 * block * block, P2=32 * block * block,
                  disp12MaxDiff=-1, preFilterCap=31, uniquenessRatio=PROTOCOL['sgbm_uniqueness_ratio'],
                  speckleWindowSize=50, speckleRange=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dleft = cv2.StereoSGBM_create(minDisparity=minimum, **common).compute(left, right).astype(np.float32) / 16.
    right_minimum = -(minimum + count - 1)
    dright = cv2.StereoSGBM_create(minDisparity=right_minimum, **common).compute(right, left).astype(np.float32) / 16.
    valid_left = dleft > max(minimum - 1, PROTOCOL['minimum_disparity_px'])
    valid_right = dright > right_minimum - 1
    dright[~valid_right] = -1e5
    return dleft, dright, valid_left, {'min_disparity': minimum, 'num_disparities': count,
                                      'right_min_disparity': right_minimum}


def rectify_points_to_world(uv, disparity, rect, left_pose):
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    other = uv - np.c_[np.asarray(disparity), np.zeros(len(uv))]
    homogeneous = cv2.triangulatePoints(rect['P1'], rect['P2'], uv.T, other.T).T
    rectified = homogeneous[:, :3] / homogeneous[:, 3:4]
    camera = rectified @ rect['R1']
    return (camera - left_pose[1].reshape(1, 3)) @ left_pose[0]


def unrectify_pixels(uv, R, P, K):
    """Invert the rectifying image homography, independent of reconstructed XYZ."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    mapping = K @ R.T @ np.linalg.inv(P[:, :3])
    raw = np.c_[uv, np.ones(len(uv))] @ mapping.T
    return raw[:, :2] / raw[:, 2:3]


def add_stereo_observations(viewer, rect, K):
    """Expose the actual matcher coordinates; never relabel predicted pixels as observations."""
    selected = viewer['surface_report']['selected_pair']
    if not selected or not viewer['points']:
        return viewer
    uv = np.array([p['rectified_reference_pixel'] for p in viewer['points']], np.float64)
    d = np.array([p['disparity_px'] for p in viewer['points']])
    left = unrectify_pixels(uv, rect['R1'], rect['P1'], K)
    right = unrectify_pixels(uv - np.c_[d, np.zeros(len(d))], rect['R2'], rect['P2'], K)
    for frame in viewer['frames']:
        frame['sparse_feature_count_in_input'] = frame.get('sparse_feature_count_in_input', frame['feature_count'])
        frame['feature_count'] = 0
        frame['observations'] = []
        side = left if frame['id'] == selected['left_frame'] else right if frame['id'] == selected['right_frame'] else None
        if side is None:
            frame['observation_note'] = 'Additional TRAIN-view photo-consistency only; no independent dense pixel matching is displayed.'
            continue
        camera = next(c for c in viewer['cameras'] if c['frame_id'] == frame['id'])
        xyz = np.array([p['xyz'] for p in viewer['points']])
        predicted, _ = project(xyz, (np.array(camera['R']), np.array(camera['t'])), K)
        frame['feature_count'] = len(side)
        frame['observation_note'] = 'Actual reciprocal stereo matcher pixels, inverse-rectified to source coordinates. Triangulation residual is a construction check, not independent accuracy.'
        frame['observations'] = [{'track_id': p['id'], 'xy': xy.tolist(), 'predicted_xy': pred.tolist(),
                                 'role': 'stereo_match', 'error_px': float(np.linalg.norm(xy - pred))}
                                for p, xy, pred in zip(viewer['points'], side, predicted)]
    return viewer


def derive_correspondence_review(output):
    """Fresh review-only derivative, preserving an already delivered raw result."""
    output = Path(output)
    target = output / 'viewer-with-correspondences.json'
    if target.exists():
        raise ValueError('Review derivative already exists')
    viewer = json.loads((output / 'viewer.json').read_text())
    with np.load(output / 'diagnostic-disparity.npz', allow_pickle=False) as data:
        rect = {k: data[k] for k in ['R1', 'R2', 'P1', 'P2']}
        K = data['K']
    add_stereo_observations(viewer, rect, K)
    viewer['review_derivation'] = {'source_viewer_sha256': sha256(output / 'viewer.json'),
                                 'source_disparity_sha256': sha256(output / 'diagnostic-disparity.npz'),
                                 'implementation_sha256': sha256(__file__),
                                 'purpose': 'Expose saved actual stereo coordinates without changing geometry, colours, triangles or support.'}
    target.write_text(json.dumps(viewer, separators=(',', ':')) + '\n')
    return target


def texture_uv(pixels, width, height):
    """OpenCV integer pixel centres to Three.js texture UV (flipY=true)."""
    uv = np.asarray(pixels, np.float64).reshape(-1, 2)
    return np.c_[(uv[:, 0] + .5) / width, 1 - (uv[:, 1] + .5) / height]


def derive_display_review(output, *, root):
    """Attested display-only half-texel correction, without rerunning stereo."""
    output = Path(output).resolve()
    root = Path(root).resolve()
    if not output.is_relative_to(root / 'data'):
        raise ValueError('Display derivative must be inside project data')
    target = output / 'viewer-display.json'
    if target.exists() or (output / 'display-provenance.json').exists():
        raise ValueError('Display derivative already exists')
    viewer = json.loads((output / 'viewer.json').read_text())
    with np.load(output / 'diagnostic-disparity.npz', allow_pickle=False) as data:
        rect = {k: data[k] for k in ['R1', 'R2', 'P1', 'P2']}
        K = data['K']
    add_stereo_observations(viewer, rect, K)
    if viewer.get('surface'):
        surface = viewer['surface']
        frame = next(f for f in viewer['frames'] if f['id'] == surface['source_frame_id'])
        camera = next(c for c in viewer['cameras'] if c['frame_id'] == surface['source_frame_id'])
        pixels, _ = project(surface['vertices'], (np.array(camera['R']), np.array(camera['t'])), K)
        surface['uv'] = texture_uv(pixels, frame['width'], frame['height']).tolist()
        surface['texture_coordinates'] = 'OpenCV integer pixel centres: u=(x+0.5)/W,v=1-(y+0.5)/H; Three.js flipY=true'
        texture_path = _local(root, surface['texture_url'])
        if sha256(texture_path) != surface['texture_sha256']:
            raise ValueError('Texture source image hash mismatch')
    notes = ['Display-only derivative adds actual inverse-rectified matcher coordinates and corrects source-image UV by half a texel.',
             'Original raw geometry, colours, accepted support, triangles and failed diagnostics remain preserved.',
             'Raw z-buffer skipped counts can include out-of-frame/behind-camera candidates and are not measured physical occlusions.']
    provenance = {'source_viewer_sha256': sha256(output / 'viewer.json'),
                  'source_protocol_sha256': sha256(output / 'protocol.json'),
                  'source_disparity_sha256': sha256(output / 'diagnostic-disparity.npz'),
                  'implementation_sha256': sha256(__file__), 'notes': notes,
                  'geometry_changed': False, 'support_changed': False, 'new_matching_or_inference': False}
    viewer['display_derivation'] = provenance
    viewer['limitations'] += [notes[-1]]
    target.write_text(json.dumps(viewer, separators=(',', ':')) + '\n')
    (output / 'display-implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'display-implementation.py').chmod(0o444)
    (output / 'display-provenance.json').write_text(json.dumps(dict(provenance, display_viewer_sha256=sha256(target)), indent=2) + '\n')
    return target


def robust_depth_bounds(points, pose, K):
    _, depths = project(points, pose, K)
    finite = depths[np.isfinite(depths) & (depths > 0)]
    if len(finite) < PROTOCOL['pair_min_common_points']:
        raise ValueError('Too few positive sparse depths for conservative disparity bounds')
    low, high = np.percentile(finite, PROTOCOL['training_depth_quantiles'])
    padding = max(float(high - low) * PROTOCOL['training_depth_range_padding_fraction'], float(low) * .01)
    lower, upper = max(1e-9, float(low - padding)), float(high + padding)
    return {'lower': lower, 'upper': upper, 'quantile_low': float(low), 'quantile_high': float(high),
            'sparse_depths': len(finite), 'excluded_sparse_outlier_count': int(((finite < lower) | (finite > upper)).sum())}


def zbuffer_visibility(points, pose, K, shape, relative_tolerance=0.02):
    """Reject occlusion by the currently observed candidate layer, never hidden geometry.

    A true return means only that no nearer *known* candidate occupies the same
    target pixel. Unobserved occluders are not invented and remain a limitation.
    """
    uv, z = project(points, pose, K)
    h, w = shape
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    xy = np.rint(np.where(np.isfinite(uv), uv, -100)).astype(int)
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    depth = np.full(h * w, np.inf)
    indices = np.where(inside)[0]
    pixel = xy[indices, 1] * w + xy[indices, 0]
    np.minimum.at(depth, pixel, z[indices])
    visible = np.zeros(len(z), bool)
    visible[indices] = z[indices] <= depth[pixel] * (1 + relative_tolerance)
    return visible


def _write_ply(path, points):
    rows = ['ply', 'format ascii 1.0', f'element vertex {len(points)}', 'property float x', 'property float y',
            'property float z', 'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
    rows.extend(' '.join(str(x) for x in p['xyz'] + p['rgb']) for p in points)
    path.write_text('\n'.join(rows) + '\n')


def visible_lattice_mesh(points, reference, other, K):
    """Connect only adjacent observed samples; no hole fill or inferred back."""
    if not points:
        return None
    stride = PROTOCOL['candidate_lattice_stride_px']
    index = {tuple(np.rint(p['rectified_reference_pixel']).astype(int)): i for i, p in enumerate(points)}
    xyz = np.array([p['xyz'] for p in points], np.float64)
    depths = np.array([p['reference_depth_arbitrary_units'] for p in points])
    triangles = []
    rejected = {'missing_observed_neighbor': 0, 'depth_jump_or_long_edge': 0, 'mask_gap': 0, 'degenerate': 0}
    for (x, y), a in index.items():
        b = index.get((x + stride, y))
        c = index.get((x, y + stride))
        d = index.get((x + stride, y + stride))
        for ids in [(a, b, c), (b, d, c)]:
            if any(i is None for i in ids):
                rejected['missing_observed_neighbor'] += 1
                continue
            p = xyz[list(ids)]
            z = depths[list(ids)]
            max_edge = PROTOCOL['mesh_max_edge_in_lattice_pixels'] * stride * float(z.mean()) / float(min(K[0, 0], K[1, 1]))
            edges = [float(np.linalg.norm(p[i] - p[j])) for i, j in [(0, 1), (1, 2), (2, 0)]]
            if max(edges) > max_edge or np.ptp(z) > PROTOCOL['mesh_max_relative_depth_jump'] * z.mean():
                rejected['depth_jump_or_long_edge'] += 1
                continue
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(normal) < 1e-12:
                rejected['degenerate'] += 1
                continue
            samples = np.vstack([p, (p[0] + p[1]) / 2, (p[1] + p[2]) / 2, (p[2] + p[0]) / 2, p.mean(axis=0)])
            a_uv, a_z = project(samples, reference['pose'], K)
            b_uv, b_z = project(samples, other['pose'], K)
            if not ((a_z > 0).all() and (b_z > 0).all() and mask_at(reference['core'], a_uv).all()
                    and mask_at(other['core'], b_uv).all()):
                rejected['mask_gap'] += 1
                continue
            # Make the observed side face the source camera; no opposite face.
            camera_center = -reference['pose'][0].T @ reference['pose'][1]
            triangle = list(ids)
            if np.dot(normal, camera_center - p.mean(axis=0)) < 0:
                triangle[1], triangle[2] = triangle[2], triangle[1]
            triangles.append(triangle)
    if not triangles:
        return None
    h, w = reference['gray'].shape
    uv, _ = project(xyz, reference['pose'], K)
    return {'kind': 'observed_partial_patch', 'vertices': xyz.tolist(), 'colors': [p['rgb'] for p in points],
            'triangles': triangles, 'texture_url': reference['image_url'],
            'texture_sha256': reference['source_row']['image_sha256'],
            'uv': texture_uv(uv, w, h).tolist(),
            'texture_coordinates': 'OpenCV integer pixel centres: u=(x+0.5)/W,v=1-(y+0.5)/H; Three.js flipY=true',
            'hole_filling': False, 'watertight': False, 'thickness_known': False,
            'triangles_rejected': rejected, 'source_frame_id': reference['id'],
            'interpretation': 'Local triangles interpolate only adjacent accepted observations; surface interiors are not separately reconstructed.'}


def run(sparse_dir, output, *, root):
    root = Path(root).resolve()
    sparse_dir = Path(sparse_dir).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root / 'data') or output.exists():
        raise ValueError('Use a fresh output directory inside project data')
    cv2.setNumThreads(2)
    frames, geometry, sparse_viewer, sparse_protocol, K = load_sparse(root, sparse_dir)
    output.mkdir(parents=True)
    policy = {'protocol': PROTOCOL, 'sparse_input': str(sparse_dir.relative_to(root)),
              'sparse_files': {name: sha256(sparse_dir / name) for name in ['protocol.json', 'viewer.json', 'geometry.json']},
              'input_views': [f['source_row'] for f in frames], 'K': K.tolist(),
              'implementation_sha256': sha256(__file__), 'source': sparse_protocol['source'],
              'source_sparse_status': sparse_viewer['status'], 'manual_coordinates': False,
              'generated_meshes_read': False, 'predicted_depth_read': False, 'old_alignment_poses_read': False}
    (output / 'protocol.json').write_text(json.dumps(policy, indent=2) + '\n')
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'implementation.py').chmod(0o444)
    selected, pair_diagnostics = select_pair(frames, geometry, K)
    points = []
    counts = {}
    surface = None
    stage = 'no_supported_training_stereo_pair'
    result = {'selected_pair': None, 'extra_views': [], 'disparity_search': None,
              'coordinate_convention': 'pose: X_camera = R X_object + t; relative R=Rb Ra.T; relative t=tb-Rrel ta; rectified X maps through R1.T then inverse left pose',
              'patch_warp_assumption': 'local reference-frontoparallel plane at candidate depth',
              'occlusion_policy': 'reject occlusion by known candidate layer z-buffer; unobserved occluders remain unknown'}
    if selected:
        _, pair_row, left, right, rect, sparse_disparity = selected
        result['selected_pair'] = pair_row
        result['valid_rectified_rois'] = {'left': list(rect['roi1']), 'right': list(rect['roi2'])}
        print(json.dumps({'phase': 'selected_pair', **pair_row}), flush=True)
        others = [f for f in frames if f['id'] not in {left['id'], right['id']}]
        # Shared sparse tracks rank extra views, without examining dense results.
        common = {t for t in geometry['points'] if str(left['index']) in geometry['tracks'][t]
                  and str(right['index']) in geometry['tracks'][t]}
        others.sort(key=lambda f: (-sum(str(f['index']) in geometry['tracks'][t] for t in common), f['index']))
        others = others[:PROTOCOL['maximum_extra_train_views_tested']]
        result['extra_views'] = [f['id'] for f in others]
        dl, dr, valid, search = stereo_disparities(rect['left_gray'], rect['right_gray'], sparse_disparity)
        result['disparity_search'] = search
        yy, xx = np.indices(dl.shape)
        counts['sgbm_valid_pixels'] = int(valid.sum())
        valid &= left_right_consistency(dl, dr, PROTOCOL['left_right_max_error_px'])
        counts['left_right_consistent_pixels'] = int(valid.sum())
        ruv = np.stack([xx - dl, yy], axis=-1)
        valid &= rect['left_core'] & mask_at(rect['right_core'], ruv.reshape(-1, 2)).reshape(dl.shape)
        counts['both_mask_core_pixels'] = int(valid.sum())
        lattice = ((xx % PROTOCOL['candidate_lattice_stride_px']) == 0) & ((yy % PROTOCOL['candidate_lattice_stride_px']) == 0)
        positions = np.argwhere(valid & lattice)
        if len(positions) > PROTOCOL['max_dense_candidates']:
            # Deterministic spatial subsampling, independent of colour/support outcomes.
            positions = positions[np.linspace(0, len(positions) - 1, PROTOCOL['max_dense_candidates'], dtype=int)]
        counts['bounded_candidates'] = len(positions)
        uv = positions[:, ::-1].astype(np.float64)
        xyzs = rectify_points_to_world(uv, dl[positions[:, 0], positions[:, 1]], rect, left['pose'])
        sparse_xyz = np.array(list(geometry['points'].values()), np.float64)
        bounds = robust_depth_bounds(sparse_xyz, left['pose'], K)
        result['training_depth_bounds'] = bounds
        _, ref_depths = project(xyzs, left['pose'], K)
        bounded = np.isfinite(xyzs).all(axis=1) & (ref_depths >= bounds['lower']) & (ref_depths <= bounds['upper'])
        counts['within_robust_training_depth_bounds'] = int(bounded.sum())
        visibility = {other['id']: zbuffer_visibility(xyzs, other['pose'], K, other['gray'].shape,
                    PROTOCOL['zbuffer_relative_depth_tolerance']) for other in others}
        extra_diagnostics = {other['id']: {'checked': 0, 'occluded_by_known_candidate_layer': 0,
                                          'passing': 0, 'rejection_reasons': {}} for other in others}
        rejected = {}
        pair_passing = 0
        for i, xyz in enumerate(xyzs):
            if not bounded[i]:
                rejected['nonfinite_or_outside_robust_training_depth_bounds'] = rejected.get('nonfinite_or_outside_robust_training_depth_bounds', 0) + 1
                continue
            pair_check = photometric_check(xyz, left, right, K)
            if not pair_check['passes']:
                reason = 'pair_' + pair_check['reason']
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            pair_passing += 1
            supporting = []
            for other in others:
                diagnostic = extra_diagnostics[other['id']]
                if not visibility[other['id']][i]:
                    diagnostic['occluded_by_known_candidate_layer'] += 1
                    continue
                diagnostic['checked'] += 1
                check = photometric_check(xyz, left, other, K, discriminate_depth=True)
                if check['passes']:
                    diagnostic['passing'] += 1
                    supporting.append({'frame_id': other['id'], 'timestamp_s': other['timestamp_s'], **check})
                else:
                    reasons = diagnostic['rejection_reasons']
                    reasons[check['reason']] = reasons.get(check['reason'], 0) + 1
            if len(supporting) < PROTOCOL['minimum_extra_train_views']:
                rejected['fewer_than_two_extra_photometric_train_views'] = rejected.get('fewer_than_two_extra_photometric_train_views', 0) + 1
                continue
            luv, depth = project(xyz[None], left['pose'], K)
            colour = sample_patch(left['image'], left['mask'], luv)[0][::-1].tolist()
            points.append({'id': f'stereo_{i:06d}', 'xyz': xyz.tolist(), 'rgb': colour,
                           'reference_frame_id': left['id'], 'reference_pixel': luv[0].tolist(),
                           'rectified_reference_pixel': uv[i].tolist(),
                           'reference_depth_arbitrary_units': float(depth[0]),
                           'disparity_px': float(dl[positions[i, 0], positions[i, 1]]),
                           'pair_zncc': pair_check['zncc'], 'extra_train_support': supporting,
                           'multiview_training_support_count': 2 + len(supporting),
                           'heldout_scored_projection_support': [],
                           'heldout_scored_projection_supported_in_at_least_two_views': False,
                           'validated': False, 'heldout_support': False})
        counts.update(pair_photometric_candidates=pair_passing, accepted_partial_surface_points=len(points), rejected=rejected)
        result['extra_view_diagnostics'] = extra_diagnostics
        surface = visible_lattice_mesh(points, left, right, K)
        counts['observed_local_triangles'] = len(surface['triangles']) if surface else 0
        if surface:
            (output / 'visible-patch.json').write_text(json.dumps(surface, separators=(',', ':')) + '\n')
        display = np.zeros_like(dl, np.uint8)
        good = valid & np.isfinite(dl)
        if good.any():
            low, high = np.percentile(dl[good], [2, 98])
            display[good] = np.clip((dl[good] - low) / max(high - low, 1e-6) * 254 + 1, 1, 255).astype(np.uint8)
        cv2.imwrite(str(output / 'rectified-left.jpg'), rect['left_image'])
        cv2.imwrite(str(output / 'rectified-right.jpg'), rect['right_image'])
        cv2.imwrite(str(output / 'disparity-preview.png'), cv2.applyColorMap(display, cv2.COLORMAP_TURBO))
        np.savez_compressed(output / 'diagnostic-disparity.npz', left=dl, right=dr, gated=valid,
                            R1=rect['R1'], R2=rect['R2'], P1=rect['P1'], P2=rect['P2'], K=K)
        stage = 'diagnostic_observed_partial_surface' if points else 'no_multiview_photo_consistent_dense_points'
    report = dict(result, status=stage, counts=counts, pair_diagnostics=pair_diagnostics,
                  camera_calibrated=False, physical_scale_known=False, complete_object=False, mesh_created=surface is not None,
                  heldout_geometry_validation=False, source_colours_only=True, new_api_calls=0,
                  limitations=['Photometric checks use only TRAIN images; they do not independently validate recovered shape.',
                               'Camera intrinsics and physical scale remain unverified; no calibrated scan is established.',
                               'Only visible, textured, multiply supported surface samples are retained. Missing surfaces remain absent.',
                               'Local frontoparallel patch warps can reject curved surfaces or tolerate some incorrect geometry.',
                               'SAM propagated masks remain fallible and unaudited; mask membership does not prove rigid part identity.'])
    (output / 'surface.json').write_text(json.dumps(report, indent=2) + '\n')
    view = {'format': 'observed-surface-review', 'version': 1, 'id': output.name,
            'title': 'Observed partial surface · diagnostic stereo', 'status': stage, 'kind': 'observed_surface',
            'reason': report['limitations'][1], 'metrics': counts,
            'frames': [dict({k: f[k] for k in ['id', 'timestamp_s', 'image_url', 'mask_url', 'width', 'height', 'registered', 'split', 'feature_count', 'audit_status']},
                       observations=[]) for f in frames],
            'cameras': [c for c in sparse_viewer['cameras'] if c['split'] == 'train'],
            'points': points, 'limitations': report['limitations'], 'surface_report': report,
            'input_protocol_sha256': sha256(output / 'protocol.json'), 'new_api_calls': 0,
            'prior_geometry_or_depth_used': False}
    if surface:
        view['surface'] = surface
    if selected:
        add_stereo_observations(view, rect, K)
    (output / 'viewer.json').write_text(json.dumps(view, separators=(',', ':')) + '\n')
    _write_ply(output / 'points.ply', points)
    print(json.dumps({'phase': 'finished', 'status': stage, 'points': len(points), 'counts': counts}), flush=True)
    return view
