"""Frozen-surface, held-out RGB consistency diagnostics; no fitting or inference.

The held-out camera poses were already estimated from sparse anchor features.
This tests new dense appearance predictions conditional on those poses, masks,
local-plane approximation and partial known visibility. It is not independent
end-to-end validation, metric ground truth, or a proof of physical visibility.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PROTOCOL = {
    'version': 'observed-surface-heldout-photo-v1',
    'geometry_frozen_before_test': True,
    'pose_fitting': False,
    'method_or_threshold_selection_from_test': False,
    'patch_radius_px': 3,
    'minimum_patch_std': 5.0,
    'minimum_zncc': 0.80,
    'mask_core_fraction': 0.15,
    'mask_core_min_px': 3.0,
    'anchor_center_guard_px': 12.0,
    'known_patch_relative_depth_tolerance': 0.02,
    'visibility_policy': 'Require every warped sample to project onto the known, front-facing triangle patch, with no nearer represented surface. Missing patch coverage is skipped. Unobserved physical occluders remain unknown.',
    'patch_warp': '7x7 source-camera-frontoparallel local plane at each frozen point depth',
    'coordinate_convention': 'OpenCV integer pixel centres in exact saved 960x540 source frames; no resize or matching search',
    'heldout_poses': 'Reuse only saved heldout_anchor_pose cameras estimated from sparse anchor correspondences; never estimate missing poses',
    'anchor_guard_limitation': '12 px centre guard does not establish disjoint SIFT descriptor support; camera estimation already used this held-out image',
    'new_api_calls': 0,
    'learned_inference_calls': 0,
    'manual_geometry_or_poses': False,
    'metric_ground_truth_available': False,
    'physical_visibility_established': False,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(xyz, pose, K):
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    camera = xyz @ pose[0].T + np.asarray(pose[1]).reshape(1, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        homogeneous = camera @ np.asarray(K).T
        xy = homogeneous[:, :2] / homogeneous[:, 2:3]
    return xy, camera[:, 2]


def plane_patch(xyz, pose, K, radius=3):
    xy, z = project(np.asarray(xyz)[None], pose, K)
    offsets = np.array([(x, y) for y in range(-radius, radius + 1)
                        for x in range(-radius, radius + 1)], np.float64)
    rays = np.c_[xy + offsets, np.ones(len(offsets))] @ np.linalg.inv(K).T
    return (rays * z[0] - np.asarray(pose[1])) @ pose[0]


def mask_core(mask):
    distance = cv2.distanceTransform((np.asarray(mask) > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return distance >= max(PROTOCOL['mask_core_min_px'], float(distance.max()) * PROTOCOL['mask_core_fraction'])


def pixel_indices(xy, shape):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    finite = np.isfinite(xy).all(axis=1)
    safe = np.where(np.isfinite(xy), xy, -100)
    ij = np.rint(safe).astype(np.int64)
    h, w = shape
    inside = finite & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    return ij, inside


def rasterize_known_patch(vertices, triangles, pose, K, shape):
    """Perspective-correct depth of saved open triangles, facing this camera.

    No closing, splatting, dilation or inferred occluder is introduced. Infinity
    means missing represented coverage, not visible empty space.
    """
    vertices = np.asarray(vertices, np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, np.int64).reshape(-1, 3)
    xy, z = project(vertices, pose, K)
    h, w = shape
    depth = np.full((h, w), np.inf)
    centre = -pose[0].T @ np.asarray(pose[1])
    for ids in triangles:
        p, v, d = vertices[ids], xy[ids], z[ids]
        if not np.isfinite(v).all() or not np.isfinite(d).all() or (d <= 0).any():
            continue
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(normal, centre - p.mean(axis=0)) <= 0:
            continue
        x0, y0 = np.maximum(np.ceil(v.min(axis=0)), [0, 0]).astype(int)
        x1, y1 = np.minimum(np.floor(v.max(axis=0)), [w - 1, h - 1]).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = v
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        wc = 1 - wa - wb
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        inverse_z = wa / d[0] + wb / d[1] + wc / d[2]
        valid = inside & (inverse_z > 0)
        patch = depth[y0:y1 + 1, x0:x1 + 1]
        patch[valid] = np.minimum(patch[valid], 1 / inverse_z[valid])
    return depth


def sample_values(image, xy):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return cv2.remap(image, xy[:, 0].astype(np.float32)[None],
                     xy[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0].astype(np.float64)


def _patch_eligibility(xy, z, frame):
    ij, inside = pixel_indices(xy, frame['gray'].shape)
    h, w = frame['gray'].shape
    if not np.isfinite(z).all() or (z <= 0).any():
        return 'behind_camera_or_nonfinite'
    if not inside.all() or (xy[:, 0] < 1).any() or (xy[:, 0] >= w - 2).any() or (xy[:, 1] < 1).any() or (xy[:, 1] >= h - 2).any():
        return 'outside_image'
    # Bilinear image samples may touch four pixels; require all four in core.
    floor = np.floor(xy).astype(int)
    for ox, oy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        if not frame['core'][floor[:, 1] + oy, floor[:, 0] + ox].all():
            return 'outside_mask_core'
    depth = frame['known_depth'][ij[:, 1], ij[:, 0]]
    if not np.isfinite(depth).all():
        return 'unknown_known_patch_coverage'
    if (z > depth * (1 + PROTOCOL['known_patch_relative_depth_tolerance'])).any():
        return 'occluded_by_represented_patch'
    if (z < depth * (1 - PROTOCOL['known_patch_relative_depth_tolerance'])).any():
        return 'local_plane_disagrees_with_represented_patch'
    return None


def check_point(xyz, reference, heldout, K):
    reference_xy, _ = project(np.asarray(xyz)[None], reference['pose'], K)
    target_xy, target_z = project(np.asarray(xyz)[None], heldout['pose'], K)
    safe_xy = lambda xy: [float(v) if np.isfinite(v) else None for v in xy]
    row = {'reference_xy': safe_xy(reference_xy[0]), 'projected_xy': safe_xy(target_xy[0]),
           'status': 'skipped', 'zncc': None, 'physical_visibility_unknown': True}
    if not np.isfinite(target_xy).all() or not np.isfinite(target_z).all() or target_z[0] <= 0:
        return dict(row, reason='target_behind_camera_or_nonfinite')
    patch = plane_patch(xyz, reference['pose'], K, PROTOCOL['patch_radius_px'])
    source_xy, source_z = project(patch, reference['pose'], K)
    target_xy, target_z = project(patch, heldout['pose'], K)
    anchors = np.asarray(heldout.get('anchor_pixels', []), np.float64).reshape(-1, 2)
    if len(anchors) and (np.linalg.norm(target_xy[:, None] - anchors[None], axis=2) < PROTOCOL['anchor_center_guard_px']).any():
        return dict(row, reason='near_sparse_pose_anchor_centre')
    for name, xy, z, frame in [('reference', source_xy, source_z, reference), ('target', target_xy, target_z, heldout)]:
        invalid = _patch_eligibility(xy, z, frame)
        if invalid:
            return dict(row, reason=name + '_' + invalid)
    a, b = sample_values(reference['gray'], source_xy), sample_values(heldout['gray'], target_xy)
    a, b = a - a.mean(), b - b.mean()
    if min(float(a.std()), float(b.std())) < PROTOCOL['minimum_patch_std']:
        return dict(row, reason='insufficient_reference_or_target_texture')
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return dict(row, zncc=score,
                status='photo_consistent' if score >= PROTOCOL['minimum_zncc'] else 'photo_inconsistent',
                reason='conditional_heldout_patch_zncc')


def local_path(root, relative):
    root = Path(root).resolve()
    value = str(relative)
    if value.startswith('/files/'):
        value = value[len('/files/'):]
    if not value.startswith('data/') or '\\' in value or '?' in value or '#' in value:
        raise ValueError('Only exact cited local data paths are permitted')
    path = (root / value).resolve()
    if not path.is_relative_to(root / 'data'):
        raise ValueError('Input escaped project data')
    return path


def _json(path):
    return json.loads(Path(path).read_text())


def verify_sparse(root, sparse):
    protocol = _json(sparse / 'protocol.json')
    viewer = _json(sparse / 'viewer.json')
    if viewer['input_protocol_sha256'] != sha256(sparse / 'protocol.json'):
        raise ValueError('Sparse protocol hash mismatch')
    if 'source_directory' in protocol:
        if protocol.get('protocol', {}).get('version') not in {'observed-shared-focal-v1', 'observed-shared-focal-v2'}:
            raise ValueError('Unknown sparse calibration provenance')
        source = local_path(root, protocol['source_directory'])
        for name, expected in protocol['source_files'].items():
            if name not in {'protocol.json', 'viewer.json', 'geometry.json', 'implementation.py'} or sha256(source / name) != expected:
                raise ValueError('Original sparse hash mismatch')
        for name, expected in protocol['code_files'].items():
            if name not in {'observed_calibration.py', 'refine_observed_camera.py', 'observed_geometry.py'} or sha256(sparse / name) != expected:
                raise ValueError('Calibration implementation hash mismatch')
        for flag in ['read_generated_mesh', 'read_predicted_depth', 'read_manual_pose']:
            if protocol.get(flag) is not False:
                raise ValueError('Calibration forbidden-input exclusion missing')
        original, _ = verify_sparse(root, source)
        if original['input_views'] != protocol['inputs']:
            raise ValueError('Calibration source split mismatch')
        program = local_path(root, protocol['source_program'])
        if sha256(program) != protocol['source_program_sha256']:
            raise ValueError('Calibration source program mismatch')
        return original, viewer
    if sha256(sparse / 'implementation.py') != protocol['implementation_sha256']:
        raise ValueError('Sparse implementation hash mismatch')
    for flag in ['manual_coordinates', 'generated_meshes_read', 'depth_maps_read', 'old_poses_read']:
        if protocol.get(flag) is not False:
            raise ValueError('Sparse forbidden-input exclusion missing')
    if sha256(local_path(root, protocol['program_path'])) != protocol['program_sha256']:
        raise ValueError('Sparse source program mismatch')
    return protocol, viewer


def _camera(camera):
    R, t, K = np.array(camera['R'], np.float64), np.array(camera['t'], np.float64), np.array(camera['K'], np.float64)
    if (R.shape != (3, 3) or t.shape != (3,) or K.shape != (3, 3)
            or not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(K).all()
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-6)):
        raise ValueError('Invalid saved camera')
    return (R, t), K


def load_inputs(root, surface_directory):
    surface = local_path(root, str(Path(surface_directory).relative_to(root)))
    protocol, viewer = _json(surface / 'protocol.json'), _json(surface / 'viewer.json')
    if viewer['input_protocol_sha256'] != sha256(surface / 'protocol.json'):
        raise ValueError('Dense protocol hash mismatch')
    if sha256(surface / 'implementation.py') != protocol['implementation_sha256']:
        raise ValueError('Dense implementation hash mismatch')
    if protocol['protocol'].get('heldout_images_read') is not False:
        raise ValueError('Surface construction held-out isolation not established')
    for flag in ['manual_coordinates', 'generated_meshes_read', 'predicted_depth_read', 'old_alignment_poses_read']:
        if protocol.get(flag) is not False:
            raise ValueError('Dense forbidden-input exclusion missing')
    sparse = local_path(root, protocol['sparse_input'])
    for name, expected in protocol['sparse_files'].items():
        if name not in {'protocol.json', 'viewer.json', 'geometry.json'} or sha256(sparse / name) != expected:
            raise ValueError('Dense sparse-input hash mismatch')
    original, sparse_viewer = verify_sparse(root, sparse)
    manifest = {f['id']: f for f in original['input_views']}
    K = np.asarray(protocol['K'], np.float64)
    points = viewer['points']
    mesh = viewer.get('surface')
    if not mesh or not points or len({p['id'] for p in points}) != len(points):
        raise ValueError('A frozen, nonempty observed partial mesh is required')
    xyz = np.array([p['xyz'] for p in points], np.float64)
    vertices = np.array(mesh['vertices'], np.float64)
    triangles = np.array(mesh['triangles'], np.int64)
    if not np.array_equal(xyz, vertices) or not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Surface point/mesh identity is invalid')
    source_id = mesh['source_frame_id']
    if any(p['reference_frame_id'] != source_id for p in points):
        raise ValueError('Mixed reference cameras are unsupported')
    frames = []
    cameras = {c['frame_id']: c for c in sparse_viewer['cameras']}
    for f in sparse_viewer['frames']:
        if f['id'] != source_id and f['split'] != 'heldout':
            continue
        row = manifest[f['id']]
        if row['split'] != f['split']:
            raise ValueError('Held-out split mismatch')
        camera = cameras.get(f['id'])
        item = {k: f[k] for k in ['id', 'timestamp_s', 'image_url', 'mask_url', 'width', 'height', 'split']}
        item['pose_available'] = camera is not None
        item['source_row'] = row
        if camera is not None:
            if camera['split'] != ('train' if f['id'] == source_id else 'heldout_anchor_pose'):
                raise ValueError('Unexpected camera provenance')
            pose, actual_K = _camera(camera)
            if not np.array_equal(K, actual_K):
                raise ValueError('Dense and source camera intrinsics differ')
            image_path, mask_path = local_path(root, f['image_url']), local_path(root, f['mask_url'])
            if sha256(image_path) != row['image_sha256'] or sha256(mask_path) != row['mask_sha256']:
                raise ValueError('Exact source RGB/mask hash mismatch')
            image, mask = cv2.imread(str(image_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or mask.shape != (f['height'], f['width']) or image.shape[:2] != mask.shape:
                raise ValueError('Source image dimensions mismatch')
            item.update(image=image, gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), core=mask_core(mask), pose=pose,
                        anchor_pixels=[o['xy'] for o in f['observations'] if o['role'] == 'pose_anchor'])
            item['known_depth'] = rasterize_known_patch(vertices, triangles, pose, K, mask.shape)
        frames.append(item)
    reference = next(f for f in frames if f['id'] == source_id)
    if reference['split'] != 'train' or not reference['pose_available']:
        raise ValueError('Missing frozen TRAIN reference camera')
    return viewer, protocol, reference, [f for f in frames if f['split'] == 'heldout'], K


def _url(root, path):
    return '/files/' + str(Path(path).relative_to(root))


def _write_overlay(output, frame, rows):
    image = frame['image'].copy()
    for row in rows:
        if row['status'] not in {'photo_consistent', 'photo_inconsistent'}:
            continue
        xy = np.rint(row['projected_xy']).astype(int)
        colour = (80, 180, 20) if row['status'] == 'photo_consistent' else (180, 30, 205)
        cv2.circle(image, tuple(xy), 1, colour, -1, cv2.LINE_AA)
    cv2.putText(image, 'Green: conditional photo agreement  Pink: disagreement', (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'Green: conditional photo agreement  Pink: disagreement', (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
    target = output / (frame['id'] + '-overlay.jpg')
    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return target


def run(surface_directory, output, *, root):
    root, output = Path(root).resolve(), Path(output).resolve()
    surface_directory = Path(surface_directory).resolve()
    if not output.is_relative_to(root / 'data') or output.exists():
        raise ValueError('Use a fresh local data output directory')
    # Write frozen protocol/code before opening held-out image pixels.
    output.mkdir(parents=True)
    source_files = {name: sha256(surface_directory / name) for name in ['viewer.json', 'protocol.json', 'implementation.py']}
    policy = {'protocol': PROTOCOL, 'source_directory': str(surface_directory.relative_to(root)),
              'source_files': source_files, 'implementation_sha256': sha256(__file__),
              'holdout_consulted_for_method_or_thresholds': False}
    (output / 'protocol.json').write_text(json.dumps(policy, indent=2) + '\n')
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'implementation.py').chmod(0o444)
    viewer, source_protocol, reference, test_frames, K = load_inputs(root, surface_directory)
    aggregate, supported, checked, frame_rows = Counter(), Counter(), set(), []
    for frame in test_frames:
        summary = {k: frame[k] for k in ['id', 'timestamp_s', 'image_url', 'mask_url', 'width', 'height', 'split', 'pose_available']}
        summary['source_image_sha256'] = frame['source_row']['image_sha256']
        summary['source_mask_sha256'] = frame['source_row']['mask_sha256']
        rows = []
        if not frame['pose_available']:
            summary.update(status='not_evaluated_no_saved_pose', counts={'not_evaluated_no_saved_pose': len(viewer['points'])},
                           zncc_median=None, coverage_fraction_of_dense_points=0, points=[])
            aggregate['not_evaluated_no_saved_pose'] += len(viewer['points'])
        else:
            counts, scores = Counter(), []
            for point in viewer['points']:
                row = dict(point_id=point['id'], **check_point(point['xyz'], reference, frame, K))
                rows.append(row)
                counts[row['status'] if row['status'] != 'skipped' else row['reason']] += 1
                if row['status'] != 'skipped':
                    checked.add(point['id'])
                    scores.append(row['zncc'])
                if row['status'] == 'photo_consistent':
                    supported[point['id']] += 1
            aggregate.update(counts)
            n = counts['photo_consistent'] + counts['photo_inconsistent']
            summary.update(status='conditional_photometric_diagnostic', counts=dict(counts), points=rows,
                           zncc_median=float(np.median(scores)) if scores else None,
                           coverage_fraction_of_dense_points=n / len(viewer['points']),
                           overlay_url=_url(root, _write_overlay(output, frame, rows)),
                           sparse_anchor_centres_guarded=len(frame['anchor_pixels']),
                           represented_patch_pixel_count=int(np.isfinite(frame['known_depth']).sum()),
                           physical_visibility_unknown=True)
        frame_rows.append(summary)
    photo_ok, photo_bad = aggregate['photo_consistent'], aggregate['photo_inconsistent']
    metrics = {'total_dense_points': len(viewer['points']), 'heldout_views_total': len(test_frames),
               'heldout_views_with_saved_anchor_pose': sum(f['pose_available'] for f in test_frames),
               'heldout_views_unregistered': sum(not f['pose_available'] for f in test_frames),
               'point_view_candidates': len(viewer['points']) * sum(f['pose_available'] for f in test_frames),
               'eligible_photo_checks': photo_ok + photo_bad, 'photo_consistent_checks': photo_ok,
               'photo_inconsistent_checks': photo_bad,
               'skipped_by_reason': {k: v for k, v in aggregate.items() if k not in {'photo_consistent', 'photo_inconsistent'}},
               'unique_points_checked': len(checked), 'unique_points_photo_consistent_in_at_least_two_views': sum(n >= 2 for n in supported.values()),
               'conditional_photo_agreement_fraction': photo_ok / (photo_ok + photo_bad) if photo_ok + photo_bad else None,
               'physical_geometry_validated': False}
    report = {'format': 'observed-surface-heldout-photo-review', 'version': 1,
              'id': output.name, 'source_run_id': viewer['id'],
              'status': 'diagnostic_heldout_photometric_check', 'metrics': metrics, 'frames': frame_rows,
              'reference_frame_id': reference['id'], 'reference_image_url': reference['image_url'],
              'K': K.tolist(), 'input_protocol_sha256': sha256(output / 'protocol.json'),
              'protocol': PROTOCOL, 'new_api_calls': 0,
              'limitations': [
                  'Held-out camera poses use sparse features from these same images; this is conditional dense photometric evaluation, not independent end-to-end validation.',
                  'No metric ground-truth shape, lens calibration, physical scale, thickness, cavity or insertion accuracy is established.',
                  'No missing camera poses are fitted; seven held-out views remain unevaluated and null scores are not zero errors.',
                  'Only known open-patch visibility is checked; unobserved occluders and segmentation errors remain possible even when appearance agrees.',
                  'The frontoparallel local-plane approximation, view-dependent materials and lighting can cause disagreement for genuine surfaces.',
                  'Mask core and anchor-neighbourhood guards skip unsupported comparisons. SIFT descriptor support may extend beyond the fixed centre guard.',
                  'Different surface candidates have different support and cameras; these percentages do not directly rank geometric accuracy.',
                  'No result is used to retune thresholds, alter dense geometry, select a focal estimate or choose a winning reconstruction.']}
    (output / 'report.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n')
    (output / 'summary.json').write_text(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2) + '\n')
    print(json.dumps({'source_run_id': viewer['id'], 'metrics': metrics}), flush=True)
    return report
