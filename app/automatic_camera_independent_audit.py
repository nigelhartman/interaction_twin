"""Independent camera-result checks; no fitting, inference or file mutation.

These checks intentionally project through OpenCV instead of the reconstruction
engine's residual implementation. Passing them establishes internal consistency,
not an accurate camera, metric object or correct hidden geometry.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import cv2
import numpy as np


def normalized_camera_to_pixels(normalized, width, height):
    """Convert centre-normalized MoGe K into integer-centre pixel coordinates."""
    matrix = np.asarray(normalized, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError('A finite normalized camera and positive integer native dimensions are required')
    # Explicit image-coordinate homography, including the half-pixel conversion.
    pixel_from_normalized = np.array([[width, 0, -.5], [0, height, -.5], [0, 0, 1.]])
    return pixel_from_normalized @ matrix


def opencv_projection(points, rotation, translation, camera):
    """Independent zero-distortion projection, rejecting skew OpenCV omits."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    K = np.asarray(camera, dtype=np.float64)
    if (R.shape != (3, 3) or K.shape != (3, 3)
            or not all(np.isfinite(q).all() for q in [xyz, R, t, K])
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-7, rtol=0)
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError('Proper finite poses and zero-skew pinhole intrinsics are required')
    image, _ = cv2.projectPoints(xyz, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
    depth = (xyz @ R.T + t)[:, 2]
    return image.reshape(-1, 2), depth


def raw_support(points, poses, tracks, training_ids):
    """Describe constraints without dropping difficult points or views."""
    train = set(training_ids)
    cameras = set(map(int, poses))
    identities = set(points)
    if not cameras.issubset(train):
        raise ValueError('Reserved camera entered the fitted model')
    if any(not set(map(int, observations)).issubset(train) for observations in tracks.values()):
        raise ValueError('Reserved observation entered the fitting tracks')
    missing = identities - set(tracks)
    if missing:
        raise ValueError('Fitted points must retain their original raw tracks')
    point_degrees = {tid: len(set(map(int, tracks[tid])) & cameras) for tid in sorted(identities)}
    camera_degrees = {camera: sum(camera in set(map(int, tracks[tid])) for tid in identities) for camera in sorted(cameras)}
    adjacency = {camera: set() for camera in cameras}
    for tid in identities:
        seen = set(map(int, tracks[tid])) & cameras
        for camera in seen:
            adjacency[camera].update(seen - {camera})
    components = []
    pending = set(cameras)
    while pending:
        stack = [min(pending)]; group = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(adjacency[current] - group)
        components.append(sorted(group)); pending -= group
    return {'training_camera_count': len(cameras), 'point_count': len(identities),
            'observations': sum(point_degrees.values()),
            'point_view_count_histogram': dict(sorted(Counter(point_degrees.values()).items())),
            'camera_point_counts': camera_degrees, 'camera_graph_components': components,
            'points_with_fewer_than_two_views': sum(degree < 2 for degree in point_degrees.values()),
            'unobserved_cameras': [camera for camera, degree in camera_degrees.items() if degree == 0],
            'reserved_observations_used': 0}


def viewer_projection_audit(viewer, *, tolerance_px=1e-6):
    """Recompute all saved predictions/errors using untouched observed pixels."""
    points = {point['id']: np.asarray(point['xyz'], dtype=np.float64) for point in viewer['points']}
    cameras = {camera['frame_id']: camera for camera in viewer['cameras']}
    errors = []
    compared = 0
    role_failures = []
    behind = 0
    max_prediction_delta = max_error_delta = 0.
    for frame in viewer['frames']:
        camera = cameras.get(frame['id'])
        for observation in frame['observations']:
            tid = observation['track_id']
            # Unmatched raw feature markers do not enter PnP or scored errors;
            # their distinct display-only role is intentionally preserved.
            if frame['split'] == 'heldout' and tid in points:
                expected = 'heldout' if int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 3 == 0 else 'pose_anchor'
                if observation['role'] != expected:
                    role_failures.append([frame['id'], tid])
            saved = observation.get('predicted_xy')
            if camera is None or tid not in points:
                if saved is not None:
                    raise ValueError('A saved prediction lacks a camera or fitted point')
                continue
            projected, depth = opencv_projection(points[tid][None], camera['R'], camera['t'], camera['K'])
            if depth[0] <= 0:
                behind += 1
            if saved is None:
                continue
            compared += 1
            delta = float(np.linalg.norm(projected[0] - np.asarray(saved)))
            max_prediction_delta = max(max_prediction_delta, delta)
            residual = float(np.linalg.norm(projected[0] - np.asarray(observation['xy'])))
            if observation.get('error_px') is None:
                raise ValueError('Saved projected observation lacks its error')
            max_error_delta = max(max_error_delta, abs(residual - observation['error_px']))
            if frame['split'] == 'train':
                errors.append(residual)
    return {'passed': compared > 0 and not role_failures and max_prediction_delta <= tolerance_px and max_error_delta <= tolerance_px,
            'projection_implementation': 'cv2.projectPoints with explicit zero distortion',
            'compared_predictions': compared, 'maximum_prediction_difference_px': max_prediction_delta,
            'maximum_saved_error_difference_px': max_error_delta,
            'behind_camera_observations': behind, 'role_failures': role_failures,
            'train_median_px': float(np.median(errors)) if errors else None,
            'train_p90_px': float(np.percentile(errors, 90)) if errors else None,
            'physical_accuracy_established': False}
