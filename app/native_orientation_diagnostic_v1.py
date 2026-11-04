"""Bounded, read-only native appearance orientation hypotheses.

This module never changes a scene, mesh, joint, approval, or model result.
Silhouettes alone cannot distinguish a textured face from a plain reverse face.
The scores below are retrospective RGB diagnostics, not physical pose evidence.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

FORMAT = 'interaction-twin-native-orientation-diagnostic'
VERSION = 1


def require(value, message):
    if not value:
        raise ValueError(message)


def pca_rotations(vertices):
    """The 24 proper signed permutations of deterministic native PCA axes."""
    vertices = np.asarray(vertices, float)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4 and np.isfinite(vertices).all(), 'Finite Nx3 geometry required')
    centered = vertices - vertices.mean(0)
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered / len(vertices))
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    for col in range(3):
        if axes[np.argmax(np.abs(axes[:, col])), col] < 0:
            axes[:, col] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    candidates = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product([-1., 1.], repeat=3):
            local = permutation_matrix @ np.diag(signs)
            if np.linalg.det(local) < .999:
                continue
            rotation = axes @ local @ axes.T
            identity = np.allclose(rotation, np.eye(3), atol=1e-10)
            candidates.append({'id': 'identity' if identity else 'pca_' + ''.join(map(str, permutation)) + '_' + ''.join('p' if x > 0 else 'm' for x in signs),
                               'rotation': rotation.tolist(), 'pca_axis_permutation': list(permutation), 'pca_axis_signs': list(signs),
                               'angle_rad': float(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))})
    candidates.sort(key=lambda row: (row['id'] != 'identity', row['id']))
    require(len(candidates) == 24 and sum(row['id'] == 'identity' for row in candidates) == 1, 'Proper PCA group construction failed')
    return candidates


def centered_transform(rotation, center):
    rotation = np.asarray(rotation, float); center = np.asarray(center, float)
    require(rotation.shape == (3, 3) and center.shape == (3,) and np.isfinite(rotation).all() and np.isfinite(center).all(), 'Finite orientation and center required')
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(rotation), 1, atol=1e-8), 'Only proper rigid rotations are allowed')
    matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = center - rotation @ center
    return matrix


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator < 1e-9 else float(np.clip(np.sum(a * b) / denominator, -1, 1))


def appearance_metrics(source_rgb, observed_mask, rendered_rgb, predicted_mask):
    """Fixed grayscale/gradient/edge comparison; low-information stays unknown."""
    source = np.asarray(source_rgb); rendered = np.asarray(rendered_rgb)
    observed = np.asarray(observed_mask, bool); predicted = np.asarray(predicted_mask, bool)
    require(source.shape == rendered.shape and source.ndim == 3 and source.shape[2] == 3 and observed.shape == predicted.shape == source.shape[:2], 'Source/render/mask raster coordinates must agree')
    require(source.dtype == rendered.dtype == np.uint8, 'Bounded byte RGB required')
    union = int(np.count_nonzero(observed | predicted)); intersection = observed & predicted
    interior = cv2.erode(intersection.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count = int(np.count_nonzero(interior)); observed_area = int(np.count_nonzero(observed))
    metrics = {'status': 'unknown', 'reason': None, 'mask_iou': int(np.count_nonzero(intersection)) / max(1, union),
               'visible_source_coverage': int(np.count_nonzero(intersection)) / max(1, observed_area),
               'pixel_count': count, 'gray_ncc': None, 'gradient_ncc': None, 'symmetric_edge_coverage': None,
               'gray_texture_std': None, 'render_gray_texture_std': None, 'objective': None}
    if count < max(64, int(.05 * observed_area)) or metrics['visible_source_coverage'] < .5:
        metrics['reason'] = 'insufficient_foreground_overlap'; return metrics
    first = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (3, 3), .65)
    second = cv2.GaussianBlur(cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (3, 3), .65)
    source_std = float(first[interior].std()); render_std = float(second[interior].std())
    metrics.update(gray_texture_std=source_std, render_gray_texture_std=render_std)
    if source_std < .012 or render_std < .012:
        metrics['reason'] = 'insufficient_native_or_source_texture'; return metrics
    metrics['gray_ncc'] = _ncc(first[interior], second[interior])
    source_gradient = np.stack([cv2.Sobel(first, cv2.CV_32F, dx, dy, ksize=3) for dx, dy in [(1, 0), (0, 1)]], -1)
    render_gradient = np.stack([cv2.Sobel(second, cv2.CV_32F, dx, dy, ksize=3) for dx, dy in [(1, 0), (0, 1)]], -1)
    source_energy = np.linalg.norm(source_gradient[interior]); render_energy = np.linalg.norm(render_gradient[interior])
    if min(source_energy, render_energy) < 1e-6:
        metrics['reason'] = 'insufficient_gradient_information'; return metrics
    metrics['gradient_ncc'] = float(np.clip(np.sum(source_gradient[interior] * render_gradient[interior]) / (source_energy * render_energy), -1, 1))
    source_magnitude = np.linalg.norm(source_gradient, axis=2); render_magnitude = np.linalg.norm(render_gradient, axis=2)
    source_edges = interior & (source_magnitude >= max(.025, float(np.percentile(source_magnitude[interior], 75))))
    render_edges = interior & (render_magnitude >= max(.025, float(np.percentile(render_magnitude[interior], 75))))
    if min(np.count_nonzero(source_edges), np.count_nonzero(render_edges)) < 8:
        metrics['reason'] = 'insufficient_distinct_texture_edges'; return metrics
    source_distance = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, 3)
    render_distance = cv2.distanceTransform((~render_edges).astype(np.uint8), cv2.DIST_L2, 3)
    recall = float(np.mean(render_distance[source_edges] <= 2.)); precision = float(np.mean(source_distance[render_edges] <= 2.))
    edge = 2 * recall * precision / max(1e-9, recall + precision)
    objective = .3 * (1 - metrics['gray_ncc']) / 2 + .3 * (1 - metrics['gradient_ncc']) / 2 + .25 * (1 - edge) + .15 * (1 - metrics['mask_iou'])
    metrics.update(status='diagnostic_available', reason=None, symmetric_edge_coverage=edge, objective=float(objective))
    return metrics


def choose_candidate(rows):
    """Rank same-frame comparisons and explicitly defer ties/weak support.

    Input rows are {id, metrics:[appearance_metrics,...]}. Every hypothesis must
    carry the SAME predeclared frame count; no candidate may select easy frames.
    """
    require(isinstance(rows, list) and 2 <= len(rows) <= 24 and all(isinstance(row, dict) and isinstance(row.get('id'), str) and isinstance(row.get('metrics'), list) for row in rows), 'Bounded candidate metric rows required')
    require(len({row['id'] for row in rows}) == len(rows) and sum(row['id'] == 'identity' for row in rows) == 1, 'Unique candidates and one identity baseline required')
    lengths = {len(row['metrics']) for row in rows}; require(len(lengths) == 1 and 1 <= next(iter(lengths)) <= 12, 'Identical bounded frame slots required')
    total = next(iter(lengths)); ranked = []
    for row in rows:
        usable = [m for m in row['metrics'] if m.get('status') == 'diagnostic_available' and m.get('objective') is not None]
        if len(usable) == total:
            ranked.append({'id': row['id'], 'median_objective': float(np.median([m['objective'] for m in usable])), 'usable_frames': len(usable), 'median_mask_iou': float(np.median([m['mask_iou'] for m in usable]))})
    ranked.sort(key=lambda row: (row['median_objective'], row['id']))
    if not ranked:
        return {'status': 'unknown', 'reason': 'no_candidate_has_all_frame_support', 'ranked': [], 'selected_id': None, 'physical_pose_verified': False}
    best = ranked[0]; second = ranked[1] if len(ranked) > 1 else None
    baseline = next((row for row in ranked if row['id'] == 'identity'), None)
    margin = None if second is None else second['median_objective'] - best['median_objective']
    improvement = None if baseline is None else baseline['median_objective'] - best['median_objective']
    status, reason, selected = 'ambiguous', 'small_top_two_margin_or_uninformative_identity', None
    if second is not None and margin >= .025 and best['median_mask_iou'] >= .4:
        if best['id'] == 'identity':
            status, reason, selected = 'identity_preferred', None, 'identity'
        elif improvement is None or improvement >= .06:
            status, reason, selected = ('appearance_preference_single_view' if total == 1 else 'appearance_preference'), None, best['id']
    return {'status': status, 'reason': reason, 'ranked': ranked, 'selected_id': selected, 'top_two_margin': margin, 'improvement_over_informative_identity': improvement,
            'identity_low_information': baseline is None, 'physical_pose_verified': False, 'fit_or_scene_modified': False}
