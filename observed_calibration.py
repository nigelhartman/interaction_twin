"""Bounded train-only shared-focal refinement of an observed sparse model.

No inference, shape prior, depth, network, or held-out selection. The first pose
and unit seed baseline remove similarity gauge; the second pose rotation and
translation direction remain free. Low training residual is NOT calibration.
"""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import observed_geometry as G

PROTOCOL = {
    'version': 'observed-shared-focal-v2',
    'focal_bounds_factors': [.4, 2.5],
    'focal_start_factors': [.65, 1., 1.6],
    'fixed_focal_profile_factors': [.65, 1., 1.6],
    'max_evaluations_per_start': 1000,
    'max_evaluations_per_profile': 500,
    'principal_point': 'fixed_image_centre', 'square_pixels': True,
    'distortion': 'none_assumed', 'robust_loss': 'soft_l1',
    'robust_scale_px': 1.5,
    'selection': 'lowest_finite_robust_train_cost_positive_depth_fraction_at_least_0.98_then_start_order',
    'minimum_positive_training_depth_fraction': .98,
    'focal_boundary_relative_margin': .03,
    'maximum_multistart_focal_relative_spread': .05,
    'minimum_profile_focal_separation_fraction': .10,
    'minimum_profile_relative_cost_increase': .05,
    'minimum_profile_mean_cost_increase': .002,
    'observability_requires_all_starts_and_profiles_converged': True,
    'heldout_used_for_selection': False,
    'fixed_first_pose': True, 'seed_baseline': 1.,
    'second_seed_rotation_and_translation_direction_free': True,
    'correspondence_support_fixed_across_starts_and_profiles': True,
    'new_api_calls': 0, 'learned_inference_calls': 0,
    'focal_calibrated': False, 'physical_scale_known': False,
}


def normalize_gauge(points, poses, first, second):
    """Make the first pose identity and seed camera baseline exactly one."""
    if first == second or first not in poses or second not in poses:
        raise ValueError('Two distinct existing seed poses are required')
    Ra, ta = poses[first]
    shifted = {i: (R @ Ra.T, t - R @ Ra.T @ ta) for i, (R, t) in poses.items()}
    baseline = float(np.linalg.norm(shifted[second][1]))
    if not np.isfinite(baseline) or baseline < 1e-8:
        raise ValueError('Zero or invalid seed baseline cannot establish scale gauge')
    result = {i: (R, t / baseline) for i, (R, t) in shifted.items()}
    result[first] = (np.eye(3), np.zeros(3))
    return ({t: (Ra @ x + ta) / baseline for t, x in points.items()}, result)


def unit_direction(angles):
    a, b = angles
    return np.array([math.cos(a) * math.cos(b), math.sin(a) * math.cos(b), math.sin(b)])


def direction_angles(t):
    t = np.asarray(t, dtype=float) / np.linalg.norm(t)
    return np.array([math.atan2(t[1], t[0]), math.asin(float(np.clip(t[2], -1., 1.)))])


def robust_cost(residual, scale=1.5):
    r = np.asarray(residual, dtype=float)
    return float(scale * scale * np.sum(np.sqrt(1. + (r / scale) ** 2) - 1.))


class BundleProblem:
    def __init__(self, model, tracks, width, height, training_indices):
        self.width, self.height = int(width), int(height)
        self.dimension = max(width, height)
        self.first, self.second = int(model['seed']['a']), int(model['seed']['b'])
        self.points, self.poses = normalize_gauge(model['points'], model['poses'], self.first, self.second)
        train = set(training_indices)
        if not set(self.poses).issubset(train):
            raise ValueError('Held-out pose must never enter calibration')
        if any(not set(obs).issubset(train) for obs in tracks.values()):
            raise ValueError('Held-out pixel tracks must never enter calibration')
        if not all(np.isfinite(x).all() for x in self.points.values()):
            raise ValueError('Finite observed points are required')
        self.ids = sorted(self.points)
        if len(self.ids) < 8 or len(self.poses) < 3:
            raise ValueError('At least eight points and three training views required')
        self.camera_ids = sorted(self.poses)
        self.free = [i for i in self.camera_ids if i not in {self.first, self.second}]
        self.camera_index = {i: j for j, i in enumerate(self.camera_ids)}
        self.point_index = {t: i for i, t in enumerate(self.ids)}
        observations = [(fi, t, np.asarray(uv, dtype=float)) for t in self.ids
                        for fi, uv in sorted(tracks[t].items()) if fi in self.poses]
        self.observations = observations
        self.obs_c = np.array([self.camera_index[fi] for fi, t, uv in observations])
        self.obs_p = np.array([self.point_index[t] for fi, t, uv in observations])
        self.obs_uv = np.array([uv for fi, t, uv in observations])
        if not np.isfinite(self.obs_uv).all():
            raise ValueError('Finite raw training pixels are required')
        self.offset = 5 + len(self.free) * 6
        self.camera_offsets = {self.second: (0, 5)}
        self.camera_offsets.update({fi: (5 + j * 6, 6) for j, fi in enumerate(self.free)})
        self.initial = np.r_[cv2.Rodrigues(self.poses[self.second][0])[0].ravel(),
                             direction_angles(self.poses[self.second][1]),
                             np.concatenate([np.r_[cv2.Rodrigues(self.poses[fi][0])[0].ravel(), self.poses[fi][1]]
                                             for fi in self.free]) if self.free else np.empty(0),
                             np.array([self.points[t] for t in self.ids]).ravel()]

    def decode(self, x):
        poses = {self.first: (np.eye(3), np.zeros(3)),
                 self.second: (cv2.Rodrigues(x[:3])[0], unit_direction(x[3:5]))}
        for j, fi in enumerate(self.free):
            q = x[5 + j * 6:5 + j * 6 + 6]
            poses[fi] = (cv2.Rodrigues(q[:3])[0], q[3:6])
        return poses, x[self.offset:self.offset + len(self.ids) * 3].reshape(-1, 3)

    def residual_and_depth(self, x, focal):
        poses, xyz = self.decode(x)
        rotations = np.array([poses[i][0] for i in self.camera_ids])
        translations = np.array([poses[i][1] for i in self.camera_ids])
        q = np.einsum('nij,nj->ni', rotations[self.obs_c], xyz[self.obs_p]) + translations[self.obs_c]
        z = q[:, 2]
        divisor = np.where(np.abs(z) > 1e-8, z, np.where(z >= 0, 1e-8, -1e-8))
        prediction = focal * q[:, :2] / divisor[:, None] + [self.width / 2, self.height / 2]
        return (prediction - self.obs_uv).ravel(), z

    def sparsity(self, focal_free):
        matrix = lil_matrix((len(self.observations) * 2, len(self.initial) + int(focal_free)), dtype=int)
        for j, (fi, tid, uv) in enumerate(self.observations):
            if fi in self.camera_offsets:
                begin, size = self.camera_offsets[fi]
                matrix[j * 2:j * 2 + 2, begin:begin + size] = 1
            begin = self.offset + self.point_index[tid] * 3
            matrix[j * 2:j * 2 + 2, begin:begin + 3] = 1
        if focal_free:
            matrix[:, -1] = 1
        return matrix.tocsr()

    def fit(self, initial_focal, *, free_focal=True, max_nfev=80):
        lower, upper = np.asarray(PROTOCOL['focal_bounds_factors']) * self.dimension
        if not lower <= initial_focal <= upper:
            raise ValueError('Focal initialization outside predeclared bounds')
        initial = np.r_[self.initial, math.log(initial_focal)] if free_focal else self.initial.copy()
        lo = np.full(len(initial), -np.inf); hi = np.full(len(initial), np.inf)
        # A spherical chart makes baseline unit norm exact; no radial gauge remains.
        lo[4], hi[4] = -np.pi / 2 + 1e-7, np.pi / 2 - 1e-7
        if free_focal:
            lo[-1], hi[-1] = math.log(lower), math.log(upper)
        def residual(x):
            return self.residual_and_depth(x, math.exp(x[-1]) if free_focal else initial_focal)[0]
        result = least_squares(residual, initial, jac_sparsity=self.sparsity(free_focal),
                               bounds=(lo, hi), loss='soft_l1', f_scale=PROTOCOL['robust_scale_px'],
                               x_scale='jac', max_nfev=max_nfev, ftol=1e-6, xtol=1e-7, gtol=1e-7)
        focal = math.exp(result.x[-1]) if free_focal else initial_focal
        r, depth = self.residual_and_depth(result.x, focal)
        poses, xyz = self.decode(result.x)
        errors = np.linalg.norm(r.reshape(-1, 2), axis=1)
        positive = float(np.mean(depth > 0))
        boundary = bool(focal <= lower * (1 + PROTOCOL['focal_boundary_relative_margin']) or
                        focal >= upper * (1 - PROTOCOL['focal_boundary_relative_margin']))
        metrics = {'initial_focal_px': float(initial_focal), 'focal_px': float(focal),
                   'focal_factor': float(focal / self.dimension), 'focal_free': free_focal,
                   'robust_train_cost': robust_cost(r), 'mean_robust_train_cost': robust_cost(r) / len(r),
                   'train_median_px': float(np.median(errors)), 'train_p90_px': float(np.percentile(errors, 90)),
                   'positive_training_depth_fraction': positive, 'at_focal_boundary': boundary,
                   'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
                   'optimizer_message': str(result.message), 'evaluations': int(result.nfev),
                   'optimality': float(result.optimality), 'training_observations': len(self.observations),
                   'heldout_observations_used': 0, 'fixed_first_pose': self.first,
                   'unit_baseline_second_pose': self.second, 'second_pose_rotation_free': True,
                   'second_pose_translation_direction_free': True,
                   'finite': bool(np.isfinite(result.x).all() and np.isfinite(r).all())}
        return {'metrics': metrics, 'model': {'seed': {'a': self.first, 'b': self.second},
                                             'poses': poses, 'points': {t: xyz[i] for i, t in enumerate(self.ids)}}}


def select_training_winner(attempts):
    eligible = [i for i, a in enumerate(attempts) if a['metrics']['finite'] and
                a['metrics']['positive_training_depth_fraction'] >= PROTOCOL['minimum_positive_training_depth_fraction']]
    if not eligible:
        return None
    return min(eligible, key=lambda i: (attempts[i]['metrics']['robust_train_cost'], i))


def observability(attempts, profiles, selected):
    """Conservative predeclared diagnostic, never a camera ground-truth claim."""
    reasons = []
    if selected is None:
        return {'supported': False, 'reasons': ['no_finite_positive_depth_training_candidate'], 'relative_focal_spread': None}
    rows = [a['metrics'] for a in attempts]
    if len(rows) != 3 or not all(r['optimizer_success'] and r['finite'] for r in rows):
        reasons.append('not_all_predeclared_starts_converged')
    if any(r['at_focal_boundary'] for r in rows):
        reasons.append('focal_solution_reaches_predeclared_boundary')
    if any(r['positive_training_depth_fraction'] < PROTOCOL['minimum_positive_training_depth_fraction'] for r in rows):
        reasons.append('a_start_lacks_positive_depth_support')
    values = [r['focal_px'] for r in rows]
    spread = (max(values) - min(values)) / max(float(np.median(values)), 1e-12) if values else None
    if spread is None or spread > PROTOCOL['maximum_multistart_focal_relative_spread']:
        reasons.append('multistart_focal_solutions_disagree')
    winner = rows[selected]
    focal = winner['focal_px']
    prows = [p['metrics'] for p in profiles]
    if len(prows) != 3 or not all(r['optimizer_success'] and r['finite'] for r in prows):
        reasons.append('not_all_predeclared_fixed_focal_profiles_converged')
    separated = [r for r in prows if abs(r['focal_px'] / focal - 1) >= PROTOCOL['minimum_profile_focal_separation_fraction']]
    if not any(r['focal_px'] < focal for r in separated) or not any(r['focal_px'] > focal for r in separated):
        reasons.append('fixed_focal_profiles_do_not_bracket_solution')
    def costs_separate(row):
        delta = row['mean_robust_train_cost'] - winner['mean_robust_train_cost']
        return delta >= max(PROTOCOL['minimum_profile_mean_cost_increase'],
                            winner['mean_robust_train_cost'] * PROTOCOL['minimum_profile_relative_cost_increase'])
    if not separated or not all(costs_separate(r) for r in separated):
        reasons.append('alternative_focals_have_similar_training_cost')
    return {'supported': not reasons, 'reasons': reasons, 'relative_focal_spread': spread,
            'camera_calibrated': False, 'ground_truth_intrinsics_available': False,
            'selection_used_heldout': False}
