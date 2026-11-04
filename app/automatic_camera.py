"""Source-bound ordinary-video camera hypotheses, never physical calibration.

Each fixed intrinsic matrix receives an independent TRAIN-only bundle fit with
identical observed support and initialization. Reserved views are consulted only
after a durable training decision. No learned depth, mesh, API or new inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import observed_calibration as C
import observed_geometry as G

SOURCE = 'data/observed-geometry-experiment-20261031/camera-body-core-v5'
PRIOR = 'data/observed-depth-completion-20261031/camera-body-moge-prior-v1'
FREE_FOCAL = 'data/observed-geometry-calibration-20261031/shared-focal-v2'
PROTOCOL = {
    'version': 'automatic-camera-fixed-hypotheses-v1',
    'candidate_order': ['assumed_image_size', 'cached_moge2_train_prior'],
    'camera_model': 'fixed_full_pinhole_K_no_distortion',
    'optimizer': 'scipy_trf_sparse_soft_l1', 'robust_scale_px': 1.5,
    'max_evaluations_per_candidate': 1000, 'maximum_seconds_per_candidate': 180,
    'ftol': 1e-6, 'xtol': 1e-7, 'gtol': 1e-7,
    'minimum_positive_training_depth_fraction': .98,
    'minimum_relative_train_cost_separation': .05,
    'minimum_mean_train_cost_separation': .002,
    'selection': 'lowest_converged_finite_positive_depth_TRAIN_cost_only_if_all_fixed_candidates_converge_and_costs_separate',
    'initialization': 'same_saved_observed_mathematical_TRAIN_model_for_every_K',
    'gauge': 'first_seed_identity_second_seed_translation_unit_norm',
    'supports': 'same_590_points_and_2621_registered_TRAIN_observations_no_filtering',
    'reserved': 'original_views_and_disjoint_hashed_anchor_score_track_roles_unchanged',
    'reserved_used_for_selection': False, 'learned_K_source_must_be_TRAIN': True,
    'learned_normalization': 'u_normalized=(u_index+0.5)/width;v_normalized=(v_index+0.5)/height',
    'learned_mapping': 'diag(native_width,native_height,1)@Knorm;subtract_0.5_from_principal_point',
    'free_focal': 'existing_ambiguous_report_diagnostic_only_never_selected_or_refit',
    'maximum_fixed_fit_branches': 2, 'new_api_calls': 0, 'new_learned_inference_calls': 0,
    'physical_scale_known': False, 'physical_shape_verified': False,
    'camera_calibrated': False, 'production_camera_changed': False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def data_file(root, relative, expected=None):
    root = Path(root).resolve()
    relative = str(relative).removeprefix('/files/')
    p = root / relative
    if not relative.startswith('data/') or p.resolve() != p or not p.is_file():
        raise ValueError('Only nonsymlink project data files may be read')
    if expected is not None and digest(p) != expected:
        raise ValueError('Data identity changed: ' + relative)
    return p


def pixel_intrinsics(normalized, width, height):
    """Convert normalized pixel centres to zero-based native image indices."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in [width, height]):
        raise ValueError('Positive integer native width and height required')
    k = np.diag([width, height, 1.]) @ np.asarray(normalized, dtype=float)
    k[0, 2] -= .5
    k[1, 2] -= .5
    return validate_K(k)


def validate_K(k):
    k = np.asarray(k, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.array_equal(k[2], [0, 0, 1]):
        raise ValueError('Finite pinhole K required')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10:
        raise ValueError('Positive focal lengths and zero skew required')
    return k


class CandidateTimeout(RuntimeError):
    pass


class FixedCameraProblem(C.BundleProblem):
    def __init__(self, model, tracks, width, height, training_indices, K):
        super().__init__(model, tracks, width, height, training_indices)
        self.K = validate_K(K)

    def residual_and_depth(self, x, focal=None):
        poses, xyz = self.decode(x)
        rotations = np.array([poses[i][0] for i in self.camera_ids])
        translations = np.array([poses[i][1] for i in self.camera_ids])
        q = np.einsum('nij,nj->ni', rotations[self.obs_c], xyz[self.obs_p]) + translations[self.obs_c]
        z = q[:, 2]
        divisor = np.where(np.abs(z) > 1e-8, z, np.where(z >= 0, 1e-8, -1e-8))
        homogeneous = q @ self.K.T
        prediction = homogeneous[:, :2] / divisor[:, None]
        return (prediction - self.obs_uv).ravel(), z

    def fit_fixed(self, *, maximum_seconds=None, max_evaluations=None):
        maximum_seconds = maximum_seconds or PROTOCOL['maximum_seconds_per_candidate']
        max_evaluations = max_evaluations or PROTOCOL['max_evaluations_per_candidate']
        begin = time.monotonic()
        lo = np.full(len(self.initial), -np.inf)
        hi = np.full(len(self.initial), np.inf)
        lo[4], hi[4] = -np.pi / 2 + 1e-7, np.pi / 2 - 1e-7
        def residual(x):
            if time.monotonic() - begin > maximum_seconds:
                raise CandidateTimeout('Frozen per-branch CPU deadline reached')
            return self.residual_and_depth(x)[0]
        result = least_squares(residual, self.initial.copy(), jac_sparsity=self.sparsity(False),
            bounds=(lo, hi), loss='soft_l1', f_scale=PROTOCOL['robust_scale_px'],
            x_scale='jac', max_nfev=max_evaluations, ftol=PROTOCOL['ftol'],
            xtol=PROTOCOL['xtol'], gtol=PROTOCOL['gtol'])
        r, depth = self.residual_and_depth(result.x)
        poses, xyz = self.decode(result.x)
        errors = np.linalg.norm(r.reshape(-1, 2), axis=1)
        metrics = {'robust_train_cost': C.robust_cost(r),
            'mean_robust_train_cost': C.robust_cost(r) / len(r),
            'train_median_px': float(np.median(errors)),
            'train_p90_px': float(np.percentile(errors, 90)),
            'positive_training_depth_fraction': float(np.mean(depth > 0)),
            'optimizer_success': bool(result.success), 'optimizer_status': int(result.status),
            'optimizer_message': str(result.message), 'evaluations': int(result.nfev),
            'optimality': float(result.optimality), 'training_observations': len(self.observations),
            'training_points': len(self.ids), 'training_cameras': len(self.camera_ids),
            'heldout_observations_used': 0,
            'finite': bool(np.isfinite(result.x).all() and np.isfinite(r).all()),
            'runtime_s': time.monotonic() - begin}
        return {'metrics': metrics, 'model': {'seed': {'a': self.first, 'b': self.second},
            'poses': poses, 'points': {t: xyz[i] for i, t in enumerate(self.ids)}}}


def support_diagnostics(problem):
    point_counts = {t: 0 for t in problem.ids}
    camera_counts = {i: 0 for i in problem.camera_ids}
    adjacency = {i: set() for i in problem.camera_ids}
    point_views = {t: set() for t in problem.ids}
    for fi, tid, uv in problem.observations:
        point_counts[tid] += 1
        camera_counts[fi] += 1
        point_views[tid].add(fi)
    for views in point_views.values():
        for fi in views:
            adjacency[fi].update(views - {fi})
    reached = {problem.first}
    while True:
        expanded = reached | set().union(*(adjacency[x] for x in reached))
        if expanded == reached:
            break
        reached = expanded
    return {'points': len(problem.ids), 'training_observations': len(problem.observations),
        'registered_train_cameras': len(problem.camera_ids),
        'point_minimum_training_views': min(point_counts.values()),
        'points_with_two_training_views': sum(n == 2 for n in point_counts.values()),
        'points_with_at_least_three_training_views': sum(n >= 3 for n in point_counts.values()),
        'camera_minimum_observations': min(camera_counts.values()),
        'camera_observation_counts': {str(i): n for i, n in camera_counts.items()},
        'registered_camera_graph_connected': len(reached) == len(adjacency),
        'unregistered_train_views_do_not_enter_fit': True,
        'two_view_points_are_not_independent_shape_validation': True}


def select_training(attempts):
    """No reserved data accepted; comparison never promotes a production camera."""
    reasons = []
    if len(attempts) != 2:
        reasons.append('both_predeclared_fixed_hypotheses_not_available')
    eligible = []
    for row in attempts:
        m = row.get('training', {})
        ok = bool(m.get('optimizer_success') and m.get('finite') and
            m.get('positive_training_depth_fraction', 0) >= PROTOCOL['minimum_positive_training_depth_fraction'])
        if ok:
            eligible.append(row)
        else:
            reasons.append(row['id'] + '_not_converged_finite_positive_depth')
    preferred = None
    ranking = sorted(eligible, key=lambda r: (r['training']['robust_train_cost'], r['id']))
    gap = None
    if len(ranking) == 2:
        best, other = ranking
        delta = other['training']['mean_robust_train_cost'] - best['training']['mean_robust_train_cost']
        minimum = max(PROTOCOL['minimum_mean_train_cost_separation'],
            best['training']['mean_robust_train_cost'] * PROTOCOL['minimum_relative_train_cost_separation'])
        gap = {'mean_robust_cost_difference': delta, 'required_difference': minimum}
        if delta >= minimum and not reasons:
            preferred = best['id']
        else:
            reasons.append('fixed_hypotheses_have_similar_TRAIN_cost')
    return {'training_preferred_hypothesis_id': preferred,
        'selected_production_camera_id': None,
        'status': 'conditional_TRAIN_fit_preference' if preferred else 'ambiguous_or_unconverged',
        'reasons': reasons, 'cost_separation': gap, 'reserved_used_for_selection': False,
        'camera_calibrated': False, 'physical_shape_verified': False, 'physical_scale_known': False,
        'production_camera_changed': False}


def pack_model(model):
    return {'seed': model['seed'], 'points': {t: x.tolist() for t, x in model['points'].items()},
        'training_poses': {str(fi): {'R': p[0].tolist(), 't': p[1].tolist()} for fi, p in model['poses'].items()}}


def load_model(raw, original):
    return {'seed': original['seed'], 'points': {t: np.array(x) for t, x in raw['points'].items()},
        'poses': {int(fi): (np.array(p['R']), np.array(p['t'])) for fi, p in raw['training_poses'].items()}}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / 'data' / 'automatic-camera-20261101') or output.exists():
        raise ValueError('Use a fresh output below data/automatic-camera-20261101')
    source = root / SOURCE
    old_protocol = json.loads((source / 'protocol.json').read_text())
    original = json.loads((source / 'viewer.json').read_text())
    raw = json.loads((source / 'geometry.json').read_text())
    if original['input_protocol_sha256'] != digest(source / 'protocol.json'):
        raise ValueError('Source viewer protocol hash mismatch')
    if digest(source / 'implementation.py') != old_protocol['implementation_sha256']:
        raise ValueError('Frozen source implementation hash mismatch')
    inputs = {}
    def pin(relative, expected=None):
        path = data_file(root, relative, expected)
        inputs[str(path.relative_to(root))] = digest(path)
        return path
    for name in ['protocol.json', 'geometry.json', 'viewer.json', 'implementation.py']:
        pin(SOURCE + '/' + name)
    source_video = pin(old_protocol['source']['path'], old_protocol['source']['sha256'])
    program = json.loads(pin(old_protocol['program_path'], old_protocol['program_sha256']).read_text())
    for f in old_protocol['input_views']:
        for kind in ['image', 'mask']:
            pin(f[kind + '_url'].removeprefix('/files/'), f[kind + '_sha256'])
    # Hashes only above: reserved images are not decoded before the TRAIN choice.
    if len(original['frames']) != len(old_protocol['input_views']):
        raise ValueError('Frame index correspondence changed')
    for frame, frozen in zip(original['frames'], old_protocol['input_views']):
        if any(frame[k] != frozen[k] for k in ['id', 'timestamp_s', 'image_url', 'mask_url', 'split']):
            raise ValueError('Frozen source frame/split mismatch')
    width, height = original['frames'][0]['width'], original['frames'][0]['height']
    assumed = G.camera_matrix(width, height, 1.)
    candidates = [{'id': 'assumed_image_size', 'label': 'Image-size camera assumption',
        'kind': 'assumed', 'K': assumed.tolist(), 'available': True,
        'learned_prior': False, 'source_frame_id': None}]
    prior_check = {'status': 'unavailable', 'reasons': []}
    prior_root = root / PRIOR
    try:
        pp = json.loads(pin(PRIOR + '/protocol.json').read_text())
        prior = json.loads(pin(PRIOR + '/completion.json').read_text())
        pin(PRIOR + '/implementation.py', pp['implementation_sha256'])
        pin(PRIOR + '/moge-v2-implementation.py', pp['model_module_sha256'])
        if prior['model_module_sha256'] != pp['model_module_sha256']:
            raise ValueError('Cached model-module identity mismatch')
        for relative, sha in pp['input_sha256'].items():
            if relative.startswith('models/'):
                p = root / relative
                if p.resolve() != p or not p.is_file() or digest(p) != sha:
                    raise ValueError('Cached model identity changed')
                inputs[relative] = sha
            else:
                pin(relative, sha)
        source_frame = pp['source_frame']
        matches = [(i, f) for i, f in enumerate(old_protocol['input_views'])
            if f['id'] == source_frame['id'] and f['image_url'] == source_frame['image_url']
            and f['timestamp_s'] == source_frame['timestamp_s']]
        if len(matches) != 1 or matches[0][1]['split'] != 'train' or source_frame['split'] != 'train':
            raise ValueError('Learned prior is not bound to one exact TRAIN frame')
        i, f = matches[0]
        if pp['input_sha256'].get(f['image_url'].removeprefix('/files/')) != f['image_sha256']:
            raise ValueError('Prior raw JPEG binding differs')
        if [source_frame['width'], source_frame['height']] != [width, height]:
            raise ValueError('Prior raw source dimensions differ')
        # The pinned completed implementation resizes the entire cv2-decoded RGB,
        # applies no crop, no rotation, no pad and calls model.infer once on TRAIN.
        expected_size = [round(width * min(1., 512 / max(width, height))),
            round(height * min(1., 512 / max(width, height)))]
        if prior['model_input_size_wh'] != expected_size or pp['protocol']['heldout_video_frames_used']:
            raise ValueError('Cached prior preprocessing or split binding differs')
        k = pixel_intrinsics(prior['model_intrinsics_normalized_record_only'], width, height)
        prior_check = {'status': 'source_bound_cached_TRAIN_prior', 'reasons': [],
            'source_frame_index': i, 'source_frame_id': f['id'], 'source_pts_s': f['timestamp_s'],
            'source_image_sha256': f['image_sha256'], 'source_image_size_wh': [width, height],
            'model_input_size_wh': expected_size, 'normalized_K': prior['model_intrinsics_normalized_record_only'],
            'native_pixel_K': k.tolist(), 'crop_or_rotation_or_padding_applied': False,
            'preprocessing': 'full_frame_cv2_INTER_AREA_resize_then_BGR_to_RGB_CHW_divide_255',
            'heldout_images_used_for_prior': False, 'model_provenance': pp['model_provenance'],
            'prior_depth_or_mesh_used_in_fit': False}
        candidates.append({'id': 'cached_moge2_train_prior', 'label': 'Learned camera prior · MoGe-2',
            'kind': 'learned_intrinsics_prior', 'K': k.tolist(), 'available': True,
            'learned_prior': True, 'source_frame_id': f['id']})
    except (ValueError, KeyError, FileNotFoundError) as exc:
        prior_check['reasons'].append(str(exc))
        candidates.append({'id': 'cached_moge2_train_prior', 'label': 'Learned camera prior · MoGe-2',
            'kind': 'learned_intrinsics_prior', 'K': None, 'available': False,
            'reason': 'cached_prior_source_binding_not_established'})
    tracks = {t: {int(fi): np.array(uv) for fi, uv in obs.items()} for t, obs in raw['tracks'].items()}
    train = {i for i, f in enumerate(old_protocol['input_views']) if f['split'] == 'train'}
    problem = FixedCameraProblem(load_model(raw, original), tracks, width, height, train, assumed)
    support = support_diagnostics(problem)
    if support['points'] != 590 or support['training_observations'] != 2621 or not support['registered_camera_graph_connected']:
        raise ValueError('Frozen 590-point / 2621-observation connected support changed')
    free_root = root / FREE_FOCAL
    free_record = {'id': 'existing_free_focal', 'label': 'Existing free-focal mathematical estimate',
        'kind': 'previous_ambiguous_diagnostic', 'eligible_for_selection': False}
    if not (free_root / 'training-selection.json').exists():
        free_root = root / 'data/observed-geometry-calibration-20261031/shared-focal-v1'
    free_protocol = json.loads(pin(str((free_root / 'protocol.json').relative_to(root))).read_text())
    if free_protocol['source_directory'] != SOURCE or any(
            digest(source / name) != sha for name, sha in free_protocol['source_files'].items()):
        raise ValueError('Preserved free-focal diagnostic belongs to another source model')
    free_selection = json.loads(pin(str((free_root / 'training-selection.json').relative_to(root))).read_text())
    free_record.update(status='ambiguous_not_selected', observability=free_selection['observability'],
        prior_training_selection=free_selection, source_path=str(free_root.relative_to(root)))
    code = {'automatic_camera.py': Path(__file__), 'observed_calibration.py': Path(C.__file__),
        'observed_geometry.py': Path(G.__file__), 'compare_automatic_cameras.py': root / 'scripts/compare_automatic_cameras.py',
        'refine_observed_camera.py': root / 'scripts/refine_observed_camera.py'}
    policy = {'format': 'automatic-camera-protocol', 'version': 1, 'protocol': copy.deepcopy(PROTOCOL),
        'case_id': program['case_id'], 'track_id': old_protocol['track_id'],
        'source': old_protocol['source'], 'source_directory': SOURCE,
        'image_size_wh': [width, height], 'input_files': inputs,
        'source_frames': old_protocol['input_views'], 'registered_train_indices': problem.camera_ids,
        'same_initialization_sha256': hashlib.sha256(problem.initial.tobytes()).hexdigest(),
        'support': support, 'candidates': candidates, 'cached_prior_binding': prior_check,
        'existing_free_focal': free_record, 'code_files': {n: digest(p) for n, p in code.items()},
        'frozen_before_fixed_camera_fit': True, 'new_api_calls': 0, 'new_learned_inference_calls': 0}
    output.mkdir(parents=True)
    dump(output / 'protocol.json', policy)
    dump(output / 'report-schema.json', {'format': 'automatic-camera-comparison', 'version': 1,
        'required': ['case_id', 'track_id', 'source', 'input_protocol_sha256', 'hypotheses',
            'training_decision', 'source_frames', 'limitations', 'physical_scale_known',
            'physical_shape_verified', 'camera_calibrated', 'production_camera_changed'],
        'hypotheses': 'two independent fixed-K TRAIN-only fits plus preserved unselected ambiguous free-focal diagnostic',
        'reserved_results': 'report only, calculated after training-decision.json durable freeze'})
    for name, path in code.items():
        (output / name).write_bytes(path.read_bytes())
        (output / name).chmod(0o444)
    return policy


def frozen_inputs(root, output, expected):
    if digest(output / 'protocol.json') != expected:
        raise ValueError('Frozen comparison protocol hash changed')
    p = json.loads((output / 'protocol.json').read_text())
    for relative, sha in p['input_files'].items():
        path = root / relative
        if path.resolve() != path or digest(path) != sha:
            raise ValueError('A frozen input changed: ' + relative)
    for name, sha in p['code_files'].items():
        path = root / ('scripts/' + name if name in {'compare_automatic_cameras.py', 'refine_observed_camera.py'} else name)
        if digest(path) != sha or digest(output / name) != sha:
            raise ValueError('Comparison code changed after freeze: ' + name)
    return p


def run(root, output, expected_protocol_sha256):
    root, output = Path(root).resolve(), Path(output).resolve()
    p = frozen_inputs(root, output, expected_protocol_sha256)
    if (output / 'training-decision.json').exists() or (output / 'report.json').exists():
        raise ValueError('Comparison already attempted; preserve outputs and use a new version')
    source = root / p['source_directory']
    raw = json.loads((source / 'geometry.json').read_text())
    original = json.loads((source / 'viewer.json').read_text())
    tracks = {t: {int(fi): np.array(uv) for fi, uv in obs.items()} for t, obs in raw['tracks'].items()}
    train = {i for i, f in enumerate(p['source_frames']) if f['split'] == 'train'}
    width, height = p['image_size_wh']
    cv2.setNumThreads(2)
    np.random.seed(3102026)
    cv2.setRNGSeed(3102026)
    rows, fits = [], {}
    for candidate in p['candidates']:
        row = copy.deepcopy(candidate)
        if not row['available']:
            row['status'] = 'source_binding_unavailable'
            rows.append(row)
            continue
        problem = FixedCameraProblem(load_model(raw, original), tracks, width, height, train, row['K'])
        if hashlib.sha256(problem.initial.tobytes()).hexdigest() != p['same_initialization_sha256']:
            raise ValueError('Candidate initial gauge/state changed')
        begin = time.monotonic()
        print(json.dumps({'phase': 'fit_start', 'candidate_id': row['id']}), flush=True)
        try:
            fit = problem.fit_fixed()
            row['training'] = fit['metrics']
            row['status'] = 'converged_TRAIN_fit' if fit['metrics']['optimizer_success'] else 'bounded_fit_not_converged'
            fits[row['id']] = fit
            dump(output / (row['id'] + '-geometry.json'), pack_model(fit['model']))
        except CandidateTimeout as exc:
            row.update(status='bounded_cpu_timeout', reason=str(exc), runtime_s=time.monotonic() - begin)
        dump(output / (row['id'] + '-training.json'), row)
        rows.append(row)
        print(json.dumps({'phase': 'fit_complete', **row}), flush=True)
    decision = select_training(rows)
    decision['frozen_before_reserved_evaluation'] = True
    decision['protocol_sha256'] = expected_protocol_sha256
    dump(output / 'training-decision.json', decision)
    # Imported only after the durable TRAIN decision. This helper uses existing
    # deterministic hashed disjoint pose anchors and scoring observations.
    from scripts.refine_observed_camera import evaluate_fixed_heldout
    for row in rows:
        if row['id'] not in fits:
            continue
        fit = fits[row['id']]
        K = np.asarray(row['K'])
        cv2.setRNGSeed(3102026)
        heldout, predictions, errors = evaluate_fixed_heldout(fit['model'], K, original)
        finite = [e for e in errors if np.isfinite(e)]
        invalid = len(errors) - len(finite)
        row['reserved'] = {'views_total': len(heldout), 'views_registered': sum(x['pose_registered'] for x in heldout),
            'views_passing': sum(x['passes'] for x in heldout), 'scored_observations': len(errors),
            'invalid_or_behind_camera': invalid,
            'median_px': float(np.median(errors)) if errors and not invalid else None,
            'p90_px': float(np.percentile(errors, 90)) if errors and not invalid else None,
            'rows': heldout, 'used_for_selection': False, 'pose_anchor_score_tracks_disjoint': True}
        view = copy.deepcopy(original)
        view.update(id=output.name + '-' + row['id'], title='Camera hypothesis · ' + row['label'],
            status='automatic_camera_diagnostic_only', kind='automatic_camera_hypothesis',
            reason='Independent TRAIN-only fit; intrinsics and physical shape remain unverified.',
            input_protocol_sha256=expected_protocol_sha256, calibration=copy.deepcopy(decision),
            bundle_adjustment=fit['metrics'], heldout_evaluation=heldout, focal_sensitivity=[])
        for point in view['points']:
            point['xyz'] = fit['model']['points'][point['id']].tolist()
            point['heldout_scored_projection_support'] = []
            point['heldout_scored_projection_supported_in_at_least_two_views'] = False
            point['max_training_triangulation_angle_deg'] = G.maximum_track_angle(
                fit['model']['points'][point['id']], tracks[point['id']], fit['model']['poses'])
        view['cameras'] = []
        for fi, frame in enumerate(view['frames']):
            pose = fit['model']['poses'].get(fi) if frame['split'] == 'train' else predictions[fi]['pose']
            frame['registered'] = pose is not None
            if pose is not None:
                view['cameras'].append({'frame_id': frame['id'], 'R': pose[0].tolist(), 't': pose[1].tolist(),
                    'K': K.tolist(), 'split': 'train' if frame['split'] == 'train' else 'heldout_anchor_pose'})
            for observation in frame['observations']:
                tid = observation['track_id']
                observation['predicted_xy'] = observation['error_px'] = None
                if pose is not None and tid in fit['model']['points']:
                    uv, z = G.project(fit['model']['points'][tid][None], pose, K)
                    if np.isfinite(uv).all():
                        observation['predicted_xy'] = uv[0].tolist()
                        observation['error_px'] = float(np.linalg.norm(uv[0] - observation['xy']))
        view['metrics'].update(recovered_points=len(fit['model']['points']), focal_px=float(K[0, 0]),
            heldout_views_passing=row['reserved']['views_passing'],
            heldout_scored_observations=len(errors), heldout_median_px=row['reserved']['median_px'],
            heldout_p90_px=row['reserved']['p90_px'],
            aggregate_sparse_projection_gates_passed=False, focal_sensitivity_stable=False,
            points_independently_scored_in_at_least_two_heldout_views=0,
            heldout_invalid_or_behind_camera_observations=invalid,
            final_training_point_max_angle_median_deg=float(np.median([
                x['max_training_triangulation_angle_deg'] for x in view['points']])),
            **G.shape_metrics(fit['model']['points']))
        view['limitations'] = ['This is a separate fixed-camera diagnostic; no production camera changed.',
            'Sparse points and poses are fit only to the original TRAIN pixels with an arbitrary gauge.',
            'One fixed pinhole K and zero distortion are hypotheses, not measured calibration.',
            'Reserved pose anchors are disjoint from reserved scored tracks and never enter candidate selection.',
            'Physical shape, metric scale, full surfaces, assembly, contact and dynamics remain unverified.']
        name = row['id'] + '-viewer.json'
        dump(output / name, view)
        row['viewer'] = {'path': str((output / name).relative_to(root)), 'sha256': digest(output / name)}
        row['geometry'] = {'path': str((output / (row['id'] + '-geometry.json')).relative_to(root)),
            'sha256': digest(output / (row['id'] + '-geometry.json'))}
    frozen_inputs(root, output, expected_protocol_sha256)
    report = {'format': 'automatic-camera-comparison', 'version': 1,
        'case_id': p['case_id'], 'track_id': p['track_id'], 'source': p['source'],
        'status': 'comparison_complete_diagnostic_only', 'input_protocol_sha256': expected_protocol_sha256,
        'support': p['support'], 'hypotheses': rows + [p['existing_free_focal']],
        'training_decision': decision, 'cached_prior_binding': p['cached_prior_binding'],
        'source_frames': p['source_frames'], 'image_size_wh': p['image_size_wh'],
        'new_api_calls': 0, 'new_learned_inference_calls': 0, 'manual_geometry_or_poses_used': False,
        'physical_scale_known': False, 'physical_shape_verified': False, 'camera_calibrated': False,
        'production_camera_changed': False,
        'limitations': ['Ordinary video is accepted without a calibration board or special capture.',
            'This compares conditional camera hypotheses on one existing video; it is not measured calibration.',
            'The learned K is inferred from exactly one cached TRAIN RGB image; no learned depth or mesh enters these fits.',
            'Zero lens distortion and one fixed intrinsic matrix across frames remain assumptions; stabilization and zoom can violate them.',
            'Two-view point support and one shared starting basin limit camera observability.',
            'Reserved PnP poses use disjoint anchors; reserved score pixels never choose or tune the camera.',
            'No complete physical shape, hidden surfaces, metric scale, assembly, joints, contact or dynamics are established.',
            'Original source, masks, observed models and production camera remain byte-preserved.']}
    dump(output / 'report.json', report)
    write_ui_report(root, output, report)
    return report


def write_ui_report(root, output, report):
    """Small presentation sidecar whose values come only from the saved report."""
    rows = []
    for row in report['hypotheses']:
        training = row.get('training', {})
        reserved = row.get('reserved', {})
        k = row.get('K')
        if row['id'] == 'existing_free_focal':
            old = row['prior_training_selection']
            chosen = old['selected_start_index']
            training = old['starts'][chosen] if chosen is not None else {}
            training = {**training, 'training_cameras': report['support']['registered_train_cameras']}
            focal = training.get('focal_px')
            focal_pair = [focal, focal] if focal else None
        else:
            focal_pair = [k[0][0], k[1][1]] if k else None
        rows.append({'id': row['id'], 'label': row['label'], 'method': row['kind'],
            'focal_px': focal_pair,
            'training': {'median_px': training.get('train_median_px'), 'p90_px': training.get('train_p90_px'),
                'views': training.get('training_cameras', 0), 'observations': training.get('training_observations', 0)},
            'reserved': {'median_px': reserved.get('median_px'), 'p90_px': reserved.get('p90_px'),
                'views': reserved.get('views_total', 0), 'observations': reserved.get('scored_observations', 0),
                'passing_views': reserved.get('views_passing')},
            'status': row['status'], 'reason': ('Preserved ambiguous estimate; never selected or applied.'
                if row['id'] == 'existing_free_focal' else
                'Independent fixed-K fit on identical TRAIN pixels; camera remains unverified.'),
            'viewer': row.get('viewer')})
    preferred = report['training_decision']['training_preferred_hypothesis_id']
    ui = {'format': 'automatic-camera-comparison-ui', 'version': 1,
        'case_id': report['case_id'], 'track_id': report['track_id'],
        'source': report['source'], 'image_size': report['image_size_wh'],
        'outcome': {'status': 'candidate_only' if preferred else 'ambiguous',
            'title': 'Ordinary-video camera comparison',
            'summary': ('One hypothesis has a conditional TRAIN fit preference; no production camera is changed.'
                if preferred else 'These camera hypotheses are not sufficiently separated or converged. Original geometry stays unchanged.')},
        'physical_accuracy_verified': False, 'object_metric_scale_established': False,
        'applied_to_geometry': False, 'new_api_calls': 0, 'hypotheses': rows,
        'frames': [{'id': f['id'], 'pts_s': f['timestamp_s'],
            'path': f['image_url'].removeprefix('/files/'), 'sha256': f['image_sha256'],
            'split': 'train' if f['split'] == 'train' else 'reserved',
            'role': ('Cached learned-camera input · TRAIN' if f['id'] == report['cached_prior_binding'].get('source_frame_id')
                else 'TRAIN feature evidence' if f['split'] == 'train' else 'Reserved projection check')}
            for f in report['source_frames']],
        'limitations': report['limitations'], 'provenance': {'report': {
            'path': str((output / 'report.json').relative_to(root)), 'sha256': digest(output / 'report.json')}}}
    dump(output / 'ui-report.json', ui)
    return ui
