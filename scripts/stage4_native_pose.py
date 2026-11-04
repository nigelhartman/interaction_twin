#!/usr/bin/env python3
"""Versioned native-material, CPU-only initialization for an untouched GLB.

The output is a candidate, never independent 6D truth.  The reference silhouette
is fitting evidence.  No old pose, manual corner, dimension, or depth file is read.
"""
import argparse
import hashlib
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_appearance import load_native_glb, render_native_mesh, ray_native_mesh_points, render_native_mask
from native_alignment import native_execution_receipt, native_model_provenance, native_representation_receipt


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_case_root(selection_path, source_video_sha256):
    """Resolve nested selection exports through a matching ancestor manifest."""
    for ancestor in Path(selection_path).resolve().parents:
        manifest = ancestor / 'manifest.json'
        if manifest.is_file():
            candidate = json.loads(manifest.read_text())
            if candidate.get('source', {}).get('sha256') == source_video_sha256:
                return ancestor
    raise ValueError('No ancestor case manifest matches the source video SHA256')


def mask_seed_provenance(source):
    """Preserve model-generated seeds separately from manual visual annotations."""
    provenance = source.get('mask_provenance', {})
    origin = provenance.get('prompt_origin', 'unknown')
    explicit = provenance.get('manual_seed_boxes_used')
    if isinstance(explicit, bool):
        manual = explicit
    else:
        origins = [origin, provenance.get('original_seed_origin', '')]
        origins += [p.get('origin', '') for p in provenance.get('prompts', [])]
        manual_tokens = ['assistant_visual_annotation', 'manual_annotation', 'human_annotation', 'manual_seed']
        if any(any(token in str(value) for token in manual_tokens) for value in origins):
            manual = True
        elif origin == 'model_generated_prompt':
            manual = False
        else:
            manual = None
    return {'mask_prompt_origin': origin, 'manual_seed_boxes_used': manual,
            'sam_masks_inherit_assistant_prompts': manual,
            'mask_prompt_provider': provenance.get('prompt_provider', provenance.get('original_seed_origin', 'not recorded')),
            'source_mask_provenance': provenance}


def mask_seed_note(seed):
    if seed['manual_seed_boxes_used'] is True:
        return 'Source masks inherit earlier manual visual seed annotations, as recorded in their provenance.'
    if seed['manual_seed_boxes_used'] is False:
        return f"Source mask seed origin: {seed['mask_prompt_origin']}; provider: {seed['mask_prompt_provider']}. No manual seed boxes are recorded."
    return f"Source mask seed assistance is unverified; recorded origin: {seed['mask_prompt_origin']}."


def intrinsics(width, height, horizontal_fov_deg):
    if not 10 < horizontal_fov_deg < 150:
        raise ValueError('Unsupported field of view')
    f = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.]])


def resized_intrinsics(k, source_size, target_size):
    """OpenCV pixel-centre resize, without substituting model-input crop pixels."""
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    a = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    return a @ k


def project(vertices, rotation, translation, k):
    camera = np.asarray(vertices) @ rotation.T + translation
    if np.any(camera[:, 2] <= 1e-6):
        return None
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def silhouette(native_mesh, rotation, translation, k, size, origin_offset=None):
    """Use the final native material/culling rasterizer at search resolution.

    Only optimization coordinates are centred. The original scene vertices,
    node transforms, primitive/UV/material associations and raw GLB stay intact.
    """
    width, height = size
    if origin_offset is not None:
        translation = translation - rotation @ origin_offset
    return render_native_mask(native_mesh, k, rotation, translation, width, height)['mask']


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def appearance_similarity(source_rgb, generated_rgb, overlap_mask, blur_sigma=3.):
    """Same-reference, illumination-normalized RGB correlation, not validation.

    Eroding the overlap excludes boundary/background mixing. Flat channels do
    not supply orientation evidence. Correlation is invariant to a positive
    per-channel affine exposure change, but not arbitrary recolouring.
    """
    if source_rgb.shape != generated_rgb.shape or source_rgb.shape[:2] != overlap_mask.shape:
        raise ValueError('Appearance inputs must share full-frame coordinates')
    margin = max(1, round(blur_sigma * 1.5))
    support = cv2.erode(overlap_mask.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)
    if np.count_nonzero(support) < 64:
        return {'available': False, 'reason': 'insufficient_overlap', 'correlation': None,
                'normalized_score': None, 'support_pixels': int(support.sum())}
    a = cv2.GaussianBlur(source_rgb.astype(np.float32), (0, 0), blur_sigma)[support]
    b = cv2.GaussianBlur(generated_rgb.astype(np.float32), (0, 0), blur_sigma)[support]
    a -= a.mean(0)
    b -= b.mean(0)
    denominator = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    valid = denominator > len(a)  # At least roughly one RGB level of variance.
    if not np.any(valid):
        return {'available': False, 'reason': 'flat_appearance_is_not_orientation_evidence',
                'correlation': None, 'normalized_score': None, 'support_pixels': int(support.sum())}
    correlation = float(np.clip(np.mean(np.sum(a * b, axis=0)[valid] / denominator[valid]), -1, 1))
    return {'available': True, 'correlation': correlation, 'normalized_score': .5 * (correlation + 1),
            'support_pixels': int(support.sum()), 'channels_used': int(valid.sum()),
            'blur_sigma_source_pixels': blur_sigma, 'independent_validation': False}


def reference_rank_score(reference_iou, appearance, appearance_weight):
    if not 0 <= appearance_weight <= .5:
        raise ValueError('Appearance weight must be in [0, 0.5]')
    # An unavailable channel is neutral, rather than a fabricated match.
    score = appearance['normalized_score'] if appearance['available'] else .5
    return float((1 - appearance_weight) * reference_iou + appearance_weight * score)


def mask_box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty observed mask')
    return np.array([x.min(), y.min(), x.max(), y.max()], float)


def initial_translation(vertices, rotation, k, target):
    """Generic bounding-box initialization, at one fixed arbitrary mesh gauge."""
    box = mask_box(target)
    target_centre = (box[:2] + box[2:]) / 2
    target_extent = np.maximum(box[2:] - box[:2], 2)
    rotated = vertices @ rotation.T
    z = max(2 * np.ptp(vertices, axis=0).max(),
            float(np.mean(np.diag(k)[:2] * np.ptp(rotated[:, :2], axis=0) / target_extent)))
    t = np.r_[(target_centre - k[:2, 2]) / np.diag(k)[:2] * z, z]
    for _ in range(5):
        uv = project(vertices, rotation, t, k)
        if uv is None:
            t[2] *= 2
            continue
        projected_extent = np.maximum(np.ptp(uv, axis=0), 1)
        t[2] *= float(np.sqrt(np.prod(projected_extent / target_extent)))
        t[2] = max(t[2], -rotated[:, 2].min() + .2)
        uv = project(vertices, rotation, t, k)
        centre = (uv.min(0) + uv.max(0)) / 2
        t[:2] += (target_centre - centre) * t[2] / np.diag(k)[:2]
    return t


def rotation_distance_deg(a, b):
    return float(np.rad2deg(Rotation.from_matrix(a @ b.T).magnitude()))


def generic_orientations(random_count=144):
    # Cube group ensures deterministic orthogonal views.  Uniform SO(3) samples
    # provide oblique views without an object-specific chosen orientation.
    return np.concatenate([Rotation.create_group('O').as_matrix(),
                           Rotation.random(random_count, random_state=17).as_matrix()])


def refine(native_mesh, origin_offset, k, target, rotation, translation, max_evaluations=240):
    h, w = target.shape
    observed_boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    observed_distance = cv2.distanceTransform(1 - observed_boundary, cv2.DIST_L2, 3)
    q0 = np.r_[np.zeros(3), translation[:2] / translation[2], np.log(translation[2])]
    limits = np.array([.6, .6, .6, .12, .12, .35])
    best = {'value': np.inf, 'q': q0.copy()}
    calls = 0

    def unpack(q):
        r = Rotation.from_rotvec(q[:3]).as_matrix() @ rotation
        z = np.exp(q[5])
        return r, np.r_[q[3:5] * z, z]

    def loss(q):
        nonlocal calls
        calls += 1
        r, t = unpack(q)
        rendered = silhouette(native_mesh, r, t, k, (w, h), origin_offset)
        if not rendered.any():
            return 5.
        boundary = cv2.morphologyEx(rendered.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        distance = cv2.distanceTransform(1 - boundary.astype(np.uint8), cv2.DIST_L2, 3)
        chamfer = .5 * (observed_distance[boundary].mean() + distance[observed_boundary > 0].mean())
        value = 1 - iou(rendered, target) + .25 * chamfer / np.hypot(w, h)
        if value < best['value']:
            best.update(value=float(value), q=q.copy())
        return float(value)

    result = minimize(loss, q0, method='Powell', bounds=np.c_[q0 - limits, q0 + limits],
                      options={'maxfev': max_evaluations, 'maxiter': 12, 'xtol': .001, 'ftol': .0001})
    r, t = unpack(best['q'])
    return {'rotation': r, 'translation': t, 'loss': best['value'], 'evaluations': calls,
            'optimizer_success': bool(result.success), 'budget_exhausted': bool(result.nfev >= max_evaluations)}


def usable_pose_gate(reference_iou, independent_pose_support, ambiguous):
    reasons = []
    if reference_iou < .85:
        reasons.append('reference_silhouette_iou_below_0.85')
    if ambiguous:
        reasons.append('competing_rotations_fit_the_reference_silhouette')
    if not independent_pose_support:
        reasons.append('no_independent_depth_or_multiview_pose_support')
    return {'accepted': not reasons, 'reasons': reasons,
            'reference_silhouette_is_independent_pose_truth': False}


def track_neighbors(meta, source_rgb, source_mask, vertices, faces, k, reference_fit, count, out):
    """Conditional SIFT/raycast/PnP check on real, unused neighboring images.

    Held-out 2D matches are genuine image measurements. Their canonical 3D points
    are conditional on the fitted reference and generated mesh; this is NOT an
    independent geometry, calibration, or physical 6D pose measurement.
    """
    if not count:
        return []
    selection_path = Path(meta['source']['selection_manifest_path'])
    if sha256(selection_path) != meta['source']['selection_manifest_sha256']:
        raise ValueError('Reference selection manifest changed')
    selection = json.loads(selection_path.read_text())
    case = find_case_root(selection_path, meta['source']['video_sha256'])
    reference_time = meta['source']['timestamp_s']
    items = [x for x in selection['all_ranked_candidates']
             if .3 <= reference_time - x['timestamp_s'] <= 5 and x['score'] >= .7]
    items.sort(key=lambda x: x['timestamp_s'], reverse=True)
    items = items[:count]
    sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=.025)
    reference_mask = cv2.erode(source_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) * 255
    ref_keypoints, ref_desc = sift.detectAndCompute(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY), reference_mask)
    results = []
    if ref_desc is None:
        return [{'status': 'unsupported', 'reason': 'no_reference_SIFT_descriptors'}]
    for item in items:
        rgb_path, mask_path = case / item['image_file'], case / item['mask_file']
        if sha256(rgb_path) != item['image_sha256'] or sha256(mask_path) != item['mask_sha256']:
            raise ValueError('Neighbor source hash mismatch')
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')) > 127
        result = {'timestamp_s': item['timestamp_s'], 'sample_index': item['sample_index'],
                  'image_sha256': sha256(rgb_path), 'mask_sha256': sha256(mask_path),
                  'status': 'unsupported', 'geometry_truth_is_independent': False,
                  'point_ids': 'SIFT_reference_descriptor_indices',
                  'training_and_holdout_split': 'reference_descriptor_id modulo 3; remainder zero held out'}
        kp, desc = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                        cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) * 255)
        if desc is None or len(desc) < 2:
            result['reason'] = 'no_neighbor_descriptors'
            results.append(result)
            continue
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = [a for pair in matcher.knnMatch(ref_desc, desc, k=2) if len(pair) == 2
                   for a, b in [pair] if a.distance < .7 * b.distance]
        reverse = {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(desc, ref_desc, k=2) if len(pair) == 2
                   for a, b in [pair] if a.distance < .7 * b.distance}
        matches = [x for x in forward if reverse.get(x.trainIdx) == x.queryIdx]
        ids = np.array([x.queryIdx for x in matches], dtype=int)
        ref_uv = np.array([ref_keypoints[x.queryIdx].pt for x in matches], dtype=float).reshape(-1, 2)
        observed_uv = np.array([kp[x.trainIdx].pt for x in matches], dtype=float).reshape(-1, 2)
        canonical, hits = ray_native_mesh_points(ref_uv, reference_fit['_native_mesh'], k, reference_fit['rotation'], reference_fit['translation'])
        canonical, observed_uv, ids = canonical[hits], observed_uv[hits], ids[hits]
        train, heldout = ids % 3 != 0, ids % 3 == 0
        result.update(mutual_ratio_matches=len(matches), canonical_ray_hits=int(hits.sum()),
                      training_point_ids=ids[train].tolist(), heldout_point_ids=ids[heldout].tolist())
        if train.sum() < 12 or heldout.sum() < 6:
            result['reason'] = 'insufficient_disjoint_training_or_heldout_matches'
            results.append(result)
            continue
        cv2.setRNGSeed(17)
        rvec = cv2.Rodrigues(reference_fit['rotation'])[0]
        tvec = reference_fit['translation'].copy().reshape(3, 1)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(canonical[train], observed_uv[train], k, None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, iterationsCount=200,
            reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 8:
            result['reason'] = 'training_PnP_RANSAC_failed'
            results.append(result)
            continue
        rvec, tvec = cv2.solvePnPRefineLM(canonical[train][inliers.ravel()], observed_uv[train][inliers.ravel()], k, None, rvec, tvec)
        rotation, translation = cv2.Rodrigues(rvec)[0], tvec.ravel()
        prediction = project(canonical[heldout], rotation, translation, k)
        if prediction is None:
            result['reason'] = 'heldout_points_behind_camera'
            results.append(result)
            continue
        errors = np.linalg.norm(prediction - observed_uv[heldout], axis=1)
        rendered = render_native_mesh(reference_fit['_native_mesh'], k, rotation, translation, rgb.shape[1], rgb.shape[0])
        full_iou = iou(rendered['mask'], mask)
        supported = np.median(errors) <= 3 and np.percentile(errors, 90) <= 8 and full_iou >= .65
        result.update(status='conditional_internal_support' if supported else 'unsupported',
                      reason='internal_gates_passed_not_independent_pose_truth' if supported else 'internal_holdout_or_mask_gate_failed',
                      training_inliers=len(inliers), heldout_error_median_px=float(np.median(errors)),
                      heldout_error_p90_px=float(np.percentile(errors, 90)), reference_to_neighbor_rotation_deg=rotation_distance_deg(rotation, reference_fit['rotation']),
                      full_mesh_visible_mask_iou=full_iou, rotation=rotation.tolist(), translation=translation.tolist(),
                      correspondences={'reference_uv': ref_uv[hits].tolist(), 'observed_neighbor_uv': observed_uv.tolist(),
                                       'canonical_xyz_conditional_on_reference': canonical.tolist(), 'point_ids': ids.tolist(),
                                       'heldout_errors_px': errors.tolist()})
        overlay = rgb.copy()
        raw = rendered['mask']
        overlay[raw] = (.45 * rgb[raw] + .55 * rendered['rgb'][raw]).astype(np.uint8)
        for observed, predicted in zip(observed_uv[heldout], prediction):
            cv2.circle(overlay, tuple(np.rint(observed).astype(int)), 3, (30, 255, 130), 1)
            cv2.line(overlay, tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int)), (255, 100, 60), 1)
        filename = f"neighbor-{item['sample_index']:06d}-overlay.png"
        Image.fromarray(overlay).save(out / filename)
        Image.fromarray(rendered['rgb']).save(out / f"neighbor-{item['sample_index']:06d}-raw-rgb.png")
        Image.fromarray((raw * 255).astype(np.uint8)).save(out / f"neighbor-{item['sample_index']:06d}-raw-mask.png")
        result['overlay_file'] = filename
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--search-width', type=int, default=256)
    parser.add_argument('--refine-count', type=int, default=6)
    parser.add_argument('--max-evaluations', type=int, default=220)
    parser.add_argument('--appearance-weight', type=float, default=0., help='Optional same-reference RGB ranking weight, at most 0.5')
    parser.add_argument('--neighbor-count', type=int, default=0, help='At most five earlier quality-ranked source observations')
    args = parser.parse_args()
    started = time.perf_counter()
    if args.out.exists():
        raise ValueError('Preserve prior evidence: choose a fresh output directory')
    if not 96 <= args.search_width <= 512 or not 1 <= args.refine_count <= 12 or not 30 <= args.max_evaluations <= 600:
        raise ValueError('Search settings outside the bounded pilot limits')
    if not 0 <= args.appearance_weight <= .5 or not 0 <= args.neighbor_count <= 5:
        raise ValueError('Appearance/neighbor settings outside bounded pilot limits')
    meta = json.loads(args.manifest.read_text())
    seed_provenance = mask_seed_provenance(meta['source'])
    base = args.manifest.parent
    rgb_path = base / meta['source']['image_path']
    mask_path = base / meta['source']['mask_path']
    mesh_path = base / meta['outputs']['glb_path']
    for key, path in [('image_sha256', rgb_path), ('mask_sha256', mask_path)]:
        if sha256(path) != meta['source'][key]:
            raise ValueError(f'Source hash mismatch: {key}')
    artifact_hash = next(x['sha256'] for x in meta['artifact_files'] if x['file'] == meta['outputs']['glb_path'])
    if sha256(mesh_path) != artifact_hash:
        raise ValueError('Generated GLB hash mismatch')
    mesh = load_native_glb(mesh_path)
    vertices, faces = mesh.vertices, mesh.faces
    centre = (vertices.min(0) + vertices.max(0)) / 2
    centred = vertices - centre
    rgb = np.array(Image.open(rgb_path).convert('RGB'))
    target_full = np.array(Image.open(mask_path).convert('L')) > 127
    height, width = target_full.shape
    if rgb.shape[:2] != target_full.shape:
        raise ValueError('Full-frame RGB and SAM mask differ')
    if [width, height] != [meta['source']['width'], meta['source']['height']]:
        raise ValueError('Source metadata dimensions differ from actual full-frame pixels')
    work_size = (args.search_width, round(height * args.search_width / width))
    target = cv2.resize(target_full.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    k = intrinsics(width, height, 60.)
    small_k = resized_intrinsics(k, (width, height), work_size)
    coarse = []
    for index, rotation in enumerate(generic_orientations()):
        translation = initial_translation(centred, rotation, small_k, target)
        rendered = silhouette(mesh, rotation, translation, small_k, work_size, centre)
        coarse.append({'index': index, 'rotation': rotation, 'translation': translation, 'iou': iou(rendered, target)})
    coarse.sort(key=lambda x: x['iou'], reverse=True)
    # Retain different basins instead of polishing many almost-identical seeds.
    seeds = []
    for item in coarse:
        if all(rotation_distance_deg(item['rotation'], old['rotation']) >= 35 for old in seeds):
            seeds.append(item)
        if len(seeds) == args.refine_count:
            break
    candidates = []
    for seed in seeds:
        fit = refine(mesh, centre, small_k, target, seed['rotation'], seed['translation'], args.max_evaluations)
        fit.update(seed_index=seed['index'], seed_iou=seed['iou'], horizontal_fov_deg=60.)
        fit['translation'] = fit['translation'] - fit['rotation'] @ centre
        fit['iou_search_resolution'] = iou(silhouette(mesh, fit['rotation'], fit['translation'], small_k, work_size), target)
        rendered = render_native_mesh(mesh, k, fit['rotation'], fit['translation'], width, height)
        fit['_rendered'] = rendered
        fit['iou_full_resolution'] = iou(rendered['mask'], target_full)
        fit['appearance'] = appearance_similarity(rgb, rendered['rgb'], rendered['mask'] & target_full)
        fit['reference_combined_rank_score'] = reference_rank_score(fit['iou_full_resolution'], fit['appearance'], args.appearance_weight)
        candidates.append(fit)
    rank_field = 'reference_combined_rank_score' if args.appearance_weight else 'iou_search_resolution'
    candidates.sort(key=lambda x: x[rank_field], reverse=True)
    best = candidates[0]
    full_mask = silhouette(mesh, best['rotation'], best['translation'], k, (width, height))
    fast_full_iou = iou(full_mask, target_full)
    full_iou = fast_full_iou
    alternatives = [x for x in candidates[1:] if rotation_distance_deg(x['rotation'], best['rotation']) >= 45]
    silhouette_ambiguous = bool(alternatives and any(abs(best['iou_search_resolution'] - x['iou_search_resolution']) < .03 for x in alternatives))
    combined_ambiguous = bool(alternatives and best['reference_combined_rank_score'] - max(x['reference_combined_rank_score'] for x in alternatives) < .03)
    ambiguous = combined_ambiguous if args.appearance_weight else silhouette_ambiguous
    fov_sensitivity = []
    for fov in [45., 75.]:
        other_k = intrinsics(width, height, fov)
        other_small = resized_intrinsics(other_k, (width, height), work_size)
        t = initial_translation(centred, best['rotation'], other_small, target)
        fit = refine(mesh, centre, other_small, target, best['rotation'], t, args.max_evaluations)
        fit['translation'] = fit['translation'] - fit['rotation'] @ centre
        fit.update(horizontal_fov_deg=fov, iou_search_resolution=iou(silhouette(mesh, fit['rotation'], fit['translation'], other_small, work_size), target))
        fov_sensitivity.append(fit)
    args.out.mkdir(parents=True)
    panels = []
    detail_panels = []
    box = mask_box(target_full)
    padding = max(24, round(max(box[2:] - box[:2]) * .2))
    display_crop = np.r_[np.maximum(box[:2] - padding, 0), np.minimum(box[2:] + padding + 1, [width, height])].astype(int)
    for index, fit in enumerate(candidates):
        rendered = fit['_rendered']
        raw = rendered['mask']
        if index == 0:
            full_iou = iou(raw, target_full)
        Image.fromarray((raw * 255).astype(np.uint8)).save(args.out / f'candidate-{index + 1:02d}-raw-mask.png')
        Image.fromarray(rendered['rgb']).save(args.out / f'candidate-{index + 1:02d}-raw-rgb.png')
        overlay = rgb.copy()
        overlay[raw] = np.rint(.45 * rgb[raw] + .55 * rendered['rgb'][raw]).astype(np.uint8)
        observed_edges = cv2.morphologyEx(target_full.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        predicted_edges = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        overlay[observed_edges] = [0, 230, 160]
        overlay[predicted_edges] = [255, 110, 65]
        Image.fromarray(overlay).save(args.out / f'candidate-{index + 1:02d}-overlay.png')
        source_panel = ImageOps.contain(Image.fromarray(overlay), (640, 360))
        panel = Image.new('RGB', (640, 360), (13, 19, 28))
        panel.paste(source_panel, ((640 - source_panel.width) // 2, (360 - source_panel.height) // 2))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 640, 30), fill=(13, 19, 28))
        ncc = fit['appearance']['correlation']
        appearance_label = f'{ncc:.3f}' if ncc is not None else 'unavailable'
        draw.text((10, 9), f"Candidate {index + 1} | IoU {iou(raw, target_full):.3f} | RGB correlation {appearance_label} | candidate only", fill='white')
        if index < 4:
            panels.append(panel)
            crop_panel = ImageOps.contain(Image.fromarray(overlay).crop(tuple(display_crop)), (640, 330))
            detail = Image.new('RGB', (640, 360), (13, 19, 28))
            detail.paste(crop_panel, ((640 - crop_panel.width) // 2, 30 + (330 - crop_panel.height) // 2))
            ImageDraw.Draw(detail).text((10, 9), f"Candidate {index + 1} | IoU {iou(raw, target_full):.3f} | RGB correlation {appearance_label} | display crop only", fill='white')
            detail_panels.append(detail)
    preview = Image.new('RGB', (1280, 720), (13, 19, 28))
    for index, panel in enumerate(panels):
        preview.paste(panel, ((index % 2) * 640, (index // 2) * 360))
    preview.save(args.out / 'preview.jpg', quality=92)
    detail_preview = Image.new('RGB', (1280, 720), (13, 19, 28))
    for index, panel in enumerate(detail_panels):
        detail_preview.paste(panel, ((index % 2) * 640, (index // 2) * 360))
    detail_preview.save(args.out / 'detail-preview.jpg', quality=92)
    best['_native_mesh'] = mesh
    neighbors = track_neighbors(meta, rgb, target_full, vertices, faces, k, best, args.neighbor_count, args.out)
    gate = usable_pose_gate(full_iou, independent_pose_support=False, ambiguous=ambiguous)
    def serial_fit(fit):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fit.items() if not key.startswith('_')}
    record = {
        'schema_version': '0.1', 'status': 'rejected_as_usable_pose_single_view_candidate_only',
        'object_id': meta['object_id'], 'source_timestamp_s': meta['source']['timestamp_s'],
        'source_video_sha256': meta['source']['video_sha256'],
        'inputs': {'manifest': str(args.manifest.resolve()), 'manifest_sha256': sha256(args.manifest),
                   'mesh_sha256': artifact_hash, 'rgb_sha256': sha256(rgb_path), 'mask_sha256': sha256(mask_path)},
        'method': 'deterministic generic SO(3) coarse search; bounded six-parameter projected-triangle silhouette refinement; optional generated-appearance ranking',
        'model_input_crop_used_as_full_frame': False,
        'preview': {'source_aspect_ratio_preserved': True, 'detail_display_crop_xyxy': display_crop.tolist(),
                    'display_crop_changes_projection_or_pose': False},
        'coordinates': '+X right, +Y down, +Z forward; camera_xyz = raw_native_mesh_xyz @ R.T + t',
        'fixed_native_mesh_scale': 1., 'physical_scale_known': False,
        'intrinsics': {'matrix': k.tolist(), 'horizontal_fov_deg': 60., 'status': 'generic_uncalibrated_assumption',
                       'width': width, 'height': height},
        'search': {'coarse_rotations': len(coarse), 'search_image_size': work_size,
                   'refined_distinct_seeds': len(candidates), 'max_evaluations_per_seed': args.max_evaluations,
                   'random_seed': 17, 'full_native_mesh_faces_used': len(faces)},
        'candidate': serial_fit(best), 'reference_full_resolution_iou': full_iou,
        'reference_full_resolution_fast_raster_iou': fast_full_iou,
        'final_evaluation_rasterizer': 'native_appearance.render_native_mesh; same material/culling contract as search',
        'native_appearance': native_representation_receipt(mesh),
        'alignment_backend': native_execution_receipt('stage4_native_pose.py'),
        'competing_candidates': [serial_fit(x) for x in candidates[1:]],
        'rotation_ambiguous_within_0_03_iou_and_over_45_deg': silhouette_ambiguous,
        'appearance_ranking': {'enabled': bool(args.appearance_weight), 'weight': args.appearance_weight,
             'objective': '(1-weight)*full-resolution silhouette IoU + weight*(RGB correlation+1)/2',
             'same_reference_fitting_evidence_not_independent_validation': True,
             'flat_appearance_neutral_score': .5, 'ranking_field': rank_field,
             'combined_score_ambiguous_within_0_03_and_over_45_deg': combined_ambiguous,
             'combined_score_gap_to_distinct_rotation': float(best['reference_combined_rank_score'] - max(x['reference_combined_rank_score'] for x in alternatives)) if alternatives else None},
        'intrinsics_sensitivity': [serial_fit(x) for x in fov_sensitivity],
        'neighbor_frame_checks': neighbors,
        'usable_pose_gate': gate,
        'independent_evaluation': {'available': False, 'physical_pose_truth_available': False,
                                   'neighbor_frame_tracking_run': bool(neighbors),
                                   'internal_disjoint_heldout_observations_available': any('heldout_error_median_px' in x for x in neighbors),
                                   'reason': 'No independent calibrated/depth/multiview pose measurement enters this bounded reference test.'},
        'provenance': {'manual_corner_annotations_read': False, 'old_poses_read': False,
                       'manual_geometry_read': False, 'object_dimensions_read': False,
                       'agent_visual_orientation_choice': False, 'geometry_repaired_or_replaced': False,
                       **seed_provenance,
                       **native_model_provenance(meta), 'uv_converted_to_vertex_colours': False},
        'limitations': ['Same-image SAM silhouette and generated appearance are fitting evidence, not independent 6D truth.',
                       'Single-view generated geometry is warped and hidden shape is inferred.',
                       'No independently validated depth, physical scale, contact, cavity, seating or motion is recovered.',
                       'Neighbor canonical points are raycast from the fitted reference; held-out 2D error remains conditional on that geometry and camera.',
                       'Multiple rotations and uncalibrated focal length can produce similar silhouettes.',
                       'Rendered object is not clipped to SAM or hand masks; complete raw projections are retained.'],
        'wall_seconds': time.perf_counter() - started,
    }
    (args.out / 'experiment.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    rows = ''.join(f"<tr><td>{i+1}</td><td>{fit['iou_full_resolution']:.4f}</td><td>{fit['appearance']['correlation']}</td><td>{fit['reference_combined_rank_score']:.4f}</td><td><a href='candidate-{i+1:02d}-overlay.png'>overlay</a></td></tr>" for i, fit in enumerate(candidates))
    neighbor_rows = ''.join(f"<li>{x.get('timestamp_s', 'unknown')} s: {x['status']}; median held-out error {x.get('heldout_error_median_px', 'unavailable')} px" + (f" — <a href='{x['overlay_file']}'>overlay</a>" if x.get('overlay_file') else '') + '</li>' for x in neighbors)
    (args.out / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Generated-mesh pose initialization</title>
<style>body{font:17px system-ui;background:#101722;color:#e8edf5;margin:32px;max-width:1280px}img{width:100%}a{color:#9bc4ff}p{line-height:1.55}</style>
<h1>Automatic silhouette and appearance initialization — candidate only</h1>
<p>The untouched ''' + html.escape(meta['backend']) + ''' GLB is projected over its original source frame. Green: observed SAM outline. Orange: full mesh outline. Each panel comes from a different generic rotation seed. Geometry and source-mask fitting can look plausible while rotation remains wrong.</p>
<p>Optional appearance ranking compares native material appearance with the original RGB inside an eroded overlap. This is same-reference fitting evidence, not independent validation. All alternatives and separate scores are retained.</p>
<p><strong>Rejected as a usable 6D pose:</strong> no independent depth or multiview support is available. ''' + html.escape(mask_seed_note(seed_provenance)) + ''' No old camera poses, manual corners, dimensions or handmade mesh enter this experiment.</p>
<h2>Object detail</h2><p>This display crop comes from the observed mask bounds. Fitting, camera intrinsics and raw renders all retain the original full-frame coordinates.</p><img src="detail-preview.jpg" alt="Automatically cropped detail of candidate projections">
<h2>Full source frame</h2><img src="preview.jpg" alt="Competing projected mesh pose candidates, source aspect ratio preserved"><table><thead><tr><th>Rank</th><th>Silhouette IoU</th><th>RGB correlation</th><th>Combined fit score</th><th>Full render</th></tr></thead><tbody>''' + rows + '''</tbody></table>
<h2>Conditional neighbor checks</h2><p>Green circles are held-out real image matches; orange lines show reprojection error. Canonical points come from this fitted reference and are not independent 3D measurements.</p><ul>''' + neighbor_rows + '''</ul><p><a href="experiment.json">Full fitting evidence and rejection gates</a></p>''')
    print(json.dumps({'output': str(args.out), 'reference_iou': full_iou, 'usable_pose_gate': gate, 'wall_seconds': record['wall_seconds']}))


if __name__ == '__main__':
    main()
