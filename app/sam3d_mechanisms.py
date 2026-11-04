"""Fit fixed, mesh-anchored revolute attachments to a SAM video scene.

Every child transform is evaluated through one persistent joint. Observational
poses remain fitting targets, never a free-floating fallback. Joint points are
actual native mesh vertices selected automatically near cited 2D hinge seams.
This establishes structural connectivity, not verified physical joint recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

import upgrade_bundles as U
import sam3d_common_scene as S
from native_appearance import load_native_glb

ROOT = Path(__file__).resolve().parent
FORMAT = 'sam3d-constrained-mechanisms'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def rotation_pose(matrix):
    matrix = np.asarray(matrix, float)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.array_equal(matrix[3], [0, 0, 0, 1]), 'Finite native similarity required')
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    require(scale > 0 and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3) * scale**2, atol=1e-7), 'Proper positive native scale required')
    return matrix[:3, :3] / scale, matrix[:3, 3], scale


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float)
    require(axis.shape == (3,) and np.isfinite(axis).all() and np.isclose(np.linalg.norm(axis), 1.), 'Unit joint axis required')
    return Rotation.from_rotvec(axis * float(angle)).as_matrix()


def joint_frames(parent_scale, child_scale, parent_anchor, child_anchor, zero_rotation):
    """A removes parent native scale; B contains child native scale once."""
    require(parent_scale > 0 and child_scale > 0, 'Positive native scales required')
    a = np.eye(4); a[:3, :3] /= parent_scale; a[:3, 3] = np.asarray(parent_anchor) / parent_scale
    b = np.eye(4); b[:3, :3] = np.asarray(zero_rotation) * child_scale
    b[:3, 3] = -np.asarray(zero_rotation) @ np.asarray(child_anchor)
    return a, b


def forward(parent_native, parent_from_joint, axis, angle, joint_from_child_zero):
    rotation = np.eye(4); rotation[:3, :3] = axis_rotation(axis, angle)
    return np.asarray(parent_native) @ np.asarray(parent_from_joint) @ rotation @ np.asarray(joint_from_child_zero)


def project(vertices, matrix, k):
    matrix = np.asarray(matrix)
    camera = np.asarray(vertices) @ matrix[:3, :3].T + matrix[:3, 3]
    pixels = camera @ np.asarray(k).T
    good = np.isfinite(camera).all(1) & (camera[:, 2] > 1e-6)
    return pixels[:, :2] / pixels[:, 2:], good


def line_distance(pixels, endpoints):
    a, b = np.asarray(endpoints, float); direction = b - a; length = np.linalg.norm(direction)
    require(np.isfinite(length) and length >= 3, 'Visible nondegenerate 2D hinge seam required')
    unit = direction / length; along = (pixels - a) @ unit
    perpendicular = np.abs((pixels[:, 0] - a[0]) * unit[1] - (pixels[:, 1] - a[1]) * unit[0])
    outside = np.maximum(-along, 0) + np.maximum(along - length, 0)
    return np.hypot(perpendicular, outside), np.linalg.norm(pixels - (a + b) / 2, axis=1)


def surface_candidates(vertices, poses, hints, k, *, count=24):
    """Actual vertex IDs, restricted to the model's projected hinge-seam region."""
    score_sum = np.zeros(len(vertices)); support = np.zeros(len(vertices), int); distances = np.zeros(len(vertices))
    used = []
    for hint in hints:
        pose = poses.get(hint['source_sample_index'])
        if pose is None or pose['status'] != 'estimated' or pose['camera_from_native_mesh'] is None:
            continue
        require(pose['timestamp_s'] == hint['timestamp_s'], 'Hint/source PTS differs')
        uv, valid = project(vertices, pose['camera_from_native_mesh'], k)
        d, middle = line_distance(uv, hint['endpoints_px'])
        tolerance = max(5., float(hint.get('uncertainty_px', 0)))
        near = valid & (d <= tolerance * 2)
        current = d / tolerance + .15 * middle / tolerance
        score_sum[near] += current[near]; distances[near] += d[near]; support[near] += 1
        used.append(hint)
    # One canonical hinge region must be compatible with every usable cited view,
    # rather than choosing unrelated surface patches independently per frame.
    scores = np.where(support == len(used), score_sum / max(1, len(used)), np.inf)
    eligible = np.flatnonzero(np.isfinite(scores))
    require(len(eligible) >= 3 and used, 'No native mesh surface supports the cited hinge seam')
    region = eligible[np.argsort(scores[eligible])[:min(1600, len(eligible))]]
    centered = vertices[region] - np.mean(vertices[region], axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    require(singular[0] > 1e-7, 'Degenerate canonical hinge-seam region')
    # Keep distinct native points instead of repeated nearly coincident vertices.
    ordered = eligible[np.argsort(scores[eligible])]
    candidates = []; span = np.linalg.norm(np.ptp(vertices, axis=0))
    for identity in ordered:
        if not candidates or np.min(np.linalg.norm(vertices[candidates] - vertices[identity], axis=1)) >= span * .008:
            candidates.append(int(identity))
        if len(candidates) >= count:
            break
    require(len(candidates) >= 3, 'Insufficient distinct mesh seam candidates')
    return {'vertex_ids': candidates, 'points': vertices[candidates], 'pca_axis': vt[0], 'region_vertex_count': len(region),
            'used_hints': used, 'candidate_mean_hint_distance_px': (distances[candidates] / np.maximum(support[candidates], 1)).tolist(),
            'selection': 'automatic_native_surface_vertices_within_two_hint_uncertainty_radii; no geometry edits'}


def twist_angles(relative_rotations, zero_rotation, axis):
    delta = np.asarray(relative_rotations) @ np.asarray(zero_rotation).T
    quaternion = Rotation.from_matrix(delta).as_quat()
    angle = 2 * np.arctan2(quaternion[:, :3] @ axis, quaternion[:, 3])
    return np.unwrap(angle)


def choose_axis(relative_rotations, zero_rotation, parent_region, child_region):
    axes = [parent_region['pca_axis'], zero_rotation @ child_region['pca_axis']]
    delta = Rotation.from_matrix(np.asarray(relative_rotations) @ zero_rotation.T).as_rotvec()
    if np.linalg.norm(delta) > 1e-6:
        axes.append(np.linalg.svd(delta, full_matrices=False)[2][0])
    best = None
    for axis in axes:
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        # Noisy independent silhouette poses can suggest a rotation axis that
        # cuts across the cited hinge seam. Such an axis is inadmissible even
        # when it fits those noisy rotations better than the mesh edge does.
        if abs(float(axis @ parent_region['pca_axis'])) < .85:
            continue
        angles = twist_angles(relative_rotations, zero_rotation, axis)
        predicted = Rotation.from_rotvec(angles[:, None] * axis).as_matrix() @ zero_rotation
        errors = Rotation.from_matrix(predicted @ np.asarray(relative_rotations).transpose(0, 2, 1)).magnitude()
        seam_disagreement = 1 - abs(float(axis @ parent_region['pca_axis']))
        score = float(np.median(errors)) + .2 * seam_disagreement
        if best is None or score < best['score']:
            best = {'axis': axis, 'angles': angles, 'score': score, 'rotation_residual_rad': errors,
                    'axis_origin': 'minimum_robust_relative_rotation_residual_among_parent_child_mesh_seam_PCA_and_motion_candidates'}
    return best


def choose_anchors(parent_points, child_points, predicted_rotations, relative_translations, part_span):
    """Discrete bounded optimization. Both pivots must be real mesh vertices."""
    rotated = np.einsum('tij,cj->tci', predicted_rotations, child_points)
    predicted = parent_points[None, :, None, :] - rotated[:, None, :, :]
    residual = np.linalg.norm(predicted - np.asarray(relative_translations)[:, None, None, :], axis=-1)
    robust = np.mean(np.minimum(residual, max(part_span * .3, 1e-4)), axis=0)
    flat = int(np.argmin(robust)); p, c = np.unravel_index(flat, robust.shape)
    return int(p), int(c), residual[:, p, c]


def seam_zero_rotations(parent_axis, child_axis):
    """Both possible unoriented native seam correspondences, without a bad pose prior.

    A single observed silhouette does not fix a full part orientation. In
    particular, inheriting a noisy *closed* frame can turn a long lid across its
    parent. Native seam alignment fixes that transverse orientation; rotation
    along the seam remains the angle curve's gauge, fitted from all masks.
    """
    parent_axis = np.asarray(parent_axis) / np.linalg.norm(parent_axis)
    child_axis = np.asarray(child_axis) / np.linalg.norm(child_axis)
    return [Rotation.align_vectors([parent_axis], [sign * child_axis])[0].as_matrix() for sign in [1., -1.]]


def mask_anchor_scores(parent_points, child_points, rotations, parent_rows, samples, targets, vertices, k):
    """Cheap all-pairs image objective; anchors remain unchanged mesh vertices.

    This is fitting, not held-out validation. A fixed sparse vertex subset and
    predeclared evenly spaced source samples bound CPU work for future uploads.
    """
    scores = np.zeros((len(parent_points), len(child_points))); support = 0
    for sample in samples:
        if targets.get(sample) is None:
            continue
        rp, tp, _ = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
        relative = rotations[sample]; target = targets[sample]
        shifts = parent_points[:, None, :] - np.einsum('ij,cj->ci', relative, child_points)[None, :, :]
        translations = np.einsum('ij,pcj->pci', rp, shifts) + tp
        rotated = vertices @ (rp @ relative).T
        xyz = rotated[None, None, :, :] + translations[:, :, None, :]
        uvw = xyz @ k.T
        uv = uvw[..., :2] / uvw[..., 2:]
        center = np.mean(uv, axis=2)
        extent = np.maximum(np.max(uv, axis=2) - np.min(uv, axis=2), 1.)
        loss = np.linalg.norm((center - target['center']) / target['extent'], axis=-1)
        loss += .35 * np.linalg.norm(np.log(extent / target['extent']), axis=-1)
        loss[np.min(xyz[..., 2], axis=2) <= 1e-6] = 20.
        scores += np.minimum(loss, 5.); support += 1
    require(support >= 3, 'Three source masks are required for canonical seam attachment fitting')
    return scores / support


def fit_seam_model(parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k, *, root=ROOT):
    """Fit one persistent surface-anchored seam model across the whole clip.

    Independent part rotations only seed an angle, never determine transverse
    native frame alignment. Two native seam signs, discrete mesh-surface anchor
    pairs and a complete angle search are compared against source masks. The
    same axis/anchors/native scales are then retained at every source time.
    """
    axis = np.asarray(p_region['pca_axis'], float)
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    targets = {}
    for sample in paired:
        mask_ref = child_rows[sample].get('source_mask')
        if mask_ref:
            require(S.ref(mask_ref['path'], root) == mask_ref, 'Source joint mask changed')
            targets[sample] = S.mask_shape(np.asarray(Image.open(root / mask_ref['path'])) > 0)
    vertices = child_mesh.vertices[np.linspace(0, len(child_mesh.vertices) - 1, min(700, len(child_mesh.vertices)), dtype=int)] * sc
    sparse = vertices[::max(1, len(vertices) // 100)]
    ppoints, cpoints = p_region['points'] * sp, c_region['points'] * sc
    selected = [paired[i] for i in np.linspace(0, len(paired) - 1, min(12, len(paired)), dtype=int)]
    relatives = {}
    translations = []
    for sample in paired:
        rp, tp, _ = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
        rc, tc, _ = rotation_pose(child_rows[sample]['camera_from_native_mesh'])
        relatives[sample] = rp.T @ rc; translations.append(rp.T @ (tc - tp))
    choices = []
    for seam_sign, zero in enumerate(seam_zero_rotations(axis, c_region['pca_axis'])):
        initial = twist_angles([relatives[i] for i in paired], zero, axis)
        angles = dict(zip(paired, initial)); rotations = {i: axis_rotation(axis, angles[i]) @ zero for i in paired}
        pi, ci, _ = choose_anchors(ppoints, cpoints, [rotations[i] for i in paired], translations, float(np.linalg.norm(np.ptp(parent_mesh.vertices * sp, axis=0))))
        for iteration in range(3):
            # All candidates use the SAME two surface anchors, axis and native
            # scale. Global angle search prevents inherited 90-degree mistakes.
            previous_q = None; losses = []; rows = {}
            for sample in paired:
                target = targets.get(sample)
                candidates = np.arange(-math.pi, math.pi, math.radians(10))
                if previous_q is not None:
                    candidates = np.r_[candidates, previous_q + np.deg2rad([-8, -3, 0, 3, 8])]
                best = None
                rp, tp, _ = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
                for q in candidates:
                    if previous_q is not None:
                        q += round((previous_q - q) / (2 * math.pi)) * 2 * math.pi
                    relative = axis_rotation(axis, q) @ zero
                    matrix = np.eye(4); matrix[:3, :3] = rp @ relative * sc
                    matrix[:3, 3] = tp + rp @ (ppoints[pi] - relative @ cpoints[ci])
                    uv, valid = project(vertices / sc, matrix, k)
                    if not valid.all():
                        continue
                    loss, iou = S.silhouette_score(uv, target) if target is not None else (1., None)
                    temporal = 0. if previous_q is None else min(abs(q - previous_q), math.pi) / math.pi
                    objective = loss + .035 * temporal
                    if best is None or objective < best['objective']:
                        best = {'q': float(q), 'objective': float(objective), 'convex_silhouette_fit_iou': iou,
                                'rotation_residual_deg': float(np.rad2deg(Rotation.from_matrix(relative @ relatives[sample].T).magnitude())),
                                'matrix': matrix}
                if best is not None:
                    previous_q = best['q']; rows[sample] = best; angles[sample] = best['q']; losses.append(best['objective'])
            rotations = {i: axis_rotation(axis, angles[i]) @ zero for i in paired}
            if iteration < 2:
                scores = mask_anchor_scores(ppoints, cpoints, rotations, parent_rows, selected, targets, sparse, k)
                candidates = np.argsort(scores, axis=None)[:min(8, scores.size)]
                best_pair = None
                # Re-score only a bounded shortlist with the actual convex-mask
                # objective rather than pretending moment residual is an IoU.
                for flat in candidates:
                    p, c = np.unravel_index(flat, scores.shape); values = []
                    for sample in selected:
                        if targets.get(sample) is None:
                            continue
                        rp, tp, _ = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
                        relative = rotations[sample]
                        matrix = np.eye(4); matrix[:3, :3] = rp @ relative * sc
                        matrix[:3, 3] = tp + rp @ (ppoints[p] - relative @ cpoints[c])
                        uv, valid = project(vertices / sc, matrix, k)
                        values.append(S.silhouette_score(uv, targets[sample])[0] if valid.all() else 2.)
                    score = float(np.mean(values))
                    if best_pair is None or score < best_pair[0]:
                        best_pair = (score, int(p), int(c))
                _, pi, ci = best_pair
        # Opposite seam directions can give almost identical silhouettes while
        # turning an opaque lid inside out. A weak all-frame native orientation
        # prior disambiguates that discrete sign; it cannot tilt the hinge axis
        # across the mesh seam or override the global mask fit per frame.
        orientation_prior = float(np.median([Rotation.from_matrix((axis_rotation(axis, rows[i]['q']) @ zero) @ relatives[i].T).magnitude() for i in rows])) / math.pi
        choices.append({'axis': axis, 'zero': zero, 'pi': pi, 'ci': ci, 'rows': rows,
                        'score': float(np.mean(losses)) + .10 * orientation_prior,
                        'mask_objective': float(np.mean(losses)), 'orientation_sign_prior': orientation_prior, 'seam_sign': seam_sign,
                        'target_count': len(targets), 'fitting_sample_indices': selected})
    result = min(choices, key=lambda row: row['score'])
    result['native_seam_sign_candidates'] = [{key: row[key] for key in ['seam_sign', 'score', 'mask_objective', 'orientation_sign_prior']} for row in choices]
    return result


def cavity_rim_frame(vertices, seam_axis):
    """Conditional cavity evidence from central occupancy at opposite faces.

    No semantic object name or coordinate axis is used. The sparse central
    extreme face is a candidate opening; symmetric/flat parts remain unknown.
    """
    axis = np.asarray(seam_axis, float); axis /= np.linalg.norm(axis)
    sampled = vertices[::max(1, len(vertices) // 20000)]
    normal = np.linalg.svd(sampled - sampled.mean(0), full_matrices=False)[2][-1]
    normal -= axis * float(axis @ normal); normal /= np.linalg.norm(normal)
    across = np.cross(normal, axis); basis = np.stack([axis, across, normal], axis=1)
    coordinates = vertices @ basis; low, high = np.percentile(coordinates, [1, 99], axis=0)
    normalized = (coordinates - (low + high) / 2) / (high - low)
    central = np.linalg.norm(normalized[:, :2], axis=1) < .30
    occupancies = []
    for sign in [-1., 1.]:
        face = sign * normalized[:, 2] > .3
        occupancies.append(float(np.mean(central[face])))
    opening_sign = [-1., 1.][int(np.argmin(occupancies))]
    ratio = min(occupancies) / max(max(occupancies), 1e-8)
    # Both a resolved solid opposite face and an asymmetric sparse face are
    # required. This prior is deliberately unavailable for generic flat doors.
    if ratio >= .75 or max(occupancies) < .05:
        return None
    normal *= opening_sign; across = np.cross(normal, axis); basis = np.stack([axis, across, normal], axis=1)
    coordinates = vertices @ basis; low, high = np.percentile(coordinates, [1, 99], axis=0)
    rim_center = basis @ np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]])
    rim_ids = np.flatnonzero(coordinates[:, 2] >= high[2] - .13 * (high[2] - low[2]))
    return {'axis': axis, 'normal': normal, 'rim_center': rim_center, 'rim_ids': rim_ids,
            'opposite_face_central_occupancies': occupancies, 'opening_face_occupancy_ratio': ratio,
            'method': 'conditional_sparse_central_extreme_face_and_opposite_solid_face_from_unchanged_mesh'}


def cavity_mating_model(model, parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k):
    """Align cavity rims/footprints, then select actual coincident rim vertices.

    Conditional on independently resolved cavities in BOTH native parts. This
    prevents surface-connected but permanently offset lids whose pivots were
    selected from noisy standalone translations or broad image seam hints.
    """
    parent = cavity_rim_frame(parent_mesh.vertices, model['axis'])
    child = cavity_rim_frame(child_mesh.vertices, c_region['pca_axis'])
    if parent is None or child is None:
        model['cavity_mating_prior'] = {'status': 'unavailable', 'reason': 'both_native_parts_do_not_have_resolved_asymmetric_cavity_faces'}
        return model
    relatives = []
    for sample in paired:
        rp = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])[0]
        rc = rotation_pose(child_rows[sample]['camera_from_native_mesh'])[0]
        relatives.append(rp.T @ rc)
    choices = []
    for sign in [-1., 1.]:
        zero = Rotation.align_vectors([parent['axis'], -parent['normal']], [child['axis'] * sign, child['normal']])[0].as_matrix()
        angles = twist_angles(relatives, zero, parent['axis'])
        predicted = Rotation.from_rotvec(angles[:, None] * parent['axis']).as_matrix() @ zero
        residual = Rotation.from_matrix(predicted @ np.asarray(relatives).transpose(0, 2, 1)).magnitude()
        translation = parent['rim_center'] * sp - zero @ (child['rim_center'] * sc)
        pv = parent_mesh.vertices[parent['rim_ids']] * sp
        cv = child_mesh.vertices[child['rim_ids']] * sc @ zero.T + translation
        tree = cKDTree(cv); distance, child_indices = tree.query(pv)
        span = np.linalg.norm(np.ptp(parent_mesh.vertices * sp, axis=0))
        hint_scores = np.zeros(len(pv)); hint_support = 0
        for hint in p_region['used_hints']:
            uv, good = project(parent_mesh.vertices[parent['rim_ids']], parent_rows[hint['source_sample_index']]['camera_from_native_mesh'], k)
            d, middle = line_distance(uv, hint['endpoints_px']); uncertainty = max(5., hint['uncertainty_px'])
            hint_scores += np.minimum(d / uncertainty + .1 * middle / uncertainty, 10.)
            hint_scores[~good] += 10.; hint_support += 1
        # Mating is the primary constraint; broad image hints select WHICH rim
        # edge. A weak along-axis center prior avoids an arbitrary endpoint.
        along = np.abs((pv - parent['rim_center'] * sp) @ parent['axis']) / span
        # Independent reconstructions have slightly different rim depths. A
        # millimetric nearest-neighbour advantage must not choose the front rim
        # when source-cited seam evidence identifies the opposite edge.
        score = distance / span + hint_scores / max(1, hint_support) + .03 * along
        best = int(np.argmin(score)); pi = int(parent['rim_ids'][best]); ci = int(child['rim_ids'][child_indices[best]])
        choices.append({'zero': zero, 'angles': angles, 'parent_vertex': pi, 'child_vertex': ci,
                        'closed_rim_vertex_gap': float(distance[best]), 'target_translation': translation,
                        'score': float(np.median(residual))})
    chosen = min(choices, key=lambda row: row['score'])
    for region, mesh, key in [(p_region, parent_mesh, 'parent_vertex'), (c_region, child_mesh, 'child_vertex')]:
        vertex = chosen[key]
        if vertex not in region['vertex_ids']:
            region['vertex_ids'].append(vertex); region['points'] = np.vstack([region['points'], mesh.vertices[vertex]])
            region['candidate_mean_hint_distance_px'].append(None)
    model.update(axis=parent['axis'], zero=chosen['zero'], pi=p_region['vertex_ids'].index(chosen['parent_vertex']),
                 ci=c_region['vertex_ids'].index(chosen['child_vertex']))
    for sample, angle in zip(paired, chosen['angles']):
        if sample in model['rows']:
            model['rows'][sample]['q'] = float(angle)
    model['cavity_mating_prior'] = {'status': 'estimated_geometry_prior',
        'method': 'opposed_native_cavity_normals_and_aligned_closed_rim_footprint_centers_then_actual_nearest_rim_vertices',
        'parent': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in parent.items() if key != 'rim_ids'},
        'child': {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in child.items() if key != 'rim_ids'},
        'closed_rim_vertex_gap_native_scene_units': chosen['closed_rim_vertex_gap'],
        'ideal_closed_relative_translation': chosen['target_translation'].tolist(),
        'physical_mating_verified': False}
    return model


def refine_shared_parent(model, parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k, *, root=ROOT, state_hints=()):
    """Self-occlusion-aware whole-assembly fit, still entirely estimated.

    A closed container's parent mask may be just its visible front strip. Using
    that strip as a full-object depth observation incorrectly shrinks both
    parts. The union of the two independently segmented parts supports a joint
    parent pose and angle, without scaling either mesh or detaching the lid.
    """
    axis, zero = model['axis'], model['zero']
    h = p_region['points'][model['pi']] * sp; c = c_region['points'][model['ci']] * sc
    pv = parent_mesh.vertices[np.linspace(0, len(parent_mesh.vertices) - 1, min(300, len(parent_mesh.vertices)), dtype=int)] * sp
    cv = child_mesh.vertices[np.linspace(0, len(child_mesh.vertices) - 1, min(300, len(child_mesh.vertices)), dtype=int)] * sc
    parent_normal = np.linalg.svd(pv - pv.mean(0), full_matrices=False)[2][-1]
    child_normal = np.linalg.svd(cv - cv.mean(0), full_matrices=False)[2][-1]
    closed_samples = {row['source_sample_index'] for row in state_hints if row.get('state') == 'closed'}
    targets = {}
    for sample in paired:
        masks = []
        for row in [parent_rows[sample], child_rows[sample]]:
            item = row.get('source_mask')
            if item:
                require(S.ref(item['path'], root) == item, 'Source assembly mask changed')
                masks.append(np.asarray(Image.open(root / item['path'])) > 0)
        if len(masks) == 2:
            targets[sample] = S.mask_shape(masks[0] | masks[1])
    refined_parent = {}; refined_rows = {}; previous_q = None; previous_rotation = None
    for sample in paired:
        old = model['rows'].get(sample)
        if old is None or sample not in targets:
            continue
        rp, tp, _ = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
        rotations = [rp]
        if previous_rotation is not None:
            rotations.append(previous_rotation)
        for camera_axis in np.eye(3):
            for degrees in [-15, 15]:
                rotations.append(Rotation.from_rotvec(camera_axis * math.radians(degrees)).as_matrix() @ rp)
        angle_candidates = list(np.arange(-math.pi, math.pi, math.radians(10))) + [old['q']]
        if previous_q is not None:
            angle_candidates += list(previous_q + np.deg2rad([-8, -3, 0, 3, 8]))
        best = None; target = targets[sample]
        observed_child_rotation = rotation_pose(child_rows[sample]['camera_from_native_mesh'])[0]
        child_item = child_rows[sample].get('source_mask')
        child_target = S.mask_shape(np.asarray(Image.open(root / child_item['path'])) > 0) if child_item else None
        for angle in angle_candidates:
            if previous_q is not None:
                angle += round((previous_q - angle) / (2 * math.pi)) * 2 * math.pi
            relative = axis_rotation(axis, angle) @ zero
            assembly = np.concatenate([pv, (cv - c) @ relative.T + h])
            for rotation in rotations:
                pose = S.projected_pose(assembly, rotation, 1., k, target, tp[2])
                if pose is None:
                    continue
                translation, uv, _ = pose; loss, iou = S.silhouette_score(uv, target)
                child = np.eye(4); child[:3, :3] = rotation @ relative * sc
                child[:3, 3] = translation + rotation @ (h - relative @ c)
                cuv, valid = project(cv / sc, child, k)
                child_iou = S.silhouette_score(cuv, child_target)[1] if valid.all() and child_target else None
                temporal = 0. if previous_q is None else min(abs(angle - previous_q), math.pi) / math.pi
                orientation_prior = Rotation.from_matrix(rotation @ rp.T).magnitude() / math.pi
                child_orientation_prior = Rotation.from_matrix((rotation @ relative) @ observed_child_rotation.T).magnitude() / math.pi
                # A cited closed state supplies a weak geometric plane prior,
                # not an exact matrix from its partially visible parent mask.
                # For approximately slab-like containers this resolves an
                # otherwise silhouette-ambiguous open-looking zero reference.
                if sample in closed_samples and model.get('cavity_mating_prior', {}).get('status') == 'estimated_geometry_prior':
                    # The signed cavity-mating frame defines geometric closure;
                    # antiparallel cavity faces and aligned rim footprints are
                    # required, rather than two cups both looking upward.
                    closed_plane_loss = 1 - math.cos(float(angle))
                else:
                    closed_plane_loss = 0.
                child_loss = 1 - child_iou if child_iou is not None else 1.
                objective = .65 * loss + .35 * child_loss + .035 * temporal + .08 * orientation_prior + .10 * child_orientation_prior + .35 * closed_plane_loss
                if best is None or objective < best['objective']:
                    parent = np.eye(4); parent[:3, :3] = rotation * sp; parent[:3, 3] = translation
                    best = {'q': float(angle), 'matrix': child, 'parent_matrix': parent, 'rotation': rotation,
                            'objective': float(objective), 'union_silhouette_fit_iou': iou,
                            'closed_state_plane_prior_loss': closed_plane_loss if sample in closed_samples else None,
                            'convex_silhouette_fit_iou': child_iou,
                            'rotation_residual_deg': float(np.rad2deg(Rotation.from_matrix((rotation @ relative) @ rotation_pose(child_rows[sample]['camera_from_native_mesh'])[0].T).magnitude()))}
        if best is not None:
            previous_q, previous_rotation = best['q'], best['rotation']
            refined_parent[sample] = best['parent_matrix']; refined_rows[sample] = best
    require(len(refined_rows) >= 6, 'Insufficient shared-parent source mask fits')
    model['rows'] = refined_rows; model['parent_refinements'] = refined_parent
    return model


def scene_inputs(scene_path, proposals_path, root):
    scene_path, proposals_path = U._local(scene_path, root), U._local(proposals_path, root)
    scene, proposals = U.read(scene_path), U.read(proposals_path)
    require(scene.get('format') == S.FORMAT and scene.get('accepted_motion_poses') == 0, 'Automatic estimated SAM scene required')
    require(proposals.get('format') == 'sam3d-mechanism-proposals' and proposals.get('version') == 1
            and proposals.get('source_sha256') == scene['source']['sha256']
            and proposals.get('interaction_sha256') == scene['original_program']['sha256']
            and proposals.get('physical_accuracy_verified') is False, 'Source/program-bound mechanism proposals required')
    require(isinstance(proposals.get('joints'), list) and len(proposals['joints']) <= 12, 'Bounded mechanism proposals required')
    for component in scene['components']:
        require(S.ref(component['asset']['path'], root) == component['asset'], 'Unchanged native geometry required')
    return scene, proposals, S.ref(scene_path, root), S.ref(proposals_path, root)


def fit_revolute(scene, proposal, *, root=ROOT, refine_parent=True):
    components = {c['id']: c for c in scene['components']}
    parent, child = components[proposal['parent_id']], components[proposal['child_id']]
    parent_rows = {r['source_sample_index']: r for r in parent['poses']}; child_rows = {r['source_sample_index']: r for r in child['poses']}
    paired = [i for i in sorted(parent_rows) if parent_rows[i]['status'] == child_rows.get(i, {}).get('status') == 'estimated']
    require(len(paired) >= 6, 'At least six joint-visible source estimates required')
    require(proposal.get('hinge_hints'), 'A source-cited 2D hinge-region proposal is required')
    k = np.asarray(scene['camera']['K']); parent_mesh = load_native_glb(root / parent['asset']['path']); child_mesh = load_native_glb(root / child['asset']['path'])
    p_region = surface_candidates(parent_mesh.vertices, parent_rows, proposal['hinge_hints'], k)
    c_region = surface_candidates(child_mesh.vertices, child_rows, proposal['hinge_hints'], k)
    relatives = []; translations = []
    sp, sc = parent['native_uniform_scale'], child['native_uniform_scale']
    for sample in paired:
        rp, tp, actual_sp = rotation_pose(parent_rows[sample]['camera_from_native_mesh'])
        rc, tc, actual_sc = rotation_pose(child_rows[sample]['camera_from_native_mesh'])
        require(np.isclose(sp, actual_sp) and np.isclose(sc, actual_sc), 'Fixed native scales required through sequence')
        relatives.append(rp.T @ rc); translations.append(rp.T @ (tc - tp))
    relatives, translations = np.asarray(relatives), np.asarray(translations)
    closed = [h for h in proposal.get('state_hints', []) if h['state'] == 'closed' and h['source_sample_index'] in paired]
    reference = closed[0]['source_sample_index'] if closed else next((h['source_sample_index'] for h in proposal['hinge_hints'] if h['source_sample_index'] in paired), paired[len(paired) // 2])
    reference_index = paired.index(reference); zero = relatives[reference_index]
    global_fit = fit_seam_model(parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k, root=root)
    global_fit = cavity_mating_model(global_fit, parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k)
    if refine_parent:
        global_fit = refine_shared_parent(global_fit, parent_mesh, child_mesh, parent_rows, child_rows, paired, p_region, c_region, sp, sc, k, root=root, state_hints=proposal.get('state_hints', []))
        for sample, matrix in global_fit['parent_refinements'].items():
            parent_rows[sample] = dict(parent_rows[sample], camera_from_native_mesh=matrix.tolist())
    axis, zero = global_fit['axis'], global_fit['zero']
    pi, ci = global_fit['pi'], global_fit['ci']
    parent_vertex_id, child_vertex_id = p_region['vertex_ids'][pi], c_region['vertex_ids'][ci]
    h, c = parent_mesh.vertices[parent_vertex_id] * sp, child_mesh.vertices[child_vertex_id] * sc
    q_rows = global_fit['rows']
    # Source state labels choose a transparent ANGLE GAUGE after fitting the
    # orientation; they never freeze a noisy independent closed-frame matrix.
    q_reference = q_rows[reference]['q']
    zero = axis_rotation(axis, q_reference) @ zero
    for row in q_rows.values():
        row['q'] -= q_reference
    cited_open_angles = [q_rows[h['source_sample_index']]['q'] for h in proposal.get('state_hints', [])
                         if h.get('state') == 'open' and h['source_sample_index'] in q_rows]
    axis_sign_flipped = bool(cited_open_angles and np.median(cited_open_angles) < -1e-8)
    if axis_sign_flipped:
        axis = -axis
        for row in q_rows.values():
            row['q'] = -row['q']
    a, b = joint_frames(sp, sc, h, c, zero)
    zero_time = parent_rows[reference]['timestamp_s']
    translation_errors = []
    rotation_errors = []
    for sample, row in q_rows.items():
        relative = axis_rotation(axis, row['q']) @ zero
        translation_errors.append(float(np.linalg.norm(h - relative @ c - translations[paired.index(sample)])))
        rotation_errors.append(float(Rotation.from_matrix(relative @ relatives[paired.index(sample)].T).magnitude()))
        # Recompute through the public FK, including the post-fit zero gauge.
        row['matrix'] = forward(parent_rows[sample]['camera_from_native_mesh'], a, axis, row['q'], b)
    projected_axis_cosines = []
    for hint in p_region['used_hints']:
        anchor_native = parent_mesh.vertices[parent_vertex_id]
        ends = np.array([anchor_native - axis * .1, anchor_native + axis * .1])
        uv, positive = project(ends, parent_rows[hint['source_sample_index']]['camera_from_native_mesh'], k)
        if positive.all():
            image_axis = uv[1] - uv[0]; hint_axis = np.diff(np.asarray(hint['endpoints_px']), axis=0)[0]
            denominator = np.linalg.norm(image_axis) * np.linalg.norm(hint_axis)
            if denominator > 1e-8:
                projected_axis_cosines.append(abs(float(image_axis @ hint_axis / denominator)))
    residual_rows = []
    samples = []
    for source in parent['poses']:
        sample = source['source_sample_index']; known = q_rows.get(sample)
        samples.append({'source_sample_index': sample, 'timestamp_s': source['timestamp_s'],
                        'status': 'estimated' if known is not None else 'unknown', 'q_rad': known['q'] if known else None,
                        'reason': 'fixed_mesh_surface_anchored_revolute_fit' if known else 'parent_or_joint_evidence_unavailable',
                        'accepted_motion_evidence': False})
        if known:
            _, tc, _ = rotation_pose(child_rows[sample]['camera_from_native_mesh'])
            residual_rows.append({'source_sample_index': sample, 'timestamp_s': source['timestamp_s'],
                                  'translation_difference_from_unconstrained_estimate': float(np.linalg.norm(known['matrix'][:3, 3] - tc)),
                                  'rotation_difference_deg': known['rotation_residual_deg'],
                                  'convex_silhouette_fit_iou': known['convex_silhouette_fit_iou'],
                                  'closed_state_plane_prior_loss': known.get('closed_state_plane_prior_loss'),
                                  'union_silhouette_fit_iou': known.get('union_silhouette_fit_iou')})
    known_angles = [r['q_rad'] for r in samples if r['status'] == 'estimated']
    require(len(known_angles) >= 6, 'No supported constrained angle sequence')
    state_comparisons = []
    for hint in proposal.get('state_hints', []):
        if hint.get('state') not in ('open', 'closed') or hint['source_sample_index'] not in q_rows:
            continue
        angle = float(q_rows[hint['source_sample_index']]['q'])
        state_comparisons.append({'source_sample_index': hint['source_sample_index'], 'timestamp_s': hint['timestamp_s'],
                                  'cited_state': hint['state'], 'q_rad': angle,
                                  'distance_from_closed_reference_rad': abs(angle)})
    # A transparent contradiction diagnostic, NOT a physical angle acceptance
    # gate. Fifteen degrees is a declared interpretation threshold; it is never
    # used to insert or alter source geometry/poses.
    open_comparisons = [r for r in state_comparisons if r['cited_state'] == 'open']
    angle_consistent = bool(closed and open_comparisons and all(r['distance_from_closed_reference_rad'] > math.radians(15.) for r in open_comparisons))
    angle_quality = {'angle_visual_consistency': angle_consistent,
                     'test': 'all_cited_supported_open_states_separated_from_cited_closed_zero_by_more_than_declared_threshold',
                     'declared_separation_threshold_rad': math.radians(15.), 'state_comparisons': state_comparisons,
                     'angle_interpretation': 'estimated_opening_configuration_not_physically_calibrated' if angle_consistent else 'model_configuration_parameter_not_verified_opening_angle',
                     'confidence': 'low', 'physical_angle_verified': False,
                     'zero_state_alignment_verified': False,
                     'source_alignment_verified': False,
                     'quality_flags': [] if angle_consistent else ['cited_open_closed_states_not_reliably_separated_by_fitted_configuration', 'parent_rotation_and_hinge_angle_are_ambiguous_in_symmetric_or_occluded_views']}
    # Numerical exactness of BOTH mesh anchors is checked for every sample.
    coincidence = []
    for sample, estimate in q_rows.items():
        p = np.asarray(parent_rows[sample]['camera_from_native_mesh']); child_matrix = estimate['matrix']
        parent_anchor = p @ np.r_[parent_mesh.vertices[parent_vertex_id], 1.]
        child_anchor = child_matrix @ np.r_[child_mesh.vertices[child_vertex_id], 1.]
        coincidence.append(float(np.linalg.norm(parent_anchor[:3] - child_anchor[:3])))
    require(max(coincidence) < 1e-8, 'Persistent mesh-anchor connectivity failed')
    return {'id': proposal['id'], 'type': 'revolute', 'parent_id': parent['id'], 'child_id': child['id'], 'status': 'estimated',
            'axis': axis.tolist(), 'parent_from_joint': a.tolist(), 'joint_from_child_zero': b.tolist(),
            'parent_anchor_native': parent_mesh.vertices[parent_vertex_id].tolist(), 'child_anchor_native': child_mesh.vertices[child_vertex_id].tolist(),
            'parent_anchor_vertex_id': parent_vertex_id, 'child_anchor_vertex_id': child_vertex_id,
            'parent_native_scale': sp, 'child_native_scale': sc, 'limits_rad': [min(known_angles), max(known_angles)],
            'angle_quality': angle_quality, 'axis_sign_flipped_for_positive_median_cited_open': axis_sign_flipped,
            'limits_kind': 'observed_estimated_range_not_verified_physical_stops', 'physical_limits_rad': None,
            'zero_reference': {'source_sample_index': reference, 'timestamp_s': zero_time, 'state': 'estimated_closed' if closed else 'model_reference_state_unknown'},
            'samples': samples, 'evidence_ids': proposal.get('evidence_ids', []), 'hinge_hints': proposal['hinge_hints'],
            'shared_parent_refinement': {'performed': bool(global_fit.get('parent_refinements')),
                'method': 'joint_parent_and_angle_fit_to_union_of_parent_child_source_masks; fixed_native_scales_and_mesh_surface_anchors',
                'status': 'estimated', 'accepted_motion_evidence': False,
                'samples': [{'source_sample_index': i, 'timestamp_s': parent_rows[i]['timestamp_s'],
                             'camera_from_native_mesh': matrix.tolist(), 'status': 'estimated'}
                            for i, matrix in global_fit.get('parent_refinements', {}).items()]},
            'cavity_mating_prior': global_fit.get('cavity_mating_prior'),
            'anchor_selection': {'parent': {k: v for k, v in p_region.items() if k not in ('points', 'pca_axis')},
                                 'child': {k: v for k, v in c_region.items() if k not in ('points', 'pca_axis')},
                                 'parent_selected_hint_distance_px': p_region['candidate_mean_hint_distance_px'][pi],
                                 'child_selected_hint_distance_px': c_region['candidate_mean_hint_distance_px'][ci],
                                 'points_on_unchanged_mesh_vertices': True, 'geometry_physically_measured': False},
            'fit_diagnostics': {'anchor_coincidence_max_native_scene_units': max(coincidence),
                                'relative_translation_median_before_angle_mask_refinement': float(np.median(translation_errors)),
                                'relative_rotation_median_deg': float(np.rad2deg(np.median(rotation_errors))),
                                'axis_parent_seam_PCA_cosine': abs(float(axis @ p_region['pca_axis'])),
                                'axis_estimation': 'native_mesh_seam_PCA_with_transverse_native_frame_alignment_and_global_mask_fitting',
                                'orientation_estimation': 'both_unoriented_native_seam_signs_global_convex_silhouette_fit; closed_state_defines_angle_zero_after_fitting',
                                'fitting_sample_indices': global_fit['fitting_sample_indices'],
                                'native_seam_sign_choice': global_fit['seam_sign'],
                                'native_seam_sign_candidates': global_fit['native_seam_sign_candidates'],
                                'angle_visual_consistency': angle_consistent,
                                'angle_interpretation': angle_quality['angle_interpretation'],
                                'projected_axis_hint_direction_cosines': projected_axis_cosines,
                                'samples': residual_rows, 'physical_joint_recovery_verified': False,
                                'structural_connectivity_verified': True, 'source_alignment_verified': False},
            'matrix_convention': 'parent_native_pose * parent_from_joint(inverse_parent_scale) * axis_rotation(q) * joint_from_child_zero(child_scale_once)',
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'limitations': ['Mesh surface anchors and one fixed axis enforce attachment; they do not establish the true physical hinge.',
                            'All observed independent poses and convex silhouettes are estimated fitting inputs.',
                            'Angle limits describe this estimated clip only; physical stops and contact remain unknown.']}


def build(scene_path, proposals_path, output, *, root=ROOT):
    root = Path(root).resolve(); start = time.monotonic()
    scene, proposals, source_ref, proposal_ref = scene_inputs(scene_path, proposals_path, root)
    out = Path(output); out = out if out.is_absolute() else root / out
    require(out.is_relative_to(root / 'data') and out.resolve() == out, 'Ordinary data output required')
    require(not any(p.is_symlink() for p in [out, *out.parents]), 'Symlink constrained output refused')
    dependencies = [S.source_ref(root / name, root) for name in ['sam3d_mechanisms.py', 'sam3d_common_scene.py', 'native_appearance.py']]
    inputs_by_path = {item['path']: item for item in [source_ref, proposal_ref, *scene.get('inputs', []),
                                                     *[component['asset'] for component in scene['components']]]}
    inputs = [inputs_by_path[key] for key in sorted(inputs_by_path)]
    for item in inputs:
        require(S.ref(item['path'], root) == item, 'Frozen mechanism source/model/mask input changed')
    signature = U.digest(U.encoded({'inputs': inputs, 'dependencies': dependencies}))
    if out.exists():
        require((out / 'build.json').is_file(), 'Existing mechanism output lacks a verified receipt; choose a fresh version')
        previous = U.read(out / 'build.json')
        require(previous.get('format') == 'sam3d-constrained-mechanism-build' and previous.get('input_signature') == signature
                and previous.get('inputs') == inputs and previous.get('dependencies') == dependencies,
                'Existing mechanism inputs or implementation changed; choose a fresh version')
        for key in ['manifest', 'mechanics']:
            require(S.ref(previous['result'][key]['path'], root) == previous['result'][key], 'Cached mechanism output changed')
        cached_scene = U.read(out / 'scene.json'); cached_mechanics = U.read(out / 'mechanics.json')
        require(cached_scene.get('mechanics') == cached_mechanics and cached_mechanics.get('source_scene') == source_ref
                and cached_mechanics.get('proposal') == proposal_ref, 'Cached scene/mechanism binding differs')
        return dict(previous['result'], reused_verified_cache=True, wall_seconds=time.monotonic() - start)
    components = {c['id']: c for c in scene['components']}; parents = {}; fits = []; unresolved = []
    for proposal in proposals['joints']:
        if proposal.get('attachment') != 'persistent':
            continue
        p, c = proposal['parent_id'], proposal['child_id']
        require(p in components and c in components and p != c and c not in parents, 'Unique known mechanism parent required')
        ancestor = p
        while ancestor in parents:
            require(ancestor != c, 'Mechanism graph cannot contain a cycle'); ancestor = parents[ancestor]
        require(ancestor != c, 'Mechanism graph cannot contain a cycle'); parents[c] = p
        if proposal.get('joint_type') != 'revolute':
            unresolved.append({'id': proposal['id'], 'parent_id': p, 'child_id': c, 'type': proposal.get('joint_type'),
                               'status': 'unknown', 'reason': 'persistent_joint_type_not_yet_supported'})
            continue
        try:
            # Refining a branched or already attached parent independently for
            # each child would give inconsistent parent poses. Those topologies
            # retain the shared input parent until a joint tree optimizer exists.
            persistent = [j for j in proposals['joints'] if j.get('attachment') == 'persistent']
            can_refine_parent = sum(j['parent_id'] == p for j in persistent) == 1 and not any(j['child_id'] == p for j in persistent)
            fits.append(fit_revolute(scene, proposal, root=root, refine_parent=can_refine_parent))
        except (ValueError, KeyError, StopIteration) as error:
            unresolved.append({'id': proposal['id'], 'parent_id': p, 'child_id': c, 'type': proposal.get('joint_type'),
                               'status': 'unknown', 'reason': 'persistent_joint_fit_not_established', 'diagnostic': str(error)[:500]})
    mechanics = {'format': FORMAT, 'version': 1, 'source_scene': source_ref, 'proposal': proposal_ref,
                 'joints': fits, 'unresolved_children': unresolved, 'physical_accuracy_verified': False,
                 'structural_connectivity_verified': bool(fits) and not unresolved, 'simulation_ready': False,
                 'free_floating_child_fallback': False, 'manual_joint_coordinates_used': False,
                 'implementation': S.source_ref(root / 'sam3d_mechanisms.py', root), 'wall_seconds': time.monotonic() - start}
    output_scene = dict(scene, mechanics=mechanics)
    # Store FK poses for inspection/export, but the viewer MUST evaluate parent
    # plus scalar joint angle at intermediate times instead of interpolating
    # these world-space child matrices independently.
    ordered = []; remaining = list(fits)
    while remaining:
        level = [j for j in remaining if not any(other['child_id'] == j['parent_id'] for other in remaining)]
        require(level, 'Constrained mechanism topology is cyclic')
        ordered.extend(level); remaining = [j for j in remaining if j not in level]
    mechanics['joints'] = ordered
    for joint in ordered:
        parent = components[joint['parent_id']]; child = components[joint['child_id']]
        refinement = {row['source_sample_index']: row for row in joint.get('shared_parent_refinement', {}).get('samples', [])}
        if refinement:
            parent['poses'] = [dict(row, camera_from_native_mesh=refinement[row['source_sample_index']]['camera_from_native_mesh'],
                                    method='shared_parent_and_joint_union_silhouette_fit',
                                    reason='assembly_union_fit_handles_parent_child_self_occlusion', accepted_motion_evidence=False)
                               if row['source_sample_index'] in refinement else dict(row) for row in parent['poses']]
        p_rows = {r['source_sample_index']: r for r in parent['poses']}; q_rows = {r['source_sample_index']: r for r in joint['samples']}
        constrained = []
        for old in child['poses']:
            current = dict(old); p = p_rows[old['source_sample_index']]; q = q_rows[old['source_sample_index']]
            if p['status'] == q['status'] == 'estimated':
                current.update(status='estimated', camera_from_native_mesh=forward(p['camera_from_native_mesh'], joint['parent_from_joint'], joint['axis'], q['q_rad'], joint['joint_from_child_zero']).tolist(),
                               method='persistent_mesh_surface_anchored_revolute_FK', reason='child_pose_derived_from_parent_and_joint_angle',
                               joint_id=joint['id'], accepted_motion_evidence=False)
            else:
                current.update(status='unknown', camera_from_native_mesh=None, reason='attached_child_hidden_when_parent_or_joint_unknown', joint_id=joint['id'])
            constrained.append(current)
        child['poses'] = constrained
    for pending in unresolved:
        child = components[pending['child_id']]
        child['poses'] = [dict(row, status='unknown', camera_from_native_mesh=None,
                               reason='persistent_connection_declared_but_joint_unresolved_no_free_fallback',
                               method='unresolved_attachment', joint_id=pending['id']) for row in child['poses']]
    output_scene['components'] = list(components.values())
    output_scene['limitations'] = list(scene['limitations']) + ['Persistent connected child poses are replaced by a fixed mesh-anchored joint model. Physical hinge recovery remains unverified.']
    out.mkdir(parents=True); S.write(out / 'scene.json', output_scene); S.write(out / 'mechanics.json', mechanics)
    result = {'status': 'estimated_constrained_mechanism_scene_built' if fits and not unresolved else 'scene_with_unresolved_mechanisms',
            'manifest': S.ref(out / 'scene.json', root),
            'source_sha256': scene['source']['sha256'], 'mechanics': S.ref(out / 'mechanics.json', root),
            'joints': len(fits), 'unresolved_joints': len(unresolved), 'free_floating_child_fallback': False,
            'structural_connectivity_verified': bool(fits) and not unresolved,
            'physical_accuracy_verified': False, 'wall_seconds': time.monotonic() - start, 'reused_verified_cache': False}
    S.write(out / 'build.json', {'format': 'sam3d-constrained-mechanism-build', 'version': 1, 'input_signature': signature,
                               'inputs': inputs, 'dependencies': dependencies, 'result': result})
    return result
