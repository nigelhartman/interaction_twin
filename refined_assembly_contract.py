"""Declared geometric derivatives, with bounded source-mask refitting.

This additive contract never changes the native object-program validator or
claims that a derived shell is a new SAM inference. It keeps the original
program and meshes available and treats convex silhouette agreement as a
conditional image-space diagnostic, not physical validation.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import native_assembly_refinement_v1 as N
from native_appearance import load_native_glb
import object_program as O
from sam3d_common_scene import mask_shape, silhouette_score

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-declared-derived-assembly'
VERSION = 1
POLICY = {
    'maximum_parent_rotation_component_rad': .1745,
    'maximum_parent_translation_extent_fraction': .18,
    'maximum_parent_depth_fraction': .18,
    'maximum_fit_evaluations': 450,
    'sparse_vertex_target': 1400,
    'angle_grid_count': 65,
    'minimum_child_convex_mask_iou': .55,
    'minimum_union_convex_mask_iou': .60,
    'maximum_child_iou_regression': .12,
    'minimum_retained_samples': 12,
    'minimum_retained_fraction_of_original_supported': .50,
    'minimum_supported_child_median_iou': .75,
    'minimum_supported_union_median_iou': .65,
    'maximum_joint_speed_rad_per_s': 12.,
    'cavity_facing_ambiguity_gate': 'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away',
    'minimum_parent_cavity_facing_cosine': .15,
    'minimum_child_cavity_facing_cosine_for_bracketed_open': .05,
    'fill_original_unknowns': False,
    'mask_clipping_or_mesh_hiding': False,
    'physical_accuracy_verified': False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path, root=ROOT):
    return O.ref(Path(path), Path(root).resolve())


def read(reference, root=ROOT, maximum=32*1024*1024):
    path = O.checked_ref(reference, Path(root).resolve(), maximum)
    return json.loads(path.read_bytes())


def write(path, value):
    path = Path(path)
    require(not path.exists(), 'Derived assembly outputs are immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def project(vertices, matrix, k):
    points = np.asarray(vertices) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
    require(np.isfinite(points).all() and np.min(points[:, 2]) > 1e-5, 'Projection crosses the camera near plane')
    uvw = points @ np.asarray(k).T
    return uvw[:, :2] / uvw[:, 2:]


def hull_mask(vertices, matrix, camera):
    uv = project(vertices, matrix, camera['K'])
    require(np.max(abs(uv)) < 100000, 'Unbounded projected mesh')
    mask = np.zeros((camera['height'], camera['width']), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
    return mask > 0


def iou(first, second):
    return float(np.count_nonzero(first & second)/max(1, np.count_nonzero(first | second)))


def eligible_row(row, policy=POLICY):
    scores = row['mask_iou']
    return (scores['child'] >= policy['minimum_child_convex_mask_iou'] and
            scores['union'] >= policy['minimum_union_convex_mask_iou'] and
            scores['child'] >= row['original_child_convex_mask_iou'] - policy['maximum_child_iou_regression'] and
            row['facing_ambiguity_rejected'] is False)


def state_bracket(index, hints):
    lower = [i for i in hints if i <= index]; upper = [i for i in hints if i >= index]
    if not lower or not upper:
        return {'state': None, 'source_sample_indices': []}
    first, last = max(lower), min(upper)
    return {'state': hints[first] if hints[first] == hints[last] else None,
            'source_sample_indices': [first, last]}


def cavity_facing(matrix, native_normal):
    matrix = np.asarray(matrix); normal = matrix[:3,:3] @ np.asarray(native_normal)
    view = -matrix[:3,3]
    require(np.linalg.norm(normal) > 1e-9 and np.linalg.norm(view) > 1e-9, 'Finite cavity-facing geometry required')
    return float(normal @ view / np.linalg.norm(normal) / np.linalg.norm(view))


def facing_ambiguity(bracket, parent_cosine, child_cosine, policy=POLICY):
    return bool(bracket['state'] == 'open' and parent_cosine > policy['minimum_parent_cavity_facing_cosine'] and
                child_cosine < policy['minimum_child_cavity_facing_cosine_for_bracketed_open'])


def summarize(rows, original_supported, policy=POLICY):
    accepted = [r for r in rows if r['retained']]
    child = float(np.median([r['mask_iou']['child'] for r in accepted])) if accepted else 0.
    union = float(np.median([r['mask_iou']['union'] for r in accepted])) if accepted else 0.
    eligible = (len(accepted) >= policy['minimum_retained_samples'] and
                len(accepted) >= original_supported*policy['minimum_retained_fraction_of_original_supported'] and
                child >= policy['minimum_supported_child_median_iou'] and
                union >= policy['minimum_supported_union_median_iou'])
    return {'original_supported_samples': original_supported, 'retained_samples': len(accepted),
            'newly_withheld_samples': original_supported-len(accepted),
            'supported_child_median_convex_mask_iou': child,
            'supported_union_median_convex_mask_iou': union,
            'default_eligible': eligible, 'physical_accuracy_verified': False}


def _inputs(base_object_program_ref, refinement_report_ref, root):
    base = read(base_object_program_ref, root)
    O.validate(base, root=root)
    report = N.verify_report(refinement_report_ref, root)
    require(base['constrained_scene'] == report['original_scene'], 'Derivative is not based on the exact native program scene')
    require(base['evidence_program'] == report['original_program'], 'Derivative source procedure differs')
    original = read(report['original_scene'], root)
    derived = read(report['scene_ref'], root)
    return base, report, original, derived


def _fit(original, derived, report, root):
    parts = {c['id']: c for c in derived['components']}
    old_parts = {c['id']: c for c in original['components']}
    joint = report['joint']['refined']
    pid, cid = joint['parent_id'], joint['child_id']
    full = {part: load_native_glb(O.checked_ref(parts[part]['asset'], root, 32*1024*1024).read_bytes()).vertices for part in [pid, cid]}
    old_child = load_native_glb(O.checked_ref(old_parts[cid]['asset'], root, 32*1024*1024).read_bytes()).vertices
    sparse = {part: vertices[::max(1, len(vertices)//POLICY['sparse_vertex_target'])] for part, vertices in full.items()}
    camera = derived['camera']; k = np.asarray(camera['K'])
    a, b = np.asarray(joint['parent_from_joint']), np.asarray(joint['joint_from_child_zero'])
    parents = {row['source_sample_index']: row for row in parts[pid]['poses']}
    old_children = {row['source_sample_index']: row for row in old_parts[cid]['poses']}
    angles = {row['source_sample_index']: row for row in joint['samples']}
    proposal = read(derived['mechanics']['proposal'], root)
    semantic = next(j for j in proposal['joints'] if j['id'] == joint['id'])
    hints = {row['source_sample_index']: row['state'] for row in semantic.get('state_hints', [])}
    cavity = report['joint']['original']['cavity_mating_prior']
    rows = []
    for child in parts[cid]['poses']:
        if child['camera_from_native_mesh'] is None:
            continue
        index = child['source_sample_index']; parent = parents[index]
        mp = np.asarray(Image.open(O.checked_ref(parent['source_mask'], root))) > 0
        mc = np.asarray(Image.open(O.checked_ref(child['source_mask'], root))) > 0
        tp, tc, tu = mask_shape(mp), mask_shape(mc), mask_shape(mp | mc)
        original_parent = np.asarray(parent['camera_from_native_mesh'])
        old_q = angles[index]['q_rad']
        low, high = joint['limits_rad']
        state = hints.get(index)
        if state == 'open':
            low = max(low, math.pi/12)
        if state == 'closed':
            low = high = 0.
        def q_score(q):
            matrix = original_parent @ a @ N.axis_rotation(joint['axis'], q) @ b
            return silhouette_score(project(sparse[cid], matrix, k), tc)[0] + abs(q-old_q)*.001
        candidates = np.r_[np.linspace(low, high, POLICY['angle_grid_count']), np.clip(old_q, low, high)]
        q = min(candidates, key=q_score)
        for level in range(2):
            radius = (high-low)/64/(4**level)
            q = min(np.clip(np.linspace(q-radius, q+radius, 17), low, high), key=q_score)
        ratio = original_parent[2, 3]/k[0, 0]
        dx = tu['extent'][0]*POLICY['maximum_parent_translation_extent_fraction']*ratio
        dy = tu['extent'][1]*POLICY['maximum_parent_translation_extent_fraction']*ratio
        dz = original_parent[2, 3]*POLICY['maximum_parent_depth_fraction']
        bounds = [(-POLICY['maximum_parent_rotation_component_rad'], POLICY['maximum_parent_rotation_component_rad'])]*3 + [(-dx, dx), (-dy, dy), (-dz, dz)]
        variable_q = high > low
        if variable_q:
            bounds.append((low, high))
        def matrices(x):
            p = original_parent.copy()
            p[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ original_parent[:3, :3]
            p[:3, 3] += np.asarray(x[3:6])
            angle = float(x[6]) if variable_q else float(low)
            return p, p @ a @ N.axis_rotation(joint['axis'], angle) @ b, angle
        def loss(x):
            p, c, angle = matrices(x)
            u, v = project(sparse[pid], p, k), project(sparse[cid], c, k)
            lp, lc, lu = [silhouette_score(uv, target)[0] for uv, target in [(u, tp), (v, tc), (np.r_[u, v], tu)]]
            reg = np.sum(np.square(np.asarray(x[:3])/.1745))*.003 + sum((x[i]/[dx, dy, dz][i-3])**2 for i in range(3, 6))*.002 + abs(angle-q)*.001
            return lu*.55 + lc*.35 + lp*.10 + reg
        initial = np.array([0., 0., 0., 0., 0., 0., *([q] if variable_q else [])])
        fit = minimize(loss, initial, method='Powell', bounds=bounds,
                       options={'maxfev': POLICY['maximum_fit_evaluations'], 'maxiter': 4, 'xtol': .002, 'ftol': .0001})
        x = fit.x if fit.fun < loss(initial) else initial
        p, c, q = matrices(x)
        pm, cm = hull_mask(full[pid], p, camera), hull_mask(full[cid], c, camera)
        old_mask = hull_mask(old_child, old_children[index]['camera_from_native_mesh'], camera)
        row = {'source_sample_index': index, 'timestamp_s': child['timestamp_s'],
               'q_rad': q, 'semantic_state_hint': state, 'parent_camera_from_native_mesh': p.tolist(),
               'child_camera_from_native_mesh': c.tolist(), 'parent_pose_delta': x[:6].tolist(),
               'parameter_bounds': [list(pair) for pair in bounds[:6]], 'fit_evaluations': int(fit.nfev),
               'parent_mask': parent['source_mask'], 'child_mask': child['source_mask'],
               'mask_iou': {'parent': iou(pm, mp), 'child': iou(cm, mc), 'union': iou(pm | cm, mp | mc)},
               'original_child_convex_mask_iou': iou(old_mask, mc)}
        row['source_state_bracket'] = state_bracket(index, hints)
        row['cavity_facing_cosine'] = {'parent': cavity_facing(p,cavity['parent']['normal']),
                                     'child': cavity_facing(c,cavity['child']['normal'])}
        row['facing_ambiguity_rejected'] = facing_ambiguity(row['source_state_bracket'], row['cavity_facing_cosine']['parent'], row['cavity_facing_cosine']['child'])
        row['retained'] = eligible_row(row)
        rows.append(row)
    return rows


def build(base_object_program_ref, refinement_report_ref, output, *, root=ROOT):
    root = Path(root).resolve(); folder = O.data_path(root, output)
    require(not folder.exists(), 'Fresh declared-assembly namespace required')
    base, report, original, derived = _inputs(base_object_program_ref, refinement_report_ref, root)
    rows = _fit(original, derived, report, root)
    summary = summarize(rows, len(rows))
    require(summary['default_eligible'], 'Source-mask refit is not eligible for default presentation')
    joint = derived['mechanics']['joints'][0]
    pid, cid = joint['parent_id'], joint['child_id']
    parts = {c['id']: c for c in derived['components']}; by_index = {r['source_sample_index']: r for r in rows}
    original_unknown = sum(p['camera_from_native_mesh'] is None for p in parts[cid]['poses'])
    for parent in parts[pid]['poses']:
        row = by_index.get(parent['source_sample_index'])
        if row and row['retained']:
            parent.update(camera_from_native_mesh=row['parent_camera_from_native_mesh'], reason='bounded_joint_parent_source_mask_refit', accepted_motion_evidence=False)
    for child in parts[cid]['poses']:
        row = by_index.get(child['source_sample_index'])
        if row and row['retained']:
            child.update(camera_from_native_mesh=row['child_camera_from_native_mesh'], convex_silhouette_fit_iou=row['mask_iou']['child'],
                         reason='fixed_derived_joint_FK_after_bounded_shared_parent_source_mask_refit', status='estimated', accepted_motion_evidence=False)
        else:
            child.update(camera_from_native_mesh=None, status='unknown', reason='withheld_by_source_mask_eligibility_or_original_unknown')
    for sample in joint['samples']:
        row = by_index.get(sample['source_sample_index'])
        if row and row['retained']:
            sample.update(q_rad=row['q_rad'], status='estimated', reason='source_mask_refit_in_declared_geometric_closed_gauge', accepted_motion_evidence=False)
        else:
            sample.update(q_rad=None, status='unknown', reason='withheld_by_source_mask_eligibility_or_original_unknown')
    closed = [r for r in rows if r['retained'] and r['semantic_state_hint'] == 'closed' and r['q_rad'] == 0.]
    require(closed, 'A source-cited closed frame must pass at the geometric closed reference')
    joint['zero_reference'] = {'state': 'estimated_geometric_closed', 'source_sample_index': closed[-1]['source_sample_index'],
                               'timestamp_s': closed[-1]['timestamp_s'], 'physical_closure_verified': False}
    joint['limits_kind'] = 'bounded_estimated_source_configuration_range_not_physical_stops'
    joint['zero_state_alignment_verified'] = False
    joint['limitations'] = ['Declared derived-lid geometry and bounded convex source-mask fit; geometric closure is not measured physical clearance.',
                            'Unknown, low-agreement or cavity-facing-ambiguous samples remain withheld. Angles are model configuration estimates.']
    derived['assembly_refinement']['source_alignment_revalidated'] = True
    derived['assembly_refinement']['source_alignment_kind'] = 'conditional_convex_mask_diagnostic_not_physical_or_full_surface_validation'
    folder.mkdir(parents=True)
    implementation = folder/'provenance/refined_assembly_contract.py.txt'; implementation.parent.mkdir(); implementation.write_bytes(Path(__file__).read_bytes())
    audit = {'format': 'interaction-twin-derived-assembly-source-audit', 'version': 1,
             'base_object_program_ref': base_object_program_ref, 'refinement_report_ref': refinement_report_ref,
             'policy': copy.deepcopy(POLICY), 'rows': rows, 'summary': {**summary,
                'original_unknown_samples': original_unknown, 'total_unknown_samples': original_unknown + summary['newly_withheld_samples']},
             'implementation': ref(implementation, root), 'manual_geometry_or_poses_used': False,
             'new_api_calls': 0, 'new_gpu_calls': 0, 'physical_accuracy_verified': False}
    write(folder/'source-audit.json', audit)
    derived['assembly_refinement']['source_audit'] = ref(folder/'source-audit.json', root)
    derived['limitations'] += ['Source agreement is a convex silhouette diagnostic with bounded parent/angle refitting; it cannot validate depth, hidden geometry or physical contact.']
    write(folder/'scene.json', derived)
    document = {'format': FORMAT, 'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'reviewed_geometric_derivative_with_gaps', 'source': base['source'], 'evidence_program': base['evidence_program'],
                'base_object_program_ref': base_object_program_ref, 'refinement_report_ref': refinement_report_ref,
                'scene_ref': ref(folder/'scene.json', root), 'source_alignment_audit_ref': ref(folder/'source-audit.json', root),
                'component_assets': [{'component_id': c['id'], 'native_glb': c['asset'], 'native_metadata': c['native_metadata'],
                    'native_uniform_scale': c['native_uniform_scale'], 'original_native_glb': next(a['original_asset'] for a in report['assets'] if a['id'] == c['id']),
                    'geometry_origin': 'automatic_bounded_native_rim_derivative' if c['id'] == cid else 'unchanged_SAM_3D_inference'} for c in derived['components']],
                'joints': [O.projected_joint(j) for j in derived['mechanics']['joints']],
                'removable_relationships': copy.deepcopy(base['removable_relationships']),
                'unresolved_connections': copy.deepcopy(base['unresolved_connections']),
                'source_audit_summary': copy.deepcopy(audit['summary']),
                'claims': {'original_native_assets_preserved': True, 'declared_derivative_geometry_used': True,
                           'manual_geometry_or_poses_used': False, 'conditional_convex_source_mask_check_performed': True,
                           'physical_accuracy_verified': False, 'simulation_ready': False, 'physical_robot_execution_available': False},
                'physical_accuracy_verified': False, 'simulation_ready': False,
                'limitations': ['The original native descriptive program remains separately available.',
                    'One lid is an explicitly declared bounded mathematical derivative, not a new SAM-generated mesh.',
                    'Mask agreement is conditional and depth, metric scale, contact, grasp and physical simulation remain unverified.']}
    write(folder/'derived-assembly.json', document)
    validate(document, root=root)
    return {'artifact': ref(folder/'derived-assembly.json', root), 'scene': document['scene_ref'],
            'audit': document['source_alignment_audit_ref'], 'summary': audit['summary']}


def validate(document, *, root=ROOT):
    """Recheck provenance, every retained FK and actual mask projection values."""
    root = Path(root).resolve()
    json.dumps(document, allow_nan=False)
    require(document.get('format') == FORMAT and document.get('version') == VERSION, 'Unknown derived assembly contract')
    require(document.get('physical_accuracy_verified') is False and document.get('simulation_ready') is False, 'Derived assembly is not physical certification')
    require(document.get('claims') == {'original_native_assets_preserved': True, 'declared_derivative_geometry_used': True,
            'manual_geometry_or_poses_used': False, 'conditional_convex_source_mask_check_performed': True,
            'physical_accuracy_verified': False, 'simulation_ready': False, 'physical_robot_execution_available': False}, 'Derived claims changed')
    base, report, original, unfit = _inputs(document['base_object_program_ref'], document['refinement_report_ref'], root)
    scene, audit = read(document['scene_ref'], root), read(document['source_alignment_audit_ref'], root)
    json.dumps([scene, audit], allow_nan=False)
    require(scene['source'] == document['source'] == base['source'] and document['evidence_program'] == base['evidence_program'] == scene['original_program'], 'Derived source binding changed')
    require(audit['base_object_program_ref'] == document['base_object_program_ref'] and audit['refinement_report_ref'] == document['refinement_report_ref'] and audit['policy'] == POLICY, 'Source audit provenance or quality policy changed')
    O.checked_ref(audit['implementation'], root, 256*1024)
    require(audit.get('new_api_calls') == audit.get('new_gpu_calls') == 0 and audit.get('manual_geometry_or_poses_used') is False, 'Unexpected derivative execution claim')
    pid, cid = report['joint']['parent_id'], report['joint']['child_id']
    parts, old_parts = {c['id']: c for c in scene['components']}, {c['id']: c for c in unfit['components']}
    require(set(parts) == set(old_parts), 'Derived component inventory changed')
    require(scene['camera'] == unfit['camera'] and scene.get('physical_accuracy_verified') is False and scene.get('simulation_ready') is False,
            'The source camera or research limitations changed')
    for part in parts:
        for key in ['asset', 'native_metadata', 'native_uniform_scale']:
            require(parts[part][key] == old_parts[part][key], 'Source refitting changed the declared native/derived geometry')
        if part not in {pid, cid}:
            require(parts[part] == old_parts[part], 'Source refitting moved an unrelated component')
    expected_assets = [{'component_id': c['id'], 'native_glb': c['asset'], 'native_metadata': c['native_metadata'],
                       'native_uniform_scale': c['native_uniform_scale'], 'original_native_glb': next(a['original_asset'] for a in report['assets'] if a['id'] == c['id']),
                       'geometry_origin': 'automatic_bounded_native_rim_derivative' if c['id'] == cid else 'unchanged_SAM_3D_inference'} for c in scene['components']]
    require(document['component_assets'] == expected_assets and document['removable_relationships'] == base['removable_relationships'] and document['unresolved_connections'] == base['unresolved_connections'], 'Derived descriptive bindings changed')
    joint = scene['mechanics']['joints'][0]; fixed = report['joint']['refined']
    for field in ['id', 'parent_id', 'child_id', 'axis', 'parent_from_joint', 'joint_from_child_zero', 'limits_rad', 'parent_native_scale', 'child_native_scale']:
        require(joint[field] == fixed[field], 'Fixed geometric mechanism changed during source fitting')
    require(document['joints'] == [O.projected_joint(j) for j in scene['mechanics']['joints']], 'Derived description differs from scene joint')
    rows = audit['rows']; require(len(rows) <= 120 and len({r['source_sample_index'] for r in rows}) == len(rows), 'Bounded unique audit rows required')
    by_index = {r['source_sample_index']: r for r in rows}
    original_lid = {r['source_sample_index']: r for r in old_parts[cid]['poses']}
    original_parent = {r['source_sample_index']: r for r in old_parts[pid]['poses']}
    parent = {r['source_sample_index']: r for r in parts[pid]['poses']}
    full = {part: load_native_glb(O.checked_ref(parts[part]['asset'], root, 32*1024*1024).read_bytes()).vertices for part in [pid, cid]}
    original_original = read(report['original_scene'], root)
    original_child = next(c for c in original_original['components'] if c['id'] == cid)
    original_mesh = load_native_glb(O.checked_ref(original_child['asset'], root, 32*1024*1024).read_bytes()).vertices
    original_poses = {r['source_sample_index']: r for r in original_child['poses']}
    proposal = read(unfit['mechanics']['proposal'], root)
    semantic = next(j for j in proposal['joints'] if j['id'] == joint['id'])
    hints = {r['source_sample_index']:r['state'] for r in semantic.get('state_hints',[])}
    cavity = report['joint']['original']['cavity_mating_prior']
    require(len(parts[cid]['poses']) == len(old_parts[cid]['poses']), 'Original sample clock changed')
    samples = {r['source_sample_index']: r for r in joint['samples']}
    require(len(samples) == len(joint['samples']) == len(parts[cid]['poses']), 'Joint curve changes the original sample inventory')
    previous = None
    for pose, old in zip(parts[cid]['poses'], old_parts[cid]['poses']):
        index = pose['source_sample_index']; row = by_index.get(index)
        require(pose['timestamp_s'] == old['timestamp_s'] and index == old['source_sample_index'], 'Original source clock changed')
        if old['camera_from_native_mesh'] is None:
            require(row is None and pose['camera_from_native_mesh'] is None and pose['status'] == 'unknown' and
                    samples[index]['status'] == 'unknown' and samples[index]['q_rad'] is None, 'An original unknown was filled')
            continue
        require(row is not None and row['timestamp_s'] == pose['timestamp_s'] and row['parent_mask'] == old_parts[pid]['poses'][next(i for i,p in enumerate(old_parts[pid]['poses']) if p['source_sample_index'] == index)]['source_mask'] and row['child_mask'] == old['source_mask'], 'Audit/source mask binding changed')
        require(row['retained'] == eligible_row(row), 'Retained sample bypasses image-space quality gate')
        x = np.asarray(row['parent_pose_delta']); bounds = np.asarray(row['parameter_bounds'])
        require(x.shape == (6,) and bounds.shape == (6,2) and np.isfinite(x).all() and np.all(x >= bounds[:,0]-1e-10) and np.all(x <= bounds[:,1]+1e-10), 'Parent fit exceeds declared finite bounds')
        source_parent_mask = np.asarray(Image.open(O.checked_ref(row['parent_mask'], root))) > 0
        source_child_mask = np.asarray(Image.open(O.checked_ref(row['child_mask'], root))) > 0
        extent = mask_shape(source_parent_mask | source_child_mask)['extent']
        camera_k = np.asarray(scene['camera']['K']); original_depth = original_parent[index]['camera_from_native_mesh'][2][3]
        translation_bounds = [extent[0]*POLICY['maximum_parent_translation_extent_fraction']*original_depth/camera_k[0,0],
                              extent[1]*POLICY['maximum_parent_translation_extent_fraction']*original_depth/camera_k[0,0],
                              original_depth*POLICY['maximum_parent_depth_fraction']]
        expected_bounds = [[-POLICY['maximum_parent_rotation_component_rad'], POLICY['maximum_parent_rotation_component_rad']]]*3 + [[-v,v] for v in translation_bounds]
        require(np.allclose(bounds, expected_bounds, atol=1e-12) and type(row['fit_evaluations']) is int and
                0 < row['fit_evaluations'] <= POLICY['maximum_fit_evaluations'], 'Source-derived fit budget/bounds changed')
        p = np.asarray(original_parent[index]['camera_from_native_mesh']).copy()
        p[:3,:3] = Rotation.from_rotvec(x[:3]).as_matrix() @ p[:3,:3]; p[:3,3] += x[3:]
        q = row['q_rad']; require(joint['limits_rad'][0] <= q <= joint['limits_rad'][1], 'Estimated angle exceeds original declared range')
        c = p @ np.asarray(joint['parent_from_joint']) @ N.axis_rotation(joint['axis'], q) @ np.asarray(joint['joint_from_child_zero'])
        require(np.allclose(row['parent_camera_from_native_mesh'], p, atol=1e-9) and np.allclose(row['child_camera_from_native_mesh'], c, atol=1e-9), 'Audit differs from fixed forward kinematics')
        bracket = state_bracket(index,hints)
        facing = {'parent':cavity_facing(p,cavity['parent']['normal']), 'child':cavity_facing(c,cavity['child']['normal'])}
        require(row['source_state_bracket'] == bracket and all(abs(row['cavity_facing_cosine'][key]-facing[key]) < 1e-12 for key in facing) and
                row['facing_ambiguity_rejected'] == facing_ambiguity(bracket,facing['parent'],facing['child']), 'Cavity-facing ambiguity gate differs from source/native geometry')
        mp = np.asarray(Image.open(O.checked_ref(row['parent_mask'], root))) > 0; mc = np.asarray(Image.open(O.checked_ref(row['child_mask'], root))) > 0
        pm, cm = hull_mask(full[pid], p, scene['camera']), hull_mask(full[cid], c, scene['camera'])
        scores = {'parent': iou(pm,mp), 'child': iou(cm,mc), 'union': iou(pm|cm,mp|mc)}
        require(all(abs(row['mask_iou'][key]-scores[key])<1e-12 for key in scores), 'Saved source-mask score differs from actual projected mesh')
        old_mask = hull_mask(original_mesh, original_poses[index]['camera_from_native_mesh'], scene['camera'])
        require(abs(row['original_child_convex_mask_iou']-iou(old_mask,mc))<1e-12, 'Original mask comparison changed')
        if row['retained']:
            require(pose['status'] == 'estimated' and np.allclose(pose['camera_from_native_mesh'], c, atol=1e-9) and np.allclose(parent[index]['camera_from_native_mesh'], p, atol=1e-9), 'Retained scene differs from audited FK')
            require(samples[index]['status'] == 'estimated' and samples[index]['timestamp_s'] == pose['timestamp_s'] and
                    samples[index]['q_rad'] == q, 'Joint curve differs from audited angle')
            if previous:
                require((pose['timestamp_s']-previous[0]) > 0 and abs(q-previous[1])/(pose['timestamp_s']-previous[0]) <= POLICY['maximum_joint_speed_rad_per_s'],
                        'Estimated joint curve exceeds bounded source-time speed')
            previous = (pose['timestamp_s'], q)
        else:
            require(pose['camera_from_native_mesh'] is None and pose['status'] == 'unknown' and
                    samples[index]['status'] == 'unknown' and samples[index]['q_rad'] is None and
                    parent[index] == original_parent[index], 'Rejected source fit remained visible or moved its parent')
    summary = {**summarize(rows, len(rows)), 'original_unknown_samples': sum(p['camera_from_native_mesh'] is None for p in old_parts[cid]['poses'])}
    summary['total_unknown_samples'] = summary['original_unknown_samples'] + summary['newly_withheld_samples']
    require(summary == audit['summary'] == document['source_audit_summary'] and summary['default_eligible'], 'Derived default eligibility changed')
    require(any(r['retained'] and r['semantic_state_hint'] == 'closed' and r['q_rad'] == 0 for r in rows), 'Source-cited geometric closed state missing')
    return document
