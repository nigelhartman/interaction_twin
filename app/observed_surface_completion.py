"""Explicitly inferred, source-visible interior patches; never observed geometry.

This optional layer smooths only enclosed gaps in a saved observed surface.
It does not recover physical cavities, hidden surfaces, thickness or moving
parts. No API, learned model, manual geometry or previous generated asset is
read. The original observed mesh is never mutated.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


PROTOCOL = {
    'version': 'source-visible-interior-completion-v1',
    'layer': 'inferred_only_not_observed_geometry',
    'mask_core_min_px': 4,
    'other_component_guard_px': 4,
    'maximum_hole_area_px': 4096,
    'maximum_hole_bbox_diagonal_px': 96,
    'minimum_boundary_pixels': 6,
    'maximum_boundary_relative_depth_range': 0.12,
    'maximum_affine_inverse_depth_p90_relative_residual': 0.02,
    'maximum_mesh_relative_depth_edge': 0.02,
    'maximum_total_inferred_pixels': 18000,
    'maximum_total_triangles': 50000,
    'interpolation': 'four-neighbour Dirichlet harmonic inverse depth',
    'require_closed_observed_boundary': True,
    'extrapolate_beyond_observed_boundary': False,
    'source_colours_are_appearance_not_depth_evidence': True,
    'unknown_component_boundaries_require_review': True,
    'physical_confidence': 'unknown',
    'hidden_surface_completion': False,
    'cavity_or_thickness_estimated': False,
    'new_api_calls': 0,
    'learned_model_calls': 0,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local(root, url):
    if not isinstance(url, str) or not url.startswith('/files/data/') or '?' in url or '#' in url:
        raise ValueError('Expected a local data evidence URL')
    path = (Path(root) / url[len('/files/'):]).resolve()
    if not path.is_relative_to((Path(root) / 'data').resolve()):
        raise ValueError('Evidence must stay inside project data')
    return path


def project(xyz, pose, K):
    camera = np.asarray(xyz, float) @ np.asarray(pose[0], float).T + np.asarray(pose[1], float)
    homogeneous = camera @ np.asarray(K, float).T
    return homogeneous[:, :2] / homogeneous[:, 2:], camera[:, 2]


def unproject(pixels, depth, pose, K):
    pixels = np.asarray(pixels, float)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T
    camera = rays * np.asarray(depth)[:, None]
    return (camera - np.asarray(pose[1])) @ np.asarray(pose[0])


def rasterize_depth(surface, pose, K, shape):
    """Perspective-correct source-pixel-centre depth of existing triangles."""
    xyz = np.asarray(surface['vertices'], float)
    triangles = np.asarray(surface['triangles'], int).reshape(-1, 3)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Observed vertices must be finite XYZ')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(xyz)):
        raise ValueError('Invalid observed triangle index')
    uv, z = project(xyz, pose, K)
    h, w = shape
    depth = np.full((h, w), np.nan, float)
    for indices in triangles:
        p, tz = uv[indices], z[indices]
        if not np.isfinite(p).all() or not np.isfinite(tz).all() or (tz <= 0).any():
            continue
        low = np.maximum(np.ceil(p.min(axis=0)).astype(int), [0, 0])
        high = np.minimum(np.floor(p.max(axis=0)).astype(int), [w - 1, h - 1])
        if (low > high).any():
            continue
        matrix = np.vstack([p.T, np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-10:
            continue
        yy, xx = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
        bary = np.linalg.solve(matrix, np.vstack([xx.ravel(), yy.ravel(), np.ones(xx.size)]))
        inside = (bary >= -1e-8).all(axis=0)
        inv_depth = (bary / tz[:, None]).sum(axis=0)
        valid = inside & np.isfinite(inv_depth) & (inv_depth > 0)
        found = np.full(xx.size, np.nan)
        found[valid] = 1 / inv_depth[valid]
        tile = depth[low[1]:high[1] + 1, low[0]:high[0] + 1]
        found = found.reshape(tile.shape)
        nearer = np.isfinite(found) & (~np.isfinite(tile) | (found < tile))
        tile[nearer] = found[nearer]
    return depth


def component_domain(target_mask, excluded_masks=(), component_id=1):
    mask = np.asarray(target_mask) > 0
    if mask.ndim != 2 or component_id < 1:
        raise ValueError('A two-dimensional target component mask is required')
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    domain = distance >= PROTOCOL['mask_core_min_px']
    labels = np.where(mask, component_id, 0).astype(np.int32)
    radius = PROTOCOL['other_component_guard_px']
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    for index, other in enumerate(excluded_masks):
        other = np.asarray(other) > 0
        if other.shape != mask.shape:
            raise ValueError('All component masks must use the exact same source pixels')
        guard = cv2.dilate(other.astype(np.uint8), kernel) > 0
        domain[guard] = False
        labels[guard] = component_id + index + 1
    return domain, labels


def harmonic_inverse_depth(depth, hole):
    """Unknown pixels have four-neighbour Dirichlet data on a closed boundary."""
    depth = np.asarray(depth, float)
    hole = np.asarray(hole, bool)
    yy, xx = np.where(hole)
    if not len(xx):
        return np.array([], float)
    lookup = np.full(hole.shape, -1, int)
    lookup[yy, xx] = np.arange(len(xx))
    rows, columns, values = [], [], []
    rhs = np.zeros(len(xx))
    h, w = hole.shape
    for index, (y, x) in enumerate(zip(yy, xx)):
        rows.append(index); columns.append(index); values.append(4.)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                raise ValueError('An inferred hole cannot touch the image boundary')
            if hole[ny, nx]:
                rows.append(index); columns.append(lookup[ny, nx]); values.append(-1.)
            else:
                value = depth[ny, nx]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('An inferred hole requires a complete positive-depth boundary')
                rhs[index] += 1 / value
    matrix = coo_matrix((values, (rows, columns)), shape=(len(xx), len(xx))).tocsr()
    inverse = spsolve(matrix, rhs)
    if not np.isfinite(inverse).all() or (inverse <= 0).any():
        raise ValueError('Non-finite or negative interpolated inverse depth')
    if np.max(np.abs(matrix @ inverse - rhs)) > 1e-8:
        raise ValueError('Interpolation did not satisfy its boundary constraints')
    return 1 / inverse


def infer_interior_depth(depth, domain, labels):
    """Fail closed on open, large, rough or multi-component missing regions."""
    depth = np.asarray(depth, float)
    domain = np.asarray(domain, bool)
    labels = np.asarray(labels)
    if depth.shape != domain.shape or depth.shape != labels.shape:
        raise ValueError('Depth, component mask and labels must have identical dimensions')
    observed = np.isfinite(depth) & (depth > 0) & domain
    missing = domain & ~observed
    count, connected = cv2.connectedComponents(missing.astype(np.uint8), connectivity=4)
    inferred = np.zeros(depth.shape, bool)
    completed = depth.copy()
    regions, rejected = [], Counter()
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    total = 0
    for index in range(1, count):
        hole = connected == index
        yy, xx = np.where(hole)
        area = len(xx)
        boundary = (cv2.dilate(hole.astype(np.uint8), cross) > 0) & ~hole
        reason = None
        diagonal = float(np.hypot(np.ptp(xx) + 1, np.ptp(yy) + 1))
        if (boundary & ~domain).any() or (xx == 0).any() or (yy == 0).any() or (xx == depth.shape[1]-1).any() or (yy == depth.shape[0]-1).any():
            reason = 'open_to_mask_or_image_boundary'
        elif not observed[boundary].all():
            reason = 'not_enclosed_by_observed_depth'
        elif area > PROTOCOL['maximum_hole_area_px'] or diagonal > PROTOCOL['maximum_hole_bbox_diagonal_px']:
            reason = 'gap_exceeds_frozen_size_bounds'
        elif int(boundary.sum()) < PROTOCOL['minimum_boundary_pixels']:
            reason = 'insufficient_boundary_support'
        elif len(np.unique(labels[hole | boundary])) != 1 or int(labels[yy[0], xx[0]]) <= 0:
            reason = 'crosses_component_labels'
        elif total + area > PROTOCOL['maximum_total_inferred_pixels']:
            reason = 'total_completion_budget_exceeded'
        else:
            z = depth[boundary]
            relative_range = float(np.ptp(z) / np.median(z))
            by, bx = np.where(boundary)
            coordinates = np.c_[bx - bx.mean(), by - by.mean(), np.ones(len(bx))]
            inv = 1 / z
            coefficients = np.linalg.lstsq(coordinates, inv, rcond=None)[0]
            residual = float(np.percentile(np.abs(coordinates @ coefficients - inv), 90) / np.median(inv))
            if relative_range > PROTOCOL['maximum_boundary_relative_depth_range']:
                reason = 'boundary_depth_discontinuity'
            elif residual > PROTOCOL['maximum_affine_inverse_depth_p90_relative_residual']:
                reason = 'boundary_not_smooth_enough_for_local_hypothesis'
            else:
                candidate = harmonic_inverse_depth(depth, hole)
                if (candidate < z.min() - 1e-8).any() or (candidate > z.max() + 1e-8).any():
                    reason = 'interpolation_extrapolates_boundary_depth_range'
                else:
                    completed[hole] = candidate
                    inferred[hole] = True
                    total += area
                    regions.append({'region_id': int(index), 'inferred_pixels': area,
                                    'boundary_pixels': int(boundary.sum()), 'bbox_diagonal_px': diagonal,
                                    'component_label': int(labels[yy[0], xx[0]]),
                                    'boundary_relative_depth_range': relative_range,
                                    'inverse_depth_affine_relative_p90_residual': residual,
                                    'physical_confidence': 'unknown', 'depth_observed': False})
        if reason:
            rejected[reason] += 1
    return completed, inferred, {'candidate_regions': count - 1, 'accepted_regions': regions,
                                'rejected_regions': dict(rejected), 'inferred_pixels': total}


def inferred_mesh(completed, inferred, domain, labels, image, pose, K, source_frame):
    """Only triangles touching inferred pixels are exported; every face inferred."""
    h, w = inferred.shape
    triangle_pixels = []
    near = cv2.dilate(inferred.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    yy, xx = np.where(near[:-1, :-1])
    for y, x in zip(yy, xx):
        for corners in (((x, y), (x+1, y), (x+1, y+1)), ((x, y), (x+1, y+1), (x, y+1))):
            px = np.array(corners, int)
            if not inferred[px[:, 1], px[:, 0]].any():
                continue
            z = completed[px[:, 1], px[:, 0]]
            part_ids = labels[px[:, 1], px[:, 0]]
            if (not domain[px[:, 1], px[:, 0]].all() or not np.isfinite(z).all() or (z <= 0).any()
                    or (part_ids <= 0).any() or len(np.unique(part_ids)) != 1
                    or np.ptp(z) / np.median(z) > PROTOCOL['maximum_mesh_relative_depth_edge']):
                continue
            triangle_pixels.append(corners)
    if len(triangle_pixels) > PROTOCOL['maximum_total_triangles']:
        raise ValueError('Inferred mesh exceeds the fixed triangle budget')
    if not triangle_pixels:
        return None
    pixel_list = sorted(set(p for triangle in triangle_pixels for p in triangle))
    lookup = {p: i for i, p in enumerate(pixel_list)}
    pixels = np.asarray(pixel_list, int)
    depth = completed[pixels[:, 1], pixels[:, 0]]
    xyz = unproject(pixels, depth, pose, K)
    camera_center = -np.asarray(pose[0]).T @ np.asarray(pose[1])
    triangles = []
    for triangle in triangle_pixels:
        indices = [lookup[p] for p in triangle]
        a, b, c = xyz[indices]
        if np.dot(np.cross(b-a, c-a), camera_center-a) < 0:
            indices[1], indices[2] = indices[2], indices[1]
        triangles.append(indices)
    colours = image[pixels[:, 1], pixels[:, 0], ::-1]
    return {'kind': 'inferred_source_visible_interior_patch', 'component_id': 'camera_body',
            'vertices': xyz.tolist(), 'triangles': triangles, 'colors': colours.tolist(),
            'uv': np.c_[(pixels[:, 0]+.5)/w, 1-(pixels[:, 1]+.5)/h].tolist(),
            'texture_url': source_frame['image_url'], 'source_frame_id': source_frame['id'],
            'vertex_source_pixels': pixels.tolist(),
            'vertex_provenance': ['inferred_harmonic_inverse_depth' if inferred[y, x] else 'observed_triangle_interpolation_boundary' for x, y in pixels],
            'face_provenance': ['inferred_harmonic_inverse_depth'] * len(triangles),
            'face_confidence': ['unknown'] * len(triangles),
            'observed_face_count': 0, 'inferred_face_count': len(triangles),
            'hole_filling': True, 'hidden_surface_completion': False, 'watertight': False,
            'thickness_known': False, 'depth_observed': False,
            'interpretation': 'Source-visible smooth depth hypotheses inside bounded enclosed gaps. Image texture is observed appearance; inferred depths and all faces remain unverified.'}


def complete(root, source_directory, output_directory, excluded_mask_urls=()):
    root = Path(root).resolve()
    source = (root / source_directory).resolve()
    output = (root / output_directory).resolve()
    if not source.is_relative_to(root / 'data') or not output.is_relative_to(root / 'data'):
        raise ValueError('Source and output must stay inside project data')
    output.mkdir(parents=True, exist_ok=False)
    original = source / 'viewer.json'
    # Display derivative has corrected half-texel UVs but identical geometry.
    view_path = source / 'viewer-display.json' if (source / 'viewer-display.json').is_file() else original
    if view_path != original:
        attestation = json.loads((source / 'display-provenance.json').read_text())
        for key, path in [('source_viewer_sha256', original),
                          ('source_protocol_sha256', source / 'protocol.json'),
                          ('display_viewer_sha256', view_path),
                          ('implementation_sha256', source / 'display-implementation.py')]:
            if sha256(path) != attestation[key]:
                raise ValueError('Source display derivative identity mismatch')
    view = json.loads(view_path.read_text())
    if view.get('kind') != 'observed_surface' or view.get('surface', {}).get('kind') != 'observed_partial_patch':
        raise ValueError('Only a genuine saved observed partial patch is eligible')
    frame = next(f for f in view['frames'] if f['id'] == view['surface']['source_frame_id'])
    camera = next(c for c in view['cameras'] if c['frame_id'] == frame['id'])
    image_path, mask_path = _local(root, frame['image_url']), _local(root, frame['mask_url'])
    excluded_paths = [_local(root, url) for url in excluded_mask_urls]
    inputs = [original, view_path, source / 'protocol.json', image_path, mask_path, *excluded_paths]
    hashes = {str(p.relative_to(root)): sha256(p) for p in inputs}
    implementation = Path(__file__).resolve()
    frozen = {'protocol': copy.deepcopy(PROTOCOL), 'source_directory': str(source.relative_to(root)),
              'source_viewer_path': str(view_path.relative_to(root)), 'input_sha256': hashes,
              'source_frame_id': frame['id'], 'component_id': 'camera_body',
              'excluded_component_mask_urls': list(excluded_mask_urls),
              'component_boundary_status': 'explicit_other_component_masks_excluded' if excluded_paths else 'target_mask_only_other_component_boundaries_unverified',
              'K': camera['K'], 'R': camera['R'], 't': camera['t'],
              'implementation_sha256': sha256(implementation),
              'frozen_before_completion': True}
    (output / 'protocol.json').write_text(json.dumps(frozen, indent=2)+'\n')
    (output / 'implementation.py').write_bytes(implementation.read_bytes())
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    others = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in excluded_paths]
    if image is None or mask is None or any(m is None for m in others) or image.shape[:2] != mask.shape or mask.shape != (frame['height'], frame['width']):
        raise ValueError('Saved evidence dimensions do not match')
    pose, K = (np.asarray(camera['R']), np.asarray(camera['t'])), np.asarray(camera['K'])
    depth = rasterize_depth(view['surface'], pose, K, mask.shape)
    domain, labels = component_domain(mask, others)
    completed, inferred, report = infer_interior_depth(depth, domain, labels)
    mesh = inferred_mesh(completed, inferred, domain, labels, image, pose, K, frame)
    if mesh:
        mesh['texture_sha256'] = sha256(image_path)
    overlay = image.copy()
    coverage = np.isfinite(depth) & (depth > 0) & domain
    overlay[coverage] = (.65 * overlay[coverage] + .35 * np.array([100, 185, 105])).astype(np.uint8)
    overlay[inferred] = (.35 * overlay[inferred] + .65 * np.array([45, 165, 245])).astype(np.uint8)
    cv2.imwrite(str(output / 'source-coverage-inferred.png'), overlay)
    cv2.imwrite(str(output / 'inferred-pixels.png'), inferred.astype(np.uint8)*255)
    counts = {'observed_triangles_unchanged': len(view['surface']['triangles']),
              'observed_source_coverage_pixels': int(coverage.sum()), 'inferred_pixels': int(inferred.sum()),
              'inferred_vertices': len(mesh['vertices']) if mesh else 0,
              'inferred_triangles': len(mesh['triangles']) if mesh else 0,
              'accepted_regions': len(report['accepted_regions']), 'observed_3d_points_added': 0,
              'camera_calibrated': False, 'physical_scale_known': False, 'complete_object': False}
    limits = ['This is an optional inferred layer, not observed reconstruction or independently validated depth.',
              'Enclosed holes may be real cavities, occlusion, segmentation errors or different parts; smooth filling can be physically wrong.',
              'Only bounded source-visible interior holes are considered; open silhouettes, unseen backs, thickness and cavities remain absent.',
              'Other component boundaries are unverified unless exact source-frame exclusion masks are supplied.',
              'Original source texture does not validate inferred shape. Camera intrinsics and scale remain ambiguous.',
              'The original observed mesh is unchanged; inferred faces retain separate provenance and unknown confidence.']
    report.update(status='inferred_candidate_requires_component_review' if mesh else 'no_eligible_enclosed_holes',
                  counts=counts, physical_confidence='unknown', limitations=limits,
                  component_boundary_status=frozen['component_boundary_status'])
    (output / 'completion.json').write_text(json.dumps(report, indent=2)+'\n')
    derived = {'format': 'observed-surface-completion-review', 'version': 1, 'id': output.name,
               'title': 'Optional source-visible gap inference', 'kind': 'inferred_surface_completion',
               'status': report['status'], 'reason': 'Separate harmonic depth hypotheses for bounded enclosed gaps; every added face is inferred and unverified.',
               'metrics': counts, 'frames': view['frames'], 'cameras': view['cameras'], 'points': [],
               'observed_surface_url': '/files/'+str(view_path.relative_to(root)),
               'inferred_surface': mesh, 'completion_report': report, 'limitations': limits,
               'source_frame_id': frame['id'], 'component_id': 'camera_body',
               'new_api_calls': 0, 'learned_model_calls': 0, 'production_geometry_changed': False}
    (output / 'viewer.json').write_text(json.dumps(derived, separators=(',', ':'))+'\n')
    for path, digest in hashes.items():
        if sha256(root / path) != digest:
            raise ValueError('Completion mutated an input artifact')
    (output / 'input-integrity.json').write_text(json.dumps({'unchanged': True, 'input_sha256': hashes}, indent=2)+'\n')
    return report
