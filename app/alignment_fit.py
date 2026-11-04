"""Fixed-size, camera-relative PnP fitting for the Stage 4 battery pilot.

Object points must already include the single chosen asset scale.  This module
never estimates scale or changes object geometry.  +X is right, +Y is down and
+Z is forward; returned poses map object points into recording-camera space.
Reprojection agreement is internal evidence, not verified physical 6D accuracy.
"""
import cv2
import numpy as np


_SEED = 20261028
_MIN_POINTS = 6


def _noncollinear(points):
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return len(singular) >= 2 and singular[0] > 0 and singular[1] / singular[0] > 1e-4


def solve_pose(object_points, image_points, K, initial_R=None, initial_t=None,
               reprojection_threshold=4.0):
    """Robust rigid pose from matching Nx3 and Nx2 observations.

    Uses deterministic EPNP RANSAC (300 iterations, confidence .999), followed
    by LM on its consensus.  A supplied initial pose provides a second LM seed
    on that same consensus; it cannot bypass RANSAC or positive-depth checks.
    Candidates are ranked by thresholded inlier count, then Huber image error.

    Invalid observation rows are excluded but original indices are preserved.
    Returns a dict with success, reason, rotation (3x3), translation (3,),
    inlier_indices (original row IDs), and residual_px (N, NaN for invalid rows).
    Unsupported numerical fits return success=False and no pose.  Malformed
    array shapes, invalid intrinsics, or half-specified initial poses raise
    ValueError.  Rotation about a textureless cylinder's axis remains ambiguous.
    """
    objects = np.asarray(object_points, dtype=np.float64)
    images = np.asarray(image_points, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if objects.ndim != 2 or objects.shape[1] != 3 or images.shape != (len(objects), 2):
        raise ValueError('Expected matching Nx3 object points and Nx2 image points')
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or not np.allclose([intrinsics[0, 1], intrinsics[1, 0]], 0)):
        raise ValueError('Expected finite, standard zero-skew pinhole intrinsics')
    if not np.isfinite(reprojection_threshold) or reprojection_threshold <= 0:
        raise ValueError('reprojection_threshold must be positive and finite')
    if (initial_R is None) != (initial_t is None):
        raise ValueError('Supply both initial_R and initial_t, or neither')
    initial = None
    if initial_R is not None:
        rotation = np.asarray(initial_R, dtype=np.float64)
        translation = np.asarray(initial_t, dtype=np.float64).reshape(-1)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
            raise ValueError('Expected a finite proper initial rotation and translation')
        initial = (cv2.Rodrigues(rotation)[0], translation.reshape(3, 1))

    valid = np.isfinite(objects).all(axis=1) & np.isfinite(images).all(axis=1)
    original_ids = np.flatnonzero(valid)
    xyz = np.ascontiguousarray(objects[valid])
    uv = np.ascontiguousarray(images[valid])
    notes = ['Camera-relative, fixed asset scale; physical scale and intrinsics remain unverified.',
             'Reprojection errors do not establish true 6D pose; cylinder axial roll may be ambiguous.']
    result = {'success': False, 'reason': None, 'rotation': None, 'translation': None,
              'inlier_indices': np.empty(0, dtype=int),
              'residual_px': np.full(len(objects), np.nan),
              'valid_indices': original_ids, 'method': 'EPNP_RANSAC_LM',
              'scale_estimated': False, 'uncertainty_notes': notes,
              'ransac_seed': _SEED, 'reprojection_threshold_px': float(reprojection_threshold)}
    if len(xyz) < _MIN_POINTS:
        result['reason'] = 'fewer_than_six_finite_correspondences'
        return result
    if (len(np.unique(xyz, axis=0)) < _MIN_POINTS
            or len(np.unique(uv, axis=0)) < _MIN_POINTS
            or not _noncollinear(xyz) or not _noncollinear(uv)):
        result['reason'] = 'degenerate_correspondence_geometry'
        return result
    singular = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    result['object_spread_singular_values'] = singular
    if singular[-1] / singular[0] < .01:
        notes.append('Object-point support is nearly planar; multiple poses may explain this view.')

    cv2.setRNGSeed(_SEED)
    try:
        ok, rvec, tvec, consensus = cv2.solvePnPRansac(
            xyz, uv, intrinsics, None, iterationsCount=300,
            reprojectionError=float(reprojection_threshold), confidence=.999,
            flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        result['reason'] = 'opencv_ransac_failed'
        return result
    if not ok or consensus is None or len(consensus) < _MIN_POINTS:
        result['reason'] = 'insufficient_ransac_consensus'
        return result
    consensus = consensus[:, 0]
    if not _noncollinear(xyz[consensus]) or not _noncollinear(uv[consensus]):
        result['reason'] = 'degenerate_ransac_consensus'
        return result
    seeds = [('ransac', rvec, tvec)]
    if initial is not None:
        seeds.append(('initial_pose', *initial))
    candidates = []
    for label, candidate_rvec, candidate_tvec in seeds:
        try:
            fitted_rvec, fitted_tvec = cv2.solvePnPRefineLM(
                np.ascontiguousarray(xyz[consensus]), np.ascontiguousarray(uv[consensus]),
                intrinsics, None, candidate_rvec.copy(), candidate_tvec.copy())
            fitted_rotation = cv2.Rodrigues(fitted_rvec)[0]
            fitted_translation = fitted_tvec.reshape(3)
            camera_points = xyz @ fitted_rotation.T + fitted_translation
            if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
                continue
            projected = cv2.projectPoints(xyz, fitted_rvec, fitted_tvec, intrinsics, None)[0][:, 0]
            residual = np.linalg.norm(projected - uv, axis=1)
        except cv2.error:
            continue
        if not np.isfinite(residual).all():
            continue
        inlier = residual <= reprojection_threshold
        if (inlier.sum() < _MIN_POINTS or not _noncollinear(xyz[inlier])
                or not _noncollinear(uv[inlier])):
            continue
        delta = float(reprojection_threshold)
        huber = np.where(residual <= delta, .5 * residual**2,
                         delta * (residual - .5 * delta))
        candidates.append(((-int(inlier.sum()), float(huber.sum())), label,
                           fitted_rotation, fitted_translation, residual, inlier))
    if not candidates:
        result['reason'] = 'no_positive_depth_supported_refinement'
        return result
    _, label, rotation, translation, residual, inlier = min(candidates, key=lambda row: row[0])
    result['residual_px'][original_ids] = residual
    result.update(success=True, reason='supported_reprojection_fit', rotation=rotation,
                  translation=translation, inlier_indices=original_ids[inlier],
                  refinement_seed=label,
                  ransac_consensus_indices=original_ids[consensus])
    return result
