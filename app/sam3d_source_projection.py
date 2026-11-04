"""Fixed source-projection math; estimated K from one inferred pointmap only."""
import numpy as np

from sam3d_coordinates import native_layout_to_cv, SAM3D_COMMIT, DECODED_SCALE_SEMANTICS


def grid_observations(pointmap, *, stride=8, offset=0):
    points = np.asarray(pointmap)
    if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1:
        raise ValueError('Pointmap must be native HxWx3')
    if type(stride) is not int or type(offset) is not int or stride < 2 or not 0 <= offset < stride:
        raise ValueError('Invalid deterministic pointmap grid')
    yy, xx = np.mgrid[offset:points.shape[0]:stride, offset:points.shape[1]:stride]
    cv = np.asarray(points[yy, xx], dtype=np.float64) * [-1, -1, 1]
    valid = np.isfinite(cv).all(-1) & (cv[..., 2] > 1e-8)
    return cv[valid], np.c_[xx[valid], yy[valid]].astype(float), int(valid.size)


def residuals(K, points, pixels):
    predicted = points @ np.asarray(K, float).T
    errors = np.linalg.norm(predicted[:, :2] / predicted[:, 2:] - pixels, axis=1)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError('Finite pointmap camera support required')
    return {'support': len(errors), 'median_px': float(np.median(errors)),
            'p95_px': float(np.percentile(errors, 95)), 'maximum_px': float(errors.max())}


def recover_estimated_camera(pointmap):
    """Train phase0; held-out phase4, both on the SAME model-inferred pointmap.

    No mask, silhouette, generated mesh, optimization or old camera enters K.
    Integer coordinates denote zero-based source-image pixel sample centres.
    """
    train, uv, total = grid_observations(pointmap, stride=8, offset=0)
    held, held_uv, held_total = grid_observations(pointmap, stride=8, offset=4)
    if min(len(train), len(held)) < 32:
        raise ValueError('At least32 finite positive-depth points required per fixed grid')
    parameters = []
    for axis in (0, 1):
        design = np.c_[train[:, axis] / train[:, 2], np.ones(len(train))]
        fit, _, rank, _ = np.linalg.lstsq(design, uv[:, axis], rcond=None)
        if rank != 2 or not np.isfinite(fit).all() or fit[0] <= 0:
            raise ValueError('Pointmap does not establish a positive full-rank pinhole prior')
        parameters.append(fit)
    K = np.array([[parameters[0][0], 0, parameters[0][1]],
                  [0, parameters[1][0], parameters[1][1]], [0, 0, 1.]])
    return {'K_estimated': K.tolist(), 'method': 'separate ordinary least squares u=fxX/Z+cx, v=fyY/Z+cy',
            'pointmap_to_cv': [-1, -1, 1], 'pointmap_shape': list(np.shape(pointmap)),
            'pixel_mapping': 'exact same native HxW; integer zero-based source pixel centres; no resize',
            'training_grid': {'stride': 8, 'offset_xy': [0, 0], 'sampled': total, **residuals(K, train, uv)},
            'heldout_grid': {'stride': 8, 'offset_xy': [4, 4], 'sampled': held_total, **residuals(K, held, held_uv)},
            'K_fitted_to_SAM_mask_or_mesh': False, 'exact_discarded_pipeline_K_recovered': False,
            'independent_camera_calibration': False,
            'heldout_scope': 'disjoint pixels of the same inferred pointmap; not independent image/physical evidence'}


def raw_layout_projection(rotation, translation, scale):
    layout = native_layout_to_cv(rotation_wxyz=rotation, translation=translation, scale_xyz=scale,
                                source_commit=SAM3D_COMMIT, scale_semantics=DECODED_SCALE_SEMANTICS)
    return {'layout': layout.metadata(), 'R': layout.rotation_cv_from_glb.tolist(),
            't': (layout.translation_cv / layout.scale).tolist(), 'renderer_scale': 1.,
            'equivalence': 'R*g+t/s projects identically to s*R*g+t; unchanged native mesh; scale applied once',
            'optimized': False, 'accepted_source_video_pose': False}
