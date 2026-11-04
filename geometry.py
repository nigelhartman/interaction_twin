"""Coordinate-explicit helpers for local Stage 3 diagnostics."""
import numpy as np


def intrinsics_from_hfov(width, height, degrees):
    if width <= 0 or height <= 0 or not 5 < degrees < 175:
        raise ValueError('Invalid image dimensions or horizontal field of view')
    f = width / (2 * np.tan(np.deg2rad(degrees) / 2))
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], dtype=float)


def lift_pixels(uv, depth, intrinsics):
    """Z-depth to camera coordinates: +X right, +Y down, +Z forward.

    Invalid/out-of-image pixels produce NaN, never clipped edge observations.
    Intrinsics and Z units must be recorded by the caller.
    """
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    depth = np.asarray(depth)
    k = np.asarray(intrinsics, dtype=float)
    if depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid depth or intrinsics')
    out = np.full((len(uv), 3), np.nan)
    good = np.isfinite(uv).all(axis=1)
    good &= (uv[:, 0] >= 0) & (uv[:, 0] <= depth.shape[1]-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= depth.shape[0]-1)
    indices = np.flatnonzero(good)
    pixels = np.rint(uv[indices]).astype(int)
    z = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(z) & (z > 0)
    indices = indices[valid]; z = z[valid]
    rays = np.c_[uv[indices], np.ones(len(indices))] @ np.linalg.inv(k).T
    out[indices] = rays * z[:, None]
    return out


def project_points(points, intrinsics):
    points = np.asarray(points, dtype=float)
    out = np.full((len(points), 2), np.nan)
    good = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    p = points[good] @ np.asarray(intrinsics).T
    out[good] = p[:, :2] / p[:, 2:]
    return out


def rigid_fit(source, target):
    """Fixed-scale least-squares fit; diagnostic only, not robust pose recovery."""
    a = np.asarray(source, dtype=float); b = np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Expected matching Nx3 arrays')
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a = a[good]; b = b[good]
    if len(a) < 3 or np.linalg.matrix_rank(a-a.mean(0), tol=1e-8) < 2 or np.linalg.matrix_rank(b-b.mean(0), tol=1e-8) < 2:
        raise ValueError('Insufficient non-collinear correspondences')
    u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = b.mean(0) - rotation @ a.mean(0)
    errors = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = translation
    return transform, errors


def write_cloud(path, points, colors):
    points = np.asarray(points); colors = np.asarray(colors)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points and matching RGB')
    with path.open('w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, colors):
            f.write(' '.join(f'{v:.7g}' for v in p)+' '+' '.join(str(int(v)) for v in c)+'\n')
