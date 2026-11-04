"""CPU pinhole rendering and exact ray hits for the Stage 4 alignment pilot.

Camera axes are +X right, +Y down and +Z forward.  Pixel samples are at integer
image coordinates, matching ``geometry.project_points`` and the saved tracks.
The asset-to-camera transform is ``scale * vertices @ R.T + t``.  Rendering does
not apply any observation mask: callers retain the complete projected silhouette
and may composite an independently estimated occluder separately.
"""

import numpy as np


def _inputs(vertices, faces, K, R, t, scale):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must be integer Mx3")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face index outside vertex array")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("invalid pinhole intrinsics")
    if not np.allclose(K[2], [0, 0, 1], atol=1e-12):
        raise ValueError("intrinsics must have bottom row [0, 0, 1]")
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("rotation must be finite 3x3")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError("R must be a proper rotation; scale is passed separately")
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("translation must be finite length 3")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    return vertices, faces.astype(np.intp, copy=False), K, R, t, float(scale)


def _clip_near(points, colors, near):
    """Clip a camera-space triangle to Z >= near, including color attributes."""
    out_points, out_colors = [], []
    for i in range(len(points)):
        a, b = points[i - 1], points[i]
        ca, cb = colors[i - 1], colors[i]
        inside_a, inside_b = a[2] >= near, b[2] >= near
        if inside_a != inside_b:
            f = (near - a[2]) / (b[2] - a[2])
            out_points.append(a + f * (b - a))
            out_colors.append(ca + f * (cb - ca))
        if inside_b:
            out_points.append(b)
            out_colors.append(cb)
    return np.asarray(out_points), np.asarray(out_colors)


def raster_triangle_candidates(points, faces, K, width, height, near=1e-8):
    """Conservatively reject faces whose raster bounding box contains no sample.

    All near-plane-crossing or nonfinite projections go to the original clipping
    loop. A generous floating-point margin keeps boundary cases in that loop as
    well. This helper changes no vertices, face ordering or fragment arithmetic;
    returned values are ORIGINAL face indices, so diagnostic IDs remain stable.
    Work is batched to keep temporary storage bounded for large native meshes.
    """
    selected=[]
    for start in range(0,len(faces),65536):
        triangles=points[faces[start:start+65536]]
        front=triangles[:,:,2]>=near;all_front=front.all(axis=1);keep=front.any(axis=1)&~all_front
        if all_front.any():
            rows=np.flatnonzero(all_front);tri=triangles[rows]
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                projected=tri@K.T;uv=projected[:,:,:2]/projected[:,:,2:]
            finite=np.isfinite(uv).all(axis=(1,2));keep[rows[~finite]]=True
            if finite.any():
                indices=rows[finite];pixels=uv[finite]
                margin=1e-7+64*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(pixels),axis=(1,2)))
                lower=np.ceil(pixels.min(axis=1)-margin[:,None]);upper=np.floor(pixels.max(axis=1)+margin[:,None])
                lower=np.maximum(lower,[0,0]);upper=np.minimum(upper,[width-1,height-1])
                keep[indices]=np.all(lower<=upper,axis=1)
        selected.append(np.flatnonzero(keep)+start)
    return np.concatenate(selected)if selected else np.empty(0,dtype=np.intp)


def render_mesh(vertices, faces, K, R, t, width, height, scale=1.0, vertex_colors=None):
    """Render full geometry with perspective-correct depth and RGB interpolation.

    Returns ``mask`` (bool), ``depth`` (float32 Z; infinity on background),
    ``rgb`` (uint8; black background), and ``face_index`` (int32; -1 background).
    Colors are optional Nx3/Nx4 in the 0..255 convention; alpha is ignored.  No
    backface culling or lighting changes are applied.  Triangles crossing the
    near plane are clipped, rather than silently dropped.
    """
    vertices, faces, K, R, t, scale = _inputs(vertices, faces, K, R, t, scale)
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)) or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    if vertex_colors is None:
        colors = np.full((len(vertices), 3), 180.0)
    else:
        colors = np.asarray(vertex_colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] not in (3, 4) or not np.isfinite(colors).all():
            raise ValueError("vertex_colors must be finite Nx3 or Nx4")
        colors = np.clip(colors[:, :3], 0, 255)
    points = scale * vertices @ R.T + t
    depth = np.full((height, width), np.inf, dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    face_index = np.full((height, width), -1, dtype=np.int32)
    near = 1e-8

    for index in raster_triangle_candidates(points,faces,K,width,height,near):
        face=faces[index]
        p, c = _clip_near(points[face], colors[face], near)
        if len(p) < 3:
            continue
        for j in range(1, len(p) - 1):
            tri = p[[0, j, j + 1]]
            color = c[[0, j, j + 1]]
            homogeneous = tri @ K.T
            uv = homogeneous[:, :2] / homogeneous[:, 2:]
            lo = np.maximum(np.ceil(uv.min(0)), [0, 0])
            hi = np.minimum(np.floor(uv.max(0)), [width - 1, height - 1])
            if np.any(lo > hi):
                continue
            x0, y0 = lo.astype(int)
            x1, y1 = hi.astype(int)
            a, b, d = uv
            denominator = (b[1] - d[1]) * (a[0] - d[0]) + (d[0] - b[0]) * (a[1] - d[1])
            if abs(denominator) < 1e-12:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            w0 = ((b[1] - d[1]) * (xx - d[0]) + (d[0] - b[0]) * (yy - d[1])) / denominator
            w1 = ((d[1] - a[1]) * (xx - d[0]) + (a[0] - d[0]) * (yy - d[1])) / denominator
            w2 = 1 - w0 - w1
            bary = np.stack([w0, w1, w2], axis=-1)
            inside = np.min(bary, axis=-1) >= -1e-10
            reciprocal_z = np.sum(bary / tri[:, 2], axis=-1)
            z = np.divide(1, reciprocal_z, out=np.full_like(reciprocal_z, np.inf), where=reciprocal_z > 0)
            target_depth = depth[y0:y1 + 1, x0:x1 + 1]
            closer = inside & (z < target_depth)
            if not closer.any():
                continue
            target_depth[closer] = z[closer]
            perspective_weights = bary[closer] / tri[:, 2]
            perspective_weights /= reciprocal_z[closer, None]
            pixel_colors = perspective_weights @ color
            rgb[y0:y1 + 1, x0:x1 + 1][closer] = np.clip(np.rint(pixel_colors), 0, 255).astype(np.uint8)
            face_index[y0:y1 + 1, x0:x1 + 1][closer] = index

    return {"mask": face_index >= 0, "depth": depth.astype(np.float32), "rgb": rgb, "face_index": face_index}


def ray_mesh_points(uv, vertices, faces, K, R, t, scale=1.0):
    """Return exact nearest canonical-space intersections, plus a valid mask.

    Rays originate at the camera centre and pass through the requested (possibly
    subpixel) image coordinates.  Non-finite pixels, misses and intersections
    behind the camera return NaN.  No backface culling is used.  This establishes
    geometric correspondences conditional on the supplied pose; it does not
    independently verify the image tracks or make hidden surfaces observable.
    """
    vertices, faces, K, R, t, scale = _inputs(vertices, faces, K, R, t, scale)
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be Nx2")
    result = np.full((len(uv), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(uv), dtype=bool)
    indices = np.flatnonzero(np.isfinite(uv).all(axis=1))
    if not len(indices) or not len(faces):
        return result, valid
    rays_camera = np.c_[uv[indices], np.ones(len(indices))] @ np.linalg.inv(K).T
    direction = rays_camera @ R / scale
    origin = -t @ R / scale
    tri = vertices[faces]
    edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    # Batch rays but keep temporary matrices modest for larger caller meshes.
    for start in range(0, len(indices), 256):
        d = direction[start:start + 256]
        h = np.cross(d[:, None, :], edge2[None, :, :])
        determinant = np.einsum("fj,nfj->nf", edge1, h)
        inverse = np.divide(1, determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-12)
        s = origin - tri[:, 0]
        u = np.einsum("fj,nfj->nf", s, h) * inverse
        q = np.cross(s, edge1)
        v = np.einsum("nj,fj->nf", d, q) * inverse
        distance = np.einsum("fj,fj->f", edge2, q)[None, :] * inverse
        hit = (np.abs(determinant) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance > 1e-8)
        distance = np.where(hit, distance, np.inf)
        nearest = distance.min(axis=1)
        good = np.isfinite(nearest)
        selected = indices[start:start + len(d)][good]
        result[selected] = origin + nearest[good, None] * d[good]
        valid[selected] = True
    return result, valid
