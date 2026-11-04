"""CPU-only conversion of pinned SAM 3D decoded instance layout to CV coordinates.

This converts conventions; it does not estimate a pose, establish metric scale,
or validate a real model result. See docs/sam3d-coordinate-contract.md.
"""
from dataclasses import dataclass

import numpy as np


SAM3D_COMMIT = "f91db411c50efee93d8db7aeb323885650f6f722"
PYTORCH3D_COMMIT = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
DECODED_SCALE_SEMANTICS = "decoded_linear_isotropic_instance_scale"
CAMERA_AXES = "+X right, +Y down, +Z forward"

# Official to_glb: row vertices_glb = vertices_raw @ RAW_TO_GLB_ROW.
RAW_TO_GLB_ROW = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
# Same flip for column and row vectors, because this matrix is symmetric.
PYTORCH3D_TO_CV = np.diag([-1., -1., 1.])
RAW_TO_GLB_ROW.setflags(write=False)
PYTORCH3D_TO_CV.setflags(write=False)


def _single_vector(value, length, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be one finite vector") from error
    if result.shape == (1, length):
        result = result[0]
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape ({length},) or (1, {length}) and be finite")
    return result.copy()


def quaternion_wxyz_to_matrix(rotation_wxyz):
    """Scalar-first quaternion matrix, matching pinned PyTorch3D's definition.

    The returned matrix has the usual column-vector quaternion definition.
    Official SAM 3D passes this matrix directly to a *row-vector* transform;
    callers must not silently transpose it at that step.
    """
    quaternion = _single_vector(rotation_wxyz, 4, "rotation_wxyz")
    # Normalize robustly without squaring very large input components.
    maximum = np.max(np.abs(quaternion))
    if maximum == 0:
        raise ValueError("rotation_wxyz must be a nonzero quaternion")
    quaternion /= maximum
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


@dataclass(frozen=True)
class CVLayout:
    """Uniform similarity transform in predicted native scene units.

    ``camera_from_glb`` includes scale. ``rotation_cv_from_glb`` excludes it.
    Both multiply column vectors; Nx3 arrays use their transposes.
    """
    camera_from_glb: np.ndarray
    rotation_cv_from_glb: np.ndarray
    translation_cv: np.ndarray
    scale: float

    def metadata(self):
        """Explicit provenance, deliberately not an accepted reviewer pose."""
        return {
            "camera_from_glb": self.camera_from_glb.tolist(),
            "rotation_cv_from_glb": self.rotation_cv_from_glb.tolist(),
            "translation_cv": self.translation_cv.tolist(),
            "uniform_scale": self.scale,
            "matrix_convention": "column vectors; scale included in camera_from_glb",
            "camera_axes": CAMERA_AXES,
            "scale_units": "native_model_predicted_scene_units",
            "metric_scale_verified": False,
            "convention_verification": "pinned source audit and CPU synthetic math only",
            "source_video_pose_validated": False,
            "real_sam3d_inference_validated": False,
            "sam3d_code_commit": SAM3D_COMMIT,
            "pytorch3d_code_commit": PYTORCH3D_COMMIT,
        }


def native_layout_to_cv(*, rotation_wxyz, translation, scale_xyz,
                        source_commit, scale_semantics):
    """Convert the decoded layout for an unmodified official GLB export.

    Accepts exactly one object's public ``rotation``, ``translation`` and
    ``scale`` outputs, not raw diffusion/pose-target values or Gaussian scales.
    The caller must explicitly attest the audited source pin and scale meaning.
    Anisotropic, non-positive, non-finite and ambiguous scale inputs fail closed.

    For raw asset row vector p, official layout is (p * s) @ Rq + t.
    Official GLB export is g = p @ A. Therefore the column-vector CV mapping is
    c = D @ Rq.T @ (s I) @ A @ g + D @ t.
    """
    if source_commit != SAM3D_COMMIT:
        raise ValueError("Unaudited SAM 3D source commit; verify conventions before conversion")
    if scale_semantics != DECODED_SCALE_SEMANTICS:
        raise ValueError("Explicit decoded linear isotropic instance scale semantics required")
    scale = _single_vector(scale_xyz, 3, "scale_xyz")
    if np.any(scale <= 0):
        raise ValueError("scale_xyz must be strictly positive linear scale")
    # The pinned public decoder repeats one mean in all three components.
    # Do not silently average an anisotropic prediction into a rigid asset.
    if not np.allclose(scale, scale[0], rtol=1e-7, atol=0):
        raise ValueError("Anisotropic scale is not supported; refusing a guessed uniform scale")
    t = _single_vector(translation, 3, "translation")
    rq = quaternion_wxyz_to_matrix(rotation_wxyz)
    rotation = PYTORCH3D_TO_CV @ rq.T @ RAW_TO_GLB_ROW
    translation_cv = PYTORCH3D_TO_CV @ t
    matrix = np.eye(4)
    matrix[:3, :3] = scale[0] * rotation
    matrix[:3, 3] = translation_cv
    # Catch overflow from input values whose individual components are finite.
    if not np.isfinite(matrix).all():
        raise ValueError("Layout conversion overflowed")
    return CVLayout(matrix, rotation, translation_cv, float(scale[0]))


def transform_glb_points(points_glb, camera_from_glb):
    """Apply an affine column-vector pose to finite Nx3 GLB points."""
    points = np.asarray(points_glb, dtype=float)
    matrix = np.asarray(camera_from_glb, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 GLB points")
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0., 0., 0., 1.])):
        raise ValueError("Expected finite affine 4x4 camera_from_glb matrix")
    return points @ matrix[:3, :3].T + matrix[:3, 3]
