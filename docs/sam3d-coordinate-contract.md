# SAM 3D native layout → video camera coordinates

This is a CPU-tested **convention conversion**, ready for the first real SAM 3D output. It does not recover motion, validate model geometry, prove a source-video pose, or establish physical dimensions. No SAM 3D checkpoint inference was run for this work.

## Audited versions and source evidence

SAM 3D Objects is pinned to `f91db411c50efee93d8db7aeb323885650f6f722`. Its PyTorch3D dependency is pinned to `75ebeeaea0908c5527e7b1e305fbc7681382db47` in `requirements.p3d.txt`.

| Evidence | What it establishes |
|---|---|
| [SAM 3D pose decoder, lines 562–565](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_utils.py#L562-L565) | Public `rotation`, `translation`, and `scale` are decoded instance pose. The public scale repeats one component across XYZ. Raw training targets, logarithmic scales and Gaussian ellipsoid scales are different values. |
| [SceneVisualizer.object_pointcloud](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/utils/visualization/scene_visualizer.py#L168-L196) and [compose_transform](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/data/dataset/tdfy/transforms_3d.py#L20-L30) | Apply scale, then the quaternion matrix, then translation using PyTorch3D's row-vector transformation. |
| [PyTorch3D quaternion conversion](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/pytorch3d/transforms/rotation_conversions.py#L43-L70) and [Rotate](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/pytorch3d/transforms/transform3d.py#L653-L687) | Quaternion order is **WXYZ**. `Rotate` inserts the supplied matrix directly into its row-vector transform. Do not silently substitute a column-vector quaternion action. |
| [Official GLB exporter, lines 665–666](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py#L665-L666) | The saved GLB has already undergone the raw Z-up → GLB Y-up axis conversion below. |
| [Official PyTorch3D → OpenCV flip](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/layout_post_optimization_utils.py#L1148-L1169) | Camera coordinates flip X and Y, preserving positive forward Z. |

## Exact math

Let `p` be a raw local asset **row vector**, `g` the exported GLB row vector, `Rq` PyTorch3D's quaternion matrix, `s` the positive uniform decoded instance scale, and `t` the decoded translation.

```text
A = [[1, 0,  0],        D = [[-1,  0, 0],
     [0, 0, -1],             [ 0, -1, 0],
     [0, 1,  0]]             [ 0,  0, 1]]

Official GLB:          g = p @ A
Official native scene: c_p3d = (p * s) @ Rq + t
Project CV scene:      c_cv = c_p3d @ D
```

Our project uses **column-vector 4×4 matrices**, with CV axes `+X right, +Y down, +Z forward`. Therefore:

```text
R_cv_from_glb = D @ Rq.T @ A
t_cv          = D @ t

camera_from_glb = [[s * R_cv_from_glb, t_cv],
                   [0, 0, 0,           1   ]]

c_cv_column = camera_from_glb @ g_homogeneous_column
```

`camera_from_glb` **includes scale exactly once**. Its top-left block is a similarity transform, not a pure rotation. Use the separate `rotation_cv_from_glb` for a rigid orientation/PnP initializer. Scale either the mesh once or use the similarity matrix; doing both is wrong.

The axis conversion has no translation offset. Although the upstream camera helper constructs a look-at transform, `compute_pointmap` uses only its rotation when transforming depth-model pointmaps. Do not introduce an extra one-unit Z offset. [Official pointmap conversion](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_pipeline_pointmap.py#L268-L282)

An external pointmap supplied to SAM 3D must already use its PyTorch3D camera convention; our existing CV/MoGe points require the same X/Y flip before passing them upstream. This helper does not supply pointmaps or modify source data.

## Usage and failure rules

```python
from sam3d_coordinates import (
    SAM3D_COMMIT, DECODED_SCALE_SEMANTICS,
    native_layout_to_cv, transform_glb_points,
)

pose = native_layout_to_cv(
    rotation_wxyz=output["rotation"],
    translation=output["translation"],
    scale_xyz=output["scale"],
    source_commit=SAM3D_COMMIT,
    scale_semantics=DECODED_SCALE_SEMANTICS,
)
camera_points = transform_glb_points(glb_vertices, pose.camera_from_glb)
```

Inputs must describe one unmodified official GLB and its matching decoded layout from the audited pin. The converter accepts `(4,)` or `(1,4)` rotation and `(3,)` or `(1,3)` translation/scale. It rejects multiple-object batches, zero quaternions, missing/non-finite values, unrecognized source pins, unspecified/raw scale semantics, scalar scale of ambiguous origin, non-positive scale, and anisotropic scale. Quaternions are normalized; sign-equivalent quaternions give the same matrix.

No old manual mesh, corners, camera dimensions, guessed physical scale or source-frame pose enters this conversion. `CVLayout.metadata()` retains predicted native scene units and marks metric scale, real SAM 3D inference validation and source-video pose validation as false. It deliberately does not emit the reviewer's `convention_verified: true` acceptance gate automatically.

Intrinsics are a separate requirement: use a camera matrix tied to the exact full source image and record whether it is estimated. Normalized intrinsics, cropped images and resized masks require their documented image-coordinate transforms. Correct 3D conventions do not repair incorrect intrinsics or a poor reconstruction.

## CPU verification performed

Eleven tests passed, with no GPU or model inference:

- Known raw basis vectors, GLB axis mapping, CV camera signs and a nonzero translation.
- A known +90° quaternion that catches the row/column transpose error.
- Independent SciPy rotations and pixel projections for 32 random poses.
- Uniform scale applied once, inverse round trips, equivalent/non-unit quaternions and invalid-input rejection.
- Execution of the exact axis expression from the pinned official GLB exporter.
- Numerical equivalence for 40 random poses against **actual pinned PyTorch3D quaternion and Transform3d CPU tensor code**, plus actual SAM 3D `compose_transform` source. Import-only dependencies were replaced to avoid compiled/GPU installation; transformation code was unmodified.

The optional reference test checks SHA-256 before executing public math sources. Source files are temporary developer reference material, not model assets or an installed runtime:

```text
rotation_conversions.py  b819672f569e489dbe73e9ff79743822f6199ff799d07399a4624c9fc0537dc0
transform3d.py            b8fdd5c9aa5dcc89b41994984472d6117cc9ccf49f8b225334197f2a373dfd84
```

Reproduce the full reference check, if those pinned files are present:

```sh
SAM3D_PYTORCH3D_REFERENCE_DIR=/private/tmp/metahack-pytorch3d-coordinate-reference \
  .venv-sam/bin/python -m unittest tests.test_sam3d_coordinates -v
```

Without that environment variable, normal offline tests run and explicitly skip the optional dependency-source execution. A future real SAM 3D result must still be projected into its source image, checked against masks/visible geometry, and accepted or rejected independently. Synthetic math equivalence does not establish that result's accuracy.
