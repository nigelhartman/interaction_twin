# First genuine SAM3D source-projection experiment

The raw-layout experiment has now run on the genuine GPU-generated earbud.
Its unchanged mesh projects onto the source frame with a same-frame silhouette
IoU of 0.774435, using an estimated camera and no silhouette or pose fitting.
This is an encouraging initializer diagnostic; temporal alignment and physical
accuracy remain unvalidated. The actual result and inspection page are described
below. The original protocol is retained to explain what was fixed before the run.

## What is already defined

`sam3d_coordinates.native_layout_to_cv` converts the pinned official decoded
layout to the saved GLB's native coordinate system. It applies model scale once
and keeps physical scale unverified. `sam3d_milestone` already records this as an
initialization candidate, without granting a source-video pose.

For an unchanged native vertex `g`, the initializer is:

```text
camera point = s * R * g + t
```

The current native renderer accepts rigid `R, t` and an unchanged mesh. For
projection only, `R, t/s` produces the same image as the similarity above because
the whole camera point is divided by the positive scalar `s`. Do not scale the
mesh and apply `s` again. Any center shift requires its exact translation
compensation; native origins should be preserved.

The native pose adapter currently searches generic rotations with an assumed
60-degree horizontal field of view. It does not consume the SAM3D initializer or
the model's camera matrix. A new geometry candidate must receive a fresh fit;
poses from the original comparison mesh are incompatible.

## Camera information still needs care

The [pinned pipeline](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_pipeline_pointmap.py#L239-L291)
has an internal camera matrix, but its returned result omits that matrix and
exports a potentially downsampled pointmap. The current adapter saves that
pointmap without saving K. Downsampling uses bilinear interpolation with
`align_corners=False`; its pixel mapping must be recorded before using points as
full-frame evidence. At a 960-pixel longest side this stage uses scale one.

The [MoGe wrapper](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/depth_models/moge.py)
uses `force_projection=False`. Therefore camera recovery from the saved points
must report projection residuals and remain an estimated prior. It cannot claim
to recover the exact discarded matrix. Optional NaN gaps must remain missing.
The [upstream camera recovery helper](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/utils/pointmap.py#L19-L97)
also handles focal/depth-shift ambiguity; do not substitute an unchecked planar
ICP alignment or silently force all points onto camera rays.

## Minimal sequence after the actual import

1. Verify the real native output/import, untouched GLB/material interpretation,
   source RGB/SAM arrays and timestamp. Keep the original candidate unchanged.
2. Inspect actual pointmap dimensions and finite support. If the native pixel
   mapping or camera recovery is unsupported, record that gap explicitly. A
   separately labelled assumed-camera control remains possible.
3. Freeze the camera hypothesis, source observations, support, seed rules and
   compute budget before fitting. Render the raw layout initializer first,
   keeping complete unclipped RGB and silhouette projections.
4. Compare the additional layout seed against the existing deterministic generic
   search using the same K, target, native mesh, objective and refinement budget.
   Same-frame silhouette/appearance scores are fitting diagnostics; they do not
   establish independent pose or shape accuracy. Preserve competing rotations.
5. If the source fit is useful, run fresh conditional tracking tied to that exact
   new mesh. Reuse raw 2D observations only when their source/query/split lineage
   matches; never reuse old canonical 3D points or camera matrices. Reserve
   independent image observations for diagnostics, rather than selecting the
   seed or K with them.
6. Export only supported conditional samples to a new inspection scene. Keep
   unsupported times as gaps and retain a separate native-only inspection mode.

A promising image overlay is progress toward interactive component inspection.
Common part frames, actual insertion, metric collision geometry, joints, mass,
contact and robotics training are separate unfinished problems.

## Actual first-object diagnostic · 2 November 2026

`sam3d_source_projection.py` and
`scripts/raw_sam3d_source_projection.py` now implement the bounded raw-layout
step. The genuine earbud output has a native 960×540×3 pointmap matching the
source image exactly, with no downsample mapping required. The fixed training
grid uses stride 8 and offset (0,0); the held-out grid uses offset (4,4). Both grids
are disjoint pixels of the **same inferred pointmap**, not independent physical
or camera-calibration evidence. No SAM mask or generated mesh enters K recovery.

The estimated matrix is approximately:

```text
1055.819404    0.000000  269.504392
   0.000000 1056.144016  479.199046
   0.000000    0.000000    1.000000
```

Training support is 8160, median residual 0.502179 px and p95 residual 1.511313 px.
Held-out support is 8040, median 0.499401 px and p95 residual 1.508718 px. These
residuals quantify pinhole consistency of inferred points; they do not recover
the exact omitted pipeline K or validate physical calibration.

One CPU render uses the unchanged 275424-vertex/550820-face native mesh and the
unrefined layout through the existing audited coordinate conversion. Its full
projected silhouette has 5450 pixels, the SAM visible mask 5225 pixels, and their
same-frame IoU is 0.774435. The mask does not clip the render or select the camera
or pose. This overlap is a descriptive single-frame initializer check, not an
independent pose, geometry or physical accuracy score.

Evidence is under
`data/sam3d-source-projection-20261102/raw-layout-v1/diagnostic.json`, with full
source RGB, full-frame projected RGB/mask, source overlay and a separately
labelled display crop. The single render completed; its initially empty console
log caused a receipt-inventory failure. The completed render was then finalized
without rerendering, as explicitly recorded. The diagnostic inventory permits
that empty log only; native model import contracts remain unchanged.

The separate hash-bound local review is
`data/sam3d-source-projection-20261102/review-v1/index.html`. Opening it runs no
inference. It offers enlarged detail and full source-frame views, source and
derived image hash verification, camera-scope details and a hash-verified
download of the unchanged native GLB. No program, catalog, preferred model or
source-video pose was promoted. Fresh motion fitting and validation remain next.
