# Native texture alignment

The new `native-base-color-v1` alignment path reads the original embedded
appearance of a GLB. It does not reconstruct an object, repaint a model, bake
video pixels, repair geometry or supply unseen motion.

## How it works

1. Validate the original GLB and supported material/sampler semantics.
2. Flatten its existing scene instances and transforms consistently for geometry,
   UVs and face/material identities. The original GLB bytes remain unchanged.
3. Search and refine a source-reference pose with the native silhouette policy.
4. Rank candidates using their actual native base-colour texture inside the
   observed overlap, when appearance ranking is enabled.
5. Raycast reference points against the same native representation. Track visible
   image points and use bounded PnP and disjoint point checks to obtain conditional
   poses. Missing or rejected observations stay unavailable.
6. Recompile the unchanged cached procedure in a separate output and compare exact
   matching source timestamps before choosing a result.

Appearance ranking uses the source frame being fitted. It is not an independent
accuracy test. Camera intrinsics, physical scale, hidden geometry and contact
remain unverified.

## Use the CPU path now

In **Model Lab**, select a saved component, choose **Refit on CPU**, inspect the
plan, then start. Geometry, masks and reasoning are reused; the native-material
fitter and provisioned CoTracker checkpoint run locally on CPU. The tracker uses
a bounded 256-pixel model size and records its exact resize/aspect rounding.
This changes computation settings, not the native geometry or source pixels.
The new comparison is a separate case and does not replace the preferred original.

Equivalent command (use the original program hash shown in Model Lab):

```sh
.venv-sam/bin/python scripts/run_upgrade.py run \
  --twin-id jabra-stage6-v1--procedure-v1 --track-id small_object_a \
  --base-sha256 CURRENT_ORIGINAL_SHA256 --mode refit_native_cpu
```

This mode sends no new frames to a reasoning provider, downloads no weights and
contacts no cluster. It operates on a saved project; it does not supply fresh
semantic discovery for an unseen video without a reasoning provider.

## Supported appearance

- Opaque triangular GLB geometry with embedded PNG/JPEG base-colour images.
- UV set zero, multiple primitives/materials, UV seams and scene instancing.
- Perspective-correct UV interpolation, including clipped near-plane vertices.
- Repeat, clamp-to-edge and mirrored-repeat wrapping with supported nearest or
  linear base-level sampling. Unsupported/mismatched or mipmap filter requests
  are rejected explicitly.
- Texture filtering in linear colour space, multiplied by linear vertex tint and
  base-colour factors, then encoded to display sRGB.
- Material sidedness, reflected-node winding, shared depth buffering and matching
  ray visibility. The pass is unlit base-colour appearance, without a light rig.

Unsupported alpha modes, shader maps, extensions, alternate UV sets or invalid
accessor semantics fail explicitly. A valid import with unsupported appearance
remains available for native inspection. Renderer/decoded image budgets are
bounded; no external GLB resource is fetched.

The historical pure vertex-colour compatibility route retains its original
colour convention and two-sided diagnostic projection exactly. It uses the old
fit/tracking scripts. Native UV/material results use the new versioned scripts.

## Preserved provenance

The import-time contract is distinct from current adapter capability. A historical
version-one receipt is still verified under its original rules, then the app
reports whether current code can align that same asset. Historical files are
never rewritten to claim a capability they did not have at import time.

A reference binds the original mesh hash, flattened vertex/face ordering and
loader/decoder/renderer/helper identities. Tracking rejects a changed
interpretation. Stage cache dependencies include the new modules, so changing
texture interpretation invalidates dependent fitting/tracking work.

SAM 3D's optional UV baking creates generated appearance from model Gaussian
renders. It is experimental and does not photograph an unseen surface in the
source video. Preparing or importing a handoff does not independently prove which
model executed. Genuine SAM 3D inference has not yet run in this project.

## Inspection and verification

Open [Texture alignment checks](../workspace/appearance.html) for a local visual
inspection page. Software fixtures have known geometry and poses and are labelled
synthetic; they are not added to the genuine model catalog. The page also compares
an unchanged saved Jabra model against the historical rendering path.

Analytic tests check perspective, colour composition, seams, scene transforms,
near clipping, visibility/occlusion and unsupported inputs. Native adapter checks
exercise automatic reference fitting and the representation join. These establish
implementation behaviour, not physical reconstruction quality or hidden insertion.
Actual local integration runs and final test totals are recorded in the latest
`output/native-appearance/` QA receipts.
