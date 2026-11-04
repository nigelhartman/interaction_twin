# Artifact contract v0.1

A case directory contains `manifest.json`, source video, source-timestamped JPEGs, `frames/index.json`, contact sheets and `review.html`. All paths in a case manifest are relative to the case directory. Index rows contain a 1-based image index, relative filename, source presentation timestamp in seconds, width and height. Sample timestamps are read from FFmpeg's selected source frames, not reconstructed from frame numbers. 1 fps is an approximate minimum inter-sample interval, not an assertion of exact integer-second source frames.

The source SHA-256 binds all derived artifacts to the acquired video bytes. Acquisition can use a selected <=720p stream rather than the uploader's original recording; this is recorded by its dimensions/codec/hash. Repeating preparation with identical settings reuses the artifact. A changed source or sampling configuration needs a new case, preserving previous evidence. Source URLs are provenance, not a license grant. Third-party video and frames stay out of Git.

Stage 1 produces `reasoning/inspection.json` with the actual provider, source hash, contact-sheet inputs, human-review flag and schema-validated inspection. Codex CLI is temporary hosted OpenAI reasoning invoked locally; it is not a Meta response or fully local model. The inspection schema is input triage only; it does not establish persistent tracks, 3D poses or action truth.

`segmentation/smoke/` holds a one-frame mask, overlay and metadata. The prompt's coordinates and origin, device, checkpoint hash, model version, wall time and mask score are recorded. Predicted mask quality is not measured IoU. A manual/assistant box is a seed annotation, not automatic object discovery. No video tracking is claimed.

Reserved artifacts for later approved stages:
- `segmentation/tracks/`: instance masks, raw tracker IDs, persistent entity map, annotations and corrections.
- `geometry/`: per-component meshes, source views and generated-surface uncertainty.
- `camera/` and `motion/`: coordinate conventions, scale convention, depth/camera calibration and trajectories.
- `alignment/`: initial similarity transforms, constant component scales, rigid poses, fit diagnostics.
- `scene/`: canonical JSON, readable YAML, physical entity IDs, slots, changing relations, evidence and pre/postconditions.
- `exports/`: versioned portable scene bundle.

Each future observation or inference must distinguish observed, inferred and user-confirmed claims. Geometry assets may be shared between distinct physical entities; IDs may not be silently merged. Raw video lacks reliable absolute scale without additional information. Frame subsampling is for coarse review; Stage 2 can extract a denser selected interval without changing this case.

## Stage 2 component records

The denser case is separate from Stage 1 and retains the same source SHA-256. `segmentation/components-v*/annotations.json` records every seed box/point, annotation origin, candidate entity set, local sample range and manually withheld interval. Versioned runs preserve earlier failures. Raw binary masks are retained unchanged, even when rejected for display. `tracks/*.json` maps every prediction back to source time.

A tracklet is a supported local observation, not necessarily a globally identified physical item. Two outward battery candidates can be distinct at the same moment; later table and inward tracklets retain both possible entity IDs until evidence resolves them. `visible_end` is an observation of an end in the compartment, not a claim about which battery it is or whether it is fully seated. Multiple tracklets do not establish additional physical batteries.

Visible-part composition gives part masks priority over the parent-camera candidate. Pixels claimed by multiple peer parts remain unknown, and occluded/oversized/empty predictions are withheld from the review output. This is a disclosed deterministic postprocess, not independent evidence that the residual camera region is a complete rigid-body segmentation. Outside a tracklet range means not assessed, never confirmed absence. Read `frames.json`, `quality.json` and raw-review sheets together.

Reasoning records contain the actual Codex provider and input artifacts, proposed events, actual evidence sample IDs and unresolved identity links. A validator rejects invented IDs, out-of-range timestamps and evidence outside the proposed event interval. This validates grounding references, not semantic truth. Events remain proposals until reviewed. The future Stage 5 compiler must not execute them as verified ground truth by default.

## Stage 3 local experiment records

`geometry/local-*/experiment.json` binds immutable input JPEGs/masks, their hashes, candidate entity IDs, source sample IDs and source PTS to explicit camera assumptions and model roles. Independent depth predictions are stored as float32 Z-depth arrays with their units and unverified scale. Partial coloured PLY surfaces use recording-camera coordinates (+X right, +Y down, +Z forward); they do not include unobserved backs.

`motion*/tracks.npz` contains T×N×2 pixel coordinates, a T×N validity mask, T×N×3 lifted camera-space points, source sample IDs and timestamps. Invalid observations are NaN, not substituted poses. CoTracker additionally preserves its raw predictions and model visibility; SAM/image-bound gates determine exported valid observations. Lucas-Kanade records forward/backward errors; that field remains NaN for CoTracker, which does not use the same check.

Recording-camera extrinsics, calibrated scale and physical attachment are not established. Diagnostic fixed-scale fits are labelled as such and never exported as accepted component motion. An explicitly procedural, unfitted OBJ may exercise asset-loading plumbing but must never be relabelled as model-generated geometry. Full alignment belongs to Stage 4.

`local-battery-v3` pools model-estimated intrinsics under a fixed-optics assumption and adds bounded relative depth-scale correction. `xyz_usable` marks which lifted points can be considered as 3D evidence; finite raw diagnostic values do not override a rejected flag. `consistency.json` records disjoint training/held-out point IDs, correction factors, reference observation and internal pair-distance errors. These errors do not measure physical accuracy. Frame 52 remains visible in diagnostic reviews but is excluded from 3D use.

`battery-fitted-v1` is an evidence-fitted shape-prior completion with source-view proportion estimates, normalized asset units, inferred hidden geometry and per-vertex appearance-evidence IDs. It is distinct from both the arbitrary unit proxy and the generative TripoSR result. The latter has actual model provenance but is explicitly rejected for wrong shape; topology validity alone must not promote it to an accepted asset.

## Stage 4 battery pilot records

`alignment/battery-pilot-v2/alignment.json` binds a fixed mesh and one scalar scale to camera-relative pose candidates. Its convention is `camera_xyz = scale * canonical_xyz @ rotation.T + translation`, with +X right, +Y down, +Z forward. Candidate poses and exact source timestamps are stored in `poses.npz`; validity comes from each JSON frame's status and reasons, never from finite matrix values alone. `reference_initialization` is separate from `supported_pilot` and `unsupported`. Unsupported pose candidates are retained for diagnosis and must not be used as accepted motion.

Reference mesh-ray misses are excluded; canonical correspondences are frozen before temporal fitting. Training and held-out IDs are disjoint but share a tracker, so reported errors are internal checks. Reference reprojection is excluded from aggregate tracking errors. Per-frame asset scaling is forbidden. Final hand-mask hashes and assistant prompt/corridor provenance are recorded. `full_masks` always retains unoccluded mesh projections; `visible_masks` applies only separate foreground hand evidence, never the target battery silhouette. Raw visible-hand masks do not by themselves establish front/behind ordering. The source's rejected `xyz_usable` evidence remains a support gate even if a 2D pose candidate can be solved.
