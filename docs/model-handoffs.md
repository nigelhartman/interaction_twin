# Prepare locally, run on the GPU, compare locally

The Model Lab prepares a private, self-contained job from one saved component.
It does not contact a server, inspect an account, download anything, run a model,
or replace the current project. The original asset remains available throughout.

## The short workflow

1. Select a saved project and component in **Model Lab**.
2. Choose **SAM 3D Objects** for a replacement reconstruction, or **SAM 3.1 · one
   image** for an anchor-image segmentation comparison.
3. For SAM 3D, choose standard **Vertex colours** or explicit experimental
   **UV texture** appearance. UV baking is generated from model Gaussian renders;
   it does not recover photographs of unseen surfaces. Prepare and download the job ZIP. It includes the exact saved full-frame RGB,
   binary SAM mask, original automatic view-selection bytes, source video hash,
   original presentation timestamp, reviewed adapters and commands. Full video,
   weights and credentials are omitted.
4. Later, unpack it inside an allocated GPU session with a compatible official
   runtime and approved weights already provisioned. Run the bundled preflight,
   then the explicit inference command. The bundle installs or submits nothing.
5. Package the untouched adapter output with the bundled CPU-only packer. Upload
   that returned ZIP beside the matching job in Model Lab.
6. A validated SAM 3D import becomes an available reconstruction candidate for a
   local refit. A SAM 3.1 import remains an image comparison; its IDs cannot become
   video track IDs or inherit the older SAM 2 audit.

The recorded timestamp is the original video PTS. The model's one-image session
index zero does not replace it. An inferred hidden surface or attractive texture
is not an observation of the original object.

## GPU commands inside an unpacked job

Use the official environments separately. Replace example absolute paths with
the real already-provisioned public repository and approved local model files.

SAM 3D Objects:

```sh
python scripts/run_upgrade_job.py preflight \
  --repo /PATH/sam-3d-objects --config /PATH/pipeline.yaml
python scripts/run_upgrade_job.py run \
  --repo /PATH/sam-3d-objects --config /PATH/pipeline.yaml
```

SAM 3.1:

```sh
python scripts/run_upgrade_job.py preflight \
  --repo /PATH/sam3 --checkpoint-manifest /PATH/checkpoint-provenance.json
python scripts/run_upgrade_job.py run \
  --repo /PATH/sam3 --checkpoint-manifest /PATH/checkpoint-provenance.json
```

Preflight defaults to a fresh `preflight.json`; inference defaults to a fresh
`output/`. Existing files stop the runner. The runner checks the shipped hashes
and invokes a fixed Python adapter as an argument array; text concepts are never
expanded as shell commands. It does not automatically run inference after a
successful preflight.

Package a successful run without calling a model:

```sh
python scripts/pack_upgrade_result.py --job job.json \
  --adapter-output output --zip returned-result.zip
```

The older first two prepared demonstration jobs use the same pinned adapters but
predate `run_upgrade_job.py`. Their `command.json` and README remain valid and
unchanged. Newly prepared jobs include the simpler runner above.

## Local CLI

Discover the binding first; the printed `base_sha256` must be supplied unchanged:

```sh
.venv-sam/bin/python scripts/upgrade_bundle.py context \
  --twin-id jabra-stage6-v1--procedure-v1 --track-id small_object_a
.venv-sam/bin/python scripts/upgrade_bundle.py prepare \
  --twin-id jabra-stage6-v1--procedure-v1 --track-id small_object_a \
  --backend sam3d_objects --base-sha256 THE_PRINTED_ORIGINAL_HASH
.venv-sam/bin/python scripts/upgrade_bundle.py list
.venv-sam/bin/python scripts/upgrade_bundle.py import \
  --job-id job-THE_SAVED_ID --zip /PATH/returned-result.zip
```

`prepare` creates a new `data/upgrade-jobs/job-…/` and sibling ZIP. `import`
creates a new `data/upgrade-results/import-…/`; it never overwrites existing
assets, procedures, experiments or imports. Input bindings and every saved
artifact are checked again when the app or local runner resolves an import.

## What validation establishes

The importer rejects fixture-labelled results in production, stale original
programs, wrong job/source/timestamp/track bindings, changed hashes, incomplete
adapter runs, unexpected code pins, incompatible checkpoint attestations,
non-CUDA runtime attestations, invalid images/masks, missing native layout,
malformed GLBs, external GLB resources, archive traversal/symlinks/duplicates,
extra files and oversized archives. The upload budget is 512 MiB compressed and
expanded, 256 MiB per member, at most 512 members. Generated GLBs also obey the
existing portable viewer's smaller bounded mesh/accessor budget. Source images
must fit the local 16-megapixel inspection budget. Native layout NPZ files are
checked as nested archives before tiny transform arrays are loaded without pickle.

A SAM 3D import preserves raw `object.glb`, splat, layout and metadata. Its
generic `asset.glb` is byte-identical to the raw generated GLB. Generic metadata
records the correct Meta backend, generated appearance, exact original source
selection, raw metadata hash and native coordinates. No scale, pose, texture,
geometry repair, shared assembly transform or insertion path is invented. The
audited native-layout conversion is retained only as an unvalidated initialization
candidate; the current generic fitting stage still needs source-image checks.

An imported native asset is checked against the actual supported appearance
policy before alignment. Standard compatible vertex colours keep the legacy
adapter. Supported opaque UV/material assets use `native-base-color-v1`, with
perspective-correct texture sampling, scene transforms, depth and matching ray
visibility. Original geometry and texture bytes stay unchanged. Unsupported
materials remain valid native imports for inspection, with their exact reason.
See [native appearance scope and checks](native-appearance.md).

A historical version-one import receipt is verified using its original contract.
The app separately computes current adapter capability without rewriting that
receipt or granting physical accuracy. New version-two imports bind their
appearance policy explicitly. Loader/renderer implementation identities are
recorded separately in the new fit/tracking outputs and cache dependencies.

SAM 3.1 preserves every returned mask, cutout, score, normalized box and model
instance ID, including a legitimate zero-instance result. Comparison IoU is
agreement with the older saved mask, **not measured segmentation accuracy**.
Neither cross-frame identity nor the older model audit is inherited.

Hashes establish byte integrity and lineage. A returned adapter receipt is an
execution attestation supplied by the runtime, not cryptographic proof that a
particular model executed. Import is not automatic promotion or evidence of
better physical reconstruction, calibrated scale, contact, seating or robotics
simulation readiness.

## Verification completed locally

The focused suite exercises exact source preservation, stale bindings, corrupted
imports, unsafe ZIPs, native layout/pickle rejection, SAM 3.1 instance/identity
separation, explicit fixture rejection and the no-shell GPU launcher. Positive
adapter-contract tests use synthetic bytes exclusively in temporary workspaces;
they are never retained as genuine model results. Actual prepared Jabra jobs use
the saved 13.001144-second earbud source image and SAM mask. No genuine GPU-returned SAM 3D or SAM 3.1 result has been imported through
this handoff route, and no gated inference was run to prepare these handoffs.
Separate hosted SAM 3.1 image tests now exist; they do not validate the CUDA
result-import route or establish improved segmentation quality.
