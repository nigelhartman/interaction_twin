# Local pipeline and workspace API

## Current default · SAM 3D and descriptive mechanisms · 2 November 2026

New uploads select `meta_sam3d_default`. Its explicit processing order is:

1. Prepare bounded source frames with original video timestamps.
2. Muse Spark discovers components; SAM 3.1 produces video masks; Muse audits them.
3. Automatically select up to three distinct eligible component observations.
4. Reconstruct unchanged native SAM 3D Objects through the configured GPU worker, with at most two workers. An unavailable worker remains pending; there is no alternate reconstruction fallback.
5. Compile the source-linked procedure into the original v0.1 evidence program.
6. **Understand connections** (`mechanism_topology`): one guarded Muse request classifies persistent joints and removable parts from cited source photographs and masks. It proposes uncertain image hints, never authored 3D axes or pivots.
7. **Connect moving parts** (`mechanism_scene`): build a private CPU estimate in a common source-camera frame, then fit supported connections. Attached child motion is generated from its parent, fixed joint frames and a scalar configuration curve. Unsupported connections retain unknown child poses; the independent estimate is never published as a fallback.
8. **Build descriptive twin** (`object_program`): export and verify a separate v0.2 object program with native assets, joint frames, configuration samples, removable relations, unresolved connections and source provenance. The original evidence program and meshes remain unchanged.
9. Refresh the workspace and sortable human/agent library. The refresh only reuses fully validated constrained exports; it never starts a model, a paid request or a new CPU fit.

The default samples three frames per second, at most 40 seconds / 120 source samples, and reconstructs up to three selected components. Visibility, mask and reconstruction gates can produce fewer assets. The explicit Run consent names hosted Meta reasoning, SAM tracking, the additional connection request and transfer of selected reconstruction inputs to the configured worker.

The connection request uses the existing persistent five-dollar task envelope at `output/constrained-kinematics-20261102/task-budget-plan.json`, inside the older increment and shared hundred-dollar estimated-cost guards. The baseline is never reset by upload, resume, startup or refresh. An exact original-program SHA identifies one durable topology namespace. A completed response is verified and reused; an interrupted, failed, modified, missing-adoption or incompatible prepared attempt stays pending and never silently triggers a paid resend. These local estimates do not measure provider credit balance.

Use the default for an explicitly requested fresh run:

```bash
.venv-sam/bin/python scripts/run_pipeline.py --profile meta_sam3d_default --input /absolute/path/to/clip.mp4
```

For an existing paid topology result, run only the new stages through `mechanism_pipeline.adopt_verified_topology(program_path, proposal_path)`, then `topology(..., allow_paid_api=False)`, `constrained_scene(program_path, proposal_ref)` and `export_scene(scene_ref)`. Adoption verifies the exact source, original program, saved response/accounting and frozen reasoning contract without another request. A final already fitted scene can be passed directly to `export_scene` to avoid another fit. The CLI equivalents are in `scripts/run_mechanism_pipeline.py`; only the explicit topology command with `--allow-paid-api` can send a hosted request. Do not replay an old whole-pipeline run merely to add these stages: changing its frozen orchestrator source intentionally invalidates its original phase caches.

Current limitations are explicit. Depth, camera motion, native shape, joint anchors and configuration curves remain estimates; joint connectivity does not prove the true hinge location or physical fit. Where angle-quality checks fail, the scalar is labelled a **model configuration**, with the physical opening angle unresolved. There is no measured metric scale, collision/contact validation or executable robot policy. The frozen common-scene fitter currently supports the original `flow-components-1` SAM run. A later segmentation attempt stays pending rather than silently reading an older mask directory. Ordinary fresh uploads use the supported first attempt.

The historical local-comparison routes and their preserved results are documented below. They remain inspectable and do not replace the SAM default.

## Historical local comparison routes

The runner executes existing model adapters in order. It does not turn an HTML export into a claim of fresh inference. New input videos use actual preparation, model discovery, SAM segmentation, a separate image audit, automatic view selection, learned reconstruction, reference-pose fitting, tracked motion, procedure proposal and strict compilation. Each phase retains its errors and intermediate files. Unsupported shape, motion, identity, contact and seating remain unknown.

## Run

From `interaction-twin`:

```bash
.venv-sam/bin/python scripts/run_pipeline.py --profile local_low --input /absolute/path/to/clip.mp4
```

This profile sends sampled contact sheets and mask crops to **hosted OpenAI through the local Codex CLI**. It uses existing local **SAM 2.1 Tiny, TripoSR and CoTracker3 on Apple MPS**. TripoSR is an explicitly non-Meta temporary comparison. No new checkpoint download, credential discovery, cluster connection, manual box, manual geometry or manual pose is performed. Existing model environments and weights must already be present. There is no CPU learned-inference fallback.

The current default budget is at most 120 seconds and 120 sampled frames at 1 fps, with up to three automatically selected component observations at mesh resolution 64. The smoke profile samples 0.25 fps, at most 20 frames/70 seconds, also budgets up to three observations, and uses mesh resolution 48. These are ceilings, not promised object counts. Resolution and audit gates can leave fewer usable candidates. Historical single-component smoke results below retain their original recorded settings. Coarse sampling can miss fast interactions.

```bash
.venv-sam/bin/python scripts/run_pipeline.py --profile local_smoke --case recording-144929-1fps-v3 --run-id my-jabra-test
```

Saved-case fresh runs reprocess the original video in a fresh case. They do not merely reuse the existing masks or model outputs. Preparation preserves the original file, applies explicit HDR-to-SDR conversion when needed and verifies all native PTS before sampling. Existing coarse timestamps are not relabelled as new dense observations.

Repeat the identical command and run ID to resume. A phase is reused only when its profile, source identity, code/dependency hashes and every recorded output file still match. Changed dependencies or output pixels invalidate it. Failed outputs remain intact; retries use fresh attempt directories. A process lock prevents two pipeline invocations from owning the model worker together. The HTTP server also allows only one active request.

Each new run also preserves the exact orchestrator source loaded by Python in `pipeline-source-{hash}.py`, with its SHA-256 in run provenance. Cache identity uses that frozen source, rather than a file that could be edited while the process is running.

## Automatic multipart selection and retained failures

The selector ranks eligible RGB/mask views, applies bounded model-label diversity and same-time mask redundancy checks, then records every selected, skipped or rejected candidate in `flow-inputs-{attempt}/component-selection.json`. Category hints guide the small budget; they are fallible model text. Suppressing redundant observations does not merge physical identities, and different tracklets may still describe the same real part.

Reconstruction, pose and motion run independently for each selected tracklet. Successful siblings survive a later component failure. Each phase writes `flow-model-receipts-{attempt}/components.json`, `flow-pose-receipts-{attempt}/components.json` or `flow-motion-receipts-{attempt}/components.json`, with per-part status, inputs, output paths, timestamps, log links and explicit errors. The run receipt links these files under `artifacts.component_receipts` and the selection under `artifacts.component_selection`. Missing geometry, rejected initialization and insufficient motion support remain gaps; none receives a manual replacement asset or forced path.

Each generated part retains its own model units. **Separate asset** inspects its native learned surface; **Selected component motion** shows only that part's conditional camera-relative poses with its matching camera. Independent units and orientations are not assembled into a common scene. **Recorded positions** is available for multipart display only when an explicit saved common gauge passes the source/track/mesh/pose checks. Even that relative gauge does not establish metric scale, physical fit or verified contact.

## Review saved evidence without inference

```bash
.venv-sam/bin/python scripts/run_pipeline.py --profile verified_evidence --case jabra-stage6-v1--procedure-v1
.venv-sam/bin/python scripts/run_pipeline.py --profile verified_evidence --case wallet-stage6-v1--procedure-v1
```

These commands strictly validate existing interaction programs and hash their cited inputs, source video and artifacts. They are explicitly **not fresh model runs**. The original saved wallet variant uses cached model discovery/action evidence. A new hosted wallet request was initially rejected by automatic review and was held without a workaround; later explicit human authorization allowed a separate fresh wallet run. Both variants remain available as failure/stress evidence.

## Local workspace

```bash
.venv-sam/bin/python scripts/serve_workspace.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/workspace/index.html`. The server binds only to loopback; it does not publish data. Files are confined to the workspace, project evidence directory and three explicitly public viewer scripts. Symlink escapes, path traversal, foreign Host/Origin writes and cross-origin preflight requests are rejected. The server supports byte ranges for normal video seeking.

The page is a local model-processing UI. Uploading alone saves a private local file. Fresh Run requires an explicit hosted-reasoning consent field; the frontend must show the profile's disclosure before requesting it. Uploaded videos have bounded basenames, accepted video extensions and a 512 MiB limit. Frames are treated as untrusted model input data, not instructions. The subprocess argument vectors are constructed from the selected profile; the browser cannot send commands or arbitrary executable paths.

API contracts:

- `GET /health` → `{status: "ready", single_worker: true, cluster_connected: false}`.
- `GET /api/config` → `{profiles, max_upload_bytes, hosted_reasoning_requires_run_consent}`.
- `GET /api/cases` → `{cases: [{id, case_id, title, video_url, interaction_url, interaction_path, scene_url, workspace_case_url, duration_s, source_sha256}]}`. Variant IDs preserve historical procedures; case IDs name their source manifest directory.
- `GET /api/runs` → `{runs: [...]}`; `GET /api/runs/{id}` → full receipt/progress.
- `POST /api/upload`, binary request with `X-Filename: clip.mp4` → `{input_path, bytes, url}`.
- `POST /api/run`, JSON `{profile, case_id | input_path, hosted_reasoning_consent: true, resume_run_id?}` → HTTP 202 `{id, status, status_url}`. Fresh input paths must be under project `data/`; the CLI can accept explicitly supplied files elsewhere.
- `GET /api/cases/{published_id}/revisions` → original program hash, current revision head and saved revision metadata.
- `GET /api/cases/{published_id}/revisions/{revision_id}` → validated human review, derived document, replay checks and readable guide.
- `POST /api/cases/{published_id}/revisions`, JSON `{base_sha256, parent_revision_id, edits: {steps: {}, relationships: {}, assertions: {}}}` → HTTP 201 immutable revision. Each edits map uses an existing original item ID and `{decision: "accepted" | "rejected" | "edited", note, patch}`. Only approved semantic fields may change. Requests are bounded to 192 KiB; stale original hashes or review parents return HTTP 409.
- `POST /api/runs/{id}/cancel`, JSON `{}` → cancellation request. The subprocess process group is terminated, with a bounded SIGTERM/SIGKILL escalation; no hidden worker survives a normal server shutdown.

Status files and logs are under `data/flow-runs/{id}`. Optional failed geometry/motion/semantics produces `complete_with_gaps`, never a made-up asset/procedure. Failure of mandatory preparation/discovery/segmentation/audit produces `failed`. Cancelled runs remain inspectable. A real phase failure is not converted into a generic successful result.

## Human-authored review revisions

**Review & correct** opens an evidence-linked editor for steps, candidate relationships and point visual states. Accept, edit or reject an interpretation, inspect its cited frames and apply it to a draft. The original automatic program remains separately selectable and exportable. Rejected point states become unknown; edited point states remain human inference at their cited sample. Acceptance does not establish factual correctness or physical completion.

Saving creates a new read-only sidecar at `data/user-revisions/{published_id}/rev-*.json`; it never overwrites the original program or an earlier review. The save binds the exact original file hash and current parent revision, serializes concurrent writers and verifies the source/evidence artifact hashes. Loading a revision repeats full evidence validation and recomputes its document, replay and guide before serving it. Review paths are looked up only through published workspace cases. Drafts can be edited and exported offline, with an explicit **unsaved** label; server-backed saving requires the loopback processor.

The automatic compiler and schema remain frozen so their recorded hashes still identify historical outputs. `revisions.validate_derivative` validates a copy through the Stage 5 contract, normalizing only the two automatic-schema review flags on that copy. The returned derivative keeps its actual human-review flags and separate authorship disclosure. All other reference, state, identity, source, geometry, pose and runtime-policy constraints remain enforced. This is a scoped review adapter, not a relaxation of the automatic-output contract.

Workspace review decisions are not automatically benchmark ground truth. They do not establish measured physical accuracy; any correctness evaluation needs its own hash-bound human assessment and declared comparison basis.

## Benchmark and agent catalog

The development workspace includes saved-evidence cohort diagnostics for source linkage, masks, tracklets, conditional motion, failures and receipt-backed runtimes. Those private diagnostic artifacts are not bundled in this source release. The public release provides no aggregate physical-accuracy benchmark. Different sampling and model budgets must remain visible in any comparison.

[The agent catalog preview](../app/workspace/agents.html) shows saved responses produced by the real Python catalog. This page does not connect an LLM or start MCP. The separate read-only MCP server exposes search, component/procedure/state/evidence/asset queries and a descriptive simulation handoff over stdio. Setup and the prepared local client configuration are in [mcp.md](mcp.md). MCP has no upload, inference, paid API, scheduler, arbitrary-path or physical-execution tools. Generated assets and demonstrations still report `simulation_ready: false`.

## Meta API profiles and separate GPU readiness

As of 30 October 2026, authentication, genuine Muse Spark requests and hosted SAM 3.1 image requests have run through the official Meta API. This does not establish improved segmentation, accurate meshes or a completed physical twin. New video results and complete fresh runs require their own preserved receipts.

The current Meta profiles are:

* `meta_low`: Muse Spark discovery, independent full-frame seed localization, mask audit and procedure proposals; SAM 2.1 Tiny, TripoSR and CoTracker3 remain local.
* `meta_smoke`: the same reasoning provider at coarse sampling for a bounded orchestration check.
* `meta_sam31_low`: Meta reasoning plus independent hosted SAM 3.1 image segmentation and conservative 2D association.
* `meta_sam31_video_low`: Meta reasoning plus one SAM 3.1 video request using unchanged model-proposed box seeds. The sampled derivative is verified against its explicit original-PTS map; missing and withheld masks remain zero/unknown. This profile is a separate video-validation route, not a claim that video quality has already passed.

```bash
.venv-sam/bin/python scripts/run_pipeline.py --profile meta_low --case recording-144929-1fps-v3
```

Meta requests use the shared private SQLite ledger and a **$100 estimated-cost allowance**, with per-request admission checks, recorded token/frame usage, durable reservations and no automatic network retries. Unknown charges keep their reservation and block further paid admission until resolved. `/api/meta/budget` exposes sanitized local estimates without a credential read or provider request. These estimates do not show actual Meta credit balance, include spending through other tools, or enforce an account-wide cap. The user reconciles them against Meta's dashboard. New-video UI requests require explicit consent naming the selected hosted provider; exported projects contain no key.

`cuda_preflight` and `meta_api_preflight` generate **readiness reports only**: they perform no authentication, credential inspection, cluster access or learned inference. Hosted SAM 3.1 does not require our cluster runtime; the separate local-checkpoint SAM 3.1 adapter and SAM 3D Objects still require their documented CUDA environment and approved weights. No SAM 3D reconstruction or validated simulation handoff is implied by the API integration. Preserve original results and compare actual replacement outputs before choosing a better model.

## Validation

`tests/test_pipeline.py` covers changed input/output cache rejection, preserved optional failures, timeout/cancellation, source time/frame bounds, profile validation, traversal/symlink escape, consent/origin rejection, localhost HTTP health/config and byte ranges. HTTP tests need permission to bind loopback in the host sandbox. They do not send footage or run a learned model.

The actual run reports are stored separately from unit fixtures. The first fresh Jabra `run-jabra-smoke-20261028` exposed a missing geometric-mask-review call after actual SAM/audit and is preserved as failed. The corrected fresh Jabra `run-jabra-smoke-v2-20261028` completed all eleven phases in 219.375 seconds. It started from the original 33.636-second HLG recording, created nine exact-PTS samples, discovered eight tracklets and produced 30 actual SAM masks. The independent audit labelled 22 plausible, six rejected and two uncertain. Deterministic selection chose one component; actual TripoSR produced a 2,212-triangle mesh and actual CoTracker attempted two coarse observations. Only the unverified reference pose survived; no non-reference motion was supported. The five-step procedure, three candidate relationships and 15 assertions strictly validate. Workspace indexing succeeded without errors. This is a complete orchestration smoke test with limited geometry/motion, not a claim of recovered physical execution.

Saved-case Jabra/wallet verification runs explicitly record `fresh_model_inference: false`. Fresh wallet processing was subsequently explicitly approved by the human user; its separate `run-wallet-smoke-20261028` completed all eleven phases in 185.287 seconds. It retained its own nine source samples and three discovered tracklets, produced 18 actual SAM masks (14 plausible, four uncertain in the independent audit), reconstructed the automatically selected round object and attempted actual CoTracker tracking. Only the unverified reference survived; two other observations were withheld. Its four-step procedure, one candidate relationship and 11 assertions strictly validate. It did not borrow the Jabra or 33-frame wallet timestamps.

The aggregate actual result report is `data/flow-runs/local-validation-20261028.json`. This separates genuine fresh model execution from saved-evidence verification and records the motion failures alongside the successful orchestration.

A final source-snapshotted Jabra run, `run-jabra-frozen-20261028`, also completed all eleven phases in 232.506 seconds: seven model-proposed tracklets, 28 actual SAM masks, one learned asset, one conditional non-reference pose plus its unverified reference, and a six-step validated procedure. These are different model-proposed tracklets from the earlier smoke run, so the pose counts are not a matched quality comparison.

The historical Stage 6 focused backend suite passed **18 tests**, including an actual active-cancellation test that exposed and then verified the fix for a simultaneous process-group termination race. That final cancellation-lock fix changes the exact orchestrator source identity, so the completed learned run is preserved under its earlier source snapshot. Whole-pipeline resume under the final code was not claimed or rerun through another learned pass. Same-code cache reuse and changed dependency/output rejection are verified by the focused tests; a different source hash intentionally invalidates model phases.


## Multipart regression on 29 October

`run-multipart-jabra-20261029` completed with gaps in **462.443 seconds**, using 34 exact-PTS samples from the Jabra recording. Its actual program is `data/flow-multipart-jabra-20261029/flow-procedure-1/interaction.json`: eight proposed tracklets, three learned assets, seven steps and 15 point assertions. The three selected assets were small-piece observations (`left_piece_initial`, `loose_piece_a`, `right_piece_later`), rather than the intended diversity of case body, lid and loose part. All three model runs were retained; successful orchestration did not establish three distinct physical identities or a physical assembly. The selection receipt makes this limitation inspectable. Later selection changes require their own fresh run and benchmark cohort; this result is preserved unchanged.


The separate `run-multipart-jabra-v2-20261029` then completed **with gaps in approximately 7.2 minutes**. Its program is `data/flow-multipart-jabra-v2-20261029/flow-procedure-1/interaction.json`: 34 source samples, three learned assets (`container_body`, `right_piece_returned`, `container_lid`), eight steps, seven candidate relationships, four variables and 15 point assertions. Each asset has two supported conditional non-reference pose samples and one unverified reference; the remaining unsupported intervals stay absent. Reference initialization and generated geometry remain physically unverified, and the parts have no validated common assembly gauge. The selection is more varied in this run, but its model-proposed tracklets differ from v1, so the two runs do not establish a matched accuracy improvement. V1 remains preserved as the selection-limit comparison.
