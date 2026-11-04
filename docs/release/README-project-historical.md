# Interaction Twin

**Turn a recorded object interaction into separate 3D drafts and a source-backed procedure that agents can inspect through MCP.**

The prototype connects Meta SAM 3.1 video segmentation and Muse Spark reasoning with SAM 3D Objects, local point tracking, mathematical reconstruction and a structured interaction program. It helps technical authors, XR developers and robotics developers prepare and review reusable interaction assets.

The result remains an authoring draft. Connected-part replay uses estimated mechanical constraints. A separate Newton CPU bench executes a bounded drop/contact experiment; robot grasping, insertion, metric fit and precise joint recovery remain unfinished. Physical accuracy and reduced authoring time have not been measured.

## Open the current build

```sh
.venv-sam/bin/python scripts/launch_workspace.py
```

Or double-click **Launch Workspace.command** and leave its Terminal open. The launcher installs nothing, contacts no cluster and reuses this workspace’s server when available. The current local instance is `http://127.0.0.1:8766/showcase/index.html#samples`; use the URL printed by the launcher when starting a different instance.

| Page | Purpose |
| --- | --- |
| [Full walkthrough](showcase/walkthrough.html) | Current three-minute narrated draft, real saved outputs and complete robot task |
| [Project page](showcase/index.html) | Interactive examples, method and current capabilities |
| [Recording library](showcase/library.html) | Six curated recording families, 19 SAM 3D models, connected previews, procedures and evidence |
| [Private owner desk](admin/index.html) | Sign in, upload, review, repair, approve and inspect feedback memory |
| [Owner workspace](workspace/index.html) | Detailed inspection, diagnostics and export after sign-in |
| [Motion Studio](workspace/motion-lab.html) | Exact-frame motion candidates beside preserved baselines |
| [Reconstruction Studio](workspace/geometry-lab.html) | Source-derived geometry, checks and separate inferred completion |
| [Agent catalog](workspace/agents.html) | Read-only interaction assets and real saved MCP workflows |
| [Model Lab](workspace/upgrades.html) | Local reprocessing and prepared GPU handoffs |

The [app guide](workspace/guide.html) explains the controls. The local server is required for uploads, processing and new report inspection. Separately exported portable viewers work without the source checkout.

## How it connects

1. **Preserve the source:** recording, sampled image bytes and actual timestamps.
2. **Separate components:** Muse Spark proposes candidates and visual prompts; SAM 3.1 produces visible masks with saved fallible audits.
3. **Recover geometry:** SAM 3D Objects generates native component candidates from an exact video frame and SAM mask through the GPU handoff. Local tracks and multiview mathematics recover supported visible patches separately. SAM 3D is the default upload and primary inspection backend. Native components are loaded independently; historical comparisons remain available as saved experiments.
4. **Check motion:** CoTracker points and rigid fitting produce camera-relative candidates. Disjoint point checks, mask checks and declared time splits limit displayed support.
5. **Recover connection rules:** Muse proposes persistent joints versus removable parts from source frames. A local geometric fit estimates native-mesh anchors, a fixed hinge axis and a source-timed angle curve. Connected children are computed through their parent and joint, including between samples. Unsupported connections are withheld rather than allowed to float independently.
6. **Compile context:** a schema-validated evidence program connects components, proposed relationships, exact-time states, steps and evidence. A separate `object-program.json` adds estimated kinematics without rewriting historical evidence. Human review creates a separate revision.
7. **Hand off:** export original GLBs and descriptive programs, or retrieve them through MCP. `get_kinematics` exposes joint frames, axes, angle samples and their provenance. The robot storyboard illustrates authoring; it does not run physics.

Independent parts and intervals retain their own gauges. Missing motion stays absent from saved evidence; optional short-gap display estimates are visibly labelled and can be disabled. Overlap or disappearance does not prove contact or insertion. SAM masks alone do not recover complete textured objects.

### Connected parts stay connected

The earlier independent mask fit could detach a lid from its case. The new replay derives a hinged child as `parent_pose × parent_from_joint × rotation(axis, angle) × joint_from_child_zero`. Anchors are estimated in native mesh coordinates; original fits use mesh vertices and separately declared refinements may fit off-surface mating frames. Anchor coincidence is checked at saved and intermediate times. The viewer exposes estimated angles, source jumps, native anchor coordinates and an optional axis overlay.

The descriptive companion distinguishes the estimated angle range observed in the clip from unknown physical stops. It includes source/model hashes, units, removable relationships, unresolved connections and explicit `simulation_ready: false`. Correct attachment by construction does not establish the correct physical hinge location or angle. Occlusion, poor masks and incompatible generated shapes can still leave a connection unresolved.

## Process a recording

The private owner desk also includes **Marketing** and **Overview**. Double-click `Open Private Studio.command` to set your owner password interactively on first use and open the desk. Marketing creates branded 15/30/45-second films and X/LinkedIn drafts from exact project footage, with local edits, optional guarded Muse editor/critic rounds and exact-version approval before exporting. Pixabay audio can be imported with provenance; nothing posts automatically. Overview reports locally estimated costs, reported tokens, processing/render jobs and explicitly refreshed GPU allocation status. See [the studio guide](docs/marketing-studio.md) and [Cobalt Motion art direction](docs/marketing-art-direction.md).

Open `/admin/`, sign in and choose **New recording**. Only the owner can upload or start processing. The upload automatically queues a private candidate after explicit processing consent. The default `meta_sam3d_default` profile uses SAM 3.1 segmentation, SAM 3D Objects geometry and Muse Spark reasoning. It accepts clips up to 40 seconds at 3 fps, with at most 120 samples and three selected generated components. Longer clips are rejected before model calls. Sampled frames/video go to Meta and selected RGB/masks to the configured GPU worker; browsing starts no inference.

Set the password once with the one-use local setup code, or run `.venv-sam/bin/python scripts/admin_access.py set-password` for a hidden local prompt. There is no default password. See [owner access](docs/admin-api.md) and [curation and memory](docs/admin-workflow.md).

Every candidate gets four deterministic structural checks and four separately requested, text-only Muse reviewer roles. These are role prompts to the same model, not independent visual experts. Incomplete hosted reviews remain explicit. Feedback can request a bounded new reasoning, tracking, reconstruction or mechanism revision; verified predecessor outputs are reused, originals are preserved, and uncertain paid requests never automatically resend. Reconstruction repairs select another audited source view when available. Later segmentation versions currently leave constrained alignment pending until supported; they never reuse old masks as a substitute.

Only an exact owner approval advances the public library and default read-only MCP catalog. The last approved revision stays public during repairs. A reasoned owner decision may acknowledge hosted objections for a research draft; source/hash/structural failures remain non-overridable. Private feedback is retrieved by recording, optional task collection or general method. It is revocable guidance, not weight training or physical evidence. Full working-catalog MCP access requires `--private-owner`.

One recording pipeline runs at a time; up to two component reconstructions run concurrently on the configured GPU worker. If that worker is unavailable, source-bound inputs remain pending without an alternate geometry backend. Progress, cancellation, per-part failures and unsupported stages remain inspectable. Start with a sharp recording of a rigid part moving out of and back into a holder. Keep fingers away from useful surfaces. Ordinary draft generation needs no checkerboard. Known dimensions and camera metadata can later support physical-scale checks.

## Actual results and limits

- The current Jabra example retains four original SAM components and a separately declared rim-derived lid for the refined replay. Automatic source-mask and facing checks retain 49 fitted joint samples and withhold 52 uncertain samples. The fixed joint remains connected; fitting used the same masks and is not a held-out accuracy benchmark. The separately exported [object-program format](docs/object-program.md) carries joint frames, axis, angle curve and source evidence through the webpage and MCP. Generated mating surfaces still leave a residual closure gap; this is not verified contact or physical articulation accuracy.
- Official SAM 3.1 image/video segmentation and Muse Spark reasoning have run on saved recordings.
- The curated library contains six recording families and 19 browsable SAM 3D component models: Jabra parts, wallet/flap/coin, camera/door/batteries, laptop base/lid/cable, phone/connector, and charger/cable/socket components. Every recording opens a source-bound connected preview. Wallet, phone and camera replays also offer source-visible surfaces made from saved RGB, SAM masks and estimated monocular depth: 91/99, 26/47 and 75/75 supported samples respectively. These are partial observations, with unknown hidden regions and numerical wallet fold parameters, rather than repaired native SAM models. Sparse phone/camera display repeats identify the actual observation time. Original GLBs remain separate downloads, including the phone’s extra plate; whole-cable deformation, physical scale and electrical compatibility remain unverified.
- The laptop also carries a separate source-conditioned motion layer: nine of fifteen missing saved rows have useful pose estimates, with fixed-joint forward kinematics and explicit view-change barriers. Original unknown rows remain unchanged. The interval around 5.464 seconds now displays continuously; one conditional mask-fit frame has a visible perspective mismatch. `get_motion_hypotheses` retrieves the same estimated layer through read-only MCP. This is not independent physical tracking accuracy.
- The genuine earbud's raw model layout projects back onto its source frame with same-frame silhouette IoU 0.774435 before pose fitting. The camera is estimated from the same inferred pointmap; this is a single-frame diagnostic, not measured 3D accuracy. [Inspect the source projection](data/sam3d-source-projection-20261102/review-v1/index.html).
- A fresh CoTracker3/MPS experiment fits this new mesh to one later source sample under the unchanged conditional gates: 24 held-out points, median image residual 0.752 px and raw silhouette IoU 0.736. Eight unaudited samples stay unsupported; the source initialization stays unverified. No old mesh poses were reused or program assets promoted. [Inspect the motion diagnostic](data/sam3d-layout-motion-20261102/earbud-a-v2/observer.html).
- Fresh Jabra/wallet software pipelines complete with their geometry/motion gaps preserved.
- Automatic part drafts and partial mathematical reconstruction work. The camera example has 5,377 observed local triangles; inferred faces are a separate optional layer.
- Short-interval motion results are mixed: base/lid improve conditional image support; the earlier baseline remains stronger for the earbud. These are image checks, not physical pose accuracy.
- A controlled lid test passes six shared non-reference timestamps at both 3 fps and 12 fps. No common-frame gain is claimed.
- Portable export/import, real local MCP consumers, review/capture guidance and an illustrative robot editor work. Read-only `get_observed_surfaces` exposes the exact source-visible program, evidence and geometry separately from native kinematics. The robot demo uses explicit visual earbud seating corrections reviewed against source frames; this does not certify cavity fit, grasping or contact.

See the [motion report](docs/short-interval-motion.md), [submission script](docs/submission-demo-20261101.md) and [positioning review](docs/positioning-review-20261101.md) for exact versions, denominators and scope. New outputs preserve old programs, model bytes and failed attempts.

## Cross-recording reuse

The [phone-charging demonstration](showcase/demos/charge-phone-v1/index.html) uses a real saved MCP client trace across the approved phone and charger recordings: 42 calls retrieve and verify six native GLBs and nine cited images. A bounded deterministic compiler combines attributed interface candidates and saved steps into a five-step task draft. A separate actual metadata-only Muse review critiques that draft. The trace, task and review are distinct artifacts; browsing never invokes inference.

USB-C and European-socket context is explicitly attributed to the owner’s capture description and bound to the exact recordings. It is not inferred from display names or treated as a measured interface. The plan retains unknown same-cable identity, plug/port dimensions, approach calibration, cable dynamics, power ratings and successful charging. No physical connection or robot execution is verified.

## Models and compute

| Stage | Implementation | Runtime |
| --- | --- | --- |
| Discovery, scope review, task reasoning | Muse Spark / official Meta API | Hosted, paid and guarded |
| Visible masks | SAM 3.1 API; preserved SAM 2.1 Tiny | Hosted / local Apple MPS |
| Point tracking | CoTracker3 | Local MPS or explicit CPU mode |
| Historical mesh experiments | TripoSR, preserved outputs | Not the default geometry backend |
| Observed geometry | Source tracks, SfM, robust fitting, eligible stereo | Local CPU |
| Optional inferred surface | MoGe-2 Small or harmonic filling | Local MPS / CPU |
| Native component generation | SAM 3D Objects, verified official source and weights | Actual Linux/A100 runs; strict return import; fresh alignment remains separate |
| Connection graph and constrained replay | Muse topology + native-mesh joint fitting | Hosted reasoning; local CPU fit and browser FK |

Meta requests share a **USD100 local estimated-cost guard**, **USD2 per-request reservation ceiling** and the current increment’s additional allowance. This is not an account-wide cap or live credit balance. Credentials stay in ignored private storage; never place them in a repository or asset export. There are no automatic retries or model downloads.

## Reproduction

This checkout has prepared runtimes and private recordings, which are not a distributable installation. Check prerequisite presence without inference:

```sh
.venv-sam/bin/python scripts/launch_workspace.py --check
.venv-sam/bin/python -m unittest discover -s tests -t . -v
```

A code-only snapshot contains no saved examples or film. Use `scripts/launch_workspace.py --workspace` after preparing its runtime; the empty workspace offers New video. A smoke check using this machine’s borrowed runtime is not proof of a fresh dependency installation.

Dependency locks are split across `requirements-sam.lock.txt`, `requirements-local-geometry.lock.txt`, `requirements-meta-api.lock.txt`, `requirements-media.lock.txt` and `requirements-mcp.lock.txt`. Install upstream code/checkpoints explicitly under their licences. See [pipeline contracts](docs/pipeline.md), [model handoff](docs/model-handoffs.md), [MCP setup](docs/mcp.md) and [portable assets](docs/portable-assets.md).

Tests cover software contracts and synthetic analytic fixtures; they do not establish real-world reconstruction accuracy. Report checks with their exact saved run. Do not apply an older test count to newer features.

## Review the custom implementation

| Concern | Starting points |
| --- | --- |
| Interaction contract and orchestration | [schema](schemas/interaction.schema.json), [pipeline](pipeline.py) |
| Source-derived geometry and inferred layers | [geometry pipeline](observed_pipeline.py), [reconstruction](observed_geometry.py), [surface validation](observed_surface_validation.py) |
| Short-interval motion and exact-frame display | [motion experiment](interval_motion.py), [strict display adapter](motion_lab.py), [Motion Studio](workspace/motion-lab.js) |
| Agent retrieval and guided inspection | [catalog](twin_catalog.py), [read-only MCP server](scripts/serve_twin_mcp.py), [bounded Meta planner](mcp_assessment.py) |
| Review UI and source-bound scene loading | [workspace](workspace/app.js), [scene loader](workspace/case-data.js), [capture feedback](workspace/capture-feedback.js) |

For a code-only package, start with the [portable reproduction guide](docs/reproduction-review-20261101.md). Saved-data audits and model inference require their separately documented inputs.

## Submission and attribution

The current 45-second overview uses original instrumental music, the Cobalt design and captions. It is separate from the participant guide’s three-minute demo requirement. The demo desk links a recording plan and delivery checklist. Nothing has been publicly deployed or submitted.

Meta and upstream authors supply the models and SDKs. The custom contribution is integration, interaction representation, source/quality gates, geometry and motion experiments, review app, exports and agent handoff. Existing video-to-articulation research includes Articulat3D; no equivalent-input benchmark establishes novelty or superiority.

Before release, select shareable captures and identify hackathon-built versus pre-existing code and all contributors. Generated catalogs and media may contain private evidence even when `data/` is ignored. Publish a deliberately prepared source package, not the entire working checkout.

The earlier development README is preserved byte-for-byte locally as `README-history-20261031.md`. Its historical stage descriptions refer to their recorded dates. That operational history is intentionally excluded from code-only review packages.

## Saved model-directed MCP inspection

Muse Spark completed a guided read-only inspection of the fresh Jabra base: two reasoning rounds and five actual SDK queries, followed by a cited, fallible assessment. It reads bounded text metadata only; it does not inspect pixels, alter geometry or execute physics. Physical flags stay false by policy, and citation checks establish returned identities rather than semantic correctness. Earlier rejected/incomplete attempts are preserved. Open `/workspace/agents.html#model-assessment` in the local app; browsing starts no inference.
