# Positioning review — 1 November 2026

Primary project pages and author-submitted arXiv records were rechecked on **1 November 2026**. This is a focused comparison of the named projects, not an exhaustive novelty or patent search. Their reported performance is not independently reproduced here.

The defensible pitch is: **Turn a recorded interaction into an inspectable, source-linked authoring package: parts, candidate states and relationships, task steps, shape drafts, and explicitly bounded motion evidence that software agents can retrieve.**

## What the related projects already do

| Project | Verified scope | Implication for our pitch |
| --- | --- | --- |
| **Articulat3D** | Reconstructs articulated twins from monocular video. It uses 3D point trajectories, moving rigid groups and refinement with joint axis, pivot and per-frame articulation variables. First arXiv submission: **12 March 2026**, revision: **24 June 2026**. [Author project page](https://maxwell-zhao.github.io/Articulat3D/), [paper/history](https://arxiv.org/abs/2603.11606). | Video-to-articulated-object reconstruction and motion constraints are established research territory. We have not demonstrated stronger reconstruction or recovered articulation. |
| **SPLATART** — Lewis et al. | Learns articulated Gaussian representations from **posed images**, with part segmentations for some images; separates part segmentation from articulation estimation and supports deeper kinematic trees. Submitted **13 June 2025**. [Author-submitted paper](https://arxiv.org/abs/2506.12184). | Part separation, connectivity and richer joint hierarchies are not sufficient novelty claims. |
| **SplArt** — Lin et al., a different paper | Uses two sets of posed RGB images in different articulation states, 3D Gaussian reconstruction and mobility parameters to estimate parts and kinematics. Submitted **4 June 2025**. [Author-submitted paper](https://arxiv.org/abs/2506.03594). | Do not confuse this similarly named method with SPLATART, or claim that articulating splats is new. |
| **φ-Scene** | Reconstructs physically grounded scenes from a single image through support topology, penetration resolution and rigid-body settling, while preserving image alignment. First submitted **19 June 2026**, revised **24 August 2026**. [Author project page](https://phi-scene.github.io/), [paper/history](https://arxiv.org/abs/2606.21596). | Physical assembly, stable contact and simulation validation are substantive capabilities we currently lack. An attractive scene or plausible placement does not establish them. |

## Our demonstrated contribution

The present implementation is a workflow and data product, with custom engineering beyond a model endpoint:

- Meta SAM 3.1 component masks and Muse Spark proposals become source-linked procedure records with original timestamps, mask/image hashes, gaps and retained failures.
- Source-pixel reconstruction, generated shape priors and optional inferred completion remain distinct. Observed camera-patch exports preserve original photographic appearance; completion is a separate layer.
- Short-interval native-model checks preserve independent gauges, disjoint scoring and absent unsupported poses. The new method is mixed: the reserved base/lid checks improve over the preserved baseline; the earbud baseline is stronger. The controlled 12 fps studies show no gain at the shared timestamps.
- Twelve bounded read-only MCP tools expose published program, evidence, model, reconstruction and motion records. A genuine **36-call scripted SDK handoff** retrieves and validates a portable package. Separately, the new **Muse Spark-directed inspection** actually chooses five SDK calls over two model rounds and produces a fallible, cited assessment from pruned text metadata. It receives no source pixels or binary models and performs no robot action.

These are supported by local artifacts, not claims that the named research lacks comparable interfaces. The focused source review does not prove that agent-readable evidence packages are unprecedented.

The current pipeline uses its own saved-source reconstruction and validation modules, native TripoSR drafts and MCP adapters. The named papers are research comparisons, not the engine behind the demonstrated outputs. Nothing here establishes that our geometry is better than theirs.

## Submission wording

Use: **“A video-to-authoring workflow that keeps parts, procedure and evidence together, and makes the resulting draft inspectable by people and agents.”**

Use: **“Toward reusable robotics and simulation assets.”** Current outputs are descriptive, unreviewed drafts in arbitrary component units.

Avoid: “first video-to-digital-twin,” “better than Articulat3D,” “fully automatic simulation-ready twins,” “recovered correct hinge/contact,” “validated robot training,” or “the entire digital-twin production problem is solved.” There is no matched benchmark or physical experiment supporting those statements.

The next evidence that would materially strengthen the claim is a calibrated, overlapping capture; verified component shape and common scale; then independently checked joints, collision/contact and actual simulator execution. More API calls alone do not supply that evidence.

## Local verification references

- [Current genuine model-directed MCP review](../output/overnight-session-20261031/meta-run-v3-independent-actual-review.json): 121 checks, 38 artifact hashes, two genuine model rounds/five actual SDK calls; reported-usage estimate **USD 0.026952**.
- [Preserved prior incomplete attempt](../output/overnight-session-20261031/meta-run-v2-independent-actual-review.json): three genuine inspection plans/eight SDK calls; final output incomplete, **USD 0.056484**. The earlier HTTP 400 attempt remains preserved at zero estimated charge. Neither is rewritten as a success.
- [Scripted portable handoff](../output/overnight-session-20261031/motion-mcp-delivery-v1.json): the separate 36-call, three-GLB, six-image, four-step package and relocated import checks.
- [Controlled density audit](../output/interval-motion-20261031/all-density-independent-audits-v1.json): fixed inputs, exact source timestamps, matching controls and preserved failures; no physical-accuracy claim.

The model assessment's “not_model_audited” wording describes the original evidence snapshot it received. A separate later interval scope audit exists; the model's text is not a universal statement that no such review occurred. Citation validation checks returned identifiers, not semantic truth.
