# Technical overview

Interaction Twin prepares reusable object-interaction context from ordinary demonstration video. The custom application connects Meta perception and reasoning with source-linked geometry, estimated joints, review and an agent-readable library.

The [public demo](https://nigelhartman.github.io/interaction_twin/) contains five owner-recorded example families. It is a static snapshot: saved results and browser rendering, with no upload, paid inference, private admin or hosted MCP service. The source branch contains the implementation; see the [review setup](source-setup.md) to run it with an empty catalog.

## From a recording to an object program

```text
Video → Muse component proposals → SAM 3.1 masks → SAM 3D mesh drafts
                                                   ↓
                         Source checks + estimated motion and connections
                                                   ↓
                              Object program + original evidence
```

**Muse Spark** proposes parts, semantic relationships and demonstrated actions. **SAM 3.1** segments visible components. **SAM 3D Objects** generates native component meshes from selected source frames and masks. CoTracker and local fitting provide additional motion observations; local validators decide which estimates can be displayed.

A [descriptive object program](object-program.md) binds component IDs and asset hashes to source timestamps, proposed relations, estimated joint frames/axes/angles and action steps. An attached lid is evaluated through its parent and fixed joint frames using forward kinematics. It cannot acquire an independent translation during replay. This enforces the mathematical construction's attachment; physical hinge location, metric fit and contact remain unverified.

Original model outputs remain available separately from declared geometry refinements, source-visible 2.5D surfaces and temporal hypotheses. Unsupported observations stay explicit. Source-image checks establish only their stated image-space support.

Source-reviewed semantic corrections are separate, source-bound overlays: the original relationship and evidence remain inspectable. A selected correction expires when its approved revision, source program or scene changes. Revising a candidate relation does not establish physical attachment or modify geometry/poses.

## Build, review, improve, approve

```text
Build → Separate reviewer prompts + code checks → You approve → Curated library
  ↑                         │
  └── targeted repair ──────┘
           at most two
```

The pipeline has four review roles: **evidence, geometry, mechanics and semantics**. Deterministic checks validate references and contracts. When paid review is explicitly enabled, separate prompts ask the same Muse model to critique those areas. This is role separation, not four independent models. A separately activated visual critic can inspect source images, masks and native renders.

The extended film V2 animates this review loop and a separate Muse → SAM 3.1 → SAM 3D → object program → MCP/viewers map. Both are architecture illustrations, not live model-event traces.

The bounded improvement policy permits an initial result followed by **at most two repairs**. Targeted stages can revisit reasoning, tracking, reconstruction or mechanics. The workflow retains comparable measured improvements and stops on insufficient evidence, regression, exhausted rounds or budget/transport uncertainty. Automatic review never substitutes for owner approval of the exact revision. The public code omits private activation receipts and budgets; opening a fresh checkout does not activate paid review.

The private review desk can compare an earlier scene with the current revision on the same source clock. Optional part, joint and timestamp targets stay private review annotations; approval remains restricted to the exact current revision.

Feedback memory retrieves scoped, revocable human guidance for a later run. Exact-recording guidance takes precedence; transfer to an object family or global method requires an explicit scope. This is context retrieval, **not model-weight training**, and feedback does not become observed fact.

## Custom implementation map

These are implementation entry points in this release. Meta and upstream authors supply their models and SDKs; the project supplies the application and contracts around them.

| Custom work | Source entry points |
| --- | --- |
| Pipeline orchestration and immutable stage receipts | [pipeline.py](../app/pipeline.py), [admin_pipeline.py](../app/admin_pipeline.py) |
| Evidence procedure and descriptive object program | [interaction schema](../app/schemas/interaction.schema.json), [stage5.py](../app/stage5.py), [object_program.py](../app/object_program.py), [object-program schema](../app/schemas/object-program.schema.json) |
| Joint proposal, constrained fitting and connected replay | [mechanism_pipeline.py](../app/mechanism_pipeline.py), [sam3d_mechanisms.py](../app/sam3d_mechanisms.py), [browser scene contract](../app/showcase/aligned-scene-contract.js) |
| Source-visible geometry and separate motion hypotheses | [surface contract](../app/source_visible_flexure_contract_v1.py), [temporal continuity](../app/source_temporal_continuity_v1.py), [curated motion adapter](../app/curated_motion_hypotheses_v1.py) |
| Review roles, bounded repair and exact owner approval | [review_agents.py](../app/review_agents.py), [Muse reviews](../app/admin_muse_reviews.py), [visual audit](../app/admin_visual_audit.py), [execution policy](../app/admin_execution_policy.py), [workflow](../app/admin_workflow.py) |
| Source-bound semantic correction overlays | [correction contract](../app/semantic_relation_corrections_v1.py), [publication adapter](../app/semantic_correction_publication_v1.py) |
| Revocable feedback retrieval | [agent_memory.py](../app/agent_memory.py) |
| Curated human/agent discovery and fixed artifact access | [library_query.py](../app/library_query.py), [curated catalog](../app/curated_twin_catalog.py), [MCP server](../app/scripts/serve_twin_mcp.py) |
| Cross-recording interface/task drafts | [cross_recording_task.py](../app/cross_recording_task.py) |
| Browser robot task authoring and validation | [robot_task_program.py](../app/robot_task_program.py), [browser compiler](../app/showcase/robot-task-plan.js), [controller](../app/showcase/robot-controller.js) |

This map identifies shipped custom implementation, not exact creation dates. The submitter confirms event-built scope, any pre-existing work and all human contributors in the final competition declaration. See [release scope](PUBLIC_RELEASE.md).

## Agent reuse and robotics

The [local MCP server](mcp.md) provides read-only discovery, exact source/program inspection, model/evidence resources and bounded task preparation. A saved scripted client performed **42 MCP calls**, retrieved six native GLBs and nine source images, and prepared a five-step phone/charger draft. A separate Muse metadata critique records unresolved compatibility. The saved trace is not a live autonomous agent or proof of successful charging.

The browser robot uses an authored five-DOF rig, inverse kinematics and typed pick/place/open/close actions. Supported instructions compile locally. Its motion is a kinematic rehearsal; it does not establish grasp, contact or insertion success. A separate [Newton CPU experiment](newton-physics-bench.md) demonstrates bounded rigid-body contact with declared proxy geometry and assumed parameters. It does not validate robot grasping.

## Verification and remaining limits

The lightweight source review checks software contracts and an empty application startup without model inference. Exact commands and locally verified scope are in [source setup](source-setup.md). An optional [CI recipe](ci/source-review.yml) is provided; remote CI is not enabled. Saved model outputs and full GPU inference require additional artifacts and runtimes.

The current prototype has limited complete geometry, arbitrary inferred scale, partial tracking, and unverified physical joint fit. Flexible wallet/cable behavior, reliable grasping and constrained insertion remain research work. There is no measured authoring-time reduction or equivalent-input accuracy benchmark.
