# From video evidence to robot task assets

**3 November update:** the public library now has separate native SAM 3D case-base, lid and earbud components, an estimated connected lid joint, and a programmable browser-workbench increment. See [robotics-workbench.md](robotics-workbench.md) for the current task/kit architecture. The dated audit below records the earlier baseline; statements about a combined case asset or an unsolved illustrative arm describe that baseline. The physical requirements and simulator-validation gates still apply.

Prepared 30 October 2026. This is an integration plan and offline readiness audit. No simulator was installed or executed, no robot was controlled, and no additional API or GPU inference was used for this document.

Our useful deliverable is a package an agent can inspect: **objects, candidate component relationships, demonstrated steps, generated meshes and the original supporting observations**. The next robotics milestone is to turn one selected interaction into a calibrated, testable simulator task. Neither generating a prettier mesh nor drawing a robot arm completes that transition.

## What exists and what the robot preview means

The current `interaction.json` contract preserves source timestamps and hashes, object tracklets, fallible reasoning proposals, visual assets and conditional camera-space poses. It deliberately declares physical scale and physical pose unverified, gaps absent, and `simulation_ready: false`. MCP already exposes `get_simulation_manifest(twin_id)` and `twin://{twin_id}/simulation` as descriptive handoffs. These are preparation inputs, not robot execution commands.

A Three.js preview may show a stylized arm approach, grip, withdraw, return and release an earbud. It is a **scripted task mockup** in a presentation coordinate system. Its attachment, path, synthetic cavity, finger closure and apparent insertion are illustrative. They are not recovered human motion, solved robot IK, contact dynamics or evidence of seating. Keep the source video and recorded evidence accessible alongside the preview; do not insert mockup poses into `interaction.json` or measured-motion exports.

```text
Video + model observations                 Presentation only
           │                              Three.js task mockup
           ▼                                      │
Evidence procedure + visual assets ────────────────┘
           │
           ▼
Readiness audit ──► explicit missing physical requirements
           │
           ▼   only after calibration and validation
Prepared object bodies + robot + task goals
           │
           ├──► MuJoCo: first controlled contact experiment
           └──► Isaac Sim / Isaac Lab: later asset and learning integration
```

## The first real simulator task

Start with **one rigid earbud, one fixed open receptacle and a parallel-jaw robot gripper**. Keep the case open and fixed so the first task tests grasp, approach and placement without requiring an unobserved hinge or latch model. Use an intentionally reviewed simulator fixture if the reconstructed cavity is inadequate, labelled as a fixture rather than a captured digital twin. Never silently repair a reconstructed object or claim the fixture is model output.

Jabra is the clearest task story but has specific gaps: the current case asset combines base and lid, cavity identity may be ambiguous, and contact/charging are invisible. The camera/battery case offers rigid components but needs a real bore and correct insertion orientation. Wallet/coin is harder as a first physical task: the wallet is deformable, a rigid proxy can block or exaggerate openings, and a thin coin on a flat surface is difficult for a parallel-jaw grasp. A coin transfer above an already-open measured pocket is a reasonable later restricted task; full wallet manipulation is separate work.

The first success claim should be narrow: **a robot lifts an object, reaches a defined target, releases it and the object remains in the verified receptacle region without excessive penetration**. Charging, magnetic engagement, latch closure and functional electrical contact require additional evidence and models. All thresholds must be defined from measured geometry and validation, not copied from an arbitrary example.

## Physical inputs that the video does not give us yet

| Requirement | What we have | What must be added before a physics claim |
| --- | --- | --- |
| Physical identity | Candidate tracks; possible swaps during occlusion | Confirm object/cavity association and keep unresolved identities explicit |
| Scale and geometry | Generated native-unit meshes | A measured dimension or calibrated capture, one fixed metric scale per object, shape and cavity review |
| Coordinates | Conditional `camera_from_native_mesh` samples | Axis conventions and calibrated world, camera, body, robot-base and target transforms |
| Collision shapes | Visual assets | Separate reviewed colliders, cavity preservation and measured clearance |
| Mass and inertia | Unknown | Measured or disclosed estimated mass, center of mass and valid inertia in body coordinates |
| Mechanics | Candidate `component_of` / `removable_from` relations | Verified mechanism type; joint anchor, axis, range and passive forces where applicable |
| Grasp | Human handling observations | End-effector-specific grasp/pregrasp frames, opening, approach clearance, stable contact |
| Insertion | Candidate visual inside/outside states | Cavity mouth, insertion axis, seated frame, tolerances and collision-aware approach |
| Contact | Unknown | Validated friction/compliance/contact settings; magnetic or latch models only if supported |
| Robot control | No physical robot controller | Pinned robot model, actuator limits, IK/reachability, collision checks and compliant control |
| Evaluation | Evidence-linked visual conditions | Reset states, geometric/contact success, failure conditions and logged simulator trials |

The transform chain must be explicit. A calibrated observation could relate `world_from_body = world_from_camera × camera_from_native_mesh × native_mesh_from_body`. Every factor must use compatible metric units. Today the independent native mesh scales and estimated camera fits do not establish that chain; no missing factor should default to identity. A semantic `removable_from` relationship does not specify an insertion axis, and an `attached_to` proposal does not authorize a fixed joint.

## MuJoCo integration: smallest useful MVP

MuJoCo has official Python bindings and a macOS path, so a later small physics experiment need not wait for a large Isaac deployment. Pin its version in a separate environment when execution is approved. Use the official [Python bindings documentation](https://mujoco.readthedocs.io/en/stable/python.html).

Use a reviewed Franka Panda or FR3 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pin its commit, and inspect that model's individual license and actuator configuration. Reuse that robot's kinematics rather than pretending the stylized browser arm is a calibrated robot.

Preparation sequence:

1. Choose one component and source-backed task segment. Freeze its evidence-procedure hash, visual-asset hash and task goal; retain all gaps.
2. Resolve scale and body frames. Prepare OBJ/STL visual geometry as appropriate and record the conversion transform. Exporting our GLB is a geometry conversion, not physical calibration.
3. Prepare separate collision pieces, mass/inertia and contact assumptions. Keep the case opening genuinely open. Standard MuJoCo mesh geoms collide using a convex hull; one hull around a hollow case can seal the cavity. Multiple reviewed collision pieces are the straightforward initial approach. See [MuJoCo mesh semantics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh).
4. Author a minimal MJCF scene: fixed open receptacle, movable rigid component, the pinned robot, collision exclusions only where justified, and named grasp/preinsert/target sites. Values remain absent until supplied or explicitly estimated and tested; the current draft is deliberately not MJCF.
5. Validate passive settling and nonpenetration before moving the robot. Solve IK for clear waypoints, check limits and collisions, then test grasp/lift. Add compliant insertion only after those tests pass.
6. Step the physics while controlling actuators. Do not teleport the object or weld it to a finger and call that a successful physical grasp. Log slip, contact, pose error, releases, instability and failures. A temporary attachment can be a labelled controller prototype, separate from physical evaluation.
7. Compare the simulator task with the demonstrated visual states without claiming to have recovered the human forces or a unique physical explanation. Export simulator version, robot revision, assumptions, source evidence and results together.

Control/contact parameters require tuning and validation. MuJoCo's own modeling guide discusses gripper slip and contact/solver behavior; increasing friction is not a substitute for establishing the right geometry and forces. See [MuJoCo modeling and contact troubleshooting](https://mujoco.readthedocs.io/en/stable/modeling.html).

## Isaac Sim and Isaac Lab: later integration

For the larger workflow, prepare a USD asset layer with body transforms, rigid-body/collision schemas, physical materials and mass properties. Record stage distance, mass, time and axis conventions explicitly. OpenUSD distinguishes rendering content from physics: rigid bodies need pose and mass distribution, while contacts need collision shapes/materials. See the [OpenUSD physics schema and units](https://openusd.org/release/api/usd_physics_page_front.html).

Bring a known robot through the official URDF importer or a pinned robot USD, preserving joint/drive and collision settings. Importer options include convex hulls and convex decomposition; validate the resulting cavity and link collisions rather than accepting an automatic conversion as correct. URDF is useful for robot link/joint structure; our separate semantic evidence format should retain task meaning and uncertainty. See [Isaac Sim URDF importer](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html).

In Isaac Lab, represent the object and receptacle as rigid objects and the robot as an articulation, with explicit reset states and world coordinates. Its rigid-object guide demonstrates spawning and resetting pose/velocity; the parent's display transform is not automatically the simulation world frame. Use [Isaac Lab rigid objects](https://isaac-sim.github.io/IsaacLab/main/source/how-to/run_rigid_object.html).

Define observations, actions, reset events, success/failure termination and rewards only after the physical scene works. A manager-based environment is a useful proposed fit for reusable task terms; it is an implementation choice, not work already completed. Run a scripted controller first, then consider imitation or reinforcement learning. Choose and pin the Isaac Lab/Sim version and physics backend together; current documentation exposes multiple backends, so do not assume every engine shares settings or contact behavior. See [Isaac Lab task workflows](https://isaac-sim.github.io/IsaacLab/main/source/concepts/task_workflows.html).

Video-derived human steps are a task specification, **not robot action demonstrations**. Robot training needs executable end-effector/joint actions and simulator trajectories, plus task validation. A domain-randomization plan can vary independently documented geometry, initial states, materials and camera conditions later; it cannot repair an incorrect cavity or unidentified physical scale.

## Offline readiness auditor and non-runnable draft

`sim_readiness.py` adds no dependency and imports no simulator. Its pure functions are:

```python
from sim_readiness import audit_simulation_readiness, build_simulation_draft

report = audit_simulation_readiness(document)
draft = build_simulation_draft(document)
assert report['simulation_ready'] is False
assert draft['simulation_runnable'] is False
```

The report inventories available observations, asset references, conditional poses and task steps. It separately lists twelve blocking physical requirements with concrete next actions. It never assigns a misleading readiness percentage. The draft preserves source IDs, candidate semantics and conditional motion, and leaves scale, world transforms, collision meshes, mass/inertia, joints, grasps, receptacle frames, robot/controller and validation values `null`.

Example, from the project root:

```sh
.venv-sam/bin/python sim_readiness.py \
  data/flow-meta-video-jabra-20261030-v1/flow-procedure-scope-normalized-v1/interaction.json \
  --output output/robotics/jabra-readiness.json \
  --draft-output output/robotics/jabra-simulation-draft.json
```

This CLI reads the selected JSON only. It does not open any paths embedded in the document, inspect keys, start models, incur API usage or install/run physics engines. Input and output destinations must differ. Exit code 0 means the preparation report was written, **not** that the task is simulation-ready.

The auditor performs structural safety checks and rejects promoted physical flags, unknown references, malformed poses and non-finite values. It is not a replacement for `stage5.validate_interaction(document, root=..., verify_files=True)` at an ingestion boundary. Its verification fields explicitly leave full contract validation and asset hashes false. A consuming exporter must verify hashes, resolve paths safely under the allowed project root, and treat labels/descriptions as data rather than commands.

The existing MCP simulation manifest now includes `readiness` and `preparation_draft` after normal catalog validation. The catalog reports that full evidence-contract validation and confined declared-artifact hashes passed, binds both additions to the program hash, and keeps physical validation false. All previous handoff fields remain available. An agent can ask: “What prevents this earbud from being tested in MuJoCo?” and receive the missing calibrated scale, cavity collider and grasp-frame requirements rather than treating mesh access as robot-training readiness.

Offline downloads for the three real API samples are in `data/simulation-readiness-20261030/index.json`, with separate `jabra-readiness.json` / `jabra-simulation-draft.json`, corresponding camera and wallet files, and SHA256 digests in the index. They include no copied video, frames or model binaries. All fourteen catalog handoffs were checked through the real validator after integration. Focused auditor/catalog tests pass (21 tests); no simulator execution was performed.

## Completion gates

| Gate | Honest deliverable | Status in this increment |
| --- | --- | --- |
| Explain the task | Evidence procedure + generated visual assets + MCP retrieval | Available, with current quality gaps |
| Visual robot concept | Clearly labelled Three.js choreography | Separate presentation feature; not physical validation |
| Prepare simulation | Readiness report + null-filled draft + documented integration path | Implemented offline |
| First physics task | Calibrated rigid object, real cavity collider, robot grasp/release trials | Not run; required physical data remains missing |
| Contact insertion | Verified approach, compliant controller, measured seating criteria | Not run |
| Train and evaluate | Logged robot trajectories/policy evaluation across validated variations | Not run |

The near-term product promise can be **“turn a demonstration into an agent-readable object and task package, with an explicit path to simulation.”** “Automatically produce validated robot-training digital twins from arbitrary video” remains the goal to prove, not a completed capability.
