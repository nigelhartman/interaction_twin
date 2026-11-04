# Robotics workbench: an object library an agent can build with

The browser workbench turns a selected library twin into a **robot task draft**. It combines the twin's exact generated components and estimated relationships with an original tabletop arm, declared staging positions and a sequence of typed actions. This is a useful place to inspect a task before preparing a physical simulator experiment.

The captured object program and the robot task program are separate documents. Changing a gripper target must never rewrite the video's observations, turn an estimated joint into a measured joint, or claim that a robot performed the demonstrated human action.

## A small complete loop

1. An agent searches the curated library for a case with a hinged lid and removable earbuds.
2. It reads the selected twin, the available robot kit and their declared resources.
3. It requests a task such as “place the case on the table, open it, then take out both earbuds.”
4. The local compiler produces a bounded task draft. Objects retain their source, component and asset hashes. Arm poses, grips, targets and timing are explicitly authored.
5. The browser solves the arm's kinematics, displays the current action and replays the attachments and object joint motion. Pause, reset and seek inspect the same deterministic task.
6. An agent can download the starter, task and declared assets to build a variation. Unsupported objects or operations return missing requirements instead of fabricated library assets.

The web prompt compiler is a small local parser for the supported vocabulary. It does not call a hosted model. A connected reasoning agent may use the typed task interface; its reasoning is not evidence of physical feasibility.

The public lab has two main views: **Recorded interaction** and **Robot demo**. Choose an object, read the short task and press play. The prompt, JSON editor, joint telemetry, checkpoints, recording and downloads stay in the collapsed **Advanced** drawer. The simple path builder and separate Newton physics experiment remain available there. Default tasks follow the selected source: laptop open/close, camera battery pick/place, phone connector pick/place or wallet coin transfer.

## The design comes from robotics workflows

| Primary reference | Useful idea | How we apply it |
| --- | --- | --- |
| [MoveIt Task Constructor: pick and place](https://moveit.github.io/moveit_task_constructor/tutorials/pick-and-place.html) | Compose approach, grasp, attachment, lift, placement and retreat stages. | Small inspectable actions rather than one opaque animation. We do not implement MoveIt's planning scene or collision planner. |
| [Isaac Lab: task workflows](https://isaac-sim.github.io/IsaacLab/main/source/concepts/task_workflows.html) | Separate scene, actions, observations, reset and task conditions. | A robot kit, task program and deterministic playback state can be inspected separately. This is not an Isaac Lab environment or a learned policy. |
| [robosuite environments](https://robosuite.ai/docs/modules/environments.html) | An environment binds a robot, objects and a manipulation task through a consistent interface. | Reusable kit plus selected source-bound assets, with a stable action vocabulary. No robosuite simulator has been run for this workbench. |
| [Three.js inverse kinematics](https://threejs.org/docs/pages/CCDIKSolver.html) | IK relates an end-effector target to an articulated chain with joint limits. | The original browser arm uses its own declared kinematics and reports reach errors; Three.js renders it. It is not a manufacturer's robot model. |
| [Rapier joints](https://rapier.rs/docs/user_guides/javascript/joints/) | A revolute joint allows rotation about an axis while restricting relative translation. | A lid follows one parent/joint/child chain rather than independent trajectories. This geometric constraint is not a dynamics solver. |
| [MuJoCo modeling](https://mujoco.readthedocs.io/en/stable/modeling.html) | Physical simulation needs collision geometry, mass/inertia, contact and actuator behavior. | These remain explicit requirements for the later simulator handoff. Visual placement alone does not satisfy them. |

## What the earbud example establishes

The useful demonstration is not a detached lid floating beside a case. The base and lid retain a connected transform chain while the arm visits authored targets and the earbuds change attachment state. Component identities and mesh provenance come from the library. The target layout and grasp frames belong to the task draft, with their origin visible.

An attachment in the browser is a kinematic state transition. It is not a frictional grasp. A completed browser plan means its supported software actions were replayed, not that the object can be grasped, seated or charged in reality.

Current Jabra staging uses one parent normalization and exact supported observations for A77 and B24. A retains its saved relative rotation. Two source-bound Muse reviews identified B's native outer cap as +Z and selected a minimal additional visual rotation that matches A's supported cap tilt while preserving B's source yaw. An exact source-mask contour supplies a 35.53% visible protrusion cue. The raw source matrices, recorded hinge and native GLBs stay unchanged. The inline `staging_reference` carries the source/native/mask bindings, nonroutable review fingerprints, rotation delta and authored rest corrections. All-vertex envelope and tray-support checks are display checks; they do not establish cavity fit, lid clearance or physical contact.

## Phone, USB-C cable and power adapter

The curated library now includes captured phone/connector and charger/cable/socket examples. The browser robot task for the phone supports connector pick/place only. It does not perform or verify charging, electrical operation or insertion. The charger example retains its captured connected replay; a robot electrical-connection task is not prepared. Flexible cable uncertainty remains explicit.

A connector task additionally needs port and plug coordinate frames, approach direction, orientation, insertion distance and collision clearance. Deformable cables require a separate representation; a rigid cable mesh cannot certify routing. A labeled authored fixture may demonstrate a UI concept, but must never appear as a video-generated library twin or a validated electrical connection.

## The next physical milestone

Use one rigid earbud and a fixed open case fixture. Establish scale, review an open cavity collider, provide mass/inertia and a gripper-specific grasp frame, then run a scripted lift and release in a pinned simulator. Log contacts, penetration, slip, target error and settling. Only then add articulated-lid manipulation or constrained insertion.

The longer integration plan and readiness audit remain in [robotics-simulation-plan.md](robotics-simulation-plan.md). The browser workbench reduces the work needed to specify and inspect a task; it does not remove the physical validation step.
