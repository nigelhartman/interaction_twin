# Newton CPU physics bench

The Robotics Lab has a fourth **Physics bench** tab at showcase/robotics-lab.html?mode=physics. It renders a genuine saved Newton 1.7.0 CPU experiment using an earbud, coin and adapter generated from the curated recordings. This adds dynamics without relabeling the existing authored robot rehearsals as physics.

Newton is developed by NVIDIA, Google DeepMind and Disney Research. Its official installation guide supports macOS CPU execution. Isaac Sim itself requires a supported Windows/Linux NVIDIA RTX machine, so this Mac runs Newton directly.

- [NVIDIA Newton](https://developer.nvidia.com/newton-physics)
- [Newton 1.7 installation](https://newton-physics.github.io/newton/1.7.0/guide/installation.html)
- [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)

## What actually ran

The isolated .venv-physics-cpu environment contains newton==1.7.0, warp-lang==1.17.0, and numpy==2.2.6. The runner explicitly selects the CPU. Newton XPBD computes 2,400 steps at 600 Hz, eight solver iterations, gravity, friction and rigid collision. Every tenth state is saved for 241 replay frames. World-space contact positions come from engine contact buffers.

Native appearance GLBs remain unchanged and are checked against curated hashes. A separate deterministic convex support approximation provides collision. Appearance and physics use identical normalization and coordinate conversion. The original SAM 3D coin review is used, rather than the historical procedure's old geometry.

The final replay-native-v3.json has 2,310 contact-bearing steps and at most 13 contacts. Across every saved state, the minimum transformed hull-vertex height is +0.00793 mm. Final maximum linear speed is 0.000162 m/s. These check the assumed experiment, not real-object accuracy. An independently extracted starter ran with isolated Python and a fresh Warp kernel cache; all 241 pose arrays matched exactly, maximum difference zero.

## What remains assumed

Release pose, dimensions, density, friction, restitution and rigid-body interpretation are authored experiment parameters. They are not recovered from video. Convex hulls fill concavities and cannot validate cavity insertion. No grasp, real robot, learned policy or electrical connection runs. The original twin's simulation-ready and physical-accuracy claims remain false.

Newton/NumPy emits floating-point status warnings during support-map preprocessing on this Mac. They are retained in execution output. Finite-state, full-hull plane-contact, ballistic-acceleration and settling checks pass. Warnings have not been suppressed or relabeled as failures.

## Reproduce

Preparation uses the existing CV environment to decode exact GLBs and build hulls. Run robotics_physics_bench_v1.py prepare with .venv-sam/bin/python and an output path. Run its run command using .venv-physics-cpu/bin/python, passing --input and --output. The publication script is scripts/build_newton_physics_bench_20261104_v1.py.

Public data is hash-pinned by showcase/physics-bench-source.js. The browser verifies the full replay and every GLB, then validates engine, claims, timing, transforms, contacts and hull arrays. It samples saved poses; it never invents contact or animates a final target placement.

The 17.2 MB starter ZIP contains runner, exact input, reference output, three original GLBs, package pins, instructions and a payload hash manifest. Only its exact path qualifies for download under the 32 MiB server guard.

The Record button captures actual 1920×1080 / 30 fps WebGL with half-speed physics and gentle camera orbit. The verified first MP4 is output/robotics-physics-20261104-v1/newton-contact-native-1080p-v1.mp4: 9.466667 seconds, 284 frames. It is dynamics footage, without slide transitions.

Validation includes five independent Python numerical/archive checks, three Node contract/security checks, existing Robotics Lab/workbench checks, and actual browser playback, hull toggle, recording/download, tab switching and pause.
