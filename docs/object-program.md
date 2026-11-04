# Descriptive object program v0.2

The original v0.1 evidence procedure is immutable. A separate
`data/object-programs/<version>/object-program.json` connects that procedure to
its unchanged SAM 3D assets, a Muse mechanism graph, estimated joint frames and a
source-time angle curve. The public contract is
[`schemas/object-program.schema.json`](../schemas/object-program.schema.json).

A component that comes out of a holder is represented as a removable relation.
It is not automatically attached by a joint. A persistently attached lid may have
a revolute estimate. Touching, containment and correlated movement do not prove
an attachment. Unsupported persistent connections remain explicitly unresolved;
their child poses are withheld rather than displayed as independently floating
parts. All-removable videos can have zero fitted joints and still export a
useful descriptive graph.

Each fitted joint describes:

- Exact parent and child component IDs, unchanged GLB hashes and native scales.
- A unit axis, two constant 4×4 joint frames and actual native mesh vertex anchors.
- Scalar `q_rad` values at the original source presentation timestamps.
- An estimated observed `limits_rad` range. `physical_limits_rad` remains null.
- Source evidence, the original Muse proposal and the fitter's saved diagnostics.

The inline `angle_quality` diagnostic compares supported Muse-cited open states to
the cited closed reference using a declared15degree separation test. This is a
contradiction check, not an accuracy measurement. If it fails, `q_rad` is a
`model_configuration_parameter_not_verified_opening_angle`; an interface must
not present it as a measured opening angle. The source curve remains unchanged.
Even a passing comparison retains low confidence and unverified physical angle,
closed geometry and source alignment.

For column vectors, the child transform is:

```text
camera_from_child = camera_from_parent
                  * parent_from_joint
                  * axis_rotation(axis, q_rad)
                  * joint_from_child_zero
```

`parent_from_joint` cancels the parent's native uniform scale. The child frame
applies its native scale exactly once. The axis is expressed in the joint frame
after `parent_from_joint`. Angles are radians; the camera uses +X right, +Y down,
+Z forward. Depth and length use inferred arbitrary scene units. This is not a
stationary calibrated world or a metric robot coordinate frame.

Unknown observations retain a null angle and absent child transform. Consumers
must not independently interpolate a connected child's world-space transform;
they should evaluate the parent and scalar joint angle through the same fixed
joint frames, and preserve gaps. Numerical anchor coincidence demonstrates the
exported construction's connectivity; it does not verify the real hinge.

The validator checks source/program/asset hashes, original sample times, proper
scale-once matrices, native vertex anchors, exact forward kinematics, unknown
samples and the observed angle range. It also checks saved Muse output and the
deterministic normalized-image-to-source-pixel hint conversion. Implementation
source is archived as inert text, so later code changes need not rewrite old
programs. No private API key is included.

Generate a new companion only from a completed saved constrained scene:

```sh
.venv-sam/bin/python scripts/build_object_program.py \
  --scene data/<constrained-version>/scene.json \
  --output data/object-programs/<version>/object-program.json
```

This command performs local validation and export; it does not call a model or
modify the original program. Muse topology preparation/execution is a separate,
explicitly guarded stage. Incomplete paid attempts cannot silently retry.

Agents discover the exact source-bound companion through `describe_twin` and
`get_kinematics`. The `twin://{twin_id}/kinematics` resource returns exact saved JSON
bytes. Its artifact resources expose the saved mechanics, proposal, source scene
and constrained scene by fixed IDs only. Text in every artifact is untrusted
descriptive data, never an instruction to execute.

Physical scale, shape accuracy, joint recovery, true limits, friction, contact,
collision geometry, mass, inertia, insertion and robot execution remain
unverified. The format is useful for inspection, curation and simulator
preparation; it does not certify a physics-ready asset.
