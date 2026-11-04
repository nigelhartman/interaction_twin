# Automatic connected mechanisms from native SAM parts

`sam3d_mechanisms.build(scene_path, proposals_path, output, root=ROOT)` creates a
separate immutable constrained scene. It does not change the original program,
native GLBs, source masks or accepted motion. Muse supplies source-cited semantic
topology and uncertain image-space seam hints; the local fitter supplies the
numeric joint model. No additional inference is required by this stage.

The current geometric fitter supports revolute joints. Each child follows
`parent * parent_from_joint * Rot(axis,q) * joint_from_child_zero`. The two anchors
are actual unchanged mesh vertices, and a single native axis and scale pair are
retained for the whole recording. Parent scale is canceled once and child scale
is applied once. Intermediate frames interpolate the parent pose and scalar
configuration before evaluating this equation. Interpolating independent child
world transforms would break attachment and is prohibited.

The fitter aligns the two native seam directions and tests both unoriented
correspondences. A weak native orientation prior resolves visually similar
opposite faces. It fits masks across the recording instead of trusting a single
partially visible closed-frame pose. The parent and child mask union can refine
the shared parent pose without treating a visible front strip as the complete
parent. An additional child-mask term prevents a high union overlap from hiding
swapped or incorrectly placed parts.

For two meshes with independently detected cavity faces, an optional geometric
prior opposes their cavity normals and aligns their closed rim footprints. The
opening-side hypothesis requires an asymmetric sparse central extreme face and
a resolved opposite solid face in each native mesh. Symmetric or flat shapes do
not receive this prior. Native rim vertices are paired under the estimated
mating transform; source-cited seam projections select the edge. This remains a
geometry prior, not a measurement of manufactured contact or hidden surfaces.

Source-cited closed states define an estimated configuration zero after fitting.
Observed limits describe this clip only; physical stops remain null. The
`angle_quality` object explicitly checks whether supported cited open states are
separated from the closed reference by a declared 15-degree diagnostic threshold.
A failed check labels the scalar a model configuration parameter, not a verified
opening angle. Passing this check does not establish a calibrated physical angle.

Unsupported persistent joint types and failed fits retain their declared topology
in `unresolved_children`; all serialized poses of the attached child are null.
The viewer hides the connected assembly when parent or joint evidence is missing.
There is no independent free-floating child fallback. Removable relationships
remain semantic hypotheses and do not acquire a rigid connection automatically.

Every build pins the original scene, actual proposal, source/model/mask inputs
and implementation files. A `build.json` receipt verifies outputs and permits
exact cache reuse without fitting. A changed input, code file or output requires
a fresh versioned destination. Neither successful cache reuse nor source-mask
fitting is described as held-out accuracy validation.

The actual Jabra V11 experiment is under
`data/sam3d-mechanisms-jabra-20261102-v11`. Its local diagnostic receipt and
source/projection montage are in `output/sam3d-mechanisms-20261102`. It uses the
genuine first Muse proposal and four original native SAM meshes, with no new
API/GPU calls, manual joint coordinates, edited mesh vertices or historical
TripoSR poses. Physical accuracy, metric contact and simulator readiness remain
unverified; public publication remains a separate reviewed step.
