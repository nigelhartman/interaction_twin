# Private semantic asset companions

An additive companion describes useful features and relations without changing
the evidence procedure, native meshes, mechanism program or public library.
It is for owner review and later agent discovery. This module calls no model,
GPU, SSH service, browser, robot or external API.

The implementation is `semantic_asset_companion.py`; the local CLI is
`scripts/build_semantic_companion.py`. Existing pipeline and curation code is
unchanged. These files do **not** make a new upload publicly available.

## What it can describe

| Feature | Representation | Deliberately unresolved |
| --- | --- | --- |
| USB-C plug and device port | Separate logical endpoint/receptacle hypotheses; connection candidates | Local port frame, insertion trajectory, clearance and certified/electrical compatibility |
| Laptop power button | Press affordance and proposed control relation to the laptop | Exact button region, actuation force and actual power-state response |
| Screw cap | Thread engagement, rotation and thread disengagement as different stages | Thread pitch, release angle, friction and contact |
| Tethered cap | Retained flexible-tether relation after thread disengagement | Tether deformation and contact; it is neither a recovered fixed hinge nor an automatically free body |
| Bottle pouring into a cup | Directional source/receiver semantic relation | Liquid geometry, volume, fluid dynamics and verified transfer |
| Flexible cable | Nonrigidity hypothesis kept separate from its static draft mesh | A deforming centerline, material model or reconstructed cable dynamics |
| Charger reused across recordings | Two distinct source-bound identities connected by a possible-reuse candidate | A canonical shared asset or verified physical identity |

Unscrewing is not equivalent to complete detachment. A cap may disengage from
its threads and remain retained by a flexible tether. When a source-bound
proposal reports that tether, the companion retains it. A conflicting owner
description of a freely removed cap is recorded as a conflict for review.
No free-body trajectory or invented connector is generated to resolve it.

## Provenance

Every companion pins the original program and source video by ordinary local
path, SHA-256 and length. The existing evidence validator checks the source
manifest, presentation timestamps, frames, masks, assets and references.
Companions can be generated while geometry is deferred.

There are three distinct bases for a semantic claim:

1. **Program label classifier.** A literal label such as “USB-C port” can
   produce a typed hypothesis. The exact JSON pointer and literal label are
   preserved. A cable label does not prove its shape or stiffness. A “charging
   port” label does not automatically become USB-C.
2. **Program relation.** Typed proposed relationships can support a candidate
   relation. Unknown relationships cannot establish a positive connection.
   Tether inference requires an affirmative attached/retained tether proposal;
   a limitation merely mentioning a tether is insufficient. A separately
   referenced logical context can supply feature typing for a generic model
   label such as “blue cap”; the original model relation, evidence and status
   remain the relation's provenance, and the typing context remains unverified.
3. **Explicit context.** An immutable, separately referenced context contains
   owner information or an operator's source-review annotation. Both stay
   unverified annotations. Source-review annotations require real evidence IDs.
   Encoding an owner's description does not mean the owner confirmed the
   operator's particular component mapping.

A tiny port or button may refer to its host track and exist as a logical feature
without its own SAM mask. Its geometry explicitly says that the available mesh
belongs to a host/component track, not a verified separate feature. Metric
poses, port frames, cable curves and fluid geometry remain null.

## Create a companion

```sh
.venv-sam/bin/python scripts/build_semantic_companion.py build \
  --program data/<exact-run>/<procedure>/interaction.json \
  --output-dir output/semantic-review/<run>
```

Without context this uses only the exact program's labels and structured
relations. It does not use video filenames to guess the task.

Optional context is constructed against an **existing exact program**. Replace
these illustrative track IDs with actual compiled IDs; unknown IDs are rejected.

```python
from semantic_asset_companion import make_owner_context, write_context

context = make_owner_context(
    "data/<run>/<procedure>/interaction.json",
    note="Owner described plugging this charger into the phone.",
    entities=[
        {"id": "plug_tip", "kind": "connector_endpoint",
         "track_ids": ["cable"], "feature_label": "USB-C cable endpoint",
         "attributes": {"standard": "usb_c", "role": "plug"},
         "evidence_ids": []},
        {"id": "phone_port", "kind": "receptacle",
         "track_ids": ["phone"], "feature_label": "Phone USB-C port",
         "attributes": {"standard": "usb_c", "role": "port"},
         "evidence_ids": []},
        {"id": "desk_charger", "kind": "charger",
         "track_ids": ["charger"], "feature_label": "Owner's desk charger",
         "attributes": {}, "evidence_ids": [],
         "owner_reuse_key": "owner_desk_charger_candidate"}
    ],
    relations=[
        {"id": "plug_into_phone", "kind": "connection_candidate",
         "subject_id": "plug_tip", "object_id": "phone_port",
         "evidence_ids": []}
    ])
reference = write_context(context, "output/semantic-review/<run>")
```

Use `origin="source_review_annotation"` for an explicitly identified operator
review, with nonempty exact program evidence IDs for every entity and relation.
Do not label an operator's visual interpretation as an owner statement or as
Meta model output.

Optional `context_source_ref` and `context_source_pointer` pin the literal owner
note in an immutable input JSON document. For the capture-batch manifest the
selected recording's source hash must also match the program's normalized or
original source hash. The note and operator's component mapping remain distinct.

Then pass the returned context path with `--owner-context`. Attribute values
are narrow enums. Unknown fields such as `port_pose`, a thread pitch, certified
compatibility or a manually supplied cable trajectory are rejected.

## Cross-recording candidate catalog

```sh
.venv-sam/bin/python scripts/build_semantic_companion.py catalog \
  --companion output/semantic-review/phone/semantic-<hash>.json \
  --companion output/semantic-review/charger/semantic-<hash>.json \
  --output-dir output/semantic-review/catalog
```

Only explicitly supplied companions are read: there is no global scan, public
exposure or automatic promotion. The same owner-suggested charger reuse key in
different source recordings creates a `possible_reuse` edge. Both identities
remain intact, `canonical_asset_id` remains null, and deduplication remains
false. Matching USB-C semantic hypotheses produce connection candidates, not
certified compatibility or mating poses.

If a charger is mentioned as reused but is not visible as a track in the phone
recording, an owner-only charger entity may use
`presence: "owner_reported_offscreen"`, `track_ids: []`, `evidence_ids: []` and
an `owner_reuse_key`. It is explicitly reference-only, with no asset or geometry.
This avoids inventing a second charger mesh or treating a phone mask as the
charger. It still requires identity review before any shared asset is adopted.

Use `load_catalog` and `query_catalog` to sort/filter private entities by kind,
connector standard, geometry availability and review status. Candidate lists
are bounded and expose totals and truncation flags. No MCP/public API is added
by this module; later integration must preserve owner approval boundaries.

## Owner feedback

A feedback file has format `interaction-twin-semantic-owner-feedback`, version
1, and pins `companion_ref`, `program_ref` and `source_sha256`. Its `reviews`
list contains `{id, disposition, comment}` with dispositions `owner_supports`,
`owner_rejects` or `owner_uncertain`. It includes a bounded `note`,
`untrusted_owner_belief: true` and `training_performed: false`.

```sh
.venv-sam/bin/python scripts/build_semantic_companion.py feedback \
  --companion output/semantic-review/<run>/semantic-<hash>.json \
  --feedback output/semantic-review/<run>/owner-feedback.json \
  --output-dir output/semantic-review/<run>
```

The result is a new immutable companion referencing the previous revision.
Original files remain unchanged. Feedback is source- and revision-bound;
changed bytes, foreign IDs, wrong source revisions and malformed claims fail
closed. Owner support does not enable physical accuracy, robot execution,
automatic identity equivalence, model training or public curation.

The deterministic derivation rules belong to version 1. Future rule changes
must preserve the versioned reader or use a new descriptor version rather than
rewriting existing source-bound companions.

## Fixed four-capture follow-on

Once the named pipelines have completed, this bounded command prepares their
private contexts, companions, relation catalog and optional visible cable traces:

```sh
.venv-sam/bin/python scripts/build_capture_semantics_20261103.py
```

It reads only the `run.json` states of charger/phone/laptop version 1 and the
**corrected bottle-consistency version 2**. It accepts `complete` and
`complete_with_gaps`; any other status fails before output writes. Each program
must be the exact interaction artifact of that fixed lineage and bind the
original source recording in `output/new-captures-20261103/batch-manifest.json`.
It never searches for a “latest” run. The earlier split-body bottle branch is
excluded.

`--capture charger --capture phone` selects a completed subset. The default
private destination is `output/new-captures-20261103/semantic-review-v1`.
`--skip-cable-traces` disables the optional local mask calculation.

Literal owner notes retain the exact batch JSON pointer and source hash. Host
features are proposed against actual saved track IDs; missing or ambiguous
hosts produce warnings rather than invented component IDs, poses or geometry.
The phone's connector-only track never becomes a whole-cable reconstruction.
An offscreen charger is reference-only. Whole-cable tracing requires an explicit
cable/cord label without a connector/plug-only label and remains 2D mask evidence.

The script preserves corrected model tether relations; the owner's word
“detaches” does not establish free-body cap motion. Water/receiving-vessel
semantics remain separate from liquid simulation. No database, existing model,
program, owner approval, public catalog or pipeline state is changed.
