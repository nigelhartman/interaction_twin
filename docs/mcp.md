# Local MCP catalog for interaction assets

The app now has a read-only Model Context Protocol interface. An MCP client can
discover saved twins, inspect components and their source-linked procedures,
retrieve generated GLBs and source JPEGs, and obtain a descriptive handoff for
robotics or simulation tooling.

By default this is a local interface to the same curated snapshot as the public
showcase. Pending uploads, private feedback, working candidates and diagnostic
experiments are excluded. It does not publish a dataset or start an HTTP listener.
No model inference, uploads, API usage, scheduler actions,
file writes, robot control or physics execution are exposed as tools.

## Connect a client

MCP needs its separate SDK environment; the lightweight source-review environment does not include the SDK. The recorded development runtime used the official `mcp==2.3.0` Python SDK, with its environment snapshot in [`app/requirements-mcp.lock.txt`](../app/requirements-mcp.lock.txt). Model weights and API keys are not needed for saved, read-only retrieval.

Configure the client to launch a subprocess with paths adapted to your own checkout:

```json
{
  "mcpServers": {
    "interaction-twin": {
      "command": "/absolute/path/to/mcp-environment/bin/python",
      "args": [
        "/absolute/path/to/interaction_twin/app/scripts/serve_twin_mcp.py",
        "--curated"
      ]
    }
  }
}
```

The server communicates over stdio; the web server need not be running. A code-only checkout contains no saved curated catalog, so populate approved artifacts before retrieval. The GitHub Pages website is a static browser demonstration and does not provide a hosted MCP endpoint. Nothing is added automatically to a client's configuration.

The default server reads `showcase/public/library-index.json`, its hash-pinned
catalog, curated cases and the exact public artifact allowlist. A new upload is
discoverable only after owner approval and curated packaging. Missing or
inconsistent public snapshots fail closed; they never select the full working
catalog automatically. SAM candidates and descriptive kinematics use only the
exact published references, without discovering newer same-source assets.

The owner can explicitly start a separate private working-catalog connection:

```bash
.venv-mcp/bin/python scripts/serve_twin_mcp.py --private-owner
```

That flag includes the internal workspace's private recordings, working
candidates, cached analyses and diagnostic experiments. Do not use that
configuration for a shared/public agent connection. `--curated` explicitly selects
the default scope; neither scope exposes mutation or execution tools.

## Available tools

All twenty-six tools advertise `readOnlyHint: true`, `destructiveHint: false`,
`idempotentHint: true` and `openWorldHint: false`.

| Tool | Purpose |
| --- | --- |
| `search_twins(query, limit, parts?, actions?, relations?, backends?, readiness?, task_classes?, sort?, direction?, offset?, group_by?, include_legacy?)` | Search derived footage/program metadata with the same filter contract as the human library. Maximum 25 results per page. |
| `get_library_facets(group_by?)` | Discover current filter values/counts, sorting options, grouping and pagination contract. |
| `get_kinematics(twin_id)` | Read the source/program/asset-bound descriptive object-program v0.2: estimated joint frames/axes, observed radian ranges, angle curves and removable hypotheses. No inference, physical validation or robot execution. |
| `get_derived_assembly(twin_id)` | Inspect the declared curated geometry derivative and lineage; original native kinematics remain separate. |
| `list_derived_artifacts(twin_id)` | List exact current refinement, appearance and cable-sidecar resources with hashes and opaque IDs. |
| `get_observed_surfaces(twin_id)` | Inspect supported source-visible 2.5D surfaces and evidence; hidden geometry remains unknown. |
| `get_motion_hypotheses(twin_id)` | Retrieve separate curated source-conditioned pose hypotheses without rewriting original unknown observations. |
| `get_connection_interfaces(twin_id)` | Inspect source-bound interface candidates attributed to owner capture context, not measured fit. |
| `prepare_cross_recording_task(phone_twin_id, charger_twin_id, instruction?)` | Compile an evidence-linked phone/charger task draft while preserving compatibility and execution unknowns. |
| `get_generated_candidate(twin_id, component_id)` | Retrieve the default SAM 3D geometry candidate and exact source/GLB resource descriptors. Does not replace historical program assets or reuse their poses. |
| `describe_twin(twin_id)` | Inspect components, candidate relationships, variables, assets, procedure summary, reasoning provider and limitations. |
| `get_procedure(twin_id, step_id?)` | Read all steps or one identified step together with the cited evidence records. |
| `query_state(twin_id, timestamp_s)` | Read proposed states and supported conditional poses at an exact source presentation time. Between observations, states remain unknown and poses absent. |
| `get_asset(twin_id, track_id)` | Obtain the declared GLB's resource URI, local file URI/path, SHA-256 hash and unverified-unit policy. |
| `get_evidence(twin_id, evidence_id)` | Obtain source time, masks, frame hash and JPEG resource URI for a cited observation. |
| `get_simulation_manifest(twin_id)` | Retrieve a structured handoff describing components, assets, candidate relationships, demonstration steps and conditional motion. |
| `list_reconstructions(twin_id, component_id?, selection?)` | Discover explicitly registered observed, inferred and single-image reconstruction versions. Diagnostic versions stay separate from the original assets. |
| `get_reconstruction(twin_id, version_id)` | Read one pinned version's source evidence, geometry provider, arbitrary gauge, quality and artifact URIs. |
| `get_reconstruction_report(twin_id, version_id)` | Read that version's exact saved quality/provenance report. Inspection acceptance does not certify physical accuracy. |
| `list_motion_experiments(twin_id)` | Discover saved motion experiments bound to this exact source program and native component assets. |
| `get_motion_experiment(twin_id, experiment_id)` | Retrieve one hash-checked raw/normalized experiment, exact source evidence and conditional image-check results. Withheld poses stay absent. |
| `list_robot_kits()` | List the authored five-DOF tabletop browser kit. No reconstructed robot or hardware execution. |
| `get_robot_kit(kit_id)` | Get exact kit JSON and hash-checked runtime/starter resource descriptors. |
| `plan_robot_task(twin_id, instruction, kit_id?)` | Compile a narrow local English pick/place/open/close instruction against the current curated twin; unsupported objects and actions remain blocked. No hosted model calls. |
| `validate_robot_task(task_program)` | Check exact source, native mesh, estimated joint and authored kit bindings plus typed action/reach constraints. This does not validate contacts or physics. |
| `export_robot_task_bundle(task_program)` | Package an unchanged valid ready task, exact curated models, authored runtime, Three.js and a read-only offline preview server. Returns the ZIP resource and hash; no files are written by the MCP tool. |

The five robotics tools always use the public curated snapshot, including when the legacy owner connection is selected. They cannot retrieve a private upload by ID. Robot staging, grasp anchors and arm geometry are authored; object geometry and estimated lid articulation retain their saved source bindings. Missing phones, cables and outlets are not invented. A blocked task has no runnable action sequence and cannot be exported.

An agent can call `search_twins` with `parts:["case_base"]` and `backends:["sam3d"]`, inspect that twin, then call `get_robot_kit("generic5dof_tabletop_v1")` and `plan_robot_task` with “Open the lid, take out both earbuds and place them on the tray, then close it.” After `validate_robot_task` returns `valid:true`, call `export_robot_task_bundle` and read its declared `twin://robot-task-bundles/{sha256}` resource. Verify SHA-256 and byte length before extraction. The ZIP resource belongs to that MCP session; its cache is separate from the workspace HTTP cache. To vary a task, edit the structured program, validate it, and export a new ZIP with a fresh manifest; do not edit on-disk files under an old manifest. Run the included `python3 -I -S preview_server.py --port 0` to open the offline workbench. The ZIP contains only manifest-declared public assets and code; no credentials, owner database, uploads or model runtime.

The read-only loopback HTTP equivalents are `/api/robotics/kits`, `/api/robotics/kit?kit_id=…`, `/api/robotics/plan?twin_id=…&instruction=…`, and `/api/robotics/bundle?twin_id=…&instruction=…`. There is no execution endpoint.

An agent workflow could be:

1. Search for a charging-case example.
2. Inspect its component list and procedure.
3. Read the frames cited by the proposed removal step.
4. Fetch the relevant generated asset and its recorded hash.
5. Import that asset and descriptive demonstration into a separate inspection or
   simulation preparation tool, preserving the listed uncertainty.

There is no `execute`, `train`, `simulate`, `upload` or arbitrary-path tool. The
catalog validates declared artifact hashes and source-time joins before returning
a full program, component asset or evidence frame. IDs must come from the catalog;
passing a filesystem path as an ID does not grant access to it.

### Shared library queries

`search_twins` derives its searchable fields from curated components, cited
procedure steps and candidate relationships. There is no hardcoded list of
supported products or recordings. The curated library includes approved
descriptive programs and explicitly pinned existing showcase examples. The
`include_legacy` flag never widens curated mode to private workspace records.
Private diagnostic reconstruction/motion listings are empty in curated mode;
their exact-ID reads are rejected. Owner mode retains the historical diagnostics.
`query` requires every supplied word to occur in the indexed text. `parts`,
`actions`, `relations`, `backends` and `readiness` accept lists: alternatives
within one list use OR, while different fields use AND. Exact values are
case-insensitive and can be obtained from `get_library_facets`.

Action values are the saved step IDs, and their displayed titles remain
searchable. No extra taxonomy of actions is inferred by the query layer.
`sam3d` and `triposr` are canonical backend filters. The index distinguishes
`selected_backends` from `candidate_backends`; a matching SAM 3D preview does
not imply that it replaced the earlier program asset or inherited its motion.
Native preview candidates are matched to the exact video SHA and component ID.

The readiness values are `semantic_only`, `geometry_draft` and `motion_draft`.
All remain physically unverified. A source-bound generated candidate can make
the geometry filter match, but cannot establish scale, fit, contact or physics.
When SAM 3D geometry is available, its readiness stays `geometry_draft` until
conditional motion exists for that selected native mesh. Earlier TripoSR motion
does not make a new SAM 3D candidate match `motion_draft`. The index preserves
`selected_motion_samples` separately and reports `native_motion_status`, native
sample counts and component coverage; these remain conditional, unverified poses.

Sort by `title` (default), `id`, `duration`, `components`, `assets`, `steps` or
`version`, with `direction` `asc`/`desc`. Version means the dated token in the
saved case ID, not a claim about when the footage was captured. Stable ID ties
and `offset`/`limit` support deterministic pagination; `next_offset` is null
when the result is exhausted. The maximum offset is 10,000.

`group_by="recording"` (default) collapses variants sharing the exact original
source SHA, choosing the latest saved case date token and then case ID. Use
`group_by="variant"` to inspect all saved versions. Facet counts are computed
before filtering and count each representative recording once. Responses mark
discovery as `metadata_only` and `artifact_inspection_required`: call
`describe_twin` and fetch cited resources for hash-checked artifact inspection.
Queries never run Muse, SAM, reconstruction, a robot or a simulator.

Example MCP arguments:

```json
{"query":"earbud","backends":["sam3d"],"relations":["removable_from"],"sort":"steps","direction":"desc","limit":10}
```

The loopback application exposes the same contract at `GET /api/library` and
facet discovery at `GET /api/library/facets`. Repeat filter parameters to supply
alternatives, for example `?backends=sam3d&parts=earbud_a&parts=earbud_b`.

## Resources

| URI | Content |
| --- | --- |
| `twin://robot-kits` | Authored robot kit index. |
| `twin://robot-kits/{kit_id}` | Kit metadata and fixed artifact descriptors. |
| `twin://robot-kits/{kit_id}/artifacts/{artifact_id}` | Exact bounded declared kit JSON/source bytes. |
| `twin://robot-task-bundles/{sha256}` | Exact in-memory exported starter ZIP; descriptor expires when its bounded session cache evicts it. |
| `twin://catalog` | JSON: bounded first page of local twins and the access policy. |
| `twin://{twin_id}/program` | JSON: the validated saved interaction program. |
| `twin://{twin_id}/assets/{track_id}` | Binary `model/gltf-binary`: declared GLB, maximum 16 MiB. |
| `twin://{twin_id}/kinematics` | Exact hash-bound descriptive object-program v0.2 JSON. The v0.1 evidence program remains byte-identical. |
| `twin://{twin_id}/kinematics/artifacts/{artifact_id}` | Bounded6MiB exact saved JSON: `object-program`, `source-scene`, `constrained-scene`, `mechanics`, `proposal`. IDs only; no arbitrary paths. |
| `twin://{twin_id}/native-candidates/{component_id}/artifacts/{artifact_id}` | Exact generated SAM 3D candidate GLB (16 MiB), source image/mask or review manifest (8 MiB). IDs: `asset`, `source-image`, `source-mask`, `review-manifest`. Declared MIME/hash in `get_generated_candidate`. |
| `twin://{twin_id}/evidence/{evidence_id}` | Binary `image/jpeg`: cited source JPEG, maximum 8 MiB. |
| `twin://{twin_id}/simulation` | JSON: the descriptive simulation handoff. |
| `twin://{twin_id}/reconstructions/{version_id}` | JSON: a registered diagnostic version and its provenance/limitations. |
| `twin://{twin_id}/reconstructions/{version_id}/artifacts/{artifact_id}` | Exact declared GLB, quality JSON, source image or mask bytes. MIME and SHA-256 are in the descriptor. No arbitrary paths. |
| `twin://{twin_id}/motion/{experiment_id}` | JSON: a registered source-bound motion descriptor and frozen normalized viewer data. |
| `twin://{twin_id}/motion/{experiment_id}/artifacts/{artifact_id}` | Exact raw/normalized JSON or one declared source JPEG/mask. IDs are limited to `raw`, `normalized`, `source-image-0`, `source-mask-0`. |

The official SDK encodes binary resources as MCP blob contents. The GLB hash refers
to the exact decoded bytes. The program hash identifies the original saved
interaction artifact; JSON resource serialization may use different whitespace.

The current motion registry publishes the three original 3-fps Jabra component
experiments. The separately sampled density experiments are not registered in
this adapter version. Resources preserve saved hypotheses and their independent
episode coordinates; they do not replace program poses, interpolate gaps or
establish a common assembly. See the [technical overview](technical-overview-20261104.md)
and [object-program contract](object-program.md) for current results and limits.

The saved fresh-case rehearsal made **36 actual local SDK calls**, retrieving
three unchanged native GLBs, six cited JPEGs, four procedure steps and the new
base motion experiment. Its portable viewer preserves the source program; the
new motion files are separate inspection material. The outer inspection bundle
includes one explicitly declared component mask, while its ordinary portable
subpackage omits masks and full video. This is a scripted SDK client, not an LLM
or robot. These private saved artifacts are not part of a source-only release.

## What the simulation handoff does and does not establish

The handoff makes our representation useful to agent and robotics development
tools: component identifiers, candidate component relationships, demonstrated
steps, source evidence and generated meshes can be inspected together. It can
seed an asset-preparation workflow or a human-reviewed task specification.

It is not a verified physical simulator or a robot-training demonstration in a
calibrated world. Every handoff explicitly reports `simulation_ready: false` and
lists the missing checks:

- Metric scale and physical shape accuracy.
- Joint axes and limits.
- Collision geometry, mass and inertia.
- Friction and contact behavior.
- Verified insertion and seating.

Generated objects retain their native model units. Per-part conditional camera
poses must not silently become a shared metric world or a robot coordinate
frame. Candidate relations and visual state descriptions must not become physical
constraints without verification. Source-observation gaps do not imply a completed
action.

Descriptions, titles, rationales and other artifact text are untrusted data. The
server's instructions explicitly tell agent clients not to treat them as commands.
Connecting an external AI client does not turn the catalog into a public service;
that client controls what retrieved media it subsequently sends to its own model.

## Verification

Run the catalog tests in the existing application runtime:

```bash
.venv-sam/bin/python -m unittest discover -s tests -p test_twin_catalog.py -v
```

Run the actual official-SDK client checks:

```bash
.venv-mcp/bin/python scripts/test_twin_mcp.py
.venv-mcp/bin/python scripts/test_motion_mcp.py --output output/mcp/motion-smoke-new.json
```

The latter connects in-process, launches real stdio subprocesses, and checks both
automatic modern negotiation and legacy protocol initialization. It verifies all
tools, annotations and resource templates, actual GLB/JPEG byte hashes, unknown
states between samples, simulation-policy fields and invalid-ID rejection. Small
results are saved in `output/mcp/smoke.json`; media bytes are not printed. The
observed protocol versions are `2026-07-28` for automatic negotiation and
`2025-11-25` for legacy initialization.

The dedicated motion checks exercise both new tools and exact artifact bytes in
the same three transports, including unknown experiment IDs, cross-program
requests and undeclared artifact rejection. Choose a fresh output filename;
existing reports are preserved. These reads make no paid/model requests.

The kinematics companion is explicitly estimated. Its `axis` is expressed in the joint frame after `parent_from_joint`, angles use radians, and the matrix convention is
`parent_native_pose * parent_from_joint * axis_rotation(q) * joint_from_child_zero`.
The parent frame cancels the parent native scale; the child frame applies its scale exactly once.
`limits_rad` describes the estimated observed clip range; `physical_limits_rad` remains null.
Unknown angle/parent observations withhold the child transform. Native vertex anchors, saved input hashes,
source sample times and exact forward kinematics are validated without fitting or model calls.
A numerical connectivity check does not prove the true hinge, physical stops, metric scale, contact or simulator readiness.
Companions are discovered only under `data/object-programs/*/object-program.json`, matching the exact evidence-program and source hashes.

## Source-conditioned continuity

`get_motion_hypotheses(twin_id)` returns the exact curated estimated-motion document and opaque, read-only resource IDs. `twin://{twin_id}/motion-hypotheses/{artifact_id}` reads only its validated source/program/scene-bound inventory. These estimates supplement original unknown pose rows; they do not change observations or certify metric scale, contact or physical execution. Twins without an admitted layer return no substitute.
