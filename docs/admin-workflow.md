# Private review and curated publication

Authenticated uploads create durable private candidates. A candidate is not a public library entry. The owner explicitly consents to processing, the trusted fixed pipeline adapter runs, four automatic reviewer roles check the result, and Nigel approves an exact immutable revision before the public snapshot advances.

## State and recovery

`AdminWorkflow(root)` stores SQLite queue state and immutable revision/decision/repair-plan records beneath `.private/admin-workflow`. `FeedbackMemory(root)` stores immutable annotations/revocations beneath `.private/agent-memory`. These namespaces must be inaccessible to anonymous HTTP and public MCP tools. The HTTP layer owns owner authentication, same-origin checks and CSRF; passing `actor='nigel'` is only permitted after that boundary.

A consented upload moves through `queued → processing → awaiting_review` or `blocked_review`. Missing runtime/API/GPU prerequisites produce `needs_attention` with preserved artifacts. No automatic retry follows a failed or expired worker lease. `resume_initial(candidate_id, processing_consent=True)` explicitly resumes the **same** pipeline run, source and frozen reasoning context, with at most two recoveries. The trusted executor must use the supplied `run_id` for its existing Runner cache, and must stop if a paid attempt has uncertain accounting or incomplete preserved outputs. A new queue job ID must never become a fresh paid cache namespace.

Completed processing produces an immutable source/program/asset-bound revision. The original evidence program, generated meshes, failed experiments and previous revisions are not overwritten. The owner must supply both the current revision ID and its exact fingerprint to approve it. Approval rehashes the artifacts and reruns structural validation without model inference.

Public selection is based on the most recent intact **explicit approval**, not the candidate's current private state. A newer repair can be queued, processing, awaiting review or rejected while the last approved revision remains public. `reject` records private repair feedback; it does **not** withdraw an existing publication. Explicit unpublication is a separate future operation. If an approved artifact changes, that revision is withheld on integrity validation. The packager may fall back to a prior still-intact approval. Original showcase examples are pinned separately as existing examples and must never be labelled as newly human-approved.

## Four reviewer roles

The default roles perform real deterministic checks:

| Role | Checks | Limits |
|---|---|---|
| Evidence | Source identity, program/input hashes, exact source times and source references | A valid reference is not a correct visual interpretation |
| Geometry | Selected asset hashes, GLB2 structure and bounded embedded geometry decoding | Shape, texture, physical size and mating surfaces still require inspection |
| Mechanics | Descriptive companion, native anchors, scale-once FK, known/unknown samples and source/asset binding | Physical contact, insertion, calibrated joint axes/angles and simulation acceptance remain unverified |
| Semantics | Saved IDs, relational cycles, citations and temporal bounds | Model-proposed identity, actions and relations remain fallible |

These are four review roles, **not four independent AI models**. Structural failures block approval. Warnings remain visible to the owner.

Optional `admin_muse_reviews.build_reviewer(memory_context, root=..., allow_paid_api=True)` supplies four separate fixed Muse Spark prompts through `MechanismMetaClient`. Each request is text-only, uses the versioned V2 protocol with an 8,000-token total output cap, no image inputs, no tools, no background job and no implicit retry. The existing task ceiling of at most $5, historical increment ceiling and shared $100 ledger all apply. Construction/import does not call Meta. Production integration must be explicitly enabled by the authenticated owner's processing consent.

The original V1 protocol and source snapshot are preserved. V1 consumed its 1,000-token cap almost entirely in reasoning and produced no structured result in the three accounted attempts. Brand-new uploads use V2. An existing V1 attempt cannot silently migrate: only an explicit trusted `allow_protocol_upgrade=True` may create a distinct V2 request after verifying an accounted 1,000-output/998-reasoning-token receipt with no result/report, matching fixed payload/source/context, and the exact archived V1 implementation. The old uncertain evidence-role request remains blocked and is never resent. The new V2 plan binds the old proof references; all old files remain unchanged.

The worker accepts `worker_once(executor, hosted_reviewer_factory=factory)`, where `factory(job)` returns the reviewed callback bound to `job['memory_context']`. Transport runs outside the SQLite writer transaction. Reports preserve exact artifact bindings, prompt/schema/plan, result and accounted provider receipt. The same artifact/context/role namespace is reused across queue jobs and code changes; incomplete attempts stop for inspection rather than resending. A blocking hosted finding blocks ordinary approval. Nigel may explicitly accept only these hosted findings after inspection by supplying a nonempty override reason of at most2,000characters for that exact revision fingerprint. The immutable decision keeps the reason private and exposes only a public human_review_override flag. The original model findings and physical/simulation uncertainty remain unchanged. Structural role failures, hash/path/source mismatches, missing artifacts and changed approval records can never be overridden. An optional transport, budget, factory or already-attempted error does not discard completed pipeline output: the candidate persists with the four structural roles, explicit hosted completed/expected counts, sanitized per-role errors and missing-review warnings. Other roles continue only through their existing spending guard. No failed role is automatically resent. Completed reports are revalidated, not re-inferred, during approval; an owner can inspect and explicitly approve a structurally valid candidate whose optional reviews are incomplete. Hosted reviews still do not watch the footage or verify physical accuracy.

## Feedback and bounded repair

`reject(candidate_id, revision_id, feedback, scopes=..., repair_targets=...)` writes an immutable annotation and fixed-stage repair plan. Default scope is only the same recording. Broader transfer requires explicit `object_family` or `global_method` scope. Available object-family IDs are the same typed library task classes. Exact-recording feedback ranks ahead of family and global preferences; revocation preserves the original note but excludes it from future retrieval.

Feedback retrieval is contextual memory, **not model training**. Human statements remain untrusted preferences or criticism, not observations or physical truth. The context is digest-bound and tells the reasoning adapter to evaluate feedback against current evidence. It cannot authorize spending, execute code, change provider settings or invent hidden geometry.

`request_repair(..., processing_consent=True)` queues at most two repair attempts per candidate. Allowed targets are `reasoning`, `tracking`, `reconstruction` and `mechanics`. The trusted adapter must implement a fixed supported stage or return a clear pending prerequisite. Neither free-form feedback nor model output is executed. Completed paid/model stages are reused only when their source/implementation contracts still match; stage-specific new work produces separate outputs and a new reviewable revision.

## Shared discovery contract

`library_query.FIELDS` includes `parts`, `actions`, `relations`, `backends`, `readiness` and `task_classes`. Each typed filter accepts an array: OR within a field, AND between different fields. Matching is case-insensitive lowercase, with shared sort, source grouping and pagination. Human JavaScript queries, HTTP requests and MCP `search_twins(task_classes=...)` use the same semantics.

Task classes derive only from typed saved topology/relations, never a product name or natural-language claim:

- `hinged_assemblies`: proposed/fitted revolute connection.
- `sliding_mechanisms`: proposed/fitted prismatic connection.
- `removable_components`: typed removable relation.
- `connectors_and_insertion`: explicit typed insertion/removal-attempt annotation; moving toward a hole is insufficient.
- `flexible_or_uncertain`: explicit unknown connection or typed deformable material behavior.

A collection tag describes a proposed task family. It never implies correct geometry, successful insertion, physical validation or simulator readiness. Unknown metadata remains unclassified.

## Core adapter result

The trusted executor receives the job's fixed `run_id`, input/source reference, consent, frozen memory context, and optional immutable base revision/repair plan. It returns:

```json
{
  "status": "completed | complete_with_gaps | pending | failed",
  "run_id": "the original run ID",
  "interaction_ref": {"path":"data/...", "sha256":"...", "bytes":123},
  "object_program_ref": null,
  "scene_ref": null,
  "artifacts": [],
  "repair_execution": [],
  "diagnostic": "bounded local status",
  "new_api_calls": 0
}
```

`artifacts` contains only immutable output references, not mutable job status, live logs, indexes or current catalog aliases. Source paths must be confined ordinary `data/` files; credentials, private annotations and arbitrary local paths cannot enter revision evidence or publication.

## Trying the next recordings

The current fast profile accepts ordinary clips up to40seconds, without calibration targets. A cabinet demonstration should discover the cabinet and door separately, propose the persistent hinge, and preserve unresolved axes/angles as unknown. A phone-charging demonstration should distinguish the phone, connector tip and flexible cable; a rigid connector draft does not make the whole cable a rigid simulator asset. Visible approach/withdrawal can support a proposed task, while electrical operation, hidden contact and successful seating remain unverified. These are future capture classes, not generated examples in the current library.

Use the optional task-collection hint to retrieve family-scoped feedback before discovery. Leave it automatic when unsure. General-method annotations apply across new sources; recording-specific notes require the same original recording SHA. None of these notes train weights, supply observed geometry or grant execution permission.

The private desk distinguishes deterministic structural findings, hosted text-metadata findings and missing hosted reviews. The owner may acknowledge hosted findings with a private reason to curate a research draft; source, path, hash and structural failures cannot be acknowledged away. Model objections and physical uncertainty remain recorded.
