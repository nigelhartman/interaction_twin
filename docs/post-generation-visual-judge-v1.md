# Private post-generation visual judge

This additive quality step checks a generated component against its exact
source recording and SAM mask. It does not change the existing pipeline,
cached runs, native meshes, poses, owner approval or public library.

Prepare a fresh packet offline:

```sh
.venv-sam/bin/python scripts/review_generated_part.py prepare \
  --program data/<case>/<procedure>/interaction.json \
  --part-ledger data/part-ledgers/<hash>/part-ledger.json \
  --component <actual_component_id> \
  --output output/<new-private-packet> \
  --agent-qa output/<saved-agent-quality-notes>.json
```

The source-bound ledger must identify the exact genuine native asset and
generation anchor. The anchor's decoded original RGB and binary mask must
equal the generation input pixels, with the exact source frame index and PTS.
A new CPU contact sheet uses **all original native triangles and base colour**,
four authored orbit cameras, no convex hull, voxel reduction, new texture or
manual pose. The orbit is not video alignment or physical validation.

The existing native parser's 16MiB/triangle/material limits remain unchanged.
Oversized or unsupported assets are withheld; a decimated inspection preview
is never silently substituted for the original asset in this judge. A future
larger-native adapter requires its own explicit source/renderer contract.

Up to two alternative original RGB/mask pairs are selected from the same saved
track: earliest eligible different observation, then largest remaining mask.
An alternative must contain at least 64 foreground pixels and at least 5% of
the generation-anchor mask's area; two-pixel or similarly unusable masks are
excluded explicitly. These are
visibility hypotheses, not a measured occlusion/faithfulness score. A mask can
be wrong; a larger one can include a hand or background. Physical identity
across a gap remains unverified.

The prepared packet pins original program/ledger/source/asset hashes, source
times, immutable upload copies, render receipt and software dependencies,
strict result schema, exact provider payload hash and existing budget plan.
The existing Meta encoder's JPEG conversion is recorded; original input files
are preserved byte for byte. Preparation never constructs a model client or
uploads anything. Saved agent QA is separately pinned and **not sent in the
prompt**, so a later comparison does not merely repeat the agent's finding.

After root reviews and authorizes the exact plan SHA, a separate `run` command
may use the existing `OvernightMetaClient`, unchanged increment/global guards
and accounted provider receipts. It requires both `--allow-paid-api` and
`--authorized-plan-sha256`. A durable once-only namespace is based on the exact
asset, component, source and original anchor pixels; changing output folder,
prompt or code does not permit an uncertain paid attempt to be resent.

The strict verdict is `faithful`, `needs_repair` or `uncertain`. **Faithful means
only sampled visible appearance**, never all hidden geometry, metric scale,
contact, electrical compatibility, physics or permission to publish. Evidence
citations must exist in this packet. A better-anchor suggestion must select an
actually supplied alternative, or request an unspecified clearer future
capture. No seed boxes, dimensions, poses, code or automatic repair commands
are accepted. Each result retains human-review-required and all physical/
approval/publication/geometry-modification flags false.

`verify` checks a prepared packet offline. `verify_report()` additionally checks
the exact structured result and accounted image/payload receipt. Reports retain
the independently saved agent QA for a fallible comparison, not a truth label.

A future pipeline hook can prepare this packet after successful native import
and before owner review, then attach a separately authorized settled report.
It must preserve source/program/asset revision bindings, budget/once-only
admission and the owner curation boundary. This module does not enable that
hook, regenerate models, approve candidates, merge identities, publish artifacts
or claim an independent held-out evaluation.
