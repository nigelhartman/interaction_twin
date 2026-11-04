# SAM 3D upload default · 2 November 2026

New ordinary-video uploads use `meta_sam3d_default`. The explicit older profiles
and their outputs remain historical comparisons; they are never substituted if
the SAM 3D worker is unavailable.

## Processing path

1. Normalize the video and retain its source hash and presentation timestamps.
2. Muse Spark proposes independently moving components and full-frame seeds.
3. SAM 3.1 tracks their visible masks. Muse audits sampled source/mask evidence.
4. Deterministic quality/diversity selection chooses at most three eligible parts
   in the default 40-second, 3-fps profile.
5. `sam3d_uploads.py` freezes each selected full-frame RGB/mask in the existing
   baseline-free handoff format. No old mesh, dimensions, canonical points or
   pose enters the handoff.
6. The configured `sam3d_cluster_bridge` runs/reconciles bounded jobs in the
   existing GPU allocation and uses the strict native importer. An unavailable
   worker yields `awaiting_gpu_worker`; a failed attempted model remains
   `failed_no_retry`. Neither case produces a replacement mesh.
   At most two independent components run concurrently. The cluster bridge adds
   its own global two-slot guard across callers. Cancelling a local request
   preserves partial evidence and never cancels the parent GPU allocation.
7. Muse Spark proposes the descriptive relationships, variables, assertions and
   procedure steps. The existing deterministic compiler verifies identities,
   source timestamps, evidence and actual asset hashes. This is already real
   hosted Muse inference, rather than a planned future replacement.
8. A successful program build refreshes the workspace and robotics library.
   Semantic-only programs are still useful while their geometry is pending.
   Each imported native candidate also gets a hash-verified source/model browser
   review with the bounded lossless binary transport. If a model exceeds display
   limits, its native result remains saved and the preview is explicitly unavailable.

SAM 3D returns one canonical static candidate per component, not a newly
generated shape at every video timestamp. Native parts keep their own inferred
gauge. Generation does not establish temporal motion, common assembly, metric
scale, physical hinge axes, cavity dimensions, contact or insertion.

## Status and resumption

Full run records expose `sam3d_reconstruction_status` with generated count,
pending component IDs, and `native_motion_status`. The polling summary preserves
these fields without exposing worker commands or private runtime paths.

Resuming the identical request reconciles pending GPU reconstruction even when
all the source hashes are unchanged. The handoff identity uses component ID,
selection hash and RGB/mask manifest hash, independent of output directory or
phase attempt. The bridge keys its durable attempt by that complete handoff
hash, so another output directory is not permission to repeat a model attempt.
Completed components reuse the same native candidate. Source changes create a
different evidence identity; originals remain preserved.

The old comparison-backend pose adapter is not run on SAM 3D assets. Fresh
generic native alignment remains unfinished and is reported explicitly. The
existing single-earbud motion diagnostic is a separately scoped experiment,
not proof that arbitrary uploaded clips already have accepted 3D motion.

## GPU and cost boundary

The optional public runtime configuration contains only task-owned routing and
hash pins, never API keys or Hugging Face tokens. The browser cannot provide
SSH commands, remote hosts, scheduler resource arguments or Python programs.
The existing two-GPU allocation and its holder must not be modified or closed.

SAM 3D jobs consume cluster resources; Muse/SAM API requests retain the existing
transactional API ledger and cumulative/increment spending guards. Opening a
viewer or querying the library starts no inference. Real Meta account credits
remain separately reconciled by the account owner.

## Verification scope

Offline tests cover exact input/source/component joins, cancellation before
dispatch, stable pending/resume identity, no substitute model, changed-asset
rejection, provider routing, unchanged historical profiles, pending status in
polling, and the prohibition on the old geometry/pose paths. Cluster inference
and actual fresh upload completion require their separate execution receipts;
offline fixture passes are not evidence that a GPU model ran.
