> Run application commands below from the repository’s `app/` directory. Paths named `<checkout>` refer to that application directory.

# Local source setup

This guide was retained from the verified code-only preparation. The public release adds the README, trailer, and static deployment described in [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md). References below to a private preparation describe that earlier packaging step; the source was subsequently published at the owner's request.

# Interaction Twin — private code review snapshot

This is a deliberately selected **code-only** snapshot prepared on 4 November
2026. It is not a distributable installation or the separate saved judge demo.
The exact unmodified project README is [historical technical README](release/README-project-historical.md).
The manifest binds every included file to its original source hash, or marks it
as a newly generated package document. No original checkout file is rewritten.

## Included and excluded

Included: custom Python modules and command wrappers, source tests/helpers,
schemas, owned frontend templates, the Cobalt v2 vector mark/wordmarks and tiny
favicons, generic execution profiles, selected architecture/API/MCP/robotics
guides, dependency lock snapshots, and named Three.js runtime files plus their
existing MIT license. No project-wide license is supplied because none exists
in the current checkout; the Three.js license covers that dependency only.

Excluded: recordings, source images/masks, all generated models/results/catalogs,
saved demo tasks/traces/Muse responses, owner feedback/approval/memory/database
state, films/audio, model repositories/checkpoints, runtimes, all secrets and
environment files, cluster/account configuration, operational history/AGENTS,
logs, and the Git repository. UI templates may reference omitted demonstrations;
this package does not silently substitute fixtures for real outputs.

This is a **private repository-preparation artifact**. Public release, source
permissions, contribution dates, contributor declarations and license choice
still require the owner's final review. Existing source can contain historical
workspace paths and experimental defaults; inclusion is not permission to post
the working tree or every script publicly. No remote, commit, push, deployment
or hackathon submission was created.

## Empty workspace, with your separately prepared runtime

No fresh dependency installation has been tested for this snapshot. The split
requirements*.lock.txt files record prepared environments; some include upstream
Git dependencies and are not a verified portable installer. Install upstream
code/weights under their own terms, and explicitly prepare Python, NumPy,
Pillow, SciPy, OpenCV, trimesh/jsonschema, FFmpeg and any selected model/runtime.
The launcher expects .venv-sam/bin/python; the lower-level commands below can use
an already prepared external interpreter without copying its environment here.

```sh
IT_PYTHON=/absolute/path/to/your/prepared/python
mkdir -p data
"$IT_PYTHON" -B scripts/build_workspace.py --help
"$IT_PYTHON" -B scripts/build_workspace.py --no-replays
"$IT_PYTHON" -B scripts/serve_workspace.py --host 127.0.0.1 --port 18867
```

The build command's actual inspected options are --out and --no-replays. With
no separately supplied artifacts, it should report cases:0 and errors:[].
It creates empty local catalog files from absence of data; none are in the ZIP.
The server is loopback-only. Sign in/set up the owner account through /admin/
before opening protected /workspace/. There is no shipped password, setup code
or authenticated session. Opening the public showcase does not supply examples.
Use an unused local port and leave other running app instances untouched.

## Hosted processing and GPU setup

Opening pages and building an empty catalog do not invoke models. Uploads are
private and fresh processing requires explicit owner consent. The default
SAM3D profile sends sampled imagery to the configured Meta API and RGB/masks
to a separately configured GPU worker. This ZIP supplies neither API access,
credentials nor a working cluster configuration. It also omits saved operational
budget plans referenced by historical profile defaults; initialize/review these
through the documented owner workflow before enabling any paid processing.
Do not paste keys into source, frontend code, profile JSON or Git. Budget guards
are local estimates, not provider balances or an account-wide hard cap.
There are no automatic paid-model retries or implicit model downloads.

## What changed since the older portable guide

The included historical reproduction guide describes an earlier source snapshot,
not a test count for this one. Current code includes six-record library support,
explicit owner-curated publication, separate declared-derived assembly contracts,
a lightweight read-only lineage validator, cross-recording interface task drafts,
and portable browser-local robot instructions. Their actual saved outputs are
excluded. The separate judge-preview ZIP contains selected reviewed results.

This 4 November submission-polish snapshot also includes the source-bound
connected-scene publication gate, all-six-recording binding audit, the narrow
nonrigid-child withholding display contract and the latest private phone
reconstruction experiment source. None of their recorded inputs, private
candidate outputs, accounts or approvals are bundled. The live working library
has nineteen original SAM 3D components across six recordings; its recordings,
models and saved descriptive programs remain deliberately excluded here.

This source increment adds the read-only source-visible 2.5D observation contract,
exact source-JPEG UV appearance, CPU depth-prior research helpers, strict source
ledger barriers, separately labelled editorial frame repetition, explicit
curated observed-surface MCP resources and independent renderer lifecycle tests.
Their source footage, masks, inferred surfaces, checkpoint, approvals and all
saved proof artifacts are excluded. Numerical physical joints remain unknown.

This final source snapshot includes conservative contour-adaptive source-visible
meshes (explicit proof version2), exact cached-depth derivation and original
source-texture lineage. It includes a separate source-conditioned estimated
motion layer, continuity cuts, original-known/unknown preservation, and exact
curated read-only MCP transports. All saved data and checkpoint files remain
excluded. This is not completion of hidden geometry or physical tracking.

The current film authoring/capture source is included; final rendered films,
audio and frame captures remain excluded. Workbench recording and the explicit
capture=1 buffer opt-in are included; no robot hardware is controlled.

The main app's current complete test suite includes saved-data regressions and
is **not** expected to run fully from this code-only archive.

For this snapshot, only the fixture-only commands recorded in the adjacent
delivery receipt are newly tested from an isolated extraction, using a borrowed
development interpreter. That is not a fresh pip install, learned inference,
browser-pixel proof, physical accuracy benchmark or robot physics execution.

## Architecture and limitations

See docs/pipeline.md, docs/object-program.md, docs/admin-workflow.md and
docs/mcp.md. SAM3.1 masks identify visible components; SAM3D supplies independent
native geometry candidates; Muse proposes attributed semantics/mechanisms;
source-bound fitting and strict joint FK connect supported motion. Human review
binds exact revisions before public library/MCP exposure. Derived geometry is
declared separately from original native models. Unknown poses stay withheld.
New source includes compact source-observed display appearance states, an opaque
byte-RGB display adapter, composers and selection controls for six trailers,
an editorial native-camera capture tool, robot checkpoint/outcome/navigation
presentation helpers, a second source-clock film composer, automatic
affine/exact-subset/observed-appearance contracts,
source-bound actual critic payload verification, explicit 2D nonrigid observations,
quality-method memory and a separate Newton CPU contact runner. The private
quality-rule catalog and all historical evidence are deliberately omitted.
With no catalog, server startup marks method memory unavailable and launches no
request; a present invalid catalog fails closed. Do not transplant source-specific
rules into a fresh app without their original evidence and an explicit review.
The robot task remains authored kinematics rather than a trained policy. The
separate Newton runner supports assumed-parameter CPU contact experiments; its
private inputs/replays/models are not supplied and no new physics run is made
by creating this archive. Physical scale, fit/contact, grasp, cable dynamics, electrical safety
and successful charging remain unverified; simulation_ready stays false.
