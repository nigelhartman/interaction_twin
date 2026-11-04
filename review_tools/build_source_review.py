#!/usr/bin/env python3
"""Build an explicit code-only private review archive, never the working tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

DENIED_TOP = {'data','output','models','checkpoints','state','media','hpc','annotations','benchmarks','node_modules'}
SUFFIXES = {'.py','.js','.cjs','.mjs','.css','.html','.svg'}
GENERATED_UI = {'workspace/cases.js','workspace/cases-index.js','workspace/agent-preview.js',
                'workspace/reconstruction-preview.js','workspace/appearance-review-data.js',
                'showcase/library-index.js','showcase/physics-bench-source.js'}
DOCS = (
 'pipeline.md','data-contract.md','model-handoffs.md','mcp.md','portable-assets.md',
 'admin-api.md','admin-workflow.md','meta-budget.md','meta-sam-api.md','sam31-adapter.md',
 'sam3d-upload-default.md','sam3d-coordinate-contract.md','sam3d-mechanisms.md',
 'object-program.md','robotics-workbench.md','robotics-simulation-plan.md',
 'semantic-asset-companion-v1.md','cable-trace-2d-v1.md','native-appearance.md',
 'post-generation-visual-judge-v1.md','reproduction-review-20261101.md',
 'submission-demo-20261101.md','positioning-review-20261101.md','brand-v2.md',
 'newton-physics-bench.md','sam3d-first-source-projection-protocol.md',
 'marketing-art-direction.md','marketing-studio.md','robotics-next-captures.md',
)
PROFILES = ('cuda_preflight','meta_api_preflight','verified_evidence','local_smoke','local_low',
 'meta_low','meta_smoke','meta_sam31_low','meta_sam31_parts','meta_sam31_video_low',
 'meta_sam31_video_dense','meta_sam31_auto_camera','meta_sam3d_default')
VENDOR = ('LICENSE','manifest.json','three.module.js','three.core.js','GLTFLoader.js',
          'OrbitControls.js','BufferGeometryUtils.js')
README = '''# Interaction Twin — private code review snapshot

This is a deliberately selected **code-only** snapshot prepared on 4 November
2026. It is not a distributable installation or the separate saved judge demo.
The exact unmodified project README is [README.project.md](README.project.md).
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
'''

def sha(raw): return hashlib.sha256(raw).hexdigest()

def scan_source(raw,target):
    if len(raw)>4*1024*1024: raise ValueError('Unexpected source size: '+target)
    if Path(target).suffix!='.png' and b'\x00' in raw: raise ValueError('Unexpected binary source: '+target)
    if re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|(?:api_key|token)\s*[:=]\s*[\"\x27](?:EAA[A-Za-z0-9]{80,}|sk-[A-Za-z0-9_-]{40,})[\"\x27]',raw,re.I):
        raise ValueError('Potential embedded credential; contents withheld: '+target)
    if re.search(rb'\bhf_[A-Za-z0-9]{30,}\b|\bAKIA[A-Z0-9]{16}\b|\bAIza[A-Za-z0-9_-]{35}\b',raw):
        raise ValueError('Potential embedded provider credential; contents withheld: '+target)

def safe_relative(value):
    p=PurePosixPath(value)
    if not value or str(p)!=value or '\\' in value or p.is_absolute() or any(x in {'.','..'} or x.startswith('.') for x in p.parts):
        raise ValueError('Unsafe path')
    if p.parts[0] in DENIED_TOP or p.name=='AGENTS.md' or p.suffix.lower() in {'.env','.sqlite','.sqlite3','.mp4','.mov','.m4v','.webm','.glb','.gltf','.npz','.npy','.ply','.pt','.pth','.ckpt','.safetensors','.bin','.itcp','.u32','.wav','.mp3','.aiff','.zip','.csv','.log'}:
        raise ValueError('Excluded path')
    if p.parts[0]=='config' and str(p) not in {'config/marketing-brand-default.json','config/marketing-brand-v2.json'}:
        raise ValueError('Excluded operational configuration')
    if p.suffix.lower() in {'.jpg','.jpeg','.webp','.gif'} or (p.suffix.lower()=='.png' and str(p) not in {'showcase/brand/v2/favicon-16.png','showcase/brand/v2/favicon-32.png','showcase/brand/v2/favicon-180.png'}):
        raise ValueError('Captured or generated raster imagery excluded')
    if str(p).startswith(('showcase/public/','showcase/media/')) or str(p) in GENERATED_UI:
        raise ValueError('Generated results or catalog excluded')
    return p

def ordinary(root, relative):
    safe_relative(relative)
    p=root/relative
    if any(q.is_symlink() for q in [p,*p.parents] if q!=root.parent): raise ValueError('Symlink rejected')
    if not p.is_file() or not p.resolve().is_relative_to(root): raise ValueError('Missing ordinary source')
    return p

def selection(root):
    # Enumerate only these explicitly named ordinary directories. Never recurse
    # the checkout, private roots, a model repository, or a recording directory.
    paths={}
    def add(relative, kind='custom_source', target=None):
        if relative in GENERATED_UI: return
        p=ordinary(root,relative)
        paths[target or relative]=(p,relative,kind)
    for dirname,suffixes in [('.', {'.py'}),('scripts',{'.py'}),('tests',SUFFIXES),
       ('schemas',{'.json'}),('admin',SUFFIXES),('workspace',SUFFIXES),('showcase',SUFFIXES),
       ('portable_viewer',SUFFIXES),('native_review',SUFFIXES),('initial_native_review',SUFFIXES),
       ('initial_native_review_v2',SUFFIXES),('robot_starter',SUFFIXES)]:
        for p in sorted((root/dirname).iterdir()):
            if p.name.startswith('.') or not p.is_file() or p.suffix not in suffixes: continue
            add(p.relative_to(root).as_posix())
    for name in DOCS: add('docs/'+name,'selected_public_documentation')
    add('README.md','unmodified_project_readme','README.project.md')
    for p in sorted(root.iterdir()):
        if p.is_file() and re.fullmatch(r'requirements[a-z-]*\.lock\.txt',p.name): add(p.name,'dependency_lock_snapshot')
    for name in PROFILES: add('profiles/'+name+'.json','generic_profile')
    for name in VENDOR: add('workspace/robot-vendor/'+name,'third_party_runtime_or_license')
    for name in ('mark.svg','wordmark.svg','wordmark-dark.svg','favicon-16.png','favicon-32.png','favicon-180.png'):
        add('showcase/brand/v2/'+name,'minimal_owned_brand')
    for name in ('marketing-brand-default.json','marketing-brand-v2.json'):
        add('config/'+name,'generic_brand_configuration')
    for name in ('index.html','demo.js','demo.css'):
        add('showcase/demos/charge-phone-v1/'+name,'owned_demo_ui_without_saved_results')
    for name in ('preview_server.py','preview.html','README.txt'):
        add('judge_preview/'+name,'portable_preview_runtime_without_results')
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name.upper() in {'LICENSE','LICENSE.TXT','LICENSE.MD','COPYING'}:
            add(p.name,'existing_project_license')
    return paths

def build(root,output):
    root=root.resolve(); output=output.resolve()
    folder=output/'Interaction-Twin-source-review'
    archive=output/'Interaction-Twin-source-review-20261104-v9.zip'
    if folder.exists() or archive.exists(): raise ValueError('Immutable output already exists')
    selected=selection(root); selected_bytes={}
    for target,(p,original,kind) in selected.items():
        raw=p.read_bytes()
        # High-confidence embedded bearer/private-key indicators only; fixture
        # names and API environment-variable examples are ordinary source code.
        scan_source(raw,target)
        selected_bytes[target]=(raw,original,kind)
    selected_bytes['README.md']=(README.encode(),None,'generated_package_document')
    helper=Path(__file__).read_bytes()
    selected_bytes['review_tools/build_source_review.py']=(helper,None,'generated_package_reproduction_helper')
    inventory=[]
    for target,(raw,original,kind) in sorted(selected_bytes.items()):
        safe_relative(target)
        inventory.append({'path':target,'sha256':sha(raw),'bytes':len(raw),'original_source_path':original,
                          'origin':kind,'transformation':'none_byte_exact_copy' if original else 'new_package_only_file'})
    manifest={'format':'interaction-twin-code-only-source-review','version':9,'revision':'9',
      'private_review_only':True,'public_release_authorized':False,'source_checkout_modified':False,
      'payload_file_count':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'files':inventory,
      'manifest_self_excluded':True,'upstream_license':'workspace/robot-vendor/LICENSE',
      'project_license_provided':any(r['origin']=='existing_project_license' for r in inventory),
      'policy':{'selection':'exact manifest from named ordinary source directories, suffixes and fixed paths',
                'no_hidden_directory_traversal':True,'symlinks_allowed':False,'credentials_read':False,
                'generated_catalogs_or_saved_model_outputs_included':False,
                'new_api_calls':0,'new_gpu_calls':0,'fresh_dependency_install_tested':False,
                'quality_method_catalog_included':False,'physics_source_inputs_or_results_included':False}}
    folder.mkdir(parents=True)
    for target,(raw,_,_) in selected_bytes.items():
        p=folder/target; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    manifest_raw=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    (folder/'source-manifest.json').write_bytes(manifest_raw)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): z.write(p,folder.name+'/'+p.relative_to(folder).as_posix())
    verify(archive)
    receipt={'format':'interaction-twin-source-review-build','version':9,'revision':'9','status':'built_private_code_only',
      'archive':{'path':str(archive.relative_to(root)),'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size},
      'manifest':{'path':str((folder/'source-manifest.json').relative_to(root)),'sha256':sha(manifest_raw),'bytes':len(manifest_raw)},
      'payload_file_count':len(inventory),'zip_file_count':len(inventory)+1,'payload_bytes':manifest['payload_bytes'],
      'original_source_file_count':sum(r['original_source_path'] is not None for r in inventory),
      'new_api_calls':0,'new_gpu_calls':0,'source_files_modified':False,'public_release_performed':False}
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def verify(archive):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); prefix='Interaction-Twin-source-review/'
        if len(names)!=len(set(names)) or len(names)>1500: raise ValueError('Duplicate or excessive ZIP entries')
        for n in names:
            if not n.startswith(prefix): raise ValueError('Unexpected archive root')
            safe_relative(n[len(prefix):])
            if stat.S_ISLNK(z.getinfo(n).external_attr>>16): raise ValueError('ZIP symlink')
        m=json.loads(z.read(prefix+'source-manifest.json'))
        expected={prefix+r['path'] for r in m['files']}|{prefix+'source-manifest.json'}
        if set(names)!=expected: raise ValueError('Manifest/inventory mismatch')
        for r in m['files']:
            raw=z.read(prefix+r['path'])
            if len(raw)!=r['bytes'] or sha(raw)!=r['sha256']: raise ValueError('Hash/size mismatch')
            scan_source(raw,r['path'])
        if any(r['path'] in GENERATED_UI for r in m['files']): raise ValueError('Generated UI payload included')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(build(a.root,a.output),indent=2))
