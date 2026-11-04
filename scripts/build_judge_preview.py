#!/usr/bin/env python3
"""Freeze a standalone, read-only preview from the CURRENT curated snapshot.

No public files, source programs or owner state are modified. No working-catalog
discovery, inference, deployment, dependency installation or credential access.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import re
import posixpath
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGES = ['index.html', 'library.html', 'robotics-lab.html', 'agents.html', 'agent-setup.html',
         'native-model.html', 'aligned-scene.html', 'film.html', 'trailers.html', 'walkthrough.html', 'robot-workbench.html','physics-bench.html']
STATIC = PAGES + ['mark.svg', 'style.css', 'cobalt.css', 'studio.css', 'library.css', 'robotics-lab.css', 'agents.css',
    'app.js', 'auto-scene-camera-v1.js', 'estimated-motion-v1.js', 'source-visible-flexure-v1.js', 'source-surface-presentation-v1.js', 'agent-charge-example.js', 'library.js', 'sam3d-models.js', 'aligned-scene-contract.js', 'aligned-scene-client.js',
    'aligned-scene.js', 'native-model.js', 'robotics-lab.js', 'robotics-lab-assets.js', 'agents.js',
    'robot-workbench.js', 'robot-workbench.css', 'robot-workbench-ui-v2.js', 'robot-controller.js', 'robot-task-plan.js', 'robot-recorder.js',
    'trailers.js', 'trailers.css', 'trailer-collection.js',
    'nonrigid-observations-contract.js','native-byte-rgb-display-v1.js','source-appearance-states-v1.js','physics-bench.css','physics-bench-source.js','physics-bench-contract.js','physics-bench.js']
PHYSICS_STARTER='showcase/media/interaction-twin-newton-starter-20261104-v1.zip'
DEMO_FILES = ['index.html', 'demo.css', 'demo.js', 'task.json', 'trace.json', 'summary.json', 'muse-review.json']
MAX_OMITTED_PROVENANCE_FILES = 128
MAX_OMITTED_PROVENANCE_BYTES = 4 * 1024 * 1024
# Measured 710 MiB includes the exact contour/depth evidence, laptop motion and
# current robotics starters; leave a bounded margin for the final film. This
# local archive allowance changes no model, request or API spending guard.
MAX_PREVIEW_BYTES=768*1024*1024
# This ancestor archive is part of the newly approved derivative lineage. It is
# inert source text, kept explicitly rather than widening provenance discovery.
INCLUDED_ANCESTOR_SOURCE_TEXT=[{'path':'data/assembly-refinements/jabra-source-refit-20261103-v2/provenance/refined_assembly_contract.py.txt',
    'sha256':'d09b7345cb8ff6046ed45f4a65788c1a045d9d6552d0fcf55c4373568c287e49','bytes':29977}]
RUNTIME = ['workspace/case-data.js', 'workspace/scene.js']
PUBLIC = ['cases-index.json', 'cases-index.js', 'library-index.json', 'library-index.js',
          'library-catalog.json', 'aligned-scenes.json', 'public-assets.json']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def ref(path, root=ROOT):
    path = Path(path)
    require(path.resolve() == path and path.is_relative_to(root) and path.is_file()
            and not any(parent.is_symlink() for parent in [path, *path.parents]), 'Ordinary source file required')
    raw = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def check(path, expected, *, root=ROOT):
    value = ref(path, root=root)
    require(all(value[key] == expected[key] for key in ['sha256', 'bytes']), 'Pinned source bytes changed')
    return value


def omitted_provenance(item):
    """Only exact approved implementation source text is an optional omission."""
    path = item['path']
    if item in INCLUDED_ANCESTOR_SOURCE_TEXT:return False
    if '/provenance/' not in path or not path.endswith('.py.txt'):
        return False
    from object_program import CODE
    require(path.startswith('data/object-programs/') and Path(path).name.removesuffix('.txt') in CODE
            and type(item.get('bytes')) is int and 0 < item['bytes'] <= 512 * 1024,
            'Unrecognized implementation-provenance omission; review its declared path/type/bounds')
    return True


class _Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in {'src', 'poster', 'href'} and value)


def static_dependencies(root):
    """Resolve local HTML-linked assets without walking owner/runtime folders."""
    wanted = set('showcase/' + name for name in STATIC)
    # Fixed editorial exports only; never scan owner uploads or private renders.
    for stem in ['01-origin','02-blueprint','03-mission','04-proof','05-pocket-worlds','06-agent']:
        for suffix in ['.mp4','-poster.jpg','.vtt','-transcript.txt']:
            wanted.add('showcase/media/trailer-collection-20261104-v1/'+stem+suffix)
    wanted.update('showcase/demos/charge-phone-v1/' + name for name in DEMO_FILES)
    queue = [name for name in wanted if name.endswith('.html')]
    visited = set()
    allowed = {'.html', '.js', '.css', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico',
               '.mp4', '.webm', '.mp3', '.wav', '.vtt', '.srt', '.json', '.woff', '.woff2'}
    while queue:
        relative = queue.pop()
        if relative in visited: continue
        visited.add(relative)
        parser = _Links(); parser.feed((Path(root) / relative).read_text())
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith(('/api/', '/files/')):
                continue
            target = posixpath.normpath(parsed.path.lstrip('/') if parsed.path.startswith('/')
                                       else posixpath.join(posixpath.dirname(relative), parsed.path))
            # Creator URLs and hidden administration are not snapshot resources.
            require(target.startswith(('showcase/', 'workspace/')) and '\\' not in target and
                    not any(p.startswith('.') for p in target.split('/')),
                    'HTML references an unreviewed local resource outside preview scope')
            if target.startswith('workspace/'):
                require(target in RUNTIME, 'HTML references an owner workspace page')
            require(Path(target).suffix in allowed or target==PHYSICS_STARTER, 'Unknown local presentation asset type')
            if target not in wanted:
                wanted.add(target)
                if target.endswith('.html'): queue.append(target)
    require(len(wanted) <= 256, 'Bounded presentation dependency graph required')
    return sorted(wanted)


def physics_snapshot(root, approved):
    """Exact saved Newton experiment, three approved meshes and reviewed ZIP."""
    root=Path(root);source=root/'showcase/physics-bench-source.js';raw=source.read_text()
    match=re.fullmatch(r'globalThis\.PhysicsBenchSource=(\{.*\});\s*',raw,re.DOTALL)
    require(match is not None and len(raw)<65536,'Exact JSON-only physics pointer required')
    pointer=json.loads(match.group(1));replay=pointer['replay']
    require(pointer.get('version')==1 and re.fullmatch(r'showcase/public/physics/[A-Za-z0-9_-]+\.json',replay['path'])is not None,'Explicit saved physics replay required')
    check(root/replay['path'],replay,root=root);experiment=read(root/replay['path'])
    require(experiment.get('format')=='interaction-twin-newton-contact-experiment'and experiment.get('version')==1 and len(experiment.get('frames',[]))==241 and len(experiment.get('objects',[]))==3,'Complete three-body Newton replay required')
    claims=experiment['claims'];require(claims.get('real_physics_engine_execution')is True and all(claims.get(k)is False for k in ['metric_scale_measured','material_parameters_measured','physical_accuracy_verified','robot_grasp_validated','simulation_ready']),'Physics assumptions were silently upgraded')
    assets=[]
    for obj in experiment['objects']:
        asset=obj['asset_ref'];require(asset['path'].endswith('.glb')and 0<asset['bytes']<=32*1024*1024 and asset['path']in approved and all(asset[k]==approved[asset['path']][k]for k in ['path','sha256','bytes']),'Physics mesh is outside the current exact curated assets')
        check(root/asset['path'],asset,root=root);assets.append(asset)
    starter=ref(root/PHYSICS_STARTER,root=root)
    with ZipFile(root/PHYSICS_STARTER)as archive:
        names=archive.namelist();prefix='Interaction-Twin-Newton/'
        expected={'README.txt','input.json','manifest.json','native/adapter_early.glb','native/earbud_b.glb','native/gold_coin.glb','reference-replay.json','requirements.txt','run_experiment.py'}
        require(set(names)=={prefix+x for x in expected}and len(names)==len(expected),'Newton starter contains unexpected files')
        manifest=json.loads(archive.read(prefix+'manifest.json'));require(manifest.get('format')=='interaction-twin-newton-cpu-starter'and manifest.get('claims')==claims,'Newton starter scope differs')
        require(archive.read(prefix+'reference-replay.json')==(root/replay['path']).read_bytes(),'Starter replay differs from website replay')
        for component in manifest['source_components']:
            require(component['asset_ref']in assets,'Starter native mesh is outside the displayed three objects')
            require(archive.read(prefix+component['file'])==(root/component['asset_ref']['path']).read_bytes(),'Starter mesh bytes differ from approved native asset')
        for entry in manifest['files']:
            blob=archive.read(prefix+entry['path']);require(len(blob)==entry['bytes']and hashlib.sha256(blob).hexdigest()==entry['sha256'],'Newton starter inventory bytes changed')
    return {'source_ref':ref(source,root=root),'replay_ref':replay,'starter_ref':starter,'asset_refs':assets,'scope':'saved_actual_newton_cpu_states_not_browser_physics_or_robot_grasp_validation','claims':claims}


def verify_display_lineage(root, approved, *, catalog=None):
    """Check runtime-needed derivative/nonrigid references are downloadable."""
    root=Path(root);index=read(root/'showcase/public/aligned-scenes.json');seen=set();proofs=[]
    def exact_bytes(pointer, *, maximum=None):
        require(maximum is None or pointer['bytes']<=maximum,'Display artifact exceeds its typed byte bound')
        require(pointer['path']in approved and all(pointer[k]==approved[pointer['path']][k]for k in ['path','sha256','bytes']),'Display lineage is missing an exact approved public reference: '+pointer['path'])
        check(root/pointer['path'],pointer,root=root);return (root/pointer['path']).read_bytes()
    def exact(pointer):return json.loads(exact_bytes(pointer))
    def derivative(pointer):
        doc=exact(pointer)
        if pointer['path']in seen:return
        seen.add(pointer['path'])
        for key in ['base_object_program_ref','evidence_program','scene_ref','baseline_scene_ref','refinement_report_ref','source_alignment_audit_ref']:
            if doc.get(key):exact(doc[key])
        for binding in doc['component_assets']:
            for key in ['native_glb','original_native_glb','native_metadata']:
                if binding.get(key):
                    value=binding[key];require(value['path']in approved and all(value[k]==approved[value['path']][k]for k in ['path','sha256','bytes']),'Derived native/original asset missing from preview')
        if doc.get('parent_derived_assembly_ref'):derivative(doc['parent_derived_assembly_ref'])
        if doc.get('nonrigid_observations_ref'):
            flex=exact(doc['nonrigid_observations_ref']);exact(flex['trace_ref']);exact(flex['evidence_program'])
        if doc.get('source_appearance_states_ref'):
            appearance=exact(doc['source_appearance_states_ref'])
            require(appearance.get('format')=='interaction-twin-source-appearance-states'and appearance.get('version')==1,'Unknown typed source appearance sidecar')
            previous=exact(appearance['previous_derived_assembly_ref'])
            require(previous=={k:v for k,v in doc.items()if k!='source_appearance_states_ref'},'Appearance snapshot changed its prior descriptor')
            for key in ['scene_ref','evidence_program']:exact(appearance[key])
            exact_bytes(appearance['baseline_asset_ref'])
            exact_bytes(appearance['probe_native_vertex_ids_ref'])
            for row in appearance['keyframes']:
                for key in ['source_rgb_ref','source_mask_ref','patch_ref']:exact_bytes(row[key])
            for row in appearance['observations']:
                for key in ['source_rgb_ref','source_mask_ref']:
                    if row.get(key):exact_bytes(row[key])
        proofs.append({'document_ref':pointer,'format':doc['format'],'nonrigid_included':bool(doc.get('nonrigid_observations_ref')),'source_appearance_included':bool(doc.get('source_appearance_states_ref'))})
    def current_records():
        nonlocal catalog
        if catalog is None:
            pointer=read(root/'showcase/public/library-index.json')['catalog']
            require(re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json',pointer['path'])is not None and pointer['path']=='library-catalog-'+pointer['sha256']+'.json','Current exact observed-surface catalog required')
            check(root/'showcase/public'/pointer['path'],pointer,root=root)
            catalog=read(root/'showcase/public'/pointer['path'])
        return catalog['records']
    for row in index['scenes']:
        for key in ['manifest','object_program']:
            if row.get(key):exact(row[key])
        if row.get('derived_assembly'):derivative(row['derived_assembly'])
        require(not row.get('observed_surface_presentation')or row.get('observed_surfaces'),'Observed presentation needs its exact source-surface declaration')
        if row.get('observed_surfaces'):
            # Publication already performed expensive geometry/source proofs.
            # A frozen preview verifies their exact typed, admitted byte graph;
            # it never repeats native rasters, model inference or private scans.
            import source_visible_flexure_contract_v1 as F
            declaration=row['observed_surfaces'];surface=exact(declaration)
            validated=F.validate(surface,exact_bytes)
            matches=[r for r in current_records()if r['source']['sha256']==row['source_sha256']]
            require(len(matches)==1,'Observed display must match one current curated recording')
            record=matches[0]
            require(record.get('observed_surfaces')==declaration and
                    surface['source_program']==record['interaction']and
                    surface['source_scene']==row['manifest']and
                    surface['source']['sha256']==row['source_sha256'],
                    'Observed-surface preview differs from current catalog/program/scene')
            for pointer in validated['artifacts']:exact_bytes(pointer)
            presentation=None
            if row.get('observed_surface_presentation'):
                pointer=row['observed_surface_presentation']
                require(record.get('observed_surface_presentation')==pointer,'Observed presentation differs from current curated recording')
                from observed_surface_presentation_v1 import validate as validate_presentation
                presentation=validate_presentation(exact(pointer),surface,declaration)
            else:require(not record.get('observed_surface_presentation'),'Curated observed presentation omitted from aligned scene')
            proofs.append({'document_ref':declaration,'format':surface['format'],
                'source_sha256':surface['source']['sha256'],'source_program_ref':surface['source_program'],
                'source_scene_ref':surface['source_scene'],'observed_surfaces_included':True,
                'artifact_count':len(validated['artifacts']),
                'supported_sources':validated['supported_sources'],'unknown_sources':validated['unknown_sources'],
                'presentation_included':presentation is not None,'new_inference_calls':0,
                'geometry_proof_scope':'exact_publication_bound_observation_bytes_not_independent_shape_validation'})
        matches=[r for r in current_records()if r['source']['sha256']==row.get('source_sha256')] if row.get('motion_hypothesis') else []
        if row.get('motion_hypothesis'):
            import motion_hypothesis_contract_v1 as T
            declaration=row['motion_hypothesis'];motion=exact(declaration)
            checked=T.validate(motion,exact_bytes)
            require(len(matches)==1,'Motion display must match one current curated recording')
            record=matches[0]
            require(record.get('motion_hypothesis')==declaration and motion['twin_id']==record['id']and
                    motion['evidence_program_ref']==record['interaction']and motion['baseline_scene_ref']==row['manifest']and
                    motion['source_ref']['sha256']==row['source_sha256']==record['source']['sha256']and
                    motion['source_ref']['path']==record['source']['path'],
                    'Motion preview differs from current catalog/program/source/scene')
            for pointer in checked['artifacts']:exact_bytes(pointer)
            proofs.append({'document_ref':declaration,'format':motion['format'],'motion_hypotheses_included':True,
                'artifact_count':len(checked['artifacts']),'source_sha256':row['source_sha256'],
                'source_program_ref':motion['evidence_program_ref'],'source_scene_ref':motion['baseline_scene_ref'],
                'original_observations_modified':False,'native_geometry_modified':False,'physical_accuracy_verified':False,
                'new_inference_calls':0,'proof_scope':'exact_source_conditioned_prediction_bytes_not_physical_ground_truth'})
    for record in current_records() if catalog is not None else []:
        if record.get('motion_hypothesis'):
            matches=[r for r in index['scenes']if r.get('source_sha256')==record['source']['sha256']]
            require(len(matches)==1 and matches[0].get('motion_hypothesis')==record['motion_hypothesis'],
                    'Curated motion declaration omitted from aligned scene')
    return proofs


def robotics_snapshot(root, catalog):
    """Bake real validated supported plans; exported files contain no runtime imports."""
    from robot_task_catalog import RobotTaskCatalog, encoded
    import robot_task_program as R
    transport = RobotTaskCatalog(root)
    files = {'showcase/demos/robotics/robot-kit.json': encoded(transport._canonical_kit())}
    kit_name = 'showcase/demos/robotics/robot-kit.json'
    def reference(path):
        raw = files[path]
        return {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    tasks = []
    for record in catalog['records']:
        ids = {c['id'] for c in record['components']}
        if {'case_base', 'case_lid', 'earbud_a', 'earbud_b'} <= ids:
            instruction = 'Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.'
            slug = 'jabra'
        elif 'gold_coin' in ids:
            instruction = 'Move gold_coin to the tray.'
            slug = 'wallet'
        else:
            continue
        task = transport.plan(record['id'], instruction)
        require(task.get('status') == 'ready' and transport.validate(task).get('valid') is True,
                'A previously advertised supported rehearsal has no valid current curated plan')
        for binding in R.task_bindings(task): transport.catalog._checked(binding)
        folder = 'showcase/demos/robotics/' + slug
        name, bundle_name = folder + '/task.json', folder + '/starter.zip'
        files[name] = encoded(task)
        bundle, _ = transport._build_bundle(task)
        files[bundle_name] = bundle
        tasks.append({'twin_id': record['id'], 'title': record.get('public_title') or record['title'],
                      'instruction': instruction, 'task': reference(name), 'starter': reference(bundle_name)})
    require(1 <= len(tasks) <= 4, 'At least one exact ready robotics rehearsal is required')
    index = {'format': 'interaction-twin-frozen-robotics-index', 'version': 1,
             'kit_id': R.KIT_ID, 'kit': reference(kit_name), 'tasks': tasks,
             'scope': 'authored_kinematic_snapshot', 'inference_available': False,
             'physical_accuracy_verified': False, 'simulation_ready': False}
    files['showcase/demos/robotics/index.json'] = encoded(index)
    return files, index


def build(output, *, root=ROOT):
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    require(output.resolve() == output and output.is_relative_to(root / 'output') and not output.exists(),
            'Choose a fresh local output directory')
    public = root / 'showcase/public'
    source_refs = {name: ref(public / name, root=root) for name in PUBLIC}
    index = read(public / 'library-index.json')
    catalog_name = index['catalog']['path']
    require(re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json', catalog_name) is not None
            and catalog_name == 'library-catalog-' + index['catalog']['sha256'] + '.json', 'Current exact catalog pointer required')
    check(public / catalog_name, index['catalog'], root=root)
    require((public / 'library-catalog.json').read_bytes() == (public / catalog_name).read_bytes(), 'Current catalog aliases differ')
    catalog = read(public / catalog_name)
    cases = read(public / 'cases-index.json')
    require(cases.get('curated') is True and {c['id'] for c in cases['cases']} == {r['id'] for r in catalog['records']},
            'Only the matching CURRENT public curated snapshot is eligible')
    approved = read(public / 'public-assets.json')
    require(approved.get('format') == 'interaction-twin-public-assets' and approved.get('version') == 1,
            'Exact approved public assets required')
    records = approved['assets']
    require(len(records) <= 20000 and len({r['path'] for r in records}) == len(records), 'Bounded unique public asset list required')
    wanted = {}
    omitted = []
    approved_by_path={r['path']:r for r in records}
    physics=physics_snapshot(root,approved_by_path)
    display_lineage=verify_display_lineage(root,approved_by_path,catalog=catalog)

    def add(source, target=None, *, role, http=True, expected=None):
        source = Path(source)
        reference = check(source, expected, root=root) if expected else ref(source, root=root)
        target = target or reference['path']
        require(not Path(target).is_absolute() and not any(part.startswith('.') for part in Path(target).parts),
                'No hidden/parent bundle paths')
        require(target not in wanted, 'Repeated explicit bundle target')
        wanted[target] = {'source': source, 'source_ref': reference, 'role': role, 'http': http}

    for item in records:
        require(item['path'].startswith('data/') and not any(part.startswith('.') for part in Path(item['path']).parts),
                'Approved data path required')
        check(root / item['path'], item, root=root)
        if omitted_provenance(item):
            omitted.append({**item, 'reason': 'Historical implementation provenance source text is not part of the standalone preview runtime.'})
        else:
            add(root / item['path'], role='curated_data_download_only_html' if item['path'].endswith('.html') else 'curated_data', expected=item)
    require(len(omitted) <= MAX_OMITTED_PROVENANCE_FILES and
            sum(row['bytes'] for row in omitted) <= MAX_OMITTED_PROVENANCE_BYTES,
            'Implementation-provenance omissions exceed the reviewed file/byte bounds')
    for name in static_dependencies(root):
        if name not in wanted: add(root / name, role='showcase_static')
    if physics['replay_ref']['path']not in wanted:add(root/physics['replay_ref']['path'],role='verified_saved_newton_replay',expected=physics['replay_ref'])
    for name in PUBLIC + [catalog_name]:
        target = (public / name).relative_to(root).as_posix()
        if target not in wanted: add(public / name, role='current_curated_snapshot')
    for case in cases['cases']:
        row = case.get('scene_ref')
        if row:
            require(row['path'] == 'scenes/' + row['sha256'] + '.json', 'Exact lazy public scene required')
            path = public / row['path']
            if path.relative_to(root).as_posix() not in wanted:
                add(path, role='public_lazy_scene', expected=row)
    for name in RUNTIME:
        if name not in wanted: add(root / name, role='read_only_scene_runtime')
    vendor = root / 'workspace/robot-vendor'
    vendor_manifest = read(vendor / 'manifest.json')
    for name, sha in vendor_manifest['files'].items():
        require(name in {'three.module.js', 'three.core.js', 'OrbitControls.js', 'GLTFLoader.js', 'BufferGeometryUtils.js', 'LICENSE'},
                'Only explicit pinned Three.js files may be included')
        require(ref(vendor / name, root=root)['sha256'] == sha, 'Pinned vendor dependency differs')
        add(vendor / name, role='vendored_threejs')
    add(vendor / 'manifest.json', role='vendored_threejs_provenance')
    for name in ['library_query.py', 'task_taxonomy.py']:
        add(root / name, role='pure_read_only_query_runtime', http=False)
    for name in ['preview_server.py', 'preview.html', 'README.txt']:
        add(root / 'judge_preview' / name, target=name, role='preview_entry' if name.endswith('.html') else 'preview_runtime_or_readme',
            http=name.endswith('.html'))
    robot_files, robot_index = robotics_snapshot(root, catalog)
    for target, raw in robot_files.items():
        require(target not in wanted, 'Generated robotics snapshot collides with a source file')
        wanted[target] = {'raw': raw, 'source_ref': {'path': target, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)},
                          'role': 'frozen_robot_starter' if target.endswith('.zip') else 'frozen_robotics_snapshot', 'http': True}
    require(len(wanted) <= 4000 and sum(row['source_ref']['bytes'] for row in wanted.values()) <= MAX_PREVIEW_BYTES,
            'Preview would exceed reviewed size/file budget')
    output.mkdir(parents=True, exist_ok=False)
    folder = output / 'Interaction-Twin-judge-preview'
    folder.mkdir()
    files = []
    for target, item in sorted(wanted.items()):
        # One verified in-memory byte snapshot is written; no symlink/hardlink.
        raw = item['raw'] if 'raw' in item else item['source'].read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item['source_ref']['sha256'] and len(raw) == item['source_ref']['bytes'],
                'Source changed before snapshot copy')
        destination = folder / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw)
        files.append({'path': target, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'http': item['http'], 'role': item['role'],
                      **({'generated_from_current_curated_snapshot': True} if 'raw' in item else {'original_source_ref': item['source_ref']})})
    for name, reference in source_refs.items():
        require(ref(public / name, root=root) == reference, 'Public snapshot changed while freezing bundle')
    manifest = {'format': 'interaction-twin-frozen-judge-preview', 'version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'read_only': True,
        'inference_available': False, 'mcp_runtime_included': False, 'admin_or_upload_routes_included': False,
        'current_curated_snapshot_refs': list(source_refs.values()) + [ref(public / catalog_name, root=root)],
        'curated_ids': [row['id'] for row in catalog['records']], 'files': files,
        'omitted_internal_provenance_refs': omitted, 'original_program_bytes_modified': False,
        'provenance_omission_policy': {'exact_curated_bytes_verified': True, 'maximum_files': MAX_OMITTED_PROVENANCE_FILES,
                                      'maximum_total_bytes': MAX_OMITTED_PROVENANCE_BYTES,'explicit_included_ancestor_source_text':INCLUDED_ANCESTOR_SOURCE_TEXT},
        'frozen_robotics_snapshot': {'index_path': 'showcase/demos/robotics/index.json',
                                      'ready_twin_ids': [task['twin_id'] for task in robot_index['tasks']],
                                      'scope': 'authored_kinematic_snapshot_not_physics'},
        'frozen_physics_snapshot':physics,'verified_display_lineage':display_lineage,
        'public_files_modified': False, 'physical_accuracy_verified': False, 'simulation_ready': False,
        'data_html_policy': 'exact original bytes; sandboxed attachment only, never executable preview UI',
        'total_file_bytes': sum(row['bytes'] for row in files),
        'builder_ref': ref(Path(__file__), root=root)}
    (folder / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    module_spec = importlib.util.spec_from_file_location('judge_preview_snapshot_check', root / 'judge_preview/preview_server.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.Snapshot(folder)
    archive = output / ('Interaction-Twin-' + output.name + '.zip')
    with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as zip_file:
        for path in [folder / row['path'] for row in files] + [folder / 'bundle-manifest.json']:
            name = 'Interaction-Twin-judge-preview/' + path.relative_to(folder).as_posix()
            zip_file.write(path, name)
    with ZipFile(archive) as zip_file:
        require(len(zip_file.namelist()) == len(files) + 1 and len(set(zip_file.namelist())) == len(files) + 1,
                'Portable ZIP inventory differs')
        require(zip_file.testzip() is None, 'ZIP CRC verification failed')
    result = {'format': 'interaction-twin-judge-preview-build', 'version': 1, 'status': 'frozen_local_preview_built_not_deployed',
        'archive_ref': ref(archive, root=root), 'manifest_ref': ref(folder / 'bundle-manifest.json', root=root),
        'file_count': len(files), 'total_file_bytes': manifest['total_file_bytes'],
        'curated_data_files_included': len(records) - len(omitted), 'internal_source_text_files_omitted': len(omitted),
        'curated_ids': manifest['curated_ids'], 'public_files_modified': False,
        'new_model_calls': 0, 'new_api_calls': 0, 'deployment_performed': False,
        'remaining_verification': 'Extract to an isolated directory; run its stdlib-only server and smoke-test exact file/hash/range/privacy contracts.'}
    (output / 'build-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/judge-preview-20261103-v1')
    print(json.dumps(build(parser.parse_args().output), indent=2))
