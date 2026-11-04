"""Owner-approved public snapshots, isolated from the working pipeline catalog.

Only this packager advances the public catalog. A model run, filesystem mtime,
or a newer reconstruction cannot select itself into the curated library.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = content if isinstance(content, bytes) else content.encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.curated-', delete=False) as stream:
        temporary = Path(stream.name); stream.write(raw)
    temporary.replace(path)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')+'\n'


def ref(path, root):
    path = Path(path)
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def local(root, relative):
    require(isinstance(relative, str) and relative.startswith('data/') and '\\' not in relative,
            'Published evidence must be an ordinary data artifact')
    path = root/relative
    require(all(not p.startswith('.') for p in Path(relative).parts) and path.resolve() == path and
            not any(p.is_symlink() for p in [path, *path.parents]) and path.is_file(),
            'Escaping, hidden, missing or symlink publication artifact')
    return path


def trim_record(record):
    record = copy.deepcopy(record)
    # Internal experiments, prompts, preparation jobs and private assessments
    # are not a public browsing feature, even for an approved source recording.
    record.update(other_saved_versions=[], initial_reconstruction_inputs=[], offline_queue=None, cached_insights=[])
    return record


def approved_reviews(root, reviews, bindings):
    """A same-byte mesh cannot substitute a different generated source anchor."""
    allowed = {row['component_id']: row for row in bindings}
    result = []
    for review in reviews:
        try:
            expected = allowed.get(review['component_id'])
            if expected is None or review['native_glb']['sha256'] != expected['native_glb']['sha256']:
                continue
            manifest_ref = review['review_manifest']
            manifest_path = local(root, manifest_ref['path'])
            require(ref(manifest_path,root) == manifest_ref, 'Review manifest changed')
            receipt_ref = read(manifest_path)['import_receipt']
            receipt_path = local(root, receipt_ref['path'])
            # Native display manifests use path/hash references; older
            # manifests additionally pin size. Enforce every declared field.
            require(set(receipt_ref) in ({'path','sha256'},{'path','sha256','bytes'}) and
                    {key:value for key,value in ref(receipt_path,root).items() if key in receipt_ref}==receipt_ref,
                    'Review import receipt changed')
            metadata = ref(local(root,(receipt_path.parent/'metadata.json').relative_to(root).as_posix()),root)
            require(metadata == expected['native_metadata'], 'Review comes from a different native source anchor')
            declared = next(row for row in read(receipt_path)['import_files'] if row['path']=='metadata.json')
            require(declared['sha256']==metadata['sha256'] and declared['bytes']==metadata['bytes'], 'Review native metadata differs')
            if not any(row['component_id']==review['component_id'] for row in result):
                result.append(review)
        except (ValueError, OSError, KeyError, TypeError, StopIteration):
            # The approved native asset remains inspectable even if an optional
            # independently saved display transport is missing or ineligible.
            continue
    return result


def presentation_metadata(root):
    """Read bounded editorial labels; never infer approval or model quality."""
    path=Path(root)/'config/library-presentation.json'
    require(not any(p.is_symlink() for p in [path,*path.parents]),
            'Library presentation config must not be a symlink')
    if not path.exists():
        return {}
    require(path.resolve()==path and path.is_file() and 0<path.stat().st_size<=64*1024,
            'Library presentation config must be an ordinary file up to 64 KiB')
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate library presentation config key')
            result[key]=value
        return result
    document=json.loads(path.read_text(),object_pairs_hook=unique_object)
    require(isinstance(document,dict) and set(document)=={'version','recordings'} and
            type(document['version']) is int and document['version']==1 and
            isinstance(document['recordings'],dict),'Invalid library presentation config')
    for source,labels in document['recordings'].items():
        require(isinstance(source,str) and re.fullmatch('[0-9a-f]{64}',source) and
                isinstance(labels,dict) and set(labels)=={'title','note'},
                'Invalid library presentation source or label fields')
        for key,maximum in [('title',120),('note',400)]:
            value=labels[key]
            require(isinstance(value,str) and value==value.strip() and 0<len(value)<=maximum and
                    all(ord(c)>=32 and ord(c)!=127 for c in value),
                    'Invalid library presentation '+key)
    return document['recordings']


def observed_surface_displays(root):
    """Explicit editorial pins, never automatic discovery of working surfaces."""
    path = Path(root)/'config/observed-surface-displays.json'
    if not path.exists():
        return {}
    require(path.resolve() == path and not any(p.is_symlink() for p in [path, *path.parents]) and
            0 < path.stat().st_size <= 256*1024, 'Ordinary bounded observed-surface display config required')
    value = read(path)
    require(set(value) == {'version', 'recordings'} and type(value['version']) is int and
            value['version'] == 1 and isinstance(value['recordings'], dict) and
            len(value['recordings']) <= 1000, 'Invalid observed-surface display pins')
    import source_visible_flexure_contract_v1 as F
    for source, pointer in value['recordings'].items():
        require(isinstance(source, str) and re.fullmatch('[0-9a-f]{64}', source), 'Invalid observed-surface source identity')
        require(isinstance(pointer,dict) and set(pointer) in ({'descriptor'},{'descriptor','presentation'}),'Invalid explicit observed-surface selection')
        F.reference(pointer['descriptor'])
        if pointer.get('presentation'):F.reference(pointer['presentation'])
    return value['recordings']


def source_motion_displays(root):
    """Exact approved motion pins; working predictions are never discovered."""
    path=Path(root)/'config/source-motion-displays.json'
    if not path.exists():return {}
    require(path.resolve()==path and not any(p.is_symlink()for p in [path,*path.parents]) and
            0<path.stat().st_size<=256*1024,'Ordinary bounded motion display config required')
    value=read(path)
    require(set(value)=={'version','recordings'} and type(value['version'])is int and value['version']==1 and
            isinstance(value['recordings'],dict) and len(value['recordings'])<=1000,'Invalid motion display pins')
    import motion_hypothesis_contract_v1 as T
    for source,selection in value['recordings'].items():
        require(isinstance(source,str)and re.fullmatch('[a-f0-9]{64}',source) and
                isinstance(selection,dict)and set(selection)=={'descriptor'},'Invalid explicit motion selection')
        T.reference(selection['descriptor'],8*1024*1024)
    return value['recordings']


def seed_existing(*, root=ROOT):
    """Pin today's already-showcased examples once, without inventing approval."""
    from agent_memory import private_directory, write_immutable
    root = Path(root).resolve(); folder = private_directory(root, 'admin-workflow')
    target = folder/'legacy-showcase.json'
    if target.exists():
        return read(target)
    catalog = read(root/'showcase/library-catalog.json')
    cases = {c['id']: c for c in read(root/'workspace/cases-index.json')['cases']}
    index_path = root/'showcase/aligned-scenes.json'
    scenes = read(index_path)['scenes'] if index_path.exists() else []
    records = []
    for source in catalog['records']:
        record = trim_record(source)
        case = copy.deepcopy(cases[record['id']])
        require(case['original_sha256'] == record['interaction']['sha256'], 'Legacy showcase program differs')
        aligned = next((s for s in scenes if s['source_sha256'] == record['source']['sha256'] and
            read(local(root,s['manifest']['path']))['original_program']['sha256'] == record['interaction']['sha256']), None)
        record['curation'] = {'status':'existing_showcase_example', 'human_approval_recorded':False}
        records.append({'record':record,'case':case,'aligned':aligned})
    value = {'format':'interaction-twin-legacy-showcase-pins','version':1,'records':records,
             'new_human_approval_fabricated':False}
    write_immutable(folder,'legacy-showcase',value)
    with target.open('x') as stream:
        stream.write(encoded(value))
    target.chmod(0o600)
    return value


def _approved_records(root):
    from admin_workflow import AdminWorkflow
    from scripts.build_robotics_library import build_record
    from twin_catalog import TwinCatalog
    import object_program
    labels_by_source=presentation_metadata(root)
    legacy_path=root/'.private/admin-workflow/legacy-showcase.json'
    legacy_pins=read(legacy_path)['records'] if legacy_path.is_file() else []
    private_index = {c['id']:c for c in read(root/'workspace/cases-index.json')['cases']}
    catalog = TwinCatalog(root)
    result=[]
    for publication in AdminWorkflow(root).public_index()['records']:
        # The workflow has verified this exact immutable manifest and approval.
        pointer=publication['manifest_ref'];private=root/pointer['path']
        require(private.resolve()==private and private.is_relative_to(root/'.private/admin-workflow') and
                not private.is_symlink() and sha(private)==pointer['sha256'] and private.stat().st_size==pointer['bytes'],
                'Approved private manifest changed')
        item = read(private)['manifest']
        output = item['result'] if 'result' in item else item
        interaction = output['interaction_ref']
        matching = next((c for c in private_index.values() if c['interaction_path']==interaction['path'] and
                         c['original_sha256']==interaction['sha256']),None)
        require(matching is not None, 'Approved program is not yet indexed for presentation')
        row, doc, digest = catalog.load(matching['id'])
        record = trim_record(build_record(root,row,doc,digest,private_index))
        # Reuse an explicitly frozen display transport when it describes the
        # same exact program and native binding. Working discovery can omit a
        # component supplied by the approved immutable companion (earbud A).
        for pin in legacy_pins:
            old=pin['record']
            if old.get('interaction')==interaction and old['source']['sha256']==doc['source']['sha256']:
                record['sam3d_reviews']+=copy.deepcopy(old.get('sam3d_reviews',[]))
        companion = output.get('object_program_ref')
        aligned = None
        source_bindings=[{'component_id':a['track_id'],'native_glb':{k:a[k] for k in ['path','sha256','bytes']},
                         'native_metadata':a['metadata']} for component in record['components']
                         for a in [component.get('asset')] if a and a.get('metadata')]
        if companion:
            document = object_program.read_json(object_program.checked_ref(companion,root))
            object_program.validate(document,root=root)
            require(document['evidence_program']==interaction,'Approved object program belongs to another procedure')
            # A constrained scene may cover only articulated parts. Keep other
            # native components from this exact approved procedure inspectable,
            # while an explicit companion binding always wins for its own ID.
            companion_ids={a['component_id'] for a in document['component_assets']}
            bindings=document['component_assets']+[a for a in source_bindings if a['component_id'] not in companion_ids]
            record['sam3d_reviews']=approved_reviews(root,record['sam3d_reviews'],bindings)
            aligned={'source_sha256':doc['source']['sha256'],'manifest':document['constrained_scene'],'object_program':companion}
        else:
            record['sam3d_reviews']=approved_reviews(root,record['sam3d_reviews'],source_bindings)
        declared=[r for r in output.get('repair_execution',[]) if
                  r.get('strategy') in {'declared_source_mask_refit_assembly_adoption_v1','declared_geometry_derivative_adoption_v1'}]
        require(len(declared)<=1,'Ambiguous approved derived assembly')
        if declared:
            import refined_assembly_contract as D
            reference=declared[0]['derived_assembly_ref']
            require(reference in output['artifacts'],'Derived assembly was not bound by this approval')
            derived=D.read(reference,root)
            require(companion is not None and derived['base_object_program_ref']==companion and
                    derived['evidence_program']==interaction and
                    derived['source']['sha256']==doc['source']['sha256'] and
                    derived['source_audit_summary']['default_eligible'] is True,
                    'Approved derivative belongs to another source or is ineligible')
            # Silhouette agreement alone cannot distinguish the inside and
            # outside of an open lid. Preserve older reviewed derivatives as
            # history; select only a version with explicit facing rejection.
            facing_audit=D.read(derived['source_alignment_audit_ref'],root)
            generic=derived.get('format')=='interaction-twin-declared-geometry-derivative'
            if generic or facing_audit.get('policy',{}).get('cavity_facing_ambiguity_gate')==(
                    'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away'):
                if generic:
                    import declared_geometry_contract_v1 as G
                    G.validate_files(derived,root=root)
                else:D.validate(derived,root=root)
                record['derived_assembly']=copy.deepcopy(reference)
                aligned={'source_sha256':doc['source']['sha256'],'manifest':derived['scene_ref'],
                         'object_program':companion,'derived_assembly':copy.deepcopy(reference)}
            else:
                record['assembly_refinement_status']='withheld_facing_ambiguity'
        record['curation']={'status':'owner_approved','human_approval_recorded':True,
                            'human_guidance_used':item.get('human_guidance_used') is True,
                            'human_review_override':publication.get('human_review_override') is True,
                            'revision_id':publication['revision_id'],'fingerprint':publication['fingerprint']}
        record['task_classes']=item.get('task_classes',[])
        labels=labels_by_source.get(record['source'].get('original_sha256'))
        if labels:
            record['public_title']=labels['title']
            record['public_note']=labels['note']
            from library_query import normalized
            record['search_index']['text']+=' '+normalized(labels['title'])
        result.append({'record':record,'case':copy.deepcopy(matching),'aligned':aligned})
    return result


def build(*, root=ROOT, bootstrap_existing=False):
    """Rebuild only from frozen legacy examples and currently approved revisions."""
    root=Path(root).resolve(); private=root/'.private/admin-workflow/legacy-showcase.json'
    if bootstrap_existing:
        seed_existing(root=root)
    entries=read(private)['records'] if private.exists() else []
    approved=_approved_records(root)
    families={r['record']['source']['original_sha256']:r for r in entries}
    for entry in approved:
        families[entry['record']['source']['original_sha256']]=entry
    chosen=sorted(families.values(),key=lambda e:(e['record']['title'].casefold(),e['record']['id']))
    out=root/'showcase/public';out.mkdir(parents=True,exist_ok=True)
    policy_path=root/'config/library-display-policy.json'
    require_connected=False
    if policy_path.exists():
        require(not policy_path.is_symlink() and policy_path.stat().st_size<=4096,
                'Invalid library display policy')
        policy=read(policy_path)
        require(set(policy)=={'version','require_connected_scene'} and type(policy['version'])is int and
                policy['version']==1 and type(policy['require_connected_scene'])is bool,
                'Invalid library display policy')
        require_connected=policy['require_connected_scene']
    # A new approved procedure is a revision of the same recording. Keep its
    # public URL/MCP identity while replacing only the exact approved contents.
    # Working candidate IDs must never decide the identity of an existing twin.
    prior_ids={}
    if (out/'library-index.json').exists():
        index=read(out/'library-index.json');pointer=index.get('catalog',{})
        require(re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json',pointer.get('path','')) and
                pointer['path']=='library-catalog-'+str(pointer.get('sha256'))+'.json',
                'Invalid previous public catalog identity')
        previous=out/pointer['path']
        require(previous.is_file() and not previous.is_symlink() and
                sha(previous)==pointer['sha256'] and previous.stat().st_size==pointer['bytes'],
                'Previous public catalog changed')
        for record in read(previous)['records']:
            source=record['source']['original_sha256']
            require(source not in prior_ids,'Ambiguous previous recording identity')
            prior_ids[source]=record['id']
    assets={}
    def add(relative, expected=None, size=None):
        path=local(root,relative)
        actual=ref(path,root)
        require(expected is None or expected==actual['sha256'],'Approved artifact changed')
        require(size is None or size==actual['bytes'],'Approved artifact size changed')
        assets[relative]=actual
    def references(value):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and value['path'].startswith('data/'):
                add(value['path'],value.get('sha256'),value.get('bytes'))
            for key,child in value.items():
                if key in {'frame_path','mask_path','video_path','poster_path'} and isinstance(child,str) and child.startswith('data/'):
                    add(child)
                elif isinstance(child,(dict,list)):
                    references(child)
        elif isinstance(value,list):
            for child in value:references(child)
    import semantic_relation_corrections_v1 as S
    semantic_pins=S.selections(root);semantic_expirations=[]
    records=[];cases=[];scenes=[];surface_pins=observed_surface_displays(root);motion_pins=source_motion_displays(root)
    for entry in chosen:
        record,case=copy.deepcopy(entry['record']),copy.deepcopy(entry['case'])
        record.pop('observed_surfaces',None);record.pop('observed_surface_presentation',None)
        record.pop('motion_hypothesis',None)
        require(not require_connected or entry.get('aligned'),
                'Connected scene required before a recording enters this library')
        working_id=case['id']
        record['id']=prior_ids.get(record['source']['original_sha256'],record['id'])
        case['id']=record['id']
        if record.get('scene_case'):
            require(record['scene_case']['original_sha256']==record['interaction']['sha256'],
                    'Browser scene identity belongs to another approved procedure')
            record['scene_case']['id']=record['id']
        references(record)
        program=read(local(root,record['interaction']['path']))
        from guarded_task_metadata_v1 import derive as derive_task_classes
        object_doc=read(local(root,entry['aligned']['object_program']['path'])) if entry.get('aligned',{} ) and entry['aligned'].get('object_program') else None
        connected_doc=read(local(root,entry['aligned']['manifest']['path'])) if entry.get('aligned') else None
        record['task_classes']=derive_task_classes(program,object_program=object_doc,scene=connected_doc)
        semantic=semantic_pins.get(record['source']['sha256'])
        if semantic:
            pointer=semantic['descriptor'];descriptor=json.loads(S.read_exact(root,pointer))
            binding=S.revision_binding(descriptor)
            require(descriptor['source']['sha256']==record['source']['sha256'],'Semantic selector source differs')
            current_binding={k:record['curation'][k]for k in ['revision_id','fingerprint']}
            if binding!=current_binding:
                # Expiration is this one exact lifecycle condition. Invalid
                # descriptor bytes, policy and current bindings still fail.
                semantic_expirations.append({'source_sha256':record['source']['sha256'],'artifact_sha256':pointer['sha256'],'reason':'approved_revision_changed','previous_revision_binding':binding,'current_revision_binding':current_binding})
            else:
                review=S.validate(descriptor,record,program,entry['aligned']['manifest'],lambda r:S.read_exact(root,r))
                record=S.project(record,descriptor,pointer)
                for r in [pointer,*S.artifacts(descriptor,review)]:references(r)
        from library_query import derive_index
        record['search_index']=derive_index(record)
        # Source evidence is public; arbitrary compiler inputs, model prompts,
        # worker logs and private feedback are deliberately not traversed.
        references(program['source']);references(program['evidence']);references(program['assets'])
        case['exports']={};case['curated']=True
        if case.get('scene_ref'):
            scene_ref=case['scene_ref'];original=root/'workspace'/scene_ref['path']
            require(re.fullmatch(r'scenes/[a-f0-9]{64}\.json',scene_ref['path']) and
                    scene_ref['path']=='scenes/'+scene_ref['sha256']+'.json' and original.resolve()==original and
                    not any(p.is_symlink() for p in [original,*original.parents]),'Invalid curated scene path')
            require(original.is_file() and sha(original)==scene_ref['sha256'] and original.stat().st_size==scene_ref['bytes'],
                    'Approved browser scene changed')
            if working_id!=case['id']:
                pack=read(original)
                require(pack.get('case_id')==working_id and pack.get('interaction_sha256')==case['original_sha256'],
                        'Approved browser scene belongs to another revision')
                pack['case_id']=case['id'];raw=encoded(pack).encode()
                scene_ref={'path':'scenes/'+hashlib.sha256(raw).hexdigest()+'.json',
                           'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
                case['scene_ref']=scene_ref;atomic(out/scene_ref['path'],raw)
                if record.get('scene_case'):record['scene_case']['scene_ref']=scene_ref
            else:atomic(out/scene_ref['path'],original.read_bytes())
        if entry.get('aligned'):
            aligned=copy.deepcopy(entry['aligned']);aligned.pop('observed_surfaces',None);aligned.pop('observed_surface_presentation',None);aligned.pop('motion_hypothesis',None);scenes.append(aligned)
            scene=read(local(root,aligned['manifest']['path']))
            require(scene['original_program']==record['interaction'] and
                    scene['source']['sha256']==record['source']['sha256'],
                    'Connected scene belongs to another approved recording or program')
            surface_selection=surface_pins.get(record['source']['sha256'])
            if surface_selection:
                surface_pointer=surface_selection['descriptor']
                import source_visible_flexure_contract_v1 as F
                checked=F.validate_descriptor(surface_pointer,root=root)
                surface=checked['document']
                require(surface['source_program']==record['interaction'] and
                        surface['source_scene']==aligned['manifest'],
                        'Observed-surface display differs from the approved scene or program')
                # Full GLB/source-pixel reproduction precedes any new artifact
                # route or public pointer. The stdlib transport never refits.
                from source_visible_flexure_verify_v1 import validate_surface_files
                proof=validate_surface_files(surface_pointer,root=root)
                require(proof.get("all_passed") is True,"Observed-surface geometry/source proof failed")
                for pointer in checked['artifacts']:
                    add(pointer['path'],pointer['sha256'],pointer['bytes'])
                record['observed_surfaces']=copy.deepcopy(surface_pointer)
                aligned['observed_surfaces']=copy.deepcopy(surface_pointer)
                if surface_selection.get('presentation'):
                    from observed_surface_presentation_v1 import validate as validate_presentation
                    presentation=surface_selection['presentation'];actual=ref(local(root,presentation['path']),root)
                    require(actual==presentation,'Observed presentation bytes changed')
                    validate_presentation(read(local(root,presentation['path'])),surface,surface_pointer)
                    add(presentation['path'],presentation['sha256'],presentation['bytes'])
                    record['observed_surface_presentation']=copy.deepcopy(presentation)
                    aligned['observed_surface_presentation']=copy.deepcopy(presentation)
            motion_selection=motion_pins.get(record['source']['sha256'])
            if motion_selection:
                import motion_hypothesis_contract_v1 as T
                from incremental_motion_publication_v1 import validate_selection,historical_catalog
                pointer=motion_selection['descriptor'];actual=ref(local(root,pointer['path']),root)
                require(actual==pointer,'Motion descriptor bytes changed')
                def motion_read(reference,maximum=T.MAX_JSON):
                    path=local(root,reference['path']);actual=ref(path,root)
                    require(actual==reference and actual['bytes']<=maximum,'Motion source artifact differs or exceeds budget')
                    return path.read_bytes()
                motion=read(local(root,pointer['path']));checked=validate_selection(motion,record,aligned,motion_read)
                historical_catalog(motion,root)
                add(pointer['path'],pointer['sha256'],pointer['bytes'])
                for artifact in checked['artifacts']:add(artifact['path'],artifact['sha256'],artifact['bytes'])
                record['motion_hypothesis']=copy.deepcopy(pointer);aligned['motion_hypothesis']=copy.deepcopy(pointer)
            references(aligned)
            for component in scene['components']:
                references(component['asset']);references(component['native_metadata'])
                for pose in component['poses']:
                    for key in ['source_mask','source_rgb']:
                        if pose.get(key):references(pose[key])
            if aligned.get('object_program'):
                companion=read(local(root,aligned['object_program']['path']))
                for key in ['evidence_program','source_scene','constrained_scene','mechanics','proposal','component_assets','implementation_provenance']:
                    references(companion[key])
            if aligned.get('derived_assembly'):
                import refined_assembly_contract as D
                derived=D.read(aligned['derived_assembly'],root)
                if derived.get('format')=='interaction-twin-declared-geometry-derivative':
                    import declared_geometry_contract_v1 as G
                    G.validate_files(derived,root=root)
                else:D.validate(derived,root=root)
                references(derived)
                # These exact reports/lineage files are needed to verify the
                # declared derivative in browsers without executing fit code.
                report=read(local(root,derived['refinement_report_ref']['path']))
                references(report)
                if derived.get('format')=='interaction-twin-declared-geometry-derivative':
                    # Walk only typed, hash-bound derivative lineage. No worker
                    # prompts, logs or arbitrary JSON discovery enters public data.
                    pending=[derived['baseline_scene_ref'],derived['source_alignment_audit_ref']]
                    pending += derived.get('visual_review_refs',[])
                    for key in ['parent_derived_assembly_ref','nonrigid_observations_ref','source_appearance_states_ref']:
                        if derived.get(key):pending.append(derived[key])
                    for asset in report['assets']:
                        for operation in asset['operations']:
                            if operation.get('projection_report_ref'):pending.append(operation['projection_report_ref'])
                    seen=set()
                    while pending:
                        pointer=pending.pop(); key=pointer['path']
                        if key in seen:continue
                        seen.add(key); references(pointer)
                        document=read(local(root,key));references(document)
                        if document.get('format')=='interaction-twin-declared-derived-assembly':
                            pending += [document[k] for k in ['base_object_program_ref','refinement_report_ref','source_alignment_audit_ref','scene_ref']]
                        if document.get('format')=='interaction-twin-native-assembly-refinement':
                            pending += [document[k] for k in ['original_scene','scene_ref']]
                        if document.get('format')=='interaction-twin-object-program':
                            pending += [document[k] for k in ['proposal','constrained_scene']]
                        if document.get('format')=='interaction-twin-assembly-image-critic-review':
                            pending += [document['receipt'],document['model_result']]
                        if document.get('format')=='interaction-twin-nonrigid-observations':
                            pending.append(document['trace_ref'])
                        if document.get('format')=='interaction-twin-source-appearance-states':
                            # G.validate_files above reproduced every source RGB
                            # patch before any bytes are admitted for GET.
                            pending.append(document['previous_derived_assembly_ref'])
        records.append(record);cases.append(case)
    require(len({r['id'] for r in records})==len(records),'Ambiguous public twin identity')
    selected_assets={c['asset']['sha256'] for r in records for c in r['components'] if c['asset']}
    from library_query import CONTRACT
    document={'format':'interaction-twin-object-program-library','version':1,'selection_policy':'Owner-approved immutable revisions, plus explicitly pinned existing showcase examples.',
      'read_only':True,'no_inference':True,'published':False,'physical_accuracy_verified':False,'simulation_ready':False,
      'summary':{'recordings':len(records),'native_asset_drafts':len(selected_assets),'components':sum(len(r['components']) for r in records),
                 'steps':sum(len(r['steps']) for r in records),'eligible_saved_versions':len(records)},
      'records':records,'excluded_records':[],'query_contract':CONTRACT,
      'provenance':{'builder_sha256':sha(__file__),'curation':'explicit_owner_approval_no_latest_file_selection'}}
    if semantic_expirations:document['provenance']['expired_semantic_corrections']=semantic_expirations
    raw=encoded(document);digest=hashlib.sha256(raw.encode()).hexdigest();catalog_name='library-catalog-'+digest+'.json'
    index={'format':'interaction-twin-library-index','version':1,'catalog':{'path':catalog_name,'sha256':digest,'bytes':len(raw.encode())},'summary':document['summary']}
    # Evidence admission first, immutable content second, public pointers last.
    asset_index={'format':'interaction-twin-public-assets','version':1,'assets':sorted(assets.values(),key=lambda a:a['path'])}
    atomic(out/'public-assets.json',encoded(asset_index));atomic(out/catalog_name,raw);atomic(out/'library-catalog.json',raw)
    atomic(out/'aligned-scenes.json',encoded({'format':'sam3d-aligned-scene-index','version':1,'scenes':scenes}))
    public_cases={'version':'0.1','cases':cases,'build_errors':[],'network_dependencies':False,'physical_accuracy_verified':False,'lazy_scenes':True,'curated':True}
    atomic(out/'cases-index.json',encoded(public_cases));atomic(out/'cases-index.js','window.TWIN_CASES='+encoded(public_cases)+';\n')
    atomic(out/'library-index.json',encoded(index));atomic(out/'library-index.js','window.INTERACTION_LIBRARY_INDEX='+encoded(index)+';\n')
    return {'status':'curated_snapshot_ready','records':len(records),'approved_records':len(approved),'assets':len(assets),
            'catalog_sha256':digest,'new_api_calls':0,'new_model_calls':0,'expired_semantic_corrections':semantic_expirations}


def query(request=None, *, facets=False, root=ROOT):
    from library_query import query_records
    path=Path(root)/'showcase/public/library-catalog.json'
    records=read(path)['records'] if path.is_file() else []
    result=query_records(records,**(request or {}))
    if facets:
        return {'facets':result['facets'],'query_contract':result['query_contract'],'recordings':result['total'],'curated_only':True}
    result['curated_only']=True
    return result
