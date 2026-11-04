"""MCP view of the exact owner-curated human library; no working-catalog scan.

The ordinary TwinCatalog remains available to explicitly configured owner tools.
This adapter never discovers a newer candidate, companion, analysis or diagnostic
for an already-approved source merely because it exists on disk.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from library_query import backend_key, query_records
from twin_catalog import TwinCatalog, ROOT


class _ProgramValidator(TwinCatalog):
    """Internal validation only; never exposed as an MCP tool or resource."""
    def __init__(self, root, rows):
        super().__init__(root); self.rows = rows
    def entries(self): return self.rows


class CuratedTwinCatalog(TwinCatalog):
    curated = True

    def __init__(self, root=ROOT):
        super().__init__(root); self._manifest_cache = None; self._verified = {}

    def _json(self, relative, maximum=8*1024*1024):
        path=self.root/relative
        if (path.resolve()!=path or any(p.is_symlink() for p in [path,*path.parents]) or
                not path.is_file() or not 0<path.stat().st_size<=maximum):
            raise ValueError('Curated snapshot is unavailable; no private catalog fallback')
        raw=path.read_bytes()
        if len(raw)>maximum:raise ValueError('Curated snapshot exceeds its bounded size')
        value=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Invalid curated JSON')))
        if not isinstance(value,dict):raise ValueError('Invalid curated snapshot contract')
        return value,raw

    def _snapshot(self):
        index,_=self._json('showcase/public/library-index.json')
        reference=index.get('catalog',{});name=reference.get('path','')
        if (index.get('format')!='interaction-twin-library-index' or index.get('version')!=1 or
                not re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json',name) or
                name!='library-catalog-'+str(reference.get('sha256'))+'.json'):
            raise ValueError('Invalid curated catalog pointer')
        catalog,raw=self._json('showcase/public/'+name)
        if len(raw)!=reference.get('bytes') or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('Curated catalog bytes changed')
        cases,_=self._json('showcase/public/cases-index.json')
        records=catalog.get('records');rows=cases.get('cases')
        if (catalog.get('format')!='interaction-twin-object-program-library' or catalog.get('version')!=1 or
                cases.get('curated') is not True or not isinstance(records,list) or not isinstance(rows,list) or
                len(records)>1000 or len(rows)>1000):
            raise ValueError('Invalid curated library/case contract')
        by_id={r['id']:r for r in records};case_ids={r['id'] for r in rows}
        if len(by_id)!=len(records) or len(case_ids)!=len(rows) or set(by_id)!=case_ids:
            raise ValueError('Curated catalog and cases do not describe the same approvals')
        for row in rows:
            record=by_id[row['id']];program=record.get('interaction',{})
            if row.get('interaction_path')!=program.get('path') or row.get('original_sha256')!=program.get('sha256'):
                raise ValueError('Curated case/program binding differs')
            curation=record.get('curation',{})
            if curation.get('status') not in {'owner_approved','existing_showcase_example'}:
                raise ValueError('Unapproved record in curated library')
        return records,rows

    def _assets(self):
        path=self.root/'showcase/public/public-assets.json'
        if not path.is_file() or path.is_symlink():raise ValueError('Curated artifact manifest is unavailable')
        info=path.stat();identity=(info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns,info.st_size)
        if self._manifest_cache and self._manifest_cache[0]==identity:return self._manifest_cache[1]
        document,_=self._json('showcase/public/public-assets.json')
        items=document.get('assets')
        if document.get('format')!='interaction-twin-public-assets' or document.get('version')!=1 or not isinstance(items,list) or len(items)>20000:
            raise ValueError('Invalid curated artifact manifest')
        result={}
        for item in items:
            relative=item.get('path')
            if (not isinstance(relative,str) or not relative.startswith('data/') or '\\' in relative or
                    any(p.startswith('.') for p in Path(relative).parts) or relative in result or
                    not re.fullmatch(r'[a-f0-9]{64}',item.get('sha256','')) or type(item.get('bytes')) is not int or item['bytes']<=0):
                raise ValueError('Invalid curated artifact entry')
            result[relative]=item
        self._manifest_cache=(identity,result)
        return result

    def _checked(self, reference, *, maximum=None):
        if not isinstance(reference,dict):raise ValueError('Expected an exact curated artifact reference')
        path=self.data_path(reference.get('path'))
        declared=self._assets()[reference['path']]
        if any(reference.get(k)!=declared[k] for k in ['path','sha256','bytes']):
            raise ValueError('Artifact is not the exact curated revision')
        if maximum is not None and declared['bytes']>maximum:raise ValueError('Curated resource exceeds its size budget')
        return path

    def data_path(self, relative):
        if not isinstance(relative,str) or relative not in self._assets():
            raise ValueError('Artifact is not published in the curated library')
        path=self.root/relative;declared=self._assets()[relative]
        if path.resolve()!=path or any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file():
            raise ValueError('Curated artifact path is unavailable')
        info=path.stat()
        if info.st_size!=declared['bytes']:raise ValueError('Curated artifact bytes changed')
        identity=(relative,info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns,info.st_size,declared['sha256'])
        if identity not in self._verified:
            h=hashlib.sha256()
            with path.open('rb')as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=declared['sha256']:raise ValueError('Curated artifact bytes changed')
            if len(self._verified)>4096:self._verified.clear()
            self._verified[identity]=True
        return path

    def _bytes(self, reference, *, maximum):
        raw=self._checked(reference,maximum=maximum).read_bytes()
        if len(raw)!=reference['bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('Curated artifact changed during resource read')
        return raw

    def entries(self): return self._snapshot()[1]

    def _record(self, twin_id):
        records,rows=self._snapshot()
        record=next((r for r in records if r['id']==twin_id),None)
        if record is None:raise ValueError('Unknown curated twin; discover approved IDs with search_twins')
        return record,next(r for r in rows if r['id']==twin_id)

    def load(self, twin_id):
        record,row=self._record(twin_id)
        self._checked(record['interaction'])
        # The original compiler may validate its private internal inputs. Those
        # are not resource routes, never discovered, and never returned as bytes.
        validated,doc,digest=_ProgramValidator(self.root,[row]).load(twin_id)
        if digest!=record['interaction']['sha256']:raise ValueError('Curated original program changed')
        self.data_path(doc['source']['path'])
        for asset in doc['assets']:self.data_path(asset['path'])
        for frame in doc['evidence']:
            self.data_path(frame['frame_path'])
            for observation in frame.get('observations',[]):self.data_path(observation['mask_path'])
        return validated,doc,digest

    def library_search(self, query='', limit=10, *, automatic_only=True, **options):
        query_records([],query,limit=limit,**options)  # Validate before reading snapshots.
        records,_=self._snapshot()
        rows=[]
        for record in records:
            row=copy.deepcopy(record)
            row.update(program_sha256=record['interaction']['sha256'],program_uri=f"twin://{record['id']}/program")
            rows.append(row)
        result=query_records(rows,query,limit=limit,**options)
        result['records']=[{k:r[k] for k in ['id','title','summary','source','duration_s','counts','program_uri','program_sha256','search_index','task_classes','curation'] if k in r}
            | {'saved_variant_ids':[r['id']],'simulation_ready':False,'physical_accuracy_verified':False} for r in result['records']]
        return result|{'access':'owner_curated_public_snapshot','metadata_only':True,'artifact_inspection_required':True,
            'eligibility':'curated_approvals_and_explicit_legacy_pins_only','policy':self.policy()}

    def _task_brief(self, *args):
        return {'status':'not_available','scope':'private_cached_analyses_excluded_from_curated_library',
            'cached_only':True,'new_api_call':False,'physical_accuracy_verified':False,'simulation_ready':False}

    def describe(self,twin_id):
        result=super().describe(twin_id);record,_=self._record(twin_id)
        result['curation']=copy.deepcopy(record['curation'])
        correction=self.semantic_correction(twin_id)
        if correction:
            result['original_relationships']=copy.deepcopy(result['relationships'])
            result['relationships']=copy.deepcopy(record['relationships'])
            result['semantic_correction']={'artifact':copy.deepcopy(correction['artifact']),
                'scope':'source_reviewed_candidate_host_correction_not_verified_attachment',
                'inspection_tool':'list_derived_artifacts','original_program_uri':f'twin://{twin_id}/program',
                'human_reviewed':False,'physical_accuracy_verified':False}
        if record.get('observed_surfaces'):
            result['observed_surfaces']={'artifact':copy.deepcopy(record['observed_surfaces']),
                'scope':'source_visible_monocular_depth_observations',
                'inspection_tool':'get_observed_surfaces','original_native_program_preserved':True,
                'hidden_surfaces':'absent','physical_accuracy_verified':False,'simulation_ready':False}
        if record.get('motion_hypothesis'):
            result['motion_hypothesis']={'artifact':copy.deepcopy(record['motion_hypothesis']),
                'scope':'separate_source_conditioned_pose_hypotheses',
                'inspection_tool':'get_motion_hypotheses','original_observations_preserved':True,
                'source_unknowns_preserved':True,'native_geometry_modified':False,
                'physical_accuracy_verified':False,'simulation_ready':False}
        if record.get('derived_assembly'):
            result['derived_assembly']={'artifact':copy.deepcopy(record['derived_assembly']),
                'scope':'declared_source_mask_refit_geometric_derivative',
                'original_native_program_preserved':True,'physical_accuracy_verified':False,
                'simulation_ready':False}
        return result

    def semantic_correction(self,twin_id):
        """Exact currently selected source-review overlay; never discover one."""
        record,_=self._record(twin_id);reference=record.get('semantic_correction')
        if reference is None:return None
        import semantic_relation_corrections_v1 as S
        document=json.loads(self._bytes(reference,maximum=128*1024))
        program=json.loads(self._bytes(record['interaction'],maximum=8*1024*1024))
        index,_=self._json('showcase/public/aligned-scenes.json')
        rows=[row for row in index['scenes']if row.get('source_sha256')==record['source']['sha256']]
        if len(rows)!=1:raise ValueError('Semantic correction current scene is ambiguous')
        self._checked(rows[0]['manifest'],maximum=8*1024*1024)
        review=S.validate(document,record,program,rows[0]['manifest'],lambda r:self._bytes(r,maximum=8*1024*1024))
        # Reproduce every effective relation from its original program plus the
        # declared exact change; arbitrary corrected catalog text is not trusted.
        base=copy.deepcopy(record);base.pop('semantic_correction')
        originals={r['id']:r for r in program['relationships']}
        base['relationships']=[{**copy.deepcopy(originals[r['id']]),'citations':copy.deepcopy(r['citations'])}for r in record['relationships']]
        expected=S.project(base,document,reference)
        if expected['relationships']!=record['relationships']:raise ValueError('Effective semantic projection changed')
        return {'artifact':copy.deepcopy(reference),'document':document,'review':review}

    def simulation_manifest(self,twin_id):
        result=super().simulation_manifest(twin_id);record,_=self._record(twin_id)
        correction=self.semantic_correction(twin_id)
        if correction:
            result['original_candidate_relationships']=copy.deepcopy(result['candidate_relationships'])
            result['candidate_relationships']=copy.deepcopy(record['relationships'])
            draft=result.get('preparation_draft')
            if draft is not None:
                draft['original_candidate_relationships']=copy.deepcopy(draft['candidate_relationships'])
                draft['candidate_relationships']=copy.deepcopy(record['relationships'])
                draft['semantic_correction']=copy.deepcopy(correction['artifact'])
            result['semantic_correction']={'artifact':copy.deepcopy(correction['artifact']),
                'scope':'source_reviewed_candidate_host_correction_not_verified_attachment',
                'original_program_preserved':True,'human_reviewed':False,'physical_accuracy_verified':False}
        return result

    def _kinematics_descriptor(self,twin_id,document,digest):
        result=copy.deepcopy(super()._kinematics_descriptor(twin_id,document,digest))
        correction=self.semantic_correction(twin_id)
        if correction:
            result['original_removable_relationships']=copy.deepcopy(result.get('removable_relationships',[]))
            for edge in result.get('removable_relationships',[]):
                for c in correction['document']['corrections']:
                    old,new=c['original'],c['replacement']
                    if edge.get('child_id')==old['subject_track_id']and edge.get('parent_id')==old['object_track_id']:
                        edge['parent_id']=new['object_track_id'];edge['semantic_correction']=copy.deepcopy(correction['artifact'])
            result['semantic_correction']={'artifact':copy.deepcopy(correction['artifact']),'original_object_program_preserved':True,'scope':'source_reviewed_candidate_host_correction_not_verified_attachment'}
        return result

    def kinematics(self,twin_id):
        result=super().kinematics(twin_id);correction=self.semantic_correction(twin_id)
        if correction:
            record,_=self._record(twin_id)
            result['semantic_correction']={'artifact':copy.deepcopy(correction['artifact']),'original_object_program_preserved':True,'scope':'Apply this declared overlay for current candidate relationships; object_program is the unchanged original.'}
            result['effective_candidate_relationships']=copy.deepcopy(record['relationships'])
        return result

    def derived_assembly(self,twin_id):
        """Return only a pinned, approved geometric derivative; never discover one."""
        record,_=self._record(twin_id)
        reference=record.get('derived_assembly')
        if reference is None:return None
        import declared_assembly_view as V
        document=json.loads(self._bytes(reference,maximum=1024*1024))
        # Full geometry/pixel validation runs at publication. Transport checks
        # the pinned lineage and audit without CV dependencies or refitting.
        V.validate(document,self._bytes)
        if (document['evidence_program']!=record['interaction'] or
                document['source']['sha256']!=record['source']['sha256'] or
                document['source_audit_summary']['default_eligible'] is not True):
            raise ValueError('Curated derived assembly belongs to another source or is ineligible')
        for field in ['base_object_program_ref','refinement_report_ref','scene_ref','source_alignment_audit_ref']:
            self._checked(document[field],maximum=8*1024*1024)
        audit=json.loads(self._bytes(document['source_alignment_audit_ref'],maximum=8*1024*1024))
        if document.get('format')!='interaction-twin-declared-geometry-derivative' and audit.get('policy',{}).get('cavity_facing_ambiguity_gate')!=(
                'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away'):
            raise ValueError('Curated derivative lacks a source-facing ambiguity gate')
        for binding in document['component_assets']:
            for field in ['native_glb','original_native_glb','native_metadata']:
                self._checked(binding[field],maximum=16*1024*1024)
        return {'artifact':copy.deepcopy(reference),'document':document}

    def _kinematics_record(self, document, digest):
        import object_program
        records,_=self._snapshot()
        record=next((r for r in records if r['interaction']['sha256']==digest and r['source']['sha256']==document['source']['sha256']),None)
        if record is None:raise ValueError('Program is not in the current curated snapshot')
        index,_=self._json('showcase/public/aligned-scenes.json')
        if index.get('format')!='sam3d-aligned-scene-index' or index.get('version')!=1 or not isinstance(index.get('scenes'),list):
            raise ValueError('Invalid curated kinematics registry')
        for entry in index['scenes']:
            if entry.get('source_sha256')!=document['source']['sha256'] or not entry.get('object_program'):continue
            ref=entry['object_program'];raw=self._bytes(ref,maximum=6*1024*1024);value=json.loads(raw)
            declared=self.derived_assembly(record['id']) if entry.get('derived_assembly') else None
            selected_scene=(declared['document']['scene_ref'] if declared else value.get('constrained_scene'))
            if (value.get('evidence_program')!=record['interaction'] or selected_scene!=entry.get('manifest') or
                    declared and (declared['artifact']!=entry['derived_assembly'] or
                                  declared['document']['base_object_program_ref']!=ref)):
                raise ValueError('Curated kinematics belongs to a different approved revision')
            for name in ['source_scene','constrained_scene','mechanics','proposal']:
                self._checked(value[name],maximum=6*1024*1024)
            object_program.validate(value,root=self.root)
            return {'artifact':ref,'document':value}
        return None

    def kinematics_artifact(self,twin_id,artifact_id='object-program'):
        allowed={'object-program','source-scene','constrained-scene','mechanics','proposal'}
        if artifact_id not in allowed:raise ValueError('Unknown curated kinematics artifact ID')
        _,doc,digest=self.load(twin_id);found=self._kinematics_record(doc,digest)
        if found is None:raise ValueError('No approved descriptive kinematics available')
        ref=found['artifact'] if artifact_id=='object-program' else found['document'][artifact_id.replace('-','_')]
        return self._bytes(ref,maximum=6*1024*1024)

    def _candidate(self,twin_id,component_id):
        _,doc,_=self.load(twin_id);record,_=self._record(twin_id)
        if component_id not in {t['id'] for t in doc['tracklets']}:raise ValueError('Unknown approved component')
        candidate=next((r for r in record.get('sam3d_reviews',[]) if r['component_id']==component_id),None)
        if candidate is None:raise ValueError('No curated SAM 3D candidate for this component')
        if candidate.get('source_sha256')!=doc['source']['sha256'] or candidate.get('selected') is not False or candidate.get('physical_accuracy_verified') is not False:
            raise ValueError('Curated generated candidate source or policy differs')
        return candidate,doc

    def generated_candidate(self,twin_id,component_id):
        candidate,doc=self._candidate(twin_id,component_id);record,_=self._record(twin_id);artifacts={}
        for name,field,mime in [('asset','native_glb','model/gltf-binary'),('source-image','source_image','image/png'),('source-mask','source_mask','image/png'),('review-manifest','review_manifest','application/json')]:
            ref=candidate[field];self._checked(ref,maximum=(16 if name=='asset' else 8)*1024*1024)
            artifacts[name]={'sha256':ref['sha256'],'bytes':ref['bytes'],'mime_type':mime,'uri':f'twin://{twin_id}/native-candidates/{component_id}/artifacts/{name}'}
        selected=any(a['track_id']==component_id and a['sha256']==candidate['native_glb']['sha256'] and backend_key(a.get('backend',''))=='sam3d' for a in doc['assets'])
        return {'twin_id':twin_id,'component_id':component_id,'program_sha256':record['interaction']['sha256'],
            'source_sha256':candidate['source_sha256'],'backend':candidate['backend'],'source_sample_index':candidate['source_sample_index'],
            'source_timestamp_s':candidate['source_timestamp_s'],'geometry_preference':'sam3d_saved_program_asset' if selected else 'sam3d_candidate_for_inspection',
            'artifacts':artifacts,'selected_in_saved_program':selected,'saved_program_assets_changed':False,'existing_poses_reused':False,
            'source_video_alignment_verified':False,'physical_accuracy_verified':False,'simulation_ready':False,'policy':self.policy()}

    def generated_candidate_artifact(self,twin_id,component_id,artifact_id):
        candidate,_=self._candidate(twin_id,component_id)
        field={'asset':'native_glb','source-image':'source_image','source-mask':'source_mask','review-manifest':'review_manifest'}.get(artifact_id)
        if field is None:raise ValueError('Unknown curated candidate artifact ID')
        return self._bytes(candidate[field],maximum=(16 if artifact_id=='asset' else 8)*1024*1024)

    def asset(self,twin_id,track_id,binary=False):
        if binary:
            _,doc,_=self.load(twin_id)
            asset=next((a for a in doc['assets'] if a['track_id']==track_id),None)
            if asset is None:raise ValueError('Unknown curated generated asset')
            reference=self._assets().get(asset['path'])
            if reference is None or reference['sha256']!=asset['sha256']:raise ValueError('Asset is not the approved revision')
            if Path(reference['path']).suffix.lower()!='.glb':raise ValueError('Only declared GLBs are available')
            return self._bytes(reference,maximum=16*1024*1024)
        result=super().asset(twin_id,track_id,binary=binary)
        if not binary:result.pop('local_file_uri',None)
        return result

    def evidence(self,twin_id,evidence_id,binary=False):
        if binary:
            _,doc,_=self.load(twin_id)
            frame=next((e for e in doc['evidence'] if e['id']==evidence_id),None)
            if frame is None:raise ValueError('Unknown curated evidence ID')
            reference=self._assets().get(frame['frame_path'])
            if reference is None or reference['sha256']!=frame['frame_sha256']:raise ValueError('Evidence is not the approved revision')
            if Path(reference['path']).suffix.lower() not in {'.jpg','.jpeg'}:raise ValueError('Only declared JPEG evidence is available')
            return self._bytes(reference,maximum=8*1024*1024)
        return super().evidence(twin_id,evidence_id,binary=False)

    def reconstructions(self,twin_id,component_id=None,selection='all'):
        self.load(twin_id)
        if selection not in {'all','diagnostic','accepted'}:raise ValueError('Unknown reconstruction selection')
        return {'twin_id':twin_id,'versions':[],'scope':'private_diagnostic_versions_excluded','policy':self.policy()}

    def motion_experiments(self,twin_id):
        self.load(twin_id)
        return {'twin_id':twin_id,'experiments':[],'scope':'private_motion_experiments_excluded','policy':self.policy()}

    def reconstruction(self,twin_id,*args):
        self.load(twin_id);raise ValueError('Private diagnostic versions are unavailable in curated mode')
    reconstruction_report=reconstruction
    reconstruction_artifact=reconstruction
    motion_experiment=reconstruction
    motion_artifact=reconstruction
