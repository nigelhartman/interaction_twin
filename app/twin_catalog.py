"""Read-only catalog of published local interaction artifacts for agent clients.

Only workspace-listed programs and their data/ artifacts are eligible. This
module neither runs models nor interprets descriptive text as executable code.
"""
from __future__ import annotations

import json
import math
import copy
from pathlib import Path

import stage5
from sim_readiness import audit_simulation_readiness, build_simulation_draft
from library_query import query_records, backend_key

ROOT = Path(__file__).resolve().parent
CODE_INPUTS = {
    ('deterministic_compiler_code', 'stage5.py'),
    ('deterministic_compiler_orchestration_code', 'scripts/stage5_compile.py'),
    ('dense_semantic_compiler_adapter', 'scripts/stage5_dense_compile.py'),
    ('schema', 'schemas/interaction.schema.json'),
}
MISSING_PHYSICS = [
    'metric_scale', 'physical_shape_accuracy', 'joint_axes_and_limits',
    'collision_geometry', 'mass_and_inertia', 'friction_and_contact',
    'verified_insertion_and_seating',
]


class TwinCatalog:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()

    def data_path(self, relative):
        if not isinstance(relative, str) or not relative.startswith('data/'):
            raise ValueError('Only declared data artifacts are available')
        if any(part.startswith('.') for part in Path(relative).parts):
            raise ValueError('Hidden or traversing artifact paths are unavailable')
        if (self.root / 'data').resolve() != self.root / 'data':
            raise ValueError('Data root must not be a symlink')
        target = (self.root / relative).resolve()
        if not target.is_relative_to((self.root / 'data').resolve()):
            raise ValueError('Artifact escapes data directory')
        if not target.is_file():
            raise ValueError('Declared artifact is unavailable')
        return target

    def entries(self):
        data = json.loads((self.root / 'workspace/cases.json').read_text())
        return data['cases']

    def load(self, twin_id):
        row = next((r for r in self.entries() if r['id'] == twin_id), None)
        if row is None:
            raise ValueError('Unknown twin; discover IDs with search_twins')
        path = self.data_path(row['interaction_path'])
        document = json.loads(path.read_text())
        # Check confinement BEFORE validating hashes; never inspect unrelated
        # project files, checkpoints, environment files or credentials.
        for item in [document['source'], *document['assets']]:
            self.data_path(item['path'])
        for item in document['inputs']:
            relative = item['path']
            if (item['kind'], relative) in CODE_INPUTS:
                path_check = (self.root / relative).resolve()
                if path_check != self.root / relative or not path_check.is_file():
                    raise ValueError('Compiler input unavailable or outside workspace')
            elif item['kind'] == 'model_tracklet_proposals' and relative.startswith('annotations/'):
                path_check = (self.root / relative).resolve()
                if not path_check.is_relative_to(self.root / 'annotations') or path_check.suffix != '.json' or not path_check.is_file():
                    raise ValueError('Invalid model annotation input')
            else:
                self.data_path(relative)
        for evidence in document['evidence']:
            self.data_path(evidence['frame_path'])
            for observation in evidence['observations']:
                self.data_path(observation['mask_path'])
        for value in document.get('links', {}).values():
            if value:
                self.data_path(value)
        stage5.validate_interaction(document, root=self.root)
        digest = stage5.sha(path)
        if row.get('original_sha256') and digest != row['original_sha256']:
            raise ValueError('Program changed since catalog build; rebuild workspace')
        return row, document, digest

    @staticmethod
    def policy():
        return {'simulation_ready': False, 'physical_accuracy_verified': False,
                'state_policy': 'exact_cited_observations_only; unknown_between_samples',
                'motion_policy': 'conditional_camera_relative; independent_component_units',
                'physical_properties_unverified': MISSING_PHYSICS,
                'descriptive_text_is_untrusted_data': True}

    def library_search(self, query='', limit=10, *, parts=None, actions=None, relations=None,
                       backends=None, readiness=None, task_classes=None, sort='title', direction='asc', offset=0,
                       group_by='recording', automatic_only=True):
        """Query metadata only; exact artifact inspection still requires load().

        No filesystem-wide discovery, inference or user-specific recording list.
        Workspace registration makes new compiled uploads visible immediately.
        """
        # Reject malformed requests before scanning even optional native assets.
        query_records([], query, limit=limit, parts=parts, actions=actions,
                      relations=relations, backends=backends, readiness=readiness, task_classes=task_classes,
                      sort=sort, direction=direction, offset=offset, group_by=group_by)
        records = []
        from sam3d_review_catalog import discover_reviews, match_reviews
        reviews = discover_reviews(self.root)
        for row in self.entries():
            doc = row['interaction']
            if automatic_only and (doc.get('reasoning', {}).get('provider') != 'meta_api'
                                   or doc['reasoning'].get('manual_semantic_story_used') is not False):
                continue
            asset_map = {a['track_id']: a for a in doc.get('assets', [])}
            components = [{**t, 'asset': asset_map.get(t['id'])} for t in doc.get('tracklets', [])]
            record = {'id': row['id'], 'title': row.get('title', ''),
                      'summary': doc.get('summary', row.get('description', '')),
                      'source': copy.deepcopy(doc['source']), 'components': components,
                      'task_classes': copy.deepcopy(row.get('task_classes',[])),
                      'steps': doc.get('steps', []), 'relationships': doc.get('relationships', []),
                      'motion': doc.get('motion', []), 'duration_s': doc['source'].get('duration_s', 0),
                      'counts': {**row.get('labels', {}), 'components': len(components),
                                 'assets': len(asset_map), 'steps': len(doc.get('steps', []))},
                      'program_sha256': row.get('original_sha256'),
                      'program_uri': f"twin://{row['id']}/program"}
            record['sam3d_reviews'] = match_reviews(reviews, record)
            records.append(record)
        result = query_records(records, query, limit=limit, parts=parts, actions=actions,
                               relations=relations, backends=backends, readiness=readiness, task_classes=task_classes,
                               sort=sort, direction=direction, offset=offset, group_by=group_by)
        # Return compact discoverable IDs rather than embedding programs/media.
        result['records'] = [{k: r[k] for k in ('id', 'title', 'summary', 'source', 'duration_s',
                             'counts', 'program_uri', 'program_sha256', 'search_index', 'task_classes')}
                             | {'saved_variant_ids': r.get('saved_variant_ids', [r['id']]),
                                'simulation_ready': False, 'physical_accuracy_verified': False}
                             for r in result['records']]
        return result | {'access': 'local_only_private_recordings_not_publicly_distributed',
                         'metadata_only': True, 'artifact_inspection_required': True,
                         'eligibility': 'automatic_meta_programs' if automatic_only else 'all_workspace_programs',
                         'policy': self.policy()}

    def search(self, query='', limit=10, **options):
        # Legacy Python callers can still inspect all explicitly registered
        # variants; MCP and the human library opt into automatic Muse records.
        options.setdefault('automatic_only', False)
        result = self.library_search(query, limit, **options)
        result['twins'] = [dict(record, twin_id=record['id']) for record in result.pop('records')]
        return result

    def library_facets(self, group_by='recording'):
        result = self.library_search(limit=1, group_by=group_by)
        return {key: result[key] for key in ('facets', 'query_contract', 'total', 'metadata_only',
                                             'artifact_inspection_required', 'access', 'policy')}

    def generated_candidate(self, twin_id, component_id):
        """Inspect the default SAM 3D candidate without changing program assets."""
        _, doc, digest = self.load(twin_id)
        if component_id not in {t['id'] for t in doc['tracklets']}:
            raise ValueError('Unknown component ID in this source program')
        from sam3d_review_catalog import reviews_for_record
        record = {'source': doc['source'], 'components': doc['tracklets']}
        candidate = next((r for r in reviews_for_record(self.root, record) if r['component_id'] == component_id), None)
        if candidate is None:
            raise ValueError('No generated SAM 3D candidate for this exact source/component')
        if (candidate.get('source_sha256') != doc['source']['sha256'] or candidate.get('selected') is not False
                or candidate.get('physical_accuracy_verified') is not False):
            raise ValueError('Generated candidate source or inspection policy differs')
        artifacts = {}
        for artifact_id, field, mime in [('asset', 'native_glb', 'model/gltf-binary'),
                                         ('source-image', 'source_image', 'image/png'),
                                         ('source-mask', 'source_mask', 'image/png'),
                                         ('review-manifest', 'review_manifest', 'application/json')]:
            ref = candidate[field]
            artifacts[artifact_id] = {'sha256': ref['sha256'], 'bytes': ref['bytes'],
                'mime_type': mime, 'uri': f'twin://{twin_id}/native-candidates/{component_id}/artifacts/{artifact_id}'}
        selected = any(a['track_id'] == component_id and a['sha256'] == candidate['native_glb']['sha256']
                       and backend_key(a.get('backend', '')) == 'sam3d' for a in doc['assets'])
        return {'twin_id': twin_id, 'component_id': component_id, 'program_sha256': digest,
                'source_sha256': candidate['source_sha256'], 'backend': candidate['backend'],
                'source_sample_index': candidate['source_sample_index'],
                'source_timestamp_s': candidate['source_timestamp_s'],
                'geometry_preference': 'sam3d_saved_program_asset' if selected else 'sam3d_candidate_for_inspection', 'artifacts': artifacts,
                'selected_in_saved_program': selected, 'saved_program_assets_changed': False,
                'units': 'model_native_unverified', 'appearance_and_hidden_surfaces': 'model_generated_unverified',
                'source_video_alignment_verified': False, 'existing_poses_reused': False,
                'physical_accuracy_verified': False, 'simulation_ready': False, 'policy': self.policy()}

    def generated_candidate_artifact(self, twin_id, component_id, artifact_id):
        descriptor = self.generated_candidate(twin_id, component_id)
        declared = descriptor['artifacts'].get(artifact_id)
        if declared is None:
            raise ValueError('Unknown generated-candidate artifact ID')
        # Rebind to the same source/model identity after inspection; no client
        # path or URL is accepted, and a racing preview change fails closed.
        from sam3d_review_catalog import reviews_for_record
        _, doc, _ = self.load(twin_id)
        rows = reviews_for_record(self.root, {'source': doc['source'], 'components': doc['tracklets']})
        row = next((r for r in rows if r['component_id'] == component_id
                    and r['native_glb']['sha256'] == descriptor['artifacts']['asset']['sha256']), None)
        if row is None:
            raise ValueError('Generated candidate changed during inspection')
        field = {'asset': 'native_glb', 'source-image': 'source_image', 'source-mask': 'source_mask',
                 'review-manifest': 'review_manifest'}[artifact_id]
        ref = row[field]
        maximum = 16 * 1024 * 1024 if artifact_id == 'asset' else 8 * 1024 * 1024
        if ref['bytes'] > maximum or ref['sha256'] != declared['sha256']:
            raise ValueError('Generated candidate artifact changed or exceeds resource budget')
        data = self.data_path(ref['path']).read_bytes()
        import hashlib
        if len(data) != ref['bytes'] or hashlib.sha256(data).hexdigest() != declared['sha256']:
            raise ValueError('Generated candidate artifact changed during read')
        return data

    def describe(self, twin_id):
        row, doc, digest = self.load(twin_id)
        return {'twin_id': twin_id, 'title': row['title'], 'program_sha256': digest,
                'summary': doc['summary'], 'components': doc['tracklets'],
                'relationships': doc['relationships'], 'variables': doc['variables'],
                'assets': [self.asset_descriptor(twin_id, a) for a in doc['assets']],
                'procedure': [{'id': s['id'], 'title': s['title'], 'start_s': s['start_s'],
                               'end_s': s['end_s'], 'evidence_ids': s['evidence_ids']} for s in doc['steps']],
                'reasoning': doc['reasoning'], 'limitations': doc['limitations'],
                'task_brief': self._task_brief(twin_id, row, doc, digest),
                'kinematics': self._kinematics_descriptor(twin_id, doc, digest),
                'policy': self.policy()}

    def _kinematics_record(self, document, digest):
        from object_program import discover
        return discover(self.root,digest,document['source']['sha256'])

    def _kinematics_descriptor(self, twin_id, document, digest):
        found=self._kinematics_record(document,digest)
        if found is None:
            return {'status':'not_available','cached_only':True,'new_api_call':False,
                    'original_evidence_program_unchanged':True,'physical_accuracy_verified':False,'simulation_ready':False}
        value=found['document']
        return {'status':'estimated_descriptive_kinematics','format':value['format'],'version':value['version'],
                'artifact':{**found['artifact'],'uri':f'twin://{twin_id}/kinematics'},
                'joints':[{'id':j['id'],'type':j['type'],'parent_id':j['parent_id'],'child_id':j['child_id'],
                           'status':j['status'],'angle_units':'radians','limits_kind':j['limits_kind'],
                           'angle_quality':j['angle_quality']} for j in value['joints']],
                'removable_relationships':value['removable_relationships'],
                'unresolved_connections':value['unresolved_connections'],
                'original_evidence_program_unchanged':True,'physical_accuracy_verified':False,'simulation_ready':False,
                'robot_execution_available':False}

    def kinematics(self, twin_id):
        """Read validated object-program v0.2 estimates; never fit, infer or execute."""
        _,doc,digest=self.load(twin_id)
        found=self._kinematics_record(doc,digest)
        if found is None:
            return self._kinematics_descriptor(twin_id,doc,digest)
        artifacts={key:{**found['document'][key],'uri':f'twin://{twin_id}/kinematics/artifacts/{key.replace("_","-")}',
                        'mime_type':'application/json'} for key in ['source_scene','constrained_scene','mechanics','proposal']}
        return {'status':'estimated_descriptive_kinematics','twin_id':twin_id,'program_sha256':digest,
                'artifact':{**found['artifact'],'uri':f'twin://{twin_id}/kinematics','mime_type':'application/json'},
                'object_program':found['document'],'artifacts':artifacts,
                'original_evidence_program_unchanged':True,'policy':self.policy()}

    def kinematics_artifact(self, twin_id, artifact_id='object-program'):
        """Exact bounded artifact bytes, IDs only; ordinary data paths are confined."""
        from object_program import checked_ref
        allowed={'object-program','source-scene','constrained-scene','mechanics','proposal'}
        if artifact_id not in allowed:
            raise ValueError('Unknown kinematics artifact ID')
        _,doc,digest=self.load(twin_id)
        found=self._kinematics_record(doc,digest)
        if found is None:
            raise ValueError('No source-bound descriptive kinematics available')
        reference=found['artifact'] if artifact_id=='object-program' else found['document'][artifact_id.replace('-','_')]
        return checked_ref(reference,self.root).read_bytes()

    def _task_brief(self, twin_id, row, doc, digest):
        """Expose only a sanitized verified cached analysis; never start inference.

        The caller must first complete load() confinement and hash validation.
        Current 12-image/12k-output contracts take priority; historical contracts
        are eligible only after cached_analyses verifies their saved provenance.
        """
        absent = {'status': 'not_available', 'cached_only': True, 'new_api_call': False,
                  'physical_accuracy_verified': False, 'simulation_ready': False}
        try:
            from task_intelligence import cached_analyses, plan_analysis
            rows = cached_analyses(row['interaction_path'], root=self.root)
            eligible = [item['record'] for item in rows
                        if item.get('record', {}).get('contract', {}).get('mode') == 'analysis'
                        and item['record']['contract'].get('interaction_sha256') == digest
                        and item['record']['contract'].get('source_sha256') == doc['source']['sha256']]
            if not eligible:
                return absent
            plan = plan_analysis(row['interaction_path'], root=self.root, mode='analysis',
                                 question=None, max_images=12, max_output_tokens=12000)
            current = next((item for item in eligible if item.get('request_key') == plan['request_key']), None)
            selection = 'current_exact_contract'
            if current is None:
                historical = [item for item in eligible
                              if item['contract'].get('max_output_tokens') == 12000
                              and item['contract'].get('question') is None
                              and item['contract'].get('images') == plan['contract']['images']]
                current = historical[-1] if historical else None
                selection = 'verified_historical_contract'
            if current is None:
                return absent
        except (ImportError, OSError, ValueError, KeyError, TypeError):
            # Cache inspection cannot initiate recovery or expose local/provider
            # exception text. The validated base evidence remains available.
            return absent
        used = set()
        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'evidence_ids' and isinstance(item, list):
                        used.update(e for e in item if isinstance(e, str))
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        collect(current['result'])
        # Build source links from the validated current program, never from
        # arbitrary URLs or file paths embedded in a sidecar record.
        citations = [{'evidence_id': e['id'], 'source_index': e['source_index'],
                      'timestamp_s': e['timestamp_s'], 'frame_sha256': e['frame_sha256'],
                      'uri': f"twin://{twin_id}/evidence/{e['id']}"}
                     for e in doc['evidence'] if e['id'] in used]
        brief = {'status': 'available_unverified_model_analysis', 'selection': selection,
                'cached_only': True, 'new_api_call': False,
                'provider': current['provider'], 'model': current['model'],
                'source_sha256': doc['source']['sha256'],
                'original_source_sha256': doc['source']['original_sha256'],
                'interaction_sha256': digest, 'program_uri': f'twin://{twin_id}/program',
                'result': copy.deepcopy(current['result']), 'citations': citations,
                'human_reviewed': False, 'physical_accuracy_verified': False,
                'simulation_ready': False}
        if current.get('recovery'):
            recovery = current['recovery']
            brief['recovery'] = {'applied': True, 'policy': recovery['recovery_policy'],
                                 'removed_count': recovery['removed_count'],
                                 'all_model_items_retained': recovery['all_model_items_retained'],
                                 'original_response_sha256': recovery['original_response_sha256'],
                                 'validation': 'verified_current_source_and_grounding',
                                 'additional_api_calls': 0}
        return brief

    def procedure(self, twin_id, step_id=None):
        _, doc, digest = self.load(twin_id)
        steps = doc['steps'] if step_id is None else [s for s in doc['steps'] if s['id'] == step_id]
        if step_id is not None and not steps:
            raise ValueError('Unknown procedure step')
        ids = {eid for step in steps for eid in step['evidence_ids']}
        return {'twin_id': twin_id, 'program_sha256': digest, 'steps': steps,
                'evidence': [e for e in doc['evidence'] if e['id'] in ids],
                'policy': self.policy()}

    def state(self, twin_id, timestamp_s):
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (float, int)) or not math.isfinite(timestamp_s):
            raise ValueError('Timestamp must be a finite source presentation time')
        _, doc, digest = self.load(twin_id)
        states = stage5.state_at(doc, timestamp_s)
        nearby = sorted(doc['evidence'], key=lambda e: abs(e['timestamp_s']-timestamp_s))[:3]
        return {'twin_id': twin_id, 'program_sha256': digest, 'timestamp_s': timestamp_s,
                'states': states, 'conditional_poses': stage5.poses_at(doc, timestamp_s),
                'nearby_evidence_for_inspection_only': [{'id': e['id'], 'timestamp_s': e['timestamp_s']} for e in nearby],
                'policy': self.policy()}

    @staticmethod
    def asset_descriptor(twin_id, asset):
        return {'track_id': asset['track_id'], 'asset_id': asset['id'], 'sha256': asset['sha256'],
                'mime_type': 'model/gltf-binary', 'uri': f"twin://{twin_id}/assets/{asset['track_id']}",
                'local_workspace_path': asset['path'], 'units': 'model_native_unverified',
                'appearance_and_hidden_surfaces': 'model_generated_unverified'}

    def asset(self, twin_id, track_id, binary=False):
        _, doc, _ = self.load(twin_id)
        asset = next((a for a in doc['assets'] if a['track_id'] == track_id), None)
        if asset is None:
            raise ValueError('No generated asset for this observed tracklet')
        path = self.data_path(asset['path'])
        if path.suffix.lower() != '.glb':
            raise ValueError('Only declared generated GLB assets are available')
        if path.stat().st_size > 16*1024*1024:
            raise ValueError('Asset exceeds the local MCP resource size budget')
        if binary:
            return path.read_bytes()
        return self.asset_descriptor(twin_id, asset) | {'local_file_uri': path.as_uri(), 'policy': self.policy()}

    def evidence(self, twin_id, evidence_id, binary=False):
        _, doc, _ = self.load(twin_id)
        frame = next((e for e in doc['evidence'] if e['id'] == evidence_id), None)
        if frame is None:
            raise ValueError('Unknown source evidence ID')
        if binary:
            path = self.data_path(frame['frame_path'])
            if path.suffix.lower() not in {'.jpg', '.jpeg'} or path.stat().st_size > 8*1024*1024:
                raise ValueError('Only bounded JPEG source frames are available')
            return path.read_bytes()
        return {'twin_id': twin_id, 'evidence': frame,
                'frame_uri': f'twin://{twin_id}/evidence/{evidence_id}', 'policy': self.policy()}

    def simulation_manifest(self, twin_id):
        row, doc, digest = self.load(twin_id)
        # load() confines every declared artifact, verifies the complete evidence
        # contract/hashes and binds the current program to the catalog before
        # any preparation report is produced. This does not validate physics.
        readiness = audit_simulation_readiness(doc)
        readiness['verification'].update(full_contract_validated=True,
                                         artifact_hashes_verified=True,
                                         validated_program_sha256=digest,
                                         validation_scope='catalog_confined_declared_artifacts_and_evidence_contract_only')
        draft = build_simulation_draft(doc)
        draft['program_sha256'] = digest
        draft['program_uri'] = f'twin://{twin_id}/program'
        return {'format': 'interaction-twin-simulation-handoff', 'version': '0.1',
                'twin_id': twin_id, 'program_sha256': digest,
                'program_uri': f'twin://{twin_id}/program', 'source': doc['source'],
                'components': doc['tracklets'], 'assets': [self.asset_descriptor(twin_id, a) for a in doc['assets']],
                'candidate_relationships': doc['relationships'],
                'demonstration_steps': doc['steps'],
                'conditional_motion_samples': doc['motion'],
                'coordinate_policy': 'individual_generated_native_units_to_conditional_camera; do_not_assume_shared_metric_world',
                'verification_required_before_physics': MISSING_PHYSICS,
                'readiness': readiness,
                'preparation_draft': draft,
                'task_brief': self._task_brief(twin_id, row, doc, digest),
                'kinematics': self._kinematics_descriptor(twin_id, doc, digest),
                'policy': self.policy()}

    def _reconstruction_context(self, twin_id):
        from reconstruction_catalog import ReconstructionCatalog
        row, document, digest = self.load(twin_id)
        return ReconstructionCatalog(self.root), {
            'interaction_path': row['interaction_path'], 'document': document, 'program_sha256': digest}

    def reconstructions(self, twin_id, component_id=None, selection='all'):
        catalog, context = self._reconstruction_context(twin_id)
        return catalog.list(twin_id=twin_id, component_id=component_id, selection=selection, **context)

    def reconstruction(self, twin_id, version_id):
        catalog, context = self._reconstruction_context(twin_id)
        return catalog.describe(version_id, twin_id=twin_id, **context)

    def reconstruction_report(self, twin_id, version_id):
        catalog, context = self._reconstruction_context(twin_id)
        _, data = catalog.artifact(version_id, 'quality', **context)
        return {'twin_id': twin_id, 'version_id': version_id,
                'program_sha256': context['program_sha256'], 'report': json.loads(data),
                'report_scope': 'saved quality/provenance evidence, not physical certification',
                'policy': catalog.policy()}

    def reconstruction_artifact(self, twin_id, version_id, artifact_id):
        catalog, context = self._reconstruction_context(twin_id)
        return catalog.artifact(version_id, artifact_id, **context)[1]

    def _motion_context(self, twin_id):
        from motion_catalog import MotionCatalog
        row, document, digest = self.load(twin_id)
        return MotionCatalog(self.root), {'interaction_path': row['interaction_path'],
                                         'document': document, 'program_sha256': digest}

    def motion_experiments(self, twin_id):
        catalog, context = self._motion_context(twin_id)
        return catalog.list(twin_id=twin_id, **context)

    def motion_experiment(self, twin_id, experiment_id):
        catalog, context = self._motion_context(twin_id)
        return catalog.describe(experiment_id, twin_id=twin_id, **context)

    def motion_artifact(self, twin_id, experiment_id, artifact_id):
        catalog, context = self._motion_context(twin_id)
        return catalog.artifact(experiment_id, artifact_id, **context)
