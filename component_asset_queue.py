"""Offline SAM3D component handoffs joined to descriptive programs, never poses.

Reuses the existing source-anchor, handoff and native-import verifiers. Preparing
this queue runs no inference, submits no job and never changes preferred assets.
"""
from __future__ import annotations

import copy
from collections import Counter
import json
from pathlib import Path

import upgrade_bundles as U
from twin_catalog import TwinCatalog

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-component-asset-queue'
MAX_COMPONENTS = 64
MAX_IMPORTS = 64
DEPENDENCIES = ('component_asset_queue.py', 'upgrade_bundles.py', 'twin_catalog.py',
                'sam3d_milestone.py', 'scripts/build_sam3d_milestone_review.py',
                'scripts/stage4_native_pose.py', 'native_alignment.py', 'native_appearance.py')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _reference(path, root):
    path = U._local(path, root)
    return {'path': path.relative_to(root).as_posix(), 'sha256': U.sha(path)}


def _source_program(twin_id, root):
    row, document, digest = TwinCatalog(root).load(twin_id)
    _require(len(document['tracklets']) <= MAX_COMPONENTS, 'Component batch exceeds 64 tracks')
    _require(len({t['id'] for t in document['tracklets']}) == len(document['tracklets']), 'Duplicate component identity')
    return row, document, digest


def _candidate(reference, twin_id, track_id, binding, root):
    """Resolve only an existing verified import; never accept a supplied mesh."""
    _require(isinstance(reference, dict) and len(reference) == 1, 'Choose one existing import reference')
    if set(reference) == {'upgrade_import_id'}:
        result = U.resolve_import(reference['upgrade_import_id'], root=root)
        _require(result.get('backend') == 'sam3d_objects' and result.get('twin_id') == twin_id
                 and result.get('track_id') == track_id and result.get('base_sha256') == binding['base_sha256'],
                 'Upgrade candidate belongs to another source/component/program')
        receipt = _reference(result['import_path'], root)
        metadata_path = U._local(result['model_metadata_path'], root)
        model = U.read(metadata_path)
        fit = result['adapter_capability']
    elif set(reference) == {'native_import_directory'}:
        from scripts.build_sam3d_milestone_review import verified_import
        directory, result, source, raw, metadata = verified_import(reference['native_import_directory'], root=root)
        _require(source['binding'] == binding, 'Native candidate belongs to another anchor/source/component/program')
        receipt = _reference(directory/'import.json', root)
        metadata_path = U._local(directory/'metadata.json', root)
        model = U.read(metadata_path)
        fit = result['fit_contract']
    else:
        raise ValueError('Only validated upgrade/native import references are supported')
    _require(model.get('object_id') == track_id and model.get('backend') == 'Meta SAM 3D Objects'
             and model.get('learned_inference_executed') is True and model.get('manual_pose_used') is False
             and model.get('manual_geometry_fallback') is False and model.get('camera_pose_estimated') is False
             and model.get('metric_scale_verified') is False, 'Native candidate scope differs')
    asset = _reference(metadata_path.parent/model['outputs']['glb_path'], root)
    _require(asset['sha256'] == fit.get('raw_glb_sha256'), 'Candidate mesh differs from validated fitting contract')
    source = model.get('source')
    _require(isinstance(source,dict) and all(source.get(k) == binding['source'][k]
             for k in ('video_sha256','sample_index','timestamp_s','time_domain')), 'Candidate source identity differs')
    for role in ('image','mask'):
        actual = _reference(metadata_path.parent/source[role+'_path'],root)
        _require(actual['sha256'] == source.get(role+'_sha256') == binding[role+'_sha256'], 'Candidate source pixels differ')
    selection = _reference(source['selection_manifest_path'],root)
    _require(selection == {'path':binding['selection_path'],'sha256':binding['selection_sha256']}
             and source.get('selection_manifest_sha256') == selection['sha256'], 'Candidate source selection differs')
    return {'reference': copy.deepcopy(reference), 'import_receipt': receipt,
            'metadata': _reference(metadata_path, root), 'asset': asset,
            'backend': 'Meta SAM 3D Objects', 'fit_contract': copy.deepcopy(fit),
            'native_coordinates_preserved': True, 'selected_as_preferred': False,
            'physical_accuracy_verified': False, 'simulation_ready': False}


def plan(twin_id, *, candidate_imports=None, root=ROOT):
    """Describe all program tracks using saved automatic anchors and imports."""
    root = Path(root).resolve()
    row, document, digest = _source_program(twin_id, root)
    references = {} if candidate_imports is None else candidate_imports
    known = {t['id'] for t in document['tracklets']}
    _require(isinstance(references, dict) and len(references) <= MAX_IMPORTS and set(references) <= known,
             'Candidate references must identify existing program components')
    components = []
    for index, track in enumerate(document['tracklets'], 1):
        track_id = track['id']
        motion = [r for r in document['motion'] if r['track_id'] == track_id]
        experiment_ids = {r['experiment_input_id'] for r in motion}
        experiments = [dict(path=r['path'], sha256=r['sha256']) for r in document['inputs']
                       if r['id'] in experiment_ids and r['kind'] == 'model_conditional_motion_experiment']
        component = {'component_key': 'component-%04d' % index, 'track_id': track_id, 'label': track['label'],
                     'identity_status': track['identity_status'], 'state': 'unavailable', 'reason': None,
                     'relationships': [r['id'] for r in document['relationships']
                                       if track_id in (r['subject_track_id'], r['object_track_id'])],
                     'procedure_steps': [s['id'] for s in document['steps'] if track_id in s['participant_track_ids']],
                     'observed_motion': {'record_count': len(motion), 'status_counts': dict(sorted(Counter(r['status'] for r in motion).items())),
                                         'experiment_inputs': experiments, 'pose_matrices_copied': False,
                                         'usable_as_candidate_pose': False},
                     'anchor': None, 'candidate': None,
                     'attachment': {'state': 'unavailable', 'fresh_reference_fit_required': True,
                                    'existing_poses_reused': False, 'shared_component_gauge_established': False,
                                    'camera_pose_validated': False, 'simulation_ready': False}}
        try:
            binding = U.context(twin_id, track_id, root=root)
        except (ValueError, KeyError, TypeError) as error:
            component['reason'] = 'no_usable_saved_anchor: ' + str(error)
            _require(track_id not in references, 'Cannot attach a candidate without its current source anchor')
            components.append(component)
            continue
        _require(binding['base_sha256'] == digest, 'Program changed while resolving component anchors')
        if binding['mask_provenance'].get('manual_intervention') is not False:
            component['reason'] = 'saved_segmentation_assistance_is_not_automatic'
            _require(track_id not in references, 'Automatic queue cannot inherit assisted candidate segmentation')
            components.append(component)
            continue
        component.update(state='proposed', reason='automatic_saved_full_frame_RGB_and_SAM_mask_available', anchor=binding)
        if track_id in references:
            component['candidate'] = _candidate(references[track_id], twin_id, track_id, binding, root)
            component['state'] = 'geometry_candidate'
            component['reason'] = 'validated_native_geometry_import_is_unselected_and_has_no_video_pose'
            fit = component['candidate']['fit_contract']
            component['attachment'].update(
                state='motion_alignment_required' if fit.get('alignment_eligible') is True else 'unavailable',
                reason='fresh_source_reference_fit_then_independent_projection_checks_required'
                       if fit.get('alignment_eligible') is True else fit.get('alignment_ineligible_reason'),
                reference_adapter='scripts/stage4_native_pose.py' if fit.get('alignment_eligible') is True else None,
                camera_policy='existing_native_adapter_assumed_intrinsics_are_not_metric_calibration')
        components.append(component)
    implementations = {}
    for name in DEPENDENCIES:
        path = root/name
        _require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'Reviewed implementation unavailable')
        implementations[name] = U.sha(path)
    return {'format': FORMAT, 'version': 1, 'twin_id': twin_id,
            'program': {'path': row['interaction_path'], 'sha256': digest}, 'source': copy.deepcopy(document['source']),
            'candidate_imports': copy.deepcopy(references), 'components': components,
            'component_counts': dict(sorted(Counter(c['state'] for c in components).items())),
            'implementation_sha256': implementations,
            'read_only_source': True, 'no_inference': True, 'remote_job_submitted': False,
            'preferred_assets_changed': False, 'source_poses_rebound_to_new_geometry': False,
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'limitations': ['Program component identities and relations remain model proposals with cited evidence.',
                            'A usable saved anchor currently requires existing source reconstruction metadata; missing anchors are explicit.',
                            'No independent rigid-body certification is supplied by this batch planner.',
                            'SAM3D hidden shape and appearance are inferred; each part keeps its independent native gauge.',
                            'Old conditional motion belongs to its exact original mesh and is never copied to a new candidate.',
                            'Fresh alignment can still fail under occlusion, symmetry, wrong masks or uncertain camera intrinsics.',
                            'Metric scale, common assembly, joints, contact, collision and robot simulation remain unverified.']}


def _output(path, root):
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    relative = path.relative_to(root)
    _require(len(relative.parts) == 3 and relative.parts[:2] == ('data','component-asset-queues')
             and all(not p.startswith('.') and '\\' not in p for p in relative.parts),
             'Choose a fresh direct child of data/component-asset-queues')
    cursor = root
    for part in relative.parts:
        cursor = cursor/part
        _require(not cursor.is_symlink(), 'Symlink queue paths are forbidden')
    _require(not path.exists(), 'Preserve previous queue; choose a fresh directory')
    return path


def prepare(twin_id, output, *, candidate_imports=None, prepare_jobs=False, root=ROOT):
    """Freeze a batch; optional existing handoff preparation is local files only."""
    _require(type(prepare_jobs) is bool, 'prepare_jobs must be explicit boolean')
    root = Path(root).resolve()
    directory = _output(output, root)
    frozen = plan(twin_id, candidate_imports=candidate_imports, root=root)
    directory.mkdir(parents=True, exist_ok=False)
    U.write(directory/'plan.json', frozen)
    rows = []
    try:
        for component in frozen['components']:
            result = {'track_id': component['track_id'], 'state': component['state'], 'job': None,
                      'attachment': copy.deepcopy(component['attachment'])}
            if prepare_jobs and component['state'] == 'proposed':
                job = U.prepare_job({'twin_id': twin_id, 'track_id': component['track_id'], 'backend':'sam3d_objects',
                                     'base_sha256': frozen['program']['sha256'], 'appearance':'vertex-color'}, root=root)
                _, record = U._job_read(job['job_id'], root)
                _require(record['input_binding'] == component['anchor'], 'Prepared handoff anchor differs')
                result['job'] = {'job_id':job['job_id'], 'manifest':_reference(job['job_relative_path'],root),
                                 'zip':_reference(job['zip_relative_path'],root), 'state':'prepared_not_submitted'}
            if component['attachment']['state'] == 'motion_alignment_required':
                result['attachment']['reference_fit_argv'] = ['python', 'scripts/stage4_native_pose.py', '--manifest',
                    component['candidate']['metadata']['path'], '--out',
                    str((directory/'alignment'/component['component_key']/'reference-v1').relative_to(root)),
                    '--neighbor-count','0']
                result['attachment']['command_executed'] = False
            rows.append(result)
        _require(plan(twin_id, candidate_imports=candidate_imports, root=root) == frozen, 'Source or implementations changed during preparation')
        queue = {'format':FORMAT, 'version':1, 'status':'prepared_offline_not_executed',
                 'plan':_reference(directory/'plan.json',root), 'components':rows,
                 'prepare_jobs':prepare_jobs, 'no_inference':True, 'remote_job_submitted':False,
                 'preferred_assets_changed':False, 'physical_accuracy_verified':False, 'simulation_ready':False}
        U.write(directory/'queue.json',queue)
        return {'queue':_reference(directory/'queue.json',root), 'plan':queue['plan'],
                'component_counts':frozen['component_counts'], 'local_handoffs_prepared':sum(r['job'] is not None for r in rows),
                'download_url':'/files/'+str((directory/'queue.json').relative_to(root)), 'no_inference':True}
    except Exception as error:
        U.write(directory/'failure.json', {'status':'failed_offline_preparation_partial_files_preserved',
                  'error_type':type(error).__name__, 'no_inference':True, 'remote_job_submitted':False})
        raise


def inspect(queue_path, *, root=ROOT):
    """Revalidate a saved batch against its exact current program and imports."""
    root = Path(root).resolve()
    path = U._local(queue_path, root)
    _require(path.stat().st_size <= 8*1024*1024, 'Queue JSON exceeds inspection budget')
    queue = U.read(path)
    _require(isinstance(queue,dict) and queue.get('format') == FORMAT and type(queue.get('version')) is int and queue['version'] == 1
             and queue.get('status') == 'prepared_offline_not_executed', 'Invalid prepared queue')
    _require(queue.get('no_inference') is True and queue.get('remote_job_submitted') is False
             and queue.get('preferred_assets_changed') is False and queue.get('simulation_ready') is False
             and queue.get('physical_accuracy_verified') is False and type(queue.get('prepare_jobs')) is bool, 'Queue scope differs')
    reference = queue.get('plan')
    _require(isinstance(reference,dict), 'Queue plan binding missing')
    plan_path = U._local(reference.get('path'),root)
    _require(plan_path.stat().st_size <= 16*1024*1024, 'Plan JSON exceeds inspection budget')
    _require(plan_path == path.parent/'plan.json' and U.sha(plan_path) == reference.get('sha256'), 'Queue plan bytes differ')
    frozen = U.read(plan_path)
    _require(plan(frozen['twin_id'],candidate_imports=frozen['candidate_imports'],root=root) == frozen, 'Saved source/import/implementation binding differs')
    rows = queue.get('components')
    _require(isinstance(rows,list) and len(rows) == len(frozen['components']), 'Queue component inventory differs')
    for record, component in zip(rows,frozen['components']):
        _require(isinstance(record,dict) and set(record) == {'track_id','state','job','attachment'}, 'Queue row contract differs')
        _require(record['track_id'] == component['track_id'] and record['state'] == component['state'], 'Queue component identity/state differs')
        _require((record.get('job') is not None) == (queue['prepare_jobs'] and component['state'] == 'proposed'), 'Queue handoff preparation policy differs')
        if record.get('job') is not None:
            job = record['job']
            _require(isinstance(job,dict) and set(job) == {'job_id','manifest','zip','state'}
                     and job.get('state') == 'prepared_not_submitted', 'Queue job state differs')
            job_path, actual = U._job_read(job['job_id'],root)
            _require(actual['backend'] == 'sam3d_objects' and actual['input_binding'] == component['anchor'], 'Queue job binding differs')
            _require(job['manifest'] == _reference(job_path,root) and job['zip'] == _reference(str(job_path.parent)+'.zip',root), 'Queue handoff bytes differ')
        expected_attachment = copy.deepcopy(component['attachment'])
        if component['attachment']['state'] == 'motion_alignment_required':
            expected = ['python','scripts/stage4_native_pose.py','--manifest',component['candidate']['metadata']['path'],'--out',
                        str((path.parent/'alignment'/component['component_key']/'reference-v1').relative_to(root)),'--neighbor-count','0']
            expected_attachment.update(reference_fit_argv=expected,command_executed=False)
        _require(record['attachment'] == expected_attachment, 'Queue attachment scope or fresh fitting command differs')
    return {'status':'verified_offline_queue','queue':_reference(path,root), 'plan':reference,
            'component_counts':frozen['component_counts'],'components':rows,'no_inference':True,'simulation_ready':False}
