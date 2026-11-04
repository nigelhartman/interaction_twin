"""Adopt one already computed laptop hinge revision, without inference or approval.

The fixed production contract is deliberately separate from the frozen pipeline.
Only one exact candidate job is created and leased; the general queue is never
drained. Existing programs, native meshes, revisions and owner decisions stay intact.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sqlite3
import time

from admin_workflow import AdminWorkflow, identifier, now, owner
from agent_memory import digest
from review_agents import verified_ref, review_revision

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-local-cached-revision-adoption'
CONTRACT = {
    'candidate_id': 'candidate_a543f1ae2a864e2280868d9e0f3aebe5',
    'base_revision_id': 'revision_77efbc2e68bb4019951b2d89385e31f5',
    'base_fingerprint': '2064bd711e3cce83f041256f0710557028b5c302a33a14d62233fe26facc601d',
    'source_ref': {'path': 'data/new-captures-20261103/sources/20261103_151533_67c51eba.mp4', 'sha256': '2dbd13b04096a0f91b55ba4123c8e033812b81f924913e2df1db5dbc63fab9d2', 'bytes': 34613808},
    'interaction_ref': {'path': 'data/flow-capture-laptop-20261103-v1/flow-procedure-1/interaction.json', 'sha256': 'ad13657bba9544af8a4b0b11611ef38fe9d5da1520303354d87906003aca25ce', 'bytes': 85703},
    'scene_ref': {'path': 'data/native-orientation-diagnostics/laptop-20261103-v1-attempt2/appearance-refit-v1/constrained/scene.json', 'sha256': '4069bcb58593272d0903b6c725ad7ab62a710d3d8683cd5ed133469ee187fd51', 'bytes': 170516},
    'object_program_ref': {'path': 'data/object-programs/auto-5928ca8571f2738fd63b223db7965bed42818b470cbaf0177e09c56adaa4de66/object-program.json', 'sha256': '033e68a6722aa72c97c2d1b16996a541bb72f99f99f6e7ccc75eb62ea354b33f', 'bytes': 22330},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def readonly_base(root, contract):
    """Read immutable owner records using SQLite read-only mode, no migrations."""
    root = Path(root).resolve()
    workflow = object.__new__(AdminWorkflow)
    workflow.root = root
    workflow.folder = root / '.private/admin-workflow'
    workflow.path = workflow.folder / 'state.sqlite3'
    require(workflow.path.is_file() and not workflow.path.is_symlink(), 'Existing workflow database required')
    with sqlite3.connect(workflow.path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'], 'Exact current base revision changed; do not replay adoption')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['candidate_id'] == candidate['id'] and row['fingerprint'] == contract['base_fingerprint'], 'Exact base fingerprint changed')
        revision = workflow._verify_revision(row)
        require(revision['manifest']['source_ref'] == contract['source_ref'] and json.loads(candidate['source_json']) == contract['source_ref'], 'Exact source binding changed')
        require(revision['manifest']['interaction_ref'] == contract['interaction_ref'], 'Exact interaction binding changed')
        require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('running','queued')", (candidate['id'],)).fetchone(), 'Candidate has an active queued or running job')
        source_job = workflow._row(db, 'jobs', revision['manifest']['job_id'])
        payload = json.loads(source_job['payload_json'])
        require(payload['source_ref'] == contract['source_ref'] and payload['memory_context']['context_sha256'] == revision['manifest']['memory_context_sha256'], 'Original frozen review context changed')
        return dict(candidate), revision, payload['memory_context']


def build(*, root=ROOT, contract=None):
    root = Path(root).resolve()
    contract = copy.deepcopy(CONTRACT if contract is None else contract)
    candidate, base, memory = readonly_base(root, contract)
    for key in ['source_ref', 'interaction_ref', 'scene_ref', 'object_program_ref']:
        verified_ref(root, contract[key])
    import object_program
    object_doc = json.loads(verified_ref(root, contract['object_program_ref'], 6*1024*1024).read_text())
    object_program.validate(object_doc, root=root)
    scene = json.loads(verified_ref(root, contract['scene_ref'], 6*1024*1024).read_text())
    require(object_doc['evidence_program'] == contract['interaction_ref'] and object_doc['constrained_scene'] == contract['scene_ref'] and scene['original_program'] == contract['interaction_ref'], 'Cached companion and scene must identify the exact existing evidence program')
    evidence = json.loads(verified_ref(root, contract['interaction_ref'], 6*1024*1024).read_text())
    require(scene['source']['sha256'] == evidence['source']['sha256'] and contract['source_ref']['sha256'] in {evidence['source']['sha256'], evidence['source']['original_sha256']}, 'Cached source scene belongs to a different recording')
    require(object_doc['physical_accuracy_verified'] is False and object_doc['simulation_ready'] is False, 'Research limitations must remain explicit')

    # Bind every exact data reference the companion uses, including archived
    # implementation bytes required by portable/public descriptive downloads.
    refs = {item['path']: copy.deepcopy(item) for item in base['manifest']['artifacts']}
    seen = set()
    archived_code = {item['original_path']: item['archived_file'] for item in object_doc['implementation_provenance']}
    def references(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                if not value['path'].startswith('data/'):
                    archived = archived_code.get(value['path'])
                    require(archived is not None and value['sha256'] == archived['sha256'] and value['bytes'] == archived['bytes'], 'Non-data implementation reference lacks an exact admitted archive')
                    verified_ref(root, archived)
                    refs[archived['path']] = copy.deepcopy(archived)
                    return
                verified_ref(root, value)
                previous = refs.get(value['path'])
                require(previous is None or previous == value, 'Conflicting artifact hash at one path')
                refs[value['path']] = copy.deepcopy(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)
    references([contract['scene_ref'], contract['object_program_ref'], object_doc, scene])
    for key in ['source_scene', 'mechanics', 'proposal']:
        item = object_doc[key]
        if item['path'] not in seen:
            seen.add(item['path'])
            references(json.loads(verified_ref(root, item, 6*1024*1024).read_text()))
    require(len(refs) <= 2048, 'Bounded cached artifact inventory exceeded')
    result = {'status': 'complete_with_gaps', 'run_id': base['manifest']['run_id'],
              'interaction_ref': contract['interaction_ref'], 'object_program_ref': contract['object_program_ref'],
              'scene_ref': contract['scene_ref'], 'part_ledger_ref': base['manifest'].get('part_ledger_ref'),
              'artifacts': [refs[key] for key in sorted(refs)], 'new_api_calls': 0, 'new_gpu_calls': 0,
              'repair_execution': [{'strategy': 'explicit_local_cached_laptop_hinge_adoption',
                                    'base_revision_id': base['id'], 'base_fingerprint': base['fingerprint'],
                                    'original_artifacts_modified': False, 'new_geometry_generated': False,
                                    'new_api_calls': 0, 'new_gpu_calls': 0}],
              'diagnostic': 'Previously computed source-bound automatic laptop hinge adopted locally; no inference, geometry edits, approval or publication.'}
    review = review_revision({**result, 'source_ref': contract['source_ref']}, root=root)
    require(review['role_count'] == 4 and review['all_required_roles_completed'] and not review['blocking_failures'] and not review['hosted_roles'], 'Cached revision failed four local structural roles')
    return {'format': FORMAT, 'version': 1, 'contract': contract,
            'base_manifest_ref': base['manifest_ref'], 'frozen_memory_context': memory,
            'result': result, 'structural_preflight': review,
            'new_api_calls': 0, 'new_gpu_calls': 0, 'automatic_approval': False, 'automatic_publication': False}


def adopt(plan, *, root=ROOT, actor='nigel', contract=None):
    owner(actor)
    root = Path(root).resolve()
    expected = build(root=root, contract=contract)
    require(plan == expected, 'Prepared cached revision changed')
    contract = plan['contract']
    workflow = AdminWorkflow(root)
    token = identifier('lease')
    plan_sha = digest(plan)
    with workflow.db() as db:
        db.execute('BEGIN IMMEDIATE')
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'], 'Exact current base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['fingerprint'] == contract['base_fingerprint'] and json.loads(candidate['source_json']) == contract['source_ref'], 'Exact base/source binding changed')
        require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('running','queued')", (candidate['id'],)).fetchone(), 'Candidate has an active job')
        require(not db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?", (time.time(),)).fetchone(), 'Another owner job is running; no unrelated lease is changed')
        for prior in db.execute("SELECT payload_json FROM jobs WHERE candidate_id=? AND kind='cached_local_revision'", (candidate['id'],)):
            require(json.loads(prior['payload_json']).get('adoption_plan_sha256') != plan_sha, 'This cached adoption was already attempted; no automatic retry')
        base = workflow._revision_row(row)
        payload = {'run_id': candidate['run_id'], 'input_path': candidate['input_path'], 'source_ref': contract['source_ref'],
                   'processing_consent': False, 'base_revision': base, 'repair_plan': None,
                   'memory_context': plan['frozen_memory_context'], 'adoption_plan_sha256': plan_sha,
                   'local_cached_review_only': True}
        created = workflow._enqueue(db, candidate, 'cached_local_revision', payload, False)
        db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=? AND state='pending_consent'", (token, time.time()+300, now(), created['id']))
        db.execute("UPDATE candidates SET status='processing',updated_utc=? WHERE id=?", (now(), candidate['id']))
        job = {'id': created['id'], 'kind': created['kind'], 'candidate_id': candidate['id'], **payload}
    try:
        # No queue runner, model client, hosted factory or approval adapter.
        result = copy.deepcopy(plan['result'])
        result['repair_execution'][0]['adoption_plan_sha256'] = plan_sha
        summary = workflow._finalize_revision(job, result, token, hosted_reviewer=None)
    except BaseException as error:
        with workflow.db() as db:
            changed = db.execute("UPDATE jobs SET state='failed',diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND state='running' AND lease_token=?", ('Local cached adoption stopped: '+type(error).__name__, now(), job['id'], token)).rowcount
            if changed:
                db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?", (now(), candidate['id']))
        raise
    return {'status': 'adopted_private_cached_revision', 'candidate_id': candidate['id'],
            'base_revision_id': contract['base_revision_id'], 'job_id': job['id'], 'plan_sha256': plan_sha,
            **summary, 'new_api_calls': 0, 'new_gpu_calls': 0,
            'automatic_approval': False, 'automatic_publication': False}
