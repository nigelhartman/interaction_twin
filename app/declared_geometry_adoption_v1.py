"""Prepare/adopt a private declared geometry revision without inference.

Full geometry and source-raster reproduction precedes adoption. Native programs
and scenes remain the manifest's core; the derivative is an explicit sidecar.
No approval, publication, model retry or general queue draining occurs here.
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
import declared_geometry_contract_v1 as D

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-declared-geometry-cached-adoption'
STRATEGY = 'declared_geometry_derivative_adoption_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def readonly_base(root, contract):
    workflow = object.__new__(AdminWorkflow)
    workflow.root = Path(root).resolve()
    workflow.folder = workflow.root / '.private/admin-workflow'
    workflow.path = workflow.folder / 'state.sqlite3'
    require(workflow.path.is_file() and not workflow.path.is_symlink(),
            'Existing owner workflow required')
    with sqlite3.connect(workflow.path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'],
                'Exact current base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['fingerprint'] == contract['base_fingerprint'],
                'Exact base fingerprint changed')
        base = workflow._verify_revision(row, allow_hosted_override=True)
        require(not db.execute(
            "SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('queued','running')",
            (candidate['id'],)).fetchone(), 'Candidate has an active job')
        job = workflow._row(db, 'jobs', base['manifest']['job_id'])
        payload = json.loads(job['payload_json'])
    return dict(candidate), base, payload['memory_context']


def artifact_inventory(root, manifest, derived):
    """Follow admitted contract types; opaque provider/native JSON stays opaque."""
    original = json.loads(verified_ref(root, derived['base_object_program_ref']).read_bytes())
    archives = {row['original_path']: row['archived_file']
                for row in original['implementation_provenance']}
    refs = {row['path']: copy.deepcopy(row) for row in manifest['artifacts']}

    def collect(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                item = value
                if not item['path'].startswith('data/'):
                    replacement = archives.get(item['path'])
                    require(replacement and all(item[k] == replacement[k]
                                                for k in ['sha256', 'bytes']),
                            'Non-data reference lacks exact original code archive')
                    item = replacement
                verified_ref(root, item)
                require(item['path'] not in refs or refs[item['path']] == item,
                        'Conflicting cached artifact reference')
                refs[item['path']] = copy.deepcopy(item)
                require(len(refs) <= 2048, 'Bounded cached artifact inventory exceeded')
                return
            for child in value.values():
                if isinstance(child, (dict, list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    def read(pointer):
        collect(pointer)
        raw = verified_ref(root, pointer, 16 * 1024 * 1024).read_bytes()
        return json.loads(raw)

    def geometry_document(document, *, parent=False):
        collect(document)
        keys = ['base_object_program_ref', 'scene_ref',
                'refinement_report_ref', 'source_alignment_audit_ref']
        if not parent:
            keys.append('baseline_scene_ref')
        core = {key: read(document[key]) for key in keys}
        collect(list(core.values()))
        report = core['refinement_report_ref']
        if parent:
            # The exact older rim/source contract has an original_scene ref.
            if report.get('original_scene'):
                collect(read(report['original_scene']))
        else:
            for asset in report['assets']:
                for operation in asset['operations']:
                    if operation['kind'] == 'observed_surface_texture_projection':
                        collect(read(operation['projection_report_ref']))
        for pointer in document.get('visual_review_refs', []):
            review = read(pointer)
            # Retain actual result and accounted receipt without interpreting
            # model response prose or native runtime receipt paths as pointers.
            collect(review)
            for key in ['model_result', 'receipt']:
                verified_ref(root, review[key])
            if review.get('plan'):
                packet = read(review['plan'])
                for image in packet['images']:
                    collect({k: image[k] for k in ['path', 'sha256', 'bytes']})
                if packet.get('manifest'):
                    collect(read(packet['manifest']))
                elif packet.get('format') == 'interaction-twin-jabra-earbud-source-orientation-plan':
                    context = packet['context']
                    collect([context['source_scene'], context['native_asset']])
                    for sample in context['source_samples']:
                        collect([sample[k] for k in ['source_rgb', 'source_mask', 'source_crop']])
        if document.get('nonrigid_observations_ref'):
            nonrigid = read(document['nonrigid_observations_ref'])
            collect(nonrigid)
            trace = read(nonrigid['trace_ref'])
            collect([trace['program_ref'], trace['source_ref']])
            for sample in trace['samples']:
                collect(sample['frame_ref'])
                if sample.get('mask_ref') is not None:
                    collect(sample['mask_ref'])
        if document.get('source_appearance_states_ref'):
            # Binary patches/probe IDs remain opaque; only this typed sidecar
            # admits exact saved RGB/masks and its unchanged prior descriptor.
            appearance=read(document['source_appearance_states_ref'])
            require(appearance.get('format')=='interaction-twin-source-appearance-states'and appearance.get('version')==1,'Unknown source appearance inventory')
            collect(appearance)
            previous=read(appearance['previous_derived_assembly_ref'])
            require(previous=={k:v for k,v in document.items()if k!='source_appearance_states_ref'},'Appearance inventory changed its prior descriptor')
        if document.get('parent_derived_assembly_ref'):
            require(not parent, 'Only one exact approved parent assembly is supported')
            geometry_document(read(document['parent_derived_assembly_ref']), parent=True)

    geometry_document(derived)
    for key in ['source_scene', 'mechanics', 'proposal']:
        collect(read(original[key]))
    return [refs[key] for key in sorted(refs)]


def retained_findings(base):
    """Preserve the immediate review plus every previous retention record."""
    manifest = base['manifest']
    old = [copy.deepcopy(row['retained_prior_hosted_findings'])
           for row in manifest.get('repair_execution', [])
           if row.get('retained_prior_hosted_findings') is not None]
    require(len(old) <= 16, 'Bounded prior hosted review lineage required')
    return {
        'base_revision_id': base['id'], 'base_fingerprint': base['fingerprint'],
        'hosted_blocking_findings': base['hosted_blocking_findings'],
        'hosted_roles_completed': manifest['review']['hosted_roles_completed'],
        'hosted_review_status': manifest['review']['hosted_review_status'],
        'prior_retention_records': old,
        'original_findings_preserved_in_base_revision': True,
        'new_hosted_visual_review_performed_by_adoption': False,
        'separate_visual_review_refs_retained': True,
        'objections_are_not_resolved_by_structural_review': True,
    }


def build(*, contract, root=ROOT):
    root = Path(root).resolve()
    contract = copy.deepcopy(contract)
    require(set(contract) == {'candidate_id', 'base_revision_id', 'base_fingerprint',
                              'derived_assembly_ref'}, 'Exact geometry adoption contract required')
    candidate, base, memory = readonly_base(root, contract)
    derived = json.loads(verified_ref(root, contract['derived_assembly_ref']).read_bytes())
    validation = D.validate_files(derived, root=root)
    require(validation['source_pixels_recomputed'], 'Full source reproduction required before adoption')
    manifest = base['manifest']
    require(manifest['interaction_ref'] == derived['evidence_program'] and
            manifest['object_program_ref'] == derived['base_object_program_ref'],
            'Derivative does not match the exact candidate native program')
    source = manifest['source_ref']
    require(source['sha256'] in {derived['source']['sha256'], derived['source']['original_sha256']}
            and json.loads(candidate['source_json']) == source, 'Original owner recording changed')
    artifacts = artifact_inventory(root, manifest, derived)
    require(contract['derived_assembly_ref'] in artifacts or
            not any(r['path'] == contract['derived_assembly_ref']['path'] for r in artifacts),
            'Derivative pointer conflicted with the artifact inventory')
    if contract['derived_assembly_ref'] not in artifacts:
        artifacts.append(contract['derived_assembly_ref'])
        artifacts.sort(key=lambda row: row['path'])
    require(len(artifacts) <= 2048, 'Bounded cached artifact inventory exceeded')
    prior = retained_findings(base)
    repair = {
        'strategy': STRATEGY, 'derived_assembly_ref': contract['derived_assembly_ref'],
        'retained_prior_hosted_findings': prior,
        'native_program_and_scene_preserved': True,
        'original_native_assets_preserved': True,
        'declared_derived_assets_available': True,
        'source_mask_audit_summary': derived['source_audit_summary'],
        'full_validation_summary': validation,
        'new_api_calls': 0, 'new_gpu_calls': 0,
    }
    result = {
        'status': 'complete_with_gaps', 'run_id': manifest['run_id'],
        'interaction_ref': manifest['interaction_ref'],
        'object_program_ref': manifest['object_program_ref'],
        'scene_ref': manifest['scene_ref'], 'part_ledger_ref': manifest.get('part_ledger_ref'),
        'artifacts': artifacts, 'repair_execution': [repair],
        'new_api_calls': 0, 'new_gpu_calls': 0,
        'diagnostic': 'Cached, explicitly declared automatic geometry/kinematics derivative. '
            'Original native program and scene retained. Actual visual findings remain bound; '
            'structural reproduction does not establish physical fidelity or owner approval.',
    }
    review = review_revision({**result, 'source_ref': source}, root=root)
    require(review['role_count'] == 4 and review['all_required_roles_completed']
            and not review['blocking_failures'], 'Cached revision failed structural roles')
    return {
        'format': FORMAT, 'version': 1, 'contract': contract,
        'base_manifest_ref': base['manifest_ref'], 'frozen_memory_context': memory,
        'result': result, 'structural_preflight': review,
        'retained_prior_hosted_findings': prior,
        'new_api_calls': 0, 'new_gpu_calls': 0,
        'automatic_approval': False, 'automatic_publication': False,
    }


def adopt(plan, *, contract, root=ROOT, actor='nigel'):
    owner(actor)
    root = Path(root).resolve()
    expected = build(root=root, contract=contract)
    require(plan == expected, 'Prepared geometry adoption changed')
    contract = plan['contract']
    workflow = AdminWorkflow(root)
    token, plan_sha = identifier('lease'), digest(plan)
    with workflow.db() as db:
        db.execute('BEGIN IMMEDIATE')
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'],
                'Exact current base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['fingerprint'] == contract['base_fingerprint'], 'Exact base fingerprint changed')
        require(not db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?",
                               (time.time(),)).fetchone(), 'Another owner job is running')
        require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('running','queued')",
                               (candidate['id'],)).fetchone(), 'Candidate has an active job')
        for previous in db.execute("SELECT payload_json FROM jobs WHERE candidate_id=? AND kind='cached_local_revision'",
                                   (candidate['id'],)):
            require(json.loads(previous['payload_json']).get('adoption_plan_sha256') != plan_sha,
                    'This exact geometry adoption was already attempted; no automatic retry')
        payload = {
            'run_id': candidate['run_id'], 'input_path': candidate['input_path'],
            'source_ref': json.loads(candidate['source_json']), 'processing_consent': False,
            'base_revision': workflow._revision_row(row), 'repair_plan': None,
            'memory_context': plan['frozen_memory_context'],
            'adoption_plan_sha256': plan_sha, 'local_cached_review_only': True,
        }
        created = workflow._enqueue(db, candidate, 'cached_local_revision', payload, False)
        db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=? AND state='pending_consent'",
                   (token, time.time() + 300, now(), created['id']))
        db.execute("UPDATE candidates SET status='processing',updated_utc=? WHERE id=?",
                   (now(), candidate['id']))
        job = {'id': created['id'], 'kind': created['kind'], 'candidate_id': candidate['id'], **payload}
    try:
        result = copy.deepcopy(plan['result'])
        result['repair_execution'][0]['adoption_plan_sha256'] = plan_sha
        summary = workflow._finalize_revision(job, result, token, hosted_reviewer=None)
    except BaseException as error:
        with workflow.db() as db:
            changed = db.execute("UPDATE jobs SET state='failed',diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND state='running' AND lease_token=?",
                                 ('Cached geometry adoption stopped: ' + type(error).__name__,
                                  now(), job['id'], token)).rowcount
            if changed:
                db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",
                           (now(), candidate['id']))
        raise
    return {
        'status': 'adopted_private_declared_geometry_revision', 'candidate_id': candidate['id'],
        'base_revision_id': contract['base_revision_id'], 'job_id': job['id'],
        'plan_sha256': plan_sha, **summary,
        'retained_prior_hosted_findings': plan['retained_prior_hosted_findings'],
        'new_api_calls': 0, 'new_gpu_calls': 0,
        'automatic_approval': False, 'automatic_publication': False,
    }
