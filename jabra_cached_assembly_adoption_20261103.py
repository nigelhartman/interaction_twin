"""Prepare one cached Jabra derivative revision; never approve or run inference.

The original hosted objections remain bound through the immutable base revision
and are explicitly retained in the new repair record. A fresh structural review
does not supersede those findings or imply a fresh hosted visual judgement.
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
import refined_assembly_contract as D

ROOT = Path(__file__).resolve().parent
CONTRACT = {
    'candidate_id': 'candidate_9d79bed5fb824f1c91176fe995d3d7b5',
    'base_revision_id': 'revision_fe833c66d0284ab0aa4f45d19386dd43',
    'base_fingerprint': 'a9bb41d91eede163190641c9e26150ea449611394642e6d7f9bc5705699bfafa',
    'derived_assembly_ref': {'path': 'data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json',
        'sha256': 'd5e603a37f62bbd9d8c269f98bcd982d30d37de43b5cff8c5c4357337b95744f', 'bytes': 27216},
}


def require(value, message):
    if not value:
        raise ValueError(message)


def _base(root, contract):
    workflow = object.__new__(AdminWorkflow); workflow.root = root
    workflow.folder = root/'.private/admin-workflow'; workflow.path = workflow.folder/'state.sqlite3'
    require(workflow.path.is_file() and not workflow.path.is_symlink(), 'Existing owner workflow required')
    with sqlite3.connect(workflow.path.as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'], 'Exact current Jabra base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['fingerprint'] == contract['base_fingerprint'], 'Exact Jabra base fingerprint changed')
        # Read-only verification admits inspecting the saved hosted findings;
        # it is not an approval or dismissal of their remaining objections.
        base = workflow._verify_revision(row, allow_hosted_override=True)
        require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('queued','running')", (candidate['id'],)).fetchone(), 'Candidate has an active job')
        job = workflow._row(db, 'jobs', base['manifest']['job_id'])
        payload = json.loads(job['payload_json'])
    return dict(candidate), base, payload['memory_context']


def build(*, root=ROOT, contract=None):
    root = Path(root).resolve(); contract = copy.deepcopy(CONTRACT if contract is None else contract)
    candidate, base, memory = _base(root, contract)
    derived = D.read(contract['derived_assembly_ref'], root); D.validate(derived, root=root)
    manifest = base['manifest']
    require(manifest['interaction_ref'] == derived['evidence_program'] and manifest['object_program_ref'] == derived['base_object_program_ref'], 'Derived assembly does not match the exact candidate source program')
    source = manifest['source_ref']
    require(source['sha256'] in {derived['source']['sha256'], derived['source']['original_sha256']} and
            json.loads(candidate['source_json']) == source, 'Original owner recording changed')
    original_program = D.read(derived['base_object_program_ref'], root)
    archives = {item['original_path']: item['archived_file'] for item in original_program['implementation_provenance']}
    refs = {item['path']: copy.deepcopy(item) for item in manifest['artifacts']}
    def collect(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                item = value
                if not item['path'].startswith('data/'):
                    replacement = archives.get(item['path'])
                    require(replacement and all(item[k] == replacement[k] for k in ['sha256','bytes']), 'Non-data reference lacks exact original code archive')
                    item = replacement
                path = verified_ref(root, item)
                require(item['path'] not in refs or refs[item['path']] == item, 'Conflicting cached artifact reference')
                refs[item['path']] = copy.deepcopy(item)
                return
            for child in value.values():
                if isinstance(child, (dict,list)):
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(contract['derived_assembly_ref'])
    # Follow only the typed core contracts. Native metadata can contain remote
    # machine receipt paths; it is a bound artifact, not an instruction to walk
    # every JSON value recursively through the working directory.
    collect(derived)
    report = D.read(derived['refinement_report_ref'], root)
    scene = D.read(derived['scene_ref'], root)
    audit = D.read(derived['source_alignment_audit_ref'], root)
    collect([report, scene, audit, original_program])
    for item in [report['original_scene'], *[original_program[k] for k in ['source_scene','mechanics','proposal']]]:
        collect(D.read(item, root))
    require(len(refs) <= 2048, 'Bounded cached artifact inventory exceeded')
    retained = [item['retained_prior_hosted_findings'] for item in manifest.get('repair_execution',[])
                if item.get('strategy') == 'declared_source_mask_refit_assembly_adoption_v1' and item.get('retained_prior_hosted_findings')]
    require(len(retained) <= 1, 'Ambiguous prior hosted review lineage')
    prior = copy.deepcopy(retained[0]) if retained else {
        'base_revision_id': base['id'], 'base_fingerprint': base['fingerprint'],
        'hosted_blocking_findings': base['hosted_blocking_findings'],
        'hosted_roles_completed': manifest['review']['hosted_roles_completed'],
        'hosted_review_status': manifest['review']['hosted_review_status'],
        'original_findings_preserved_in_base_revision': True,
        'new_hosted_visual_review_performed': False,
        'objections_are_not_resolved_by_structural_review': True}
    result = {'status': 'complete_with_gaps', 'run_id': manifest['run_id'],
              'interaction_ref': manifest['interaction_ref'], 'object_program_ref': manifest['object_program_ref'],
              'scene_ref': manifest['scene_ref'], 'part_ledger_ref': manifest.get('part_ledger_ref'),
              'artifacts': [refs[key] for key in sorted(refs)], 'new_api_calls': 0, 'new_gpu_calls': 0,
              'repair_execution': [{'strategy': 'declared_source_mask_refit_assembly_adoption_v1',
                  'derived_assembly_ref': contract['derived_assembly_ref'], 'retained_prior_hosted_findings': prior,
                  'native_program_and_meshes_unchanged': True, 'derived_lid_geometry_declared': True,
                  'source_mask_audit_summary': derived['source_audit_summary'], 'new_api_calls': 0, 'new_gpu_calls': 0}],
              'diagnostic': 'Cached declared derivative and source-mask refit; original native program retained. Previous hosted objections remain unresolved; explicit owner research-draft approval required.'}
    review = review_revision({**result, 'source_ref': source}, root=root)
    require(review['role_count'] == 4 and review['all_required_roles_completed'] and not review['blocking_failures'], 'Cached revision failed structural roles')
    return {'format': 'interaction-twin-declared-assembly-cached-adoption', 'version': 1, 'contract': contract,
            'base_manifest_ref': base['manifest_ref'], 'frozen_memory_context': memory,
            'result': result, 'structural_preflight': review, 'retained_prior_hosted_findings': prior,
            'new_api_calls': 0, 'new_gpu_calls': 0, 'automatic_approval': False, 'automatic_publication': False}


def adopt(plan, *, root=ROOT, actor='nigel', contract=None):
    owner(actor); root = Path(root).resolve()
    expected = build(root=root, contract=contract)
    require(plan == expected, 'Prepared declared assembly adoption changed')
    contract = plan['contract']; workflow = AdminWorkflow(root); token = identifier('lease'); plan_sha = digest(plan)
    with workflow.db() as db:
        db.execute('BEGIN IMMEDIATE')
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        require(candidate['current_revision_id'] == contract['base_revision_id'], 'Jabra base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        require(row['fingerprint'] == contract['base_fingerprint'], 'Jabra base fingerprint changed')
        require(not db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?", (time.time(),)).fetchone(), 'Another owner job is running')
        require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('running','queued')", (candidate['id'],)).fetchone(), 'Candidate has an active job')
        for prior in db.execute("SELECT payload_json FROM jobs WHERE candidate_id=? AND kind='cached_local_revision'", (candidate['id'],)):
            require(json.loads(prior['payload_json']).get('adoption_plan_sha256') != plan_sha, 'This exact adoption was already attempted')
        payload = {'run_id': candidate['run_id'], 'input_path': candidate['input_path'], 'source_ref': json.loads(candidate['source_json']),
                   'processing_consent': False, 'base_revision': workflow._revision_row(row), 'repair_plan': None,
                   'memory_context': plan['frozen_memory_context'], 'adoption_plan_sha256': plan_sha, 'local_cached_review_only': True}
        job = workflow._enqueue(db, candidate, 'cached_local_revision', payload, False)
        db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=? AND state='pending_consent'", (token,time.time()+300,now(),job['id']))
        db.execute("UPDATE candidates SET status='processing',updated_utc=? WHERE id=?", (now(),candidate['id']))
        job = {'id':job['id'],'kind':job['kind'],'candidate_id':candidate['id'],**payload}
    try:
        result = copy.deepcopy(plan['result']);result['repair_execution'][0]['adoption_plan_sha256'] = plan_sha
        summary = workflow._finalize_revision(job, result, token, hosted_reviewer=None)
    except BaseException as error:
        with workflow.db() as db:
            changed = db.execute("UPDATE jobs SET state='failed',diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND state='running' AND lease_token=?", ('Cached assembly adoption stopped: '+type(error).__name__,now(),job['id'],token)).rowcount
            if changed:
                db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",(now(),candidate['id']))
        raise
    return {'status':'adopted_private_declared_assembly_revision','candidate_id':candidate['id'],
            'base_revision_id':contract['base_revision_id'],'job_id':job['id'],'plan_sha256':plan_sha,**summary,
            'retained_prior_hosted_findings':plan['retained_prior_hosted_findings'],
            'new_api_calls':0,'new_gpu_calls':0,'automatic_approval':False,'automatic_publication':False}
