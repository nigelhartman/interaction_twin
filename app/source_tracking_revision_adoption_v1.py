"""Adopt a verified source-tracking revision without inference or publication.

The existing exact-source cached-adoption validator supplies the bounded asset
inventory. This adapter gives source refinements their own explicit strategy and
refuses to discard a previous hosted review. It leases only its named candidate;
the general queue and every previous revision/decision remain unchanged.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from admin_workflow import AdminWorkflow, identifier, now, owner
from agent_memory import digest
import laptop_cached_revision_adoption_20261103 as Base

ROOT = Base.ROOT
FORMAT = 'interaction-twin-source-tracking-cached-adoption'
STRATEGY = 'explicit_source_tracking_refinement_adoption_v1'


def build(*, contract, root=ROOT):
    contract = copy.deepcopy(contract)
    _, previous, _ = Base.readonly_base(root, contract)
    review = previous['manifest']['review']
    Base.require(not review.get('hosted_roles') and not any(
        row.get('retained_prior_hosted_findings')
        for row in previous['manifest'].get('repair_execution', [])),
        'Prior hosted findings require their own explicit retention adapter')
    plan = Base.build(root=root, contract=contract)
    plan['format'] = FORMAT
    plan['result']['repair_execution'][0]['strategy'] = STRATEGY
    plan['result']['diagnostic'] = (
        'Previously computed source-bound SAM/Muse tracking refinement adopted '
        'locally; this adoption performs no inference, geometry edits, approval '
        'or publication. Source visibility and physical limitations remain.')
    return plan


def adopt(plan, *, contract, root=ROOT, actor='nigel'):
    owner(actor)
    root = Path(root).resolve()
    expected = build(contract=contract, root=root)
    Base.require(plan == expected, 'Prepared cached revision changed')
    contract = plan['contract']
    workflow = AdminWorkflow(root)
    token, plan_sha = identifier('lease'), digest(plan)
    with workflow.db() as db:
        db.execute('BEGIN IMMEDIATE')
        candidate = workflow._row(db, 'candidates', contract['candidate_id'])
        Base.require(candidate['current_revision_id'] == contract['base_revision_id'],
                     'Exact current base revision changed')
        row = workflow._row(db, 'revisions', contract['base_revision_id'])
        Base.require(row['fingerprint'] == contract['base_fingerprint'] and
                     json.loads(candidate['source_json']) == contract['source_ref'],
                     'Exact base/source binding changed')
        Base.require(not db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN ('running','queued')",
                                    (candidate['id'],)).fetchone(), 'Candidate has an active job')
        Base.require(not db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?",
                                    (time.time(),)).fetchone(), 'Another owner job is running')
        for prior in db.execute("SELECT payload_json FROM jobs WHERE candidate_id=? AND kind='cached_local_revision'",
                                (candidate['id'],)):
            Base.require(json.loads(prior['payload_json']).get('adoption_plan_sha256') != plan_sha,
                         'This cached adoption was already attempted; no automatic retry')
        payload = {'run_id': candidate['run_id'], 'input_path': candidate['input_path'],
                   'source_ref': contract['source_ref'], 'processing_consent': False,
                   'base_revision': workflow._revision_row(row), 'repair_plan': None,
                   'memory_context': plan['frozen_memory_context'],
                   'adoption_plan_sha256': plan_sha, 'local_cached_review_only': True}
        created = workflow._enqueue(db, candidate, 'cached_local_revision', payload, False)
        db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=? AND state='pending_consent'",
                   (token, time.time()+300, now(), created['id']))
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
                                 ('Source tracking adoption stopped: '+type(error).__name__, now(), job['id'], token)).rowcount
            if changed:
                db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",
                           (now(), candidate['id']))
        raise
    return {'status': 'adopted_private_source_tracking_revision',
            'candidate_id': candidate['id'], 'base_revision_id': contract['base_revision_id'],
            'job_id': job['id'], 'plan_sha256': plan_sha, **summary,
            'new_api_calls': 0, 'new_gpu_calls': 0,
            'automatic_approval': False, 'automatic_publication': False}
