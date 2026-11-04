"""Loopback app boundary for immutable, budgeted task understanding.

GETs inspect local artifacts only. POSTs select a published case and one bounded
operation; browsers cannot supply filesystem paths, model names or commands.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import revisions

ROOT = Path(__file__).resolve().parent
JOB_ID = re.compile(r'intel-[0-9a-f]{24}\Z')
NO_WORKER_SNAPSHOT = object()


def write(path, value):
    from meta_api import write as atomic_write
    atomic_write(path, value)


def canonical_request(body, *, root=ROOT):
    if not isinstance(body, dict) or set(body) - {'case_id', 'mode', 'question', 'hosted_reasoning_consent'}:
        raise ValueError('Choose a published case, operation and optional question')
    if body.get('mode') not in {'analysis', 'ask'}:
        raise ValueError('Choose analysis or ask')
    if body.get('hosted_reasoning_consent') is not True:
        raise ValueError('Meta analysis sends saved frames and procedure context to Meta; consent is required')
    path, doc, digest = revisions.published_source(root, body.get('case_id'))
    question = body.get('question')
    if body['mode'] == 'ask':
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1500:
            raise ValueError('Write a question of at most 1500 characters')
        question = question.strip()
    elif question is not None:
        raise ValueError('A general analysis does not accept a question')
    return {'case_id': body['case_id'], 'interaction_path': str(path.relative_to(Path(root).resolve())),
            'interaction_sha256': digest, 'mode': body['mode'], 'question': question}


def cache_rows(path, *, root=ROOT):
    from task_intelligence import cached_analyses
    result = []
    for row in cached_analyses(path, root=root):
        record = row['record']
        result.append(record)
    return result


def record(case_id, *, root=ROOT, active_job_id=NO_WORKER_SNAPSHOT):
    """Inspect saved results; optionally reconcile UI job state with this server.

    Omitting active_job_id leaves persisted jobs alone for CLI/offline callers.
    Passing None explicitly means the current server has no intelligence worker.
    An interrupted status is a read-only view: original job/accounting artifacts
    remain intact and no failed/incomplete paid attempt is automatically retried.
    """
    from sim_readiness import audit_simulation_readiness
    from task_intelligence import plan_analysis
    path, doc, digest = revisions.published_source(root, case_id)
    # Snapshot jobs BEFORE reading completed artifacts. The worker atomically
    # saves analysis.json before job=complete. If it finishes during this read,
    # returning the older running state only causes one extra harmless poll;
    # reading the cache first could report complete without its saved answer.
    jobs = []
    folder = Path(root).resolve() / 'data/task-intelligence-jobs'
    for item in folder.glob('intel-*/job.json'):
        # A malformed local job should not make saved evidence unavailable, and
        # symlinks must not turn an arbitrary/private file into a public job.
        if item.is_symlink() or item.parent.is_symlink() or not item.resolve().is_relative_to(folder):
            continue
        if not JOB_ID.fullmatch(item.parent.name) or item.stat().st_size > 16384:
            continue
        try:
            value = json.loads(item.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict) or value.get('id') != item.parent.name:
            continue
        if value.get('case_id') == case_id and value.get('interaction_sha256') == digest:
            if (active_job_id is not NO_WORKER_SNAPSHOT and value.get('status') in {'created', 'queued', 'running'}
                    and value['id'] != active_job_id):
                value = dict(value, saved_status=value['status'], status='interrupted',
                             worker_state='not_active_in_current_server', automatic_retry=False,
                             error='This saved analysis job has no active worker in the current local server. A previous request may have been charged; inspect its saved receipt and budget before another attempt. Original artifacts are preserved.')
            jobs.append(value)
    jobs.sort(key=lambda row: row.get('created_at', ''), reverse=True)
    rows = cache_rows(path, root=root)
    analyses = [row for row in rows if row.get('contract', {}).get('mode') == 'analysis']
    # Content-addressed directory order is not chronological. Prefer the exact
    # current contract, then a fully verified historical settings match.
    analysis = None
    analysis_selection = None
    if analyses:
        planned = plan_analysis(path, root=root, mode='analysis', question=None,
                                max_images=12, max_output_tokens=12000)
        expected = planned['request_key']
        analysis = next((row for row in analyses if row.get('request_key') == expected), None)
        selection_kind = 'current_exact_contract'
        if analysis is None:
            eligible = [row for row in analyses if row['contract'].get('max_output_tokens') == 12000
                        and row['contract'].get('question') is None
                        and row['contract'].get('images') == planned['contract']['images']]
            analysis = eligible[-1] if eligible else None
            selection_kind = 'verified_historical_contract'
        if analysis is None:
            # The approved CLI experiment used the same image/prompt contract
            # with a 16k output ceiling. Display that exact verified record;
            # never relabel it as the browser worker's 12k request or infer.
            alternate = plan_analysis(path, root=root, mode='analysis', question=None,
                                      max_images=12, max_output_tokens=16000)
            analysis = next((row for row in analyses if row.get('request_key') == alternate['request_key']), None)
            selection_kind = 'current_exact_16000_output_contract'
        if analysis is not None:
            analysis_selection = {'kind': selection_kind, 'requested_request_key': expected,
                                  'served_request_key': analysis['request_key'],
                                  'served_contract': analysis['contract'],
                                  'immutable_record_modified': False,
                                  'new_api_call': False}
            if analysis.get('recovery') or analysis.get('normalization'):
                analysis_selection['omission_normalization'] = analysis.get('recovery') or analysis.get('normalization')
    answers = [row for row in rows if row.get('contract', {}).get('mode') == 'ask']
    return {'case_id': case_id, 'interaction_sha256': digest,
            'analysis': analysis, 'analysis_selection': analysis_selection, 'answers': answers,
            'readiness': audit_simulation_readiness(doc), 'job': jobs[0] if jobs else None,
            'api_calls_on_get': False, 'physical_accuracy_verified': False}


class IntelligenceRunner:
    """One serial worker compatible with the workspace's model-run lock."""
    def __init__(self, request, *, root=ROOT):
        self.root = Path(root).resolve()
        self.request = request
        self.id = 'intel-' + os.urandom(12).hex()
        self.directory = self.root / 'data/task-intelligence-jobs' / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cancelled = threading.Event()
        self.state = dict(request, id=self.id, status='created',
                          created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                          physical_accuracy_verified=False, one_paid_attempt_maximum=True)
        self.save()

    def save(self):
        write(self.directory / 'job.json', self.state)

    def cancel(self):
        self.cancelled.set()

    def execute(self):
        from task_intelligence import run_analysis
        if self.cancelled.is_set():
            self.state['status'] = 'cancelled'; self.save(); return
        self.state['status'] = 'running'; self.save()
        try:
            # Revalidate the publication and source before any hosted request.
            path, _, digest = revisions.published_source(self.root, self.request['case_id'])
            if digest != self.request['interaction_sha256']:
                raise ValueError('The published program changed; reload before analysis')
            output = run_analysis(path, root=self.root, mode=self.request['mode'],
                                  question=self.request['question'], allow_paid_api=True,
                                  max_images=12, max_output_tokens=12000, timeout=600)
            self.state.update(status='complete', request_key=output['request_key'] if 'request_key' in output else output.get('contract', {}).get('request_key'),
                              result_source='immutable_task_intelligence_cache')
        except Exception:
            # Never echo provider exception text, prompts, credentials or headers.
            self.state.update(status='failed', error='Meta analysis stopped. Inspect its saved accounting receipt and budget status before another attempt. Original project outputs remain available.')
        self.state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.save()


def cached_match(request, *, root=ROOT):
    path = Path(root) / request['interaction_path']
    from task_intelligence import plan_analysis
    planned = plan_analysis(path, root=root, mode=request['mode'], question=request['question'],
                            max_images=12, max_output_tokens=12000)
    key = planned['request_key']
    for row in cache_rows(path, root=root):
        if row.get('request_key', row.get('contract', {}).get('request_key')) == key:
            return row
        # Loader/code updates do not justify charging again for an identical
        # saved inference payload. Only already fully verified records qualify;
        # source/context, exact prompt, both schemas, image bytes, model, total
        # output bound and question must match the current request contract.
        fields = ('mode', 'model', 'interaction_sha256', 'source_sha256', 'context_sha256',
                  'schema_sha256', 'provider_schema_sha256', 'provider_schema_projection_policy',
                  'prompt_sha256', 'images', 'max_output_tokens', 'question')
        contract = row.get('contract', {})
        if all(field in contract and contract[field] == planned['contract'].get(field) for field in fields):
            return row
    return None
