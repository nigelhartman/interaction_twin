"""Private bounded improvement policy and renewable execution leases.

This module never starts a model, publishes a revision or authors human memory.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
import threading
import time

TASK_PLAN = 'output/overnight-pipeline-20261102/task-budget-plan.json'
TARGETS = ('reasoning', 'tracking', 'reconstruction', 'mechanics')


def improvement_policy(root):
    from overnight_budget import plan
    plan(root)
    path = Path(root) / TASK_PLAN
    if path.is_symlink() or not path.is_file():
        raise ValueError('The authorized improvement budget plan is unavailable')
    return {'format': 'interaction-twin-authorized-improvement-policy', 'version': 1,
            'task_budget_plan': TASK_PLAN, 'task_budget_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'maximum_repairs': 2, 'minimum_improvement': .01,
            'publication_authorized': False, 'human_approval_required': True}


def validate_policy(policy, root):
    if policy != improvement_policy(root):
        raise ValueError('Authorized improvement policy changed or exceeds its fixed bounds')
    return policy


class LeaseLost(ValueError):
    pass


class WorkerLease:
    """Renew only an unexpired matching token. A stale worker cannot resurrect it."""
    def __init__(self, workflow, job_id, token, *, cancel=None, cancel_event=None, seconds=300, interval=20):
        self.workflow, self.job_id, self.token = workflow, job_id, token
        self.seconds, self.interval, self.cancel = seconds, interval, cancel
        self.cancel_event=cancel_event
        self.stop_event, self.lost = threading.Event(), threading.Event()
        self.thread = None

    def _lose(self):
        already=self.lost.is_set()
        self.lost.set()
        if self.cancel and not already:
            self.cancel()

    def guard(self, stage=None):
        if self.lost.is_set() or (self.cancel_event is not None and self.cancel_event.is_set()):
            self._lose()
            raise LeaseLost('Worker execution lease is no longer valid')
        with self.workflow.db() as db:
            row = db.execute('SELECT state,lease_token,lease_until FROM jobs WHERE id=?', (self.job_id,)).fetchone()
            valid = row and row['state'] == 'running' and row['lease_token'] == self.token and (row['lease_until'] or 0) > time.time()
            if valid and stage:
                db.execute('UPDATE jobs SET progress_json=?,updated_utc=? WHERE id=? AND lease_token=? AND state=?',
                    (json.dumps({'stage': str(stage)[:120], 'updated_utc': self.workflow.timestamp()}),
                     self.workflow.timestamp(), self.job_id, self.token, 'running'))
        if not valid:
            self._lose()
            raise LeaseLost('Worker execution lease changed or expired')

    def renew(self):
        with self.workflow.db() as db:
            changed = db.execute('UPDATE jobs SET lease_until=?,heartbeat_utc=? WHERE id=? AND state=? AND lease_token=? AND lease_until>?',
                (time.time() + self.seconds, self.workflow.timestamp(), self.job_id, 'running', self.token, time.time())).rowcount
        if changed != 1:
            self._lose()
            raise LeaseLost('Worker execution lease could not be renewed')

    def start(self):
        self.renew()
        def heartbeat():
            while not self.stop_event.wait(self.interval):
                try:
                    self.renew()
                except Exception:
                    self._lose()
                    break
        self.thread = threading.Thread(target=heartbeat, name='admin-worker-heartbeat', daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)


def review_assessment(report, manifest, *, root):
    """A trusted visual adapter supplies measured quality, never a Muse score.

    Missing/partial evidence stops automatic improvement. Model text is not an
    executable plan and cannot by itself qualify a repair as safe.
    """
    if not isinstance(report, dict) or report.get('status') != 'complete':
        return None
    from admin_muse_reviews import artifact_binding
    if report.get('artifact_binding') != artifact_binding(manifest):
        raise ValueError('Visual improvement review belongs to another revision')
    from admin_memory_context import private_path
    from agent_memory import digest
    reference = report.get('report_ref')
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Visual improvement review requires an exact private report')
    path = private_path(reference['path'], root=root)
    raw = path.read_bytes()
    if len(raw) != reference['bytes'] or len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Visual improvement review changed')
    # The wrapper is stored inside the exact report, not just transport metadata.
    saved = json.loads(raw)
    if saved.get('format')=='interaction-twin-source-bound-measured-visual-review':
        critic_ref=saved['raw_critic_ref'];critic_path=private_path(critic_ref['path'],root=root)
        critic_raw=critic_path.read_bytes()
        if len(critic_raw)!=critic_ref['bytes']or hashlib.sha256(critic_raw).hexdigest()!=critic_ref['sha256']:
            raise ValueError('Underlying visual critic report changed')
        from native_visual_recovery_v2 import verify_any_report
        verify_any_report(json.loads(critic_raw),root=root)
        from native_scene_audit import validate_manifest
        from review_agents import verified_ref
        audit=validate_manifest(verified_ref(root,saved['audit_ref'],4*1024*1024),root=root)
        if saved.get('evaluation_plan_ref'):
            from admin_evaluation_plan import verify_audit,comparison_key
            verify_audit(saved['evaluation_plan_ref'],audit,root=root)
            if saved.get('improvement_assessment',{}).get('comparison_key')!=comparison_key(saved['evaluation_plan_ref'],root=root):
                raise ValueError('Measured comparison key differs from frozen evaluation plan')
    assessment = saved.get('improvement_assessment')
    if assessment is None and report.get('improvement_assessment') is None:
        return None
    if saved.get('artifact_binding') != artifact_binding(manifest):
        raise ValueError('Measured quality report is not bound to the exact revision')
    if not isinstance(assessment, dict) or report.get('improvement_assessment') != assessment:
        raise ValueError('Visual improvement assessment is not bound to its report')
    if assessment.get('measurement_kind') != 'deterministic_local_metrics':
        return None
    comparison_key=assessment.get('comparison_key')
    if not isinstance(comparison_key,str) or not re.fullmatch('[a-f0-9]{64}',comparison_key):
        raise ValueError('Measured improvement requires a frozen comparison basis')
    fields = ('blocking_issues', 'missing_components', 'alignment_error', 'coverage')
    values = assessment.get('metrics', {})
    if set(values) != set(fields):
        return None
    for key in fields:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured improvement metric')
    if any(type(values[k]) is not int for k in fields[:2]) or values['coverage'] > 1:
        raise ValueError('Invalid counted improvement metric')
    targets = assessment.get('repair_targets', [])
    if not isinstance(targets, list) or len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
        raise ValueError('Visual improvement targets are outside trusted adapters')
    return {'metrics': values, 'comparison_key':comparison_key, 'evaluation_plan_ref':saved.get('evaluation_plan_ref'),
            'targets': [t for t in TARGETS if t in targets],
            'automatic_repair_safe': assessment.get('automatic_repair_safe') is True,
            'assessment_sha256': digest(assessment), 'report_ref': reference}


def improves(new, old, minimum=.01):
    """Conservative Pareto improvement: no measured metric may regress."""
    higher = ('coverage',)
    deltas = [(new[k] - old[k]) if k in higher else (old[k] - new[k]) for k in old]
    return all(d >= -1e-9 for d in deltas) and any(d >= minimum for d in deltas)
