"""Isolated HTTP/workflow rehearsal using retained, byte-identical evidence.

This is a development verification tool, not an alternate model pipeline. It
creates its own temporary root, fixture owner session and deferred scheduler.
Only the fixed executor's retained-import/review paths run. The real application,
owner credentials, approvals, paid ledger and GPU configuration are untouched.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CASE = 'flow-capture-phone-20261103-v1--flow-procedure-1'
FEEDBACK = 'ISOLATED_REHEARSAL_PRIVATE_FEEDBACK: inspect the extra phone plate; this test does not repair geometry.'


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def confined_evidence(root, relative):
    """Refuse hidden/external paths before even checking or hashing a file."""
    from twin_catalog import CODE_INPUTS
    codes = {p for _, p in CODE_INPUTS}
    if (not isinstance(relative, str) or '\\' in relative or
            any(p.startswith('.') for p in Path(relative).parts) or
            not (relative.startswith('data/') or
                 relative.startswith('annotations/') and relative.endswith('.json') or
                 relative in codes)):
        raise ValueError('Only explicit ordinary evidence or declared compiler inputs may be copied')
    root = Path(root).resolve()
    target = root / relative
    if target.resolve() != target or any(p.is_symlink() for p in [target, *target.parents]):
        raise ValueError('Rehearsal evidence cannot escape through a symlink')
    return target


def copy_retained_case(source_root, destination):
    """Copy exact declared phone evidence; never walk private/credential trees."""
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    rows = json.loads((source_root / 'showcase/public/cases-index.json').read_text())['cases']
    row = copy.deepcopy(next(r for r in rows if r['id'] == CASE))
    program_path = confined_evidence(source_root, row['interaction_path'])
    _require(_sha(program_path) == row['original_sha256'], 'Retained program changed')
    doc = json.loads(program_path.read_text())
    paths = {row['interaction_path'], doc['source']['path']}
    paths.update(i['path'] for i in doc['inputs'])
    paths.update(i['path'] for i in doc['assets'])
    for e in doc['evidence']:
        paths.add(e['frame_path'])
        paths.update(o['mask_path'] for o in e['observations'])
    for asset in doc['assets']:
        folder = Path(asset['path']).parent
        metadata = (folder / 'metadata.json').as_posix()
        paths.add(metadata)
        saved = json.loads(confined_evidence(source_root, metadata).read_text())
        for role in ['image', 'mask']:
            paths.add((folder / saved['source'][role + '_path']).as_posix())
    inventory = []
    for relative in sorted(paths):
        original = confined_evidence(source_root, relative)
        target = confined_evidence(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        digest = _sha(original)
        _require(_sha(target) == digest, 'Copied retained artifact differs')
        inventory.append({'path': relative, 'sha256': digest, 'bytes': target.stat().st_size})
    for name in ['workspace/cases.json', 'workspace/cases-index.json']:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'cases': [row]}))
    from admin_execution_policy import TASK_PLAN
    plan = destination / TASK_PLAN
    plan.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_root / TASK_PLAN, plan)
    return row, doc, inventory


class DeferredScheduler:
    """Record scheduling requests without executing the model queue."""
    def __init__(self): self.calls = []
    def submit(self, function, *args, **kwargs): self.calls.append((function, args, kwargs))
    def shutdown(self, **kwargs): pass


class BrowserClient:
    """Actual loopback HTTP requests; cookies stay only in temporary memory."""
    def __init__(self, port):
        self.port, self.cookie, self.csrf = port, '', ''
        self.trace = []
    def request(self, path, method='GET', body=None, *, owner=False, headers=None):
        outgoing = dict(headers or {})
        if method == 'POST':
            outgoing.setdefault('Origin', f'http://127.0.0.1:{self.port}')
            outgoing.setdefault('Content-Type', 'application/json')
        if owner:
            outgoing['Cookie'] = self.cookie
            outgoing.setdefault('X-CSRF-Token', self.csrf)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        try:
            connection.request(method, path, raw, outgoing)
            response = connection.getresponse()
            data = response.read()
            cookie = response.getheader('Set-Cookie')
            mime = response.getheader('Content-Type', '')
            value = json.loads(data) if 'application/json' in mime else data
            self.trace.append({'method': method, 'route': path.split('?')[0], 'status': response.status})
            return response.status, value, cookie
        finally:
            connection.close()
    def login(self):
        code, status, cookie = self.request('/api/admin/auth')
        _require(code == 200, 'Fixture preauth status failed')
        self.cookie, self.csrf = cookie.split(';', 1)[0], status['csrf_token']
        code, status, cookie = self.request('/api/admin/login', 'POST',
            {'password': 'Isolated rehearsal fixture password'}, owner=True)
        _require(code == 200 and status['authenticated'], 'Fixture owner login failed')
        self.cookie, self.csrf = cookie.split(';', 1)[0], status['csrf_token']


def _budget_failure_recovery(root, source):
    """Independent real ledger/lease tests, with no transport or fake usage."""
    from admin_workflow import AdminWorkflow
    from admin_execution_policy import WorkerLease, LeaseLost
    from meta_budget import BudgetLedger, BudgetError, BudgetExceeded, TextPricing, estimate_text
    workflow = AdminWorkflow(root)
    ledger = BudgetLedger(root / '.state/rehearsal-budget.sqlite3', cap_usd='2', per_request_cap_usd='1')
    pricing = TextPricing('offline-reservation-rehearsal', '1', '1', 'offline-test-only', '2026-11-03')
    estimate = estimate_text(pricing, input_tokens=1_000_000, max_output_tokens=0)
    ledger.reserve('isolated-before-dispatch', estimate, model=pricing.model, endpoint='/offline-never-sent', kind='text')
    ledger.release_no_charge('isolated-before-dispatch', reason='cancelled_before_transport')
    ledger.reserve('isolated-uncertainty-state', estimate, model=pricing.model, endpoint='/offline-never-sent', kind='text')
    ledger.mark_started('isolated-uncertainty-state')
    ledger.mark_uncertain('isolated-uncertainty-state', 'timeout')
    try:
        ledger.reserve('blocked-next', estimate, model=pricing.model, endpoint='/offline-never-sent', kind='text')
    except BudgetExceeded: pass
    else: raise AssertionError('Uncertain cost state did not block admission')
    try: ledger.mark_started('isolated-uncertainty-state')
    except BudgetError: pass
    else: raise AssertionError('Same uncertain request could be dispatched again')
    q = workflow.submit_upload(run_id='isolated-failure-and-resume', input_path=source, processing_consent=True)
    calls = []
    def unavailable(job):
        calls.append(job)
        return {'status': 'pending', 'diagnostic': 'Offline rehearsal: unavailable GPU; no model dispatched.'}
    _require(workflow.worker_once(unavailable)['status'] == 'pending', 'Unavailable worker did not preserve pending state')
    _require(workflow.worker_once(unavailable)['status'] == 'idle' and len(calls) == 1, 'Pending worker silently retried')
    workflow.resume_initial(q['candidate']['id'])
    workflow.worker_once(unavailable)
    _require(calls[0]['run_id'] == calls[1]['run_id'] and calls[0]['source_ref'] == calls[1]['source_ref'] and
        calls[0]['memory_context'] == calls[1]['memory_context'], 'Recovery changed paid cache/source/context identity')
    leased = workflow.submit_upload(run_id='isolated-lease-cancel', input_path=source, processing_consent=True)
    with workflow.db() as db:
        db.execute("UPDATE jobs SET state='running',lease_token='isolated-test',lease_until=? WHERE id=?",
                   (time.time() - 1, leased['job']['id']))
    canceled = []
    lease = WorkerLease(workflow, leased['job']['id'], 'isolated-test', cancel=lambda: canceled.append(True))
    try: lease.guard('expired-isolated-lease')
    except LeaseLost: pass
    else: raise AssertionError('Expired worker lease accepted work')
    _require(canceled == [True] and workflow.recover_expired_jobs() == 1, 'Lease loss did not stop and recover worker')
    return {'budget_unknown_state_blocks_admission': True, 'same_request_restart_refused': True,
            'pre_dispatch_reservation_released': True, 'pending_model_work_not_retried': True,
            'explicit_resume_preserves_run_source_context': True, 'expired_lease_cancels_executor': True,
            'expired_job_marked_interrupted': True, 'provider_transport_performed': False,
            'ledger_scope': 'synthetic_reservation_state_in_temporary_test_ledger_not_actual_spend'}


def run(*, source_root=ROOT):
    """Full loopback rehearsal; its approvals exist only in a disposable root."""
    import admin_auth
    import admin_http
    import admin_pipeline
    import curated_publication
    from admin_workflow import AdminWorkflow
    from curated_twin_catalog import CuratedTwinCatalog
    from object_program import ref
    from scripts import serve_workspace as server_code
    from meta_api import MetaClient
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='interaction-twin-isolated-admin-') as temporary:
        root = Path(temporary).resolve()
        row, program, inventory = copy_retained_case(source_root, root)
        admin_auth.configure_password(root, 'Isolated rehearsal fixture password')
        curated_publication.build(root=root)
        real_static, real_schedule = server_code.static_path, admin_http.schedule
        with ExitStack() as stack:
            for module in [server_code, admin_http, admin_pipeline]: stack.enter_context(patch.object(module, 'ROOT', root))
            stack.enter_context(patch.object(server_code, 'static_path', side_effect=lambda route: real_static(route, root=root)))
            stack.enter_context(patch.object(server_code, 'public_url', side_effect=lambda path: '/files/' + path.relative_to(root).as_posix()))
            stack.enter_context(patch.object(admin_http, 'schedule', side_effect=lambda server: real_schedule(server, root=root)))
            transport = stack.enter_context(patch.object(MetaClient, 'request', side_effect=AssertionError('Paid transport is forbidden during offline rehearsal')))
            class QuietHandler(server_code.Handler):
                def log_message(self, *args): pass
            server = server_code.WorkspaceServer(('127.0.0.1', 0), QuietHandler)
            server.executor.shutdown(wait=True)
            server.executor = DeferredScheduler()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = BrowserClient(server.server_address[1])
                _require(client.request('/api/admin/snapshot')[0] == 401, 'Anonymous admin access allowed')
                _require(client.request('/api/upload', 'POST', b'blocked', headers={'Content-Type': 'application/octet-stream', 'X-Filename': 'blocked.mp4'})[0] == 401, 'Anonymous upload allowed')
                _require(not (root / 'data/uploads').exists(), 'Anonymous request wrote upload data')
                client.login()
                _require(client.request('/api/upload', 'POST', b'blocked', owner=True, headers={'X-CSRF-Token': 'wrong', 'Content-Type': 'application/octet-stream', 'X-Filename': 'blocked.mp4'})[0] == 403, 'Bad CSRF allowed upload')
                code, upload, _ = client.request('/api/upload', 'POST', (root / program['source']['path']).read_bytes(), owner=True,
                    headers={'Content-Type': 'application/octet-stream', 'X-Filename': 'retained-phone-rehearsal.mp4'})
                _require(code == 201 and _sha(root / upload['input_path']) == program['source']['sha256'], 'Actual video upload changed bytes')
                _require(client.request(upload['url'])[0] == 404, 'Private owner upload exposed')
                code, submitted, _ = client.request('/api/admin/candidates', 'POST', {'input_path': upload['input_path'], 'processing_consent': False}, owner=True)
                _require(code == 202 and submitted['candidate']['status'] == 'pending_consent', 'Consent not preserved')
                cid = submitted['candidate']['id']
                workflow = AdminWorkflow(root)
                executor = admin_pipeline.FixedExecutor(root=root)
                _require(workflow.worker_once(executor)['status'] == 'idle', 'Non-consented candidate became runnable')
                # This is the application's exact retained-import path. Nothing
                # claims that the freshly uploaded bytes were newly inferred.
                result = {'status': 'complete_with_gaps', 'run_id': submitted['candidate']['run_id'],
                    'interaction_ref': ref(root / row['interaction_path'], root),
                    'artifacts': [ref(root / a['path'], root) for a in program['assets']],
                    'new_api_calls': 0, 'new_gpu_calls': 0,
                    'diagnostic': 'ISOLATED REHEARSAL: exact retained outputs; no new inference or geometry repair.'}
                from agent_memory import private_directory
                recipe = private_directory(root, 'admin-imports') / (submitted['candidate']['run_id'] + '.json')
                recipe.write_text(json.dumps({'source_ref': ref(root / upload['input_path'], root), 'result': result}))
                recipe.chmod(0o600)
                _require(client.request(f'/api/admin/candidates/{cid}/process', 'POST', {'processing_consent': True}, owner=True)[0] == 202, 'Explicit consent failed')
                _require(workflow.worker_once(executor)['status'] == 'complete', 'Retained import/review failed')
                first = workflow.detail(cid)['revisions'][0]
                _require(first['manifest']['review']['role_count'] == 4 and not first['manifest']['review']['blocking_failures'], 'Four structural roles did not complete')
                _require(client.request('/api/library')[1]['total'] == 0, 'Unapproved candidate leaked to library')
                _require(client.request('/files/' + program['assets'][0]['path'])[0] == 404, 'Unapproved model exposed')
                code, rejected, _ = client.request(f'/api/admin/candidates/{cid}/reject', 'POST', {'revision_id': first['id'], 'feedback': FEEDBACK, 'repair_targets': ['reconstruction']}, owner=True)
                _require(code == 200 and rejected['repair_plan']['targets'] == ['reconstruction'], 'Typed feedback routing failed')
                remembered = workflow.memory.retrieve(source_sha256=program['source']['sha256'])
                _require(FEEDBACK in json.dumps(remembered['lessons']), 'Source-scoped feedback was not available to a future repair')
                code, repaired, _ = client.request(f'/api/admin/candidates/{cid}/repair', 'POST', {'revision_id': first['id'], 'plan_id': rejected['repair_plan']['id'], 'processing_consent': True}, owner=True)
                _require(code == 202, 'Repair queue request failed')
                pending = workflow.worker_once(executor)
                _require(pending['status'] == 'pending' and 'no resumable pipeline receipt' in pending['result']['diagnostic'], 'Unavailable fresh repair did not stop explicitly')
                _require(workflow.worker_once(executor)['status'] == 'idle', 'Pending repair silently retried')
                _require(client.request(f'/api/admin/candidates/{cid}/review', 'POST', {'revision_id': first['id'], 'processing_consent': True}, owner=True)[0] == 202, 'Retained review revision failed to queue')
                _require(workflow.worker_once(executor)['status'] == 'complete', 'Retained-only review revision failed')
                second = workflow.detail(cid)['revisions'][0]
                _require(second['id'] != first['id'] and second['manifest']['interaction_ref'] == first['manifest']['interaction_ref'], 'Immutable review revision not distinct')
                _require(client.request(f'/api/admin/candidates/{cid}/approve', 'POST', {'revision_id': first['id'], 'expected_fingerprint': first['fingerprint']}, owner=True)[0] == 400, 'Stale revision approval accepted')
                _require(client.request(f'/api/admin/candidates/{cid}/approve', 'POST', {'revision_id': second['id'], 'expected_fingerprint': 'f' * 64}, owner=True)[0] == 400, 'Stale fingerprint accepted')
                code, approval, _ = client.request(f'/api/admin/candidates/{cid}/approve', 'POST', {'revision_id': second['id'], 'expected_fingerprint': second['fingerprint']}, owner=True)
                _require(code == 200 and approval['public_snapshot']['records'] == 1, 'Exact fixture approval/publication failed')
                public = client.request('/api/library')[1]
                _require(public['total'] == 1 and FEEDBACK not in json.dumps(public), 'Public snapshot leaked private feedback')
                descriptor = CuratedTwinCatalog(root).describe(CASE)
                _require(descriptor['policy']['simulation_ready'] is False, 'MCP lost simulation uncertainty')
                code, raw, _ = client.request('/files/' + program['assets'][0]['path'])
                _require(code == 200 and hashlib.sha256(raw).hexdigest() == program['assets'][0]['sha256'], 'Public model not exact approved bytes')
                published = json.loads((root / 'showcase/public/public-assets.json').read_text())
                _require(all(not x['path'].startswith(('.private/', '.secrets/')) for x in published['assets']), 'Private path published')
                fixture_asset = root / program['assets'][0]['path']
                saved_bytes = fixture_asset.read_bytes()
                fixture_asset.write_bytes(saved_bytes[:-1] + bytes([saved_bytes[-1] ^ 1]))
                _require(client.request('/files/' + program['assets'][0]['path'])[0] == 404, 'Changed approved model bytes were served')
                try: CuratedTwinCatalog(root).asset(CASE, program['assets'][0]['track_id'], binary=True)
                except ValueError: pass
                else: raise AssertionError('MCP accepted changed approved model bytes')
                fixture_asset.write_bytes(saved_bytes)
                _require(client.request('/files/' + program['assets'][0]['path'])[0] == 200, 'Exact retained bytes could not recover the approved route')
                _require(not transport.called, 'Rehearsal attempted paid API transport')
                safeguards = _budget_failure_recovery(root, upload['input_path'])
                return {'format': 'interaction-twin-isolated-admin-rehearsal', 'version': 1, 'status': 'passed',
                    'execution_kind': 'offline_retained_artifact_http_rehearsal_not_fresh_model_inference',
                    'case_id': CASE, 'retained_interaction': result['interaction_ref'], 'retained_artifacts': inventory,
                    'checks': {'actual_loopback_http': True, 'fixture_owner_login': True,
                        'anonymous_admin_and_upload_blocked': True, 'csrf_required': True,
                        'video_upload_byte_identical': True, 'processing_consent_required': True,
                        'deferred_model_scheduler_never_executed': True, 'four_structural_roles_completed': True,
                        'unapproved_program_and_model_withheld': True, 'private_feedback_saved': True,
                        'source_scoped_memory_available_to_future_repair': True,
                        'typed_reconstruction_repair_queued': True, 'unavailable_fresh_repair_preserved_pending': True,
                        'retained_review_revision_created': True, 'stale_revision_and_fingerprint_refused': True,
                        'exact_fixture_owner_approval_published': True, 'public_http_and_mcp_retrieval': True,
                        'feedback_never_public': True, 'physical_accuracy_and_simulation_readiness_false': True,
                        'changed_approved_model_withheld_http_and_mcp': True,
                        'exact_original_bytes_restore_approved_route': True,
                        'original_retained_artifacts_unchanged': all(_sha(confined_evidence(source_root, i['path'])) == i['sha256'] for i in inventory)},
                    'safeguards': safeguards, 'http_trace': client.trace,
                    'fixture_candidate_id': cid, 'fixture_revision_ids': [first['id'], second['id']],
                    'fixture_only_approval': True, 'production_approvals_or_publication_changed': False,
                    'real_owner_credentials_read': False, 'paid_api_transport_calls': 0, 'gpu_calls': 0,
                    'new_model_inference': False, 'feedback_applied_to_geometry': False,
                    'remaining_validation': ['fresh upload through real hosted inference and GPU adapter',
                        'actual Muse feedback repair improves measured geometry or mechanics',
                        'physical contact, scale and simulator validation'],
                    'temporary_root_deleted_after_run': True, 'elapsed_s': round(time.time() - started, 3)}
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
