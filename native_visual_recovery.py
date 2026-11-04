"""Offline omission-only salvage of an accounted native visual review.

This module cannot dispatch inference. It accepts only the frozen review/parser
protocol below, preserves the unsuccessful attempt, and drops action/target
inconsistencies without interpreting or rewriting any model advice. Saved parser
output is not a saved provider envelope; the distinction is explicit in receipts.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3

import jsonschema

import meta_api
import native_visual_review as N

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-native-visual-review-recovery-report'
PREFIX = '.private/native-visual-recoveries/'
REVIEWER_SHA256 = '4152061bdc4372e1eb98fd4cdd225b253670cf4fa3a5441f5c3156c5f6ea4b65'
PARSER_MODULE_SHA256 = '56387e95bababb14f5ed9dddf9509cecb7718779ef7127a23e17e84b7e74ac46'
PARSER_METHOD_SHA256 = 'fcf0d83e4d9fc57397e03a61a3f95ff9f3ceef7c244b4b2a7d15a24ff4710aa8'
MISMATCH = 'Visual repair action does not match its fixed target'
TARGET_ACTIONS = {
    'tracking': {'resegment', 'retain_unknown'},
    'reconstruction': {'regenerate_component', 'retain_unknown'},
    'mechanics': {'refit_mechanism', 'retain_unknown'},
    'reasoning': {'reason_again', 'retain_unknown'},
    'capture_more_evidence': {'request_capture', 'retain_unknown'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def _parser_evidence():
    reviewer = Path(N.__file__).read_bytes()
    parser = Path(meta_api.__file__).read_bytes()
    method = inspect.getsource(meta_api.MetaClient.json_result).encode()
    require(hashlib.sha256(reviewer).hexdigest() == REVIEWER_SHA256,
            'Original visual reviewer implementation changed')
    require(hashlib.sha256(parser).hexdigest() == PARSER_MODULE_SHA256
            and hashlib.sha256(method).hexdigest() == PARSER_METHOD_SHA256,
            'Known structured parser controlflow changed')
    return {'original-reviewer.py.txt': reviewer, 'original-meta-api.py.txt': parser,
            'original-json-result.py.txt': method,
            'native-visual-recovery.py.txt': Path(__file__).read_bytes()}


def _projection(result, plan):
    """Validate every citation before omitting the narrowly invalid proposals."""
    jsonschema.validate(result, N.response_schema())
    require(result['role'] == plan['role'], 'Original visual reviewer role differs')
    parts = {row['id'] for row in plan['context']['components']}
    samples = {row['sample_id']: row for row in plan['context']['samples']}
    for finding in [*result['issues'], *result['repair_proposals']]:
        require(set(finding['component_ids']) <= parts and set(finding['sample_ids']) <= samples.keys(),
                'Original finding cites a foreign component or sample')
        require(set(finding['component_ids']) <= {samples[s]['component_id'] for s in finding['sample_ids']},
                'Original finding lacks a submitted sample for its component')
    try:
        N.validate_result(result, plan)
    except ValueError as error:
        require(str(error) == MISMATCH, 'Original failure is not solely an action-target mismatch')
    else:
        raise ValueError('A valid original review needs no offline recovery')
    retained = []
    discarded = []
    projected = copy.deepcopy(result)
    projected['repair_proposals'] = []
    for index, proposal in enumerate(result['repair_proposals']):
        if proposal['action'] not in TARGET_ACTIONS[proposal['target']]:
            discarded.append({'original_index': index, 'reason': 'action_target_mismatch',
                              'original_proposal': copy.deepcopy(proposal)})
        else:
            retained.append(index)
            projected['repair_proposals'].append(copy.deepcopy(proposal))
    require(discarded and retained, 'Recovery requires an inconsistent proposal and a remaining valid proposal')
    # This catches verdict contradictions hidden behind the first mapping error.
    N.validate_result(projected, plan)
    return projected, retained, discarded


def _settled_proof(receipt, root):
    require(receipt.get('status') == 'complete_accounted' and receipt.get('provider') == 'meta_api'
            and receipt.get('kind') == 'text' and receipt.get('model') == N.MODEL
            and receipt.get('endpoint') == '/v1/chat/completions'
            and receipt.get('automatic_retries') is False
            and isinstance(receipt.get('request_id'), str)
            and re.fullmatch(r'meta-[a-f0-9]{32}', receipt['request_id']) is not None,
            'Only an accounted successful original visual request can be recovered')
    usage = receipt.get('usage')
    expected_keys = {'input_tokens', 'output_tokens', 'cached_input_tokens',
                     'reasoning_tokens', 'billable_output_tokens'}
    require(isinstance(usage, dict) and set(usage) == expected_keys
            and all(type(v) is int and v >= 0 for v in usage.values()),
            'Exact integer accounted token usage is required')
    require(isinstance(receipt.get('estimated_cost_usd'), str)
            and re.fullmatch(r'\d+\.\d{6}', receipt['estimated_cost_usd']) is not None,
            'Exact microdollar accounted cost is required')
    ledger = N.local(root, '.secrets/meta-budget.sqlite3', prefix='.secrets/')
    with sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT request_id,model,endpoint,kind,state,actual,usage_json,estimate_json '
                         'FROM requests WHERE request_id=?', (receipt['request_id'],)).fetchone()
    require(row is not None and row['state'] == 'settled' and row['model'] == N.MODEL
            and row['endpoint'] == receipt['endpoint'] and row['kind'] == 'text'
            and row['actual'] == int(receipt['estimated_cost_usd'].replace('.', ''))
            and N.digest(json.loads(row['usage_json'])) == N.digest(usage),
            'Original visual receipt must match a settled ledger entry')
    reserved = json.loads(row['estimate_json'])
    maximum = reserved.get('usage', {}).get('output_tokens')
    require(type(maximum) is int and maximum == N.MAX_OUTPUT_TOKENS,
            'Original ledger output-token limit differs')
    return {'request_id': row['request_id'], 'model': row['model'], 'endpoint': row['endpoint'],
            'kind': row['kind'], 'state': row['state'], 'actual_micro_usd': row['actual'],
            'usage': json.loads(row['usage_json']), 'reserved_estimate': reserved}


def _original(prepared, root):
    _parser_evidence()
    folder = N.local(root, prepared, prefix='.private/native-visual-reviews/', exists=False)
    plan, _, _ = N._verified_plan(folder, root)
    paths = {
        'plan_ref': folder / 'plan.json',
        'model_result_ref': folder / 'attempt/model-result.json',
        'parsed_pending_ref': folder / 'attempt/request/response.pending.json',
        'request_receipt': folder / 'attempt/request/meta-request.json',
        'original_failure_ref': folder / 'attempt/failure.json',
    }
    paths = {key: N.local(root, path, prefix='.private/native-visual-reviews/', maximum=4 * 2**20)
             for key, path in paths.items()}
    require(not (folder / 'attempt/report.json').exists(), 'Original review already has a completed report')
    require(read(paths['original_failure_ref']) == {'status': 'preserved_no_automatic_retry', 'error_type': 'ValueError'},
            'Original failure marker differs')
    intent = N.local(root, root / '.private/native-visual-attempts' / plan['attempt_namespace'] / 'intent.json',
                     prefix='.private/native-visual-attempts/', maximum=128000)
    require(read(intent) == {'request_key': plan['request_key'], 'plan_ref': N.ref(paths['plan_ref'], root),
                            'automatic_retry': False}, 'Original paid-once intent differs')
    require(not (intent.parent / 'complete.json').exists(), 'Original claim is already complete')
    paths['original_intent_ref'] = intent
    require(paths['model_result_ref'].read_bytes() == paths['parsed_pending_ref'].read_bytes(),
            'Original pending parsed JSON and saved result bytes differ')
    result = read(paths['model_result_ref'])
    projected, retained, discarded = _projection(result, plan)
    receipt = read(paths['request_receipt'])
    require(receipt.get('payload_sha256') == plan['wire_provenance']['payload_sha256']
            and receipt.get('image_inputs') == plan['wire_provenance']['encoded_image_inputs'],
            'Original request wire or exact image provenance differs')
    proof = _settled_proof(receipt, root)
    return plan, result, projected, retained, discarded, receipt, proof, paths


def _report(folder, plan, projected, retained, discarded, receipt, proof, paths, root):
    snapshots = {name: N.ref(folder / name, root) for name in _parser_evidence()}
    return {**copy.deepcopy(projected), 'format': FORMAT, 'version': 1,
            'audit_manifest': plan['context']['audit_manifest'], 'source_sha256': plan['context']['source_sha256'],
            'source_time_domain': plan['context']['source_time_domain'],
            **{key: N.ref(path, root) for key, path in paths.items()},
            'projected_result_ref': N.ref(folder / 'projected-result.json', root),
            'request_key': plan['request_key'], 'provider': 'meta_api', 'model': N.MODEL,
            'round_number': plan['round_number'], 'estimated_cost_usd': receipt['estimated_cost_usd'],
            'usage': receipt['usage'], 'settled_request_proof': proof,
            'retained_proposal_indices': retained, 'discarded_proposals': discarded,
            'projection_policy': 'omit_action_target_mismatches_only_preserve_all_other_model_content',
            'implementation_snapshots': snapshots,
            'parser_attestation': {
                'provider_envelope_preserved': False,
                'finish_stop_evidence': 'attested_local_parser_path_not_raw_provider_envelope',
                'parsed_pending_equals_saved_model_bytes': True,
                'limitation': 'Known local parser writes parsed output after stop/schema checks; raw provider envelope was not retained.'},
            'human_approval_required': True, 'automatic_retry': False, 'automatic_repair_safe': False,
            'original_attempt_preserved': True, 'original_verdict_unchanged': True,
            'new_api_calls': 0, 'additional_estimated_cost_usd': '0.000000',
            'training_performed': False}


def recover(prepared, output, *, root=ROOT):
    """Create an immutable derivative without changing the paid attempt or ledger."""
    root = Path(root).resolve()
    original = _original(prepared, root)
    plan, _, projected, retained, discarded, receipt, proof, paths = original
    folder = N.local(root, output, prefix=PREFIX, exists=False)
    if (folder / 'report.json').exists():
        report = read(folder / 'report.json')
        verify_report(report, root=root)
        require(report['plan_ref'] == N.ref(paths['plan_ref'], root), 'Recovery folder belongs to another original plan')
        return report
    folder.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, raw in _parser_evidence().items():
        with os.fdopen(os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stream:
            stream.write(raw)
    N.write_new(folder / 'projected-result.json', projected)
    report = _report(folder, plan, projected, retained, discarded, receipt, proof, paths, root)
    verify_report(report, root=root)
    N.write_new(folder / 'report.json', report)
    return report


def verify_report(report, *, root=ROOT):
    root = Path(root).resolve()
    require(isinstance(report, dict) and report.get('format') == FORMAT and report.get('version') == 1,
            'An omission-only visual recovery report is required')
    plan_path = N.checked(report['plan_ref'], root, prefix='.private/native-visual-reviews/', maximum=4 * 2**20)
    plan, _, projected, retained, discarded, receipt, proof, paths = _original(plan_path.parent, root)
    projected_path = N.checked(report['projected_result_ref'], root, prefix=PREFIX, maximum=128000)
    folder = projected_path.parent
    require(projected_path.name == 'projected-result.json' and N.digest(read(projected_path)) == N.digest(projected),
            'Projected model content changed; only omission is permitted')
    expected_snapshots = _parser_evidence()
    require(set(report['implementation_snapshots']) == set(expected_snapshots), 'Recovery implementation snapshots differ')
    for name, raw in expected_snapshots.items():
        snapshot = N.checked(report['implementation_snapshots'][name], root, prefix=PREFIX, maximum=1024 * 1024)
        require(snapshot == folder / name and snapshot.read_bytes() == raw,
                'Known parser or recovery implementation snapshot changed')
    expected = _report(folder, plan, projected, retained, discarded, receipt, proof, paths, root)
    require(N.digest(report) == N.digest(expected), 'Recovery report changed its exact projection, provenance or policy')
    return True


def verify_any_report(report, *, root=ROOT):
    if isinstance(report, dict) and report.get('format') == FORMAT:
        return verify_report(report, root=root)
    return N.verify_report(report, root=root)


def recovery_reports_for_binding(binding, *, root=ROOT):
    """Bounded private lookup; invalid or stale derivatives are never adopted."""
    root = Path(root).resolve()
    base = N.local(root, root / PREFIX, prefix='.private/', exists=False)
    if not base.is_dir():
        return []
    candidates = sorted(base.glob('*/report.json'))
    require(len(candidates) <= 4096, 'Too many private recovery reports')
    found = []
    for candidate in candidates:
        # An unrelated broken archive cannot authorize a review or prevent all
        # later lookup. A candidate with the selected exact binding is verified
        # below without swallowing any verification failure.
        try:
            candidate = N.local(root, candidate, prefix=PREFIX, maximum=256000)
            report = read(candidate)
            if not isinstance(report, dict) or report.get('format') != FORMAT:
                continue
            plan_path = N.checked(report['plan_ref'], root, prefix='.private/native-visual-reviews/', maximum=4 * 2**20)
            context = read(plan_path)['context']
            if not isinstance(context, dict):
                continue
        except (ValueError, OSError, KeyError, TypeError):
            continue
        if (context.get('interaction_ref') == binding.get('interaction_ref')
                and context.get('scene_ref') == binding.get('scene_ref')):
            verify_report(report, root=root)
            program_path = N.checked(context['interaction_ref'], root, maximum=8 * 2**20)
            source = read(program_path)['source']
            require(source.get('sha256') == context['source_sha256'], 'Recovery program normalized source differs')
            # Upload manifests bind the original bytes, while model evidence
            # binds the normalized derivative. Both hashes come from the same
            # exact verified interaction, never from browser-supplied advice.
            if binding.get('source_ref', {}).get('sha256') not in {
                    source['sha256'], source.get('original_sha256', source['sha256'])}:
                continue
            found.append({'report_ref': N.ref(candidate, root), 'report': report})
    return found
