"""Pure offline omissions with exact paid-payload/source binding and no inference."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import task_intelligence as T
import task_intelligence_recovery as R

try:
    from . import test_task_intelligence as fixtures
except ImportError:
    import test_task_intelligence as fixtures


class TaskRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TaskIntelligenceTests('test_offline_context_preserves_exact_source_identity_and_pts')
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def bad_result(self):
        value = self.fixture.result()
        op = copy.deepcopy(value['robotics_handoff']['sequence'][0])
        op['evidence_ids'] = ['source_000004']
        value['robotics_handoff']['sequence'].append(op)
        return value

    def attempt(self, raw=None):
        raw = raw or self.bad_result()
        # Simulate a paid legacy attempt that stopped at its original strict
        # grounding gate; current runs normally apply this same recovery inline.
        with patch.object(R, 'normalize_result', side_effect=T.IntelligenceError('synthetic_legacy_grounding_failure')):
            with self.assertRaises(T.IntelligenceError):
                T.run_analysis(self.fixture.path, root=self.root, allow_paid_api=True, client=self.fixture.client(raw))
        plan = T.plan_analysis(self.fixture.path, root=self.root)
        directory = self.root / 'data/task-intelligence' / plan['request_key']
        path = directory / 'request/meta-request.json'
        receipt = json.loads(path.read_text())
        receipt.update(http_status=200, payload_sha256=R.request_payload_hash(
            plan['prompt'], plan['provider_schema'], [self.root / i['path'] for i in plan['contract']['images']], plan['contract']['max_output_tokens']))
        path.write_text(json.dumps(receipt))
        return directory

    def test_invalid_operation_only_is_omitted_and_every_retained_field_is_identical(self):
        raw = self.bad_result()
        original = copy.deepcopy(raw)
        value, trace = R.normalize_result(raw, self.fixture.context())
        self.assertEqual(raw, original)
        expected = copy.deepcopy(raw)
        expected['robotics_handoff']['sequence'].pop()
        self.assertEqual(value, expected)
        self.assertEqual(trace['removed_count'], 1)
        self.assertEqual(trace['removed'][0]['section'], 'robotics_handoff.sequence')
        self.assertEqual(trace['removed'][0]['original_index'], 1)
        self.assertFalse(trace['all_model_items_retained'])
        self.assertFalse(trace['new_semantic_claims_added'])
        self.assertFalse(trace['item_fields_edited'])
        self.assertTrue(trace['full_original_validator_passed'])

    def test_invalid_claim_and_step_insight_are_omitted_whole_without_retargeting(self):
        raw = self.bad_result()
        claim = copy.deepcopy(raw['claims'][0])
        claim.update(id='bad_claim', evidence_ids=['source_000004'], source_claim_ids=[])
        raw['claims'].append(claim)
        step = copy.deepcopy(raw['step_insights'][0])
        step.update(step_id='invented_step')
        raw['step_insights'].append(step)
        value, trace = R.normalize_result(raw, self.fixture.context())
        self.assertEqual(trace['removed_count'], 3)
        self.assertEqual(value, self.fixture.result())

    def test_missing_mandatory_gap_or_all_capture_items_cannot_be_fabricated(self):
        raw = self.bad_result()
        raw['gaps'][0].update(evidence_ids=['source_000004'])
        other = copy.deepcopy(raw['gaps'][0])
        other.update(id='valid_geometry_gap', category='geometry', track_ids=[], evidence_ids=[])
        raw['gaps'].append(other)
        raw['capture_checklist'][0]['addresses_gap_ids'] = ['valid_geometry_gap']
        with self.assertRaisesRegex(T.IntelligenceError, 'missing_physical_validation_gap'):
            R.normalize_result(raw, self.fixture.context())
        raw = self.bad_result()
        raw['capture_checklist'][0]['addresses_gap_ids'] = ['nonexistent']
        import jsonschema
        with self.assertRaises(jsonschema.ValidationError):
            R.normalize_result(raw, self.fixture.context())

    def test_capture_depending_on_omitted_gap_is_omitted_entirely(self):
        raw = self.bad_result()
        gap = copy.deepcopy(raw['gaps'][0])
        gap.update(id='invalid_geometry', category='geometry', evidence_ids=['source_000004'])
        raw['gaps'].append(gap)
        capture = copy.deepcopy(raw['capture_checklist'][0])
        capture.update(id='dependent_capture', addresses_gap_ids=['g1', 'invalid_geometry'])
        raw['capture_checklist'].append(capture)
        value, trace = R.normalize_result(raw, self.fixture.context())
        self.assertEqual(value, self.fixture.result())
        self.assertEqual(trace['removed_count'], 3)
        self.assertTrue(any(r['item_id'] == 'dependent_capture' for r in trace['removed']))

    def test_offline_recovery_preserves_raw_failure_receipt_source_and_reuses_immutable_derivative(self):
        directory = self.attempt()
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        source = self.fixture.path.read_bytes()
        import meta_api
        with patch.object(meta_api.MetaClient, 'json_result', side_effect=AssertionError('No hosted inference permitted')):
            record = R.recover_attempt(directory, root=self.root)
            again = R.recover_attempt(directory, root=self.root)
        self.assertEqual(again, record)
        self.assertEqual(record['result'], self.fixture.result())
        self.assertEqual(record['recovery']['removed_count'], 1)
        self.assertFalse(record['robotics_readiness']['simulation_ready'])
        self.assertEqual(record['recovery']['additional_api_calls'], 0)
        self.assertEqual(source, self.fixture.path.read_bytes())
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes() for p in directory.rglob('*') if p.is_file()})
        self.assertEqual(R.recovered_analyses(self.fixture.path, root=self.root)[0]['record'], record)
        # MCP's isolated environment intentionally has no Pillow. Read-only
        # verification uses the bound creation proof, not image re-encoding.
        with patch.object(R, 'request_payload_hash', side_effect=AssertionError('Do not re-encode on read')):
            self.assertEqual(R.recovered_analyses(self.fixture.path, root=self.root)[0]['record'], record)

    def test_changed_paid_payload_context_source_or_trace_is_rejected(self):
        directory = self.attempt()
        receipt = directory / 'request/meta-request.json'
        original = receipt.read_bytes()
        bad = json.loads(original)
        bad['payload_sha256'] = 'f' * 64
        receipt.write_text(json.dumps(bad))
        with self.assertRaisesRegex(T.IntelligenceError, 'paid_payload_binding_changed'):
            R.recover_attempt(directory, root=self.root)
        receipt.write_bytes(original)
        record = R.recover_attempt(directory, root=self.root)
        trace = self.root / record['recovery']['trace_path']
        trace.write_text('{}')
        self.assertEqual(R.recovered_analyses(self.fixture.path, root=self.root), [])

    def test_final_full_validator_still_rejects_unhandled_physical_claim(self):
        raw = self.bad_result()
        raw['overview'] = 'The object weighs 10 grams.'
        with self.assertRaisesRegex(T.IntelligenceError, 'numeric_physical_claim'):
            R.normalize_result(raw, self.fixture.context())

    def test_future_live_adapter_normalizes_once_keeps_raw_response_and_replays_trace_on_cache_hit(self):
        raw = self.bad_result()
        record = T.run_analysis(self.fixture.path, root=self.root, allow_paid_api=True,
                                client=self.fixture.client(raw))
        self.assertEqual(record['result'], self.fixture.result())
        self.assertEqual(record['normalization']['removed_count'], 1)
        self.assertFalse(record['normalization']['all_model_items_retained'])
        directory = self.root / 'data/task-intelligence' / record['request_key']
        self.assertEqual(json.loads((directory / 'request/response.pending.json').read_text()), raw)
        again = T.run_analysis(self.fixture.path, root=self.root)
        self.assertEqual(record, again)
        self.assertEqual(len(self.fixture.calls), 1)


if __name__ == '__main__':
    unittest.main()
