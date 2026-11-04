"""No network: real local accounting and parser, with a fake provider transport."""
import copy
import json
from pathlib import Path
import sqlite3
import unittest

import meta_api
import native_visual_recovery as R
import native_visual_review as N
from tests import test_native_visual_review as fixtures


class NativeVisualRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeVisualReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.fixture.doc['source']['original_sha256'] = 'b' * 64
        meta_api.write(self.fixture.program, self.fixture.doc)
        scene = R.read(self.fixture.scene)
        scene['original_program'] = N.ref(self.fixture.program, self.root)
        meta_api.write(self.fixture.scene, scene)
        self.fixture.audit.update(interaction_ref=N.ref(self.fixture.program, self.root), scene_ref=N.ref(self.fixture.scene, self.root))
        self.fixture.publish()
        self.fixture.prepare()
        self.prepared = self.fixture.prepared
        self.plan = R.read(self.root / self.prepared / 'plan.json')
        sample = {row['component_id']: row['sample_id'] for row in self.plan['context']['samples']}
        self.result = {
            'format': N.FORMAT + '-result', 'version': 1, 'role': 'critic', 'verdict': 'repair_recommended',
            'summary': 'A visible lid discrepancy needs inspection; this is an untrusted model opinion.',
            'issues': [{'category': 'pose_alignment', 'severity': 'blocking', 'component_ids': ['lid'],
                        'sample_ids': [sample['lid']], 'visible_observation': 'Lid render is offset.',
                        'why_it_matters': 'Alignment is uncertain.', 'confidence': 'medium'}],
            'repair_proposals': [
                {'target': 'tracking', 'action': 'reason_again', 'component_ids': ['base'], 'sample_ids': [sample['base']],
                 'rationale': 'Reconsider the camera-relative state.', 'expected_visible_change': 'Review state.'},
                {'target': 'mechanics', 'action': 'refit_mechanism', 'component_ids': ['lid'], 'sample_ids': [sample['lid']],
                 'rationale': 'Refit supported hinge motion.', 'expected_visible_change': 'Better overlap.'},
                {'target': 'capture_more_evidence', 'action': 'request_capture', 'component_ids': ['base'], 'sample_ids': [sample['base']],
                 'rationale': 'More visible source evidence is needed.', 'expected_visible_change': 'None promised.'}],
            'limitations': ['Metric geometry and physical assembly remain unverified.'],
            'visual_observation_performed': True, 'independent_model_review': False, 'final_test_evidence_used': False,
            'physical_accuracy_verified': False, 'simulation_ready': False, 'automatic_publication': False, 'training_performed': False}
        self.calls = []

        def transport(endpoint, payload, timeout):
            self.calls.append(copy.deepcopy(payload))
            return {'http_status': 200, 'document': {
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.result)}}],
                'usage': {'prompt_tokens': 200, 'completion_tokens': 300}}}

        self.client = meta_api.MetaClient(ledger_path=self.root / '.secrets/meta-budget.sqlite3', transport=transport)
        with self.assertRaisesRegex(ValueError, R.MISMATCH):
            self.fixture.execute(self.client)
        self.output = '.private/native-visual-recoveries/fixture'
        self.raw = self.root / self.prepared / 'attempt/model-result.json'
        self.pending = self.root / self.prepared / 'attempt/request/response.pending.json'
        self.receipt = self.root / self.prepared / 'attempt/request/meta-request.json'

    def recover(self):
        return R.recover(self.prepared, self.output, root=self.root)

    def replace_original_result(self, value):
        for path in [self.raw, self.pending]:
            meta_api.write(path, value)

    def test_exact_omission_preserves_original_bytes_receipt_ledger_and_failure(self):
        before_files = {str(p): p.read_bytes() for base in [self.root / self.prepared, self.root / '.private/native-visual-attempts']
                        for p in base.rglob('*') if p.is_file()}
        before_budget = self.client.ledger.status()
        report = self.recover()
        self.assertTrue(R.verify_report(report, root=self.root))
        self.assertEqual(report['repair_proposals'], self.result['repair_proposals'][1:])
        self.assertEqual(report['issues'], self.result['issues'])
        self.assertEqual(report['summary'], self.result['summary'])
        self.assertEqual(report['verdict'], self.result['verdict'])
        self.assertEqual(report['retained_proposal_indices'], [1, 2])
        self.assertEqual(report['discarded_proposals'], [{'original_index': 0, 'reason': 'action_target_mismatch',
                                                       'original_proposal': self.result['repair_proposals'][0]}])
        self.assertFalse(report['parser_attestation']['provider_envelope_preserved'])
        self.assertEqual(report['parser_attestation']['finish_stop_evidence'],
                         'attested_local_parser_path_not_raw_provider_envelope')
        self.assertEqual(self.client.ledger.status(), before_budget)
        self.assertTrue(all(Path(p).read_bytes() == raw for p, raw in before_files.items()))
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(report['automatic_repair_safe'])
        self.assertEqual(report['new_api_calls'], 0)
        self.assertEqual(report['additional_estimated_cost_usd'], '0.000000')
        self.assertTrue(report['human_approval_required'])

    def test_original_validator_and_once_only_marker_still_reject(self):
        self.recover()
        with self.assertRaisesRegex(ValueError, R.MISMATCH):
            N.validate_result(R.read(self.raw), self.plan)
        old_report = {**self.result, 'format': N.FORMAT + '-report',
                      'plan_ref': N.ref(self.root / self.prepared / 'plan.json', self.root),
                      'request_key': self.plan['request_key'], 'model_result_ref': N.ref(self.raw, self.root)}
        with self.assertRaisesRegex(ValueError, R.MISMATCH):
            R.verify_any_report(old_report, root=self.root)
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            self.fixture.execute(self.client)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.raw.parent / 'report.json').exists())

    def test_repeated_recovery_and_dispatcher_are_read_only(self):
        report = self.recover()
        self.assertEqual(self.recover(), report)
        self.assertTrue(R.verify_any_report(report, root=self.root))
        with self.assertRaisesRegex(ValueError, 'new visual review report'):
            R.verify_any_report(self.result, root=self.root)
        binding = {'source_ref': {'sha256': 'a' * 64}, 'interaction_ref': self.fixture.audit['interaction_ref'],
                   'scene_ref': self.fixture.audit['scene_ref']}
        found = R.recovery_reports_for_binding(binding, root=self.root)
        self.assertEqual(len(found), 1)
        binding['source_ref']['sha256'] = 'b' * 64
        self.assertEqual(R.recovery_reports_for_binding(binding, root=self.root), found)
        binding['source_ref']['sha256'] = 'f' * 64
        self.assertEqual(R.recovery_reports_for_binding(binding, root=self.root), [])
        self.assertEqual(len(self.calls), 1)

    def test_unrelated_malformed_archive_does_not_block_lookup_but_selected_corruption_fails(self):
        report = self.recover()
        broken = self.root / '.private/native-visual-recoveries/unrelated-broken/report.json'
        broken.parent.mkdir()
        broken.write_text('not json')
        binding = {'source_ref': {'sha256': 'b' * 64}, 'interaction_ref': self.fixture.audit['interaction_ref'],
                   'scene_ref': self.fixture.audit['scene_ref']}
        self.assertEqual(len(R.recovery_reports_for_binding(binding, root=self.root)), 1)
        meta_api.write(broken, [])
        self.assertEqual(len(R.recovery_reports_for_binding(binding, root=self.root)), 1)
        bad_plan = self.root / '.private/native-visual-reviews/unrelated-plan/plan.json'
        meta_api.write(bad_plan, {'context': []})
        meta_api.write(broken, {'format': R.FORMAT, 'plan_ref': N.ref(bad_plan, self.root)})
        self.assertEqual(len(R.recovery_reports_for_binding(binding, root=self.root)), 1)
        report['estimated_cost_usd'] = '0.000000'
        meta_api.write(self.root / self.output / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'exact projection'):
            R.recovery_reports_for_binding(binding, root=self.root)

    def test_lookup_never_accepts_another_program_or_scene_with_same_source(self):
        self.recover()
        binding = {'source_ref': {'sha256': 'b' * 64}, 'interaction_ref': copy.deepcopy(self.fixture.audit['interaction_ref']),
                   'scene_ref': self.fixture.audit['scene_ref']}
        binding['interaction_ref']['sha256'] = 'f' * 64
        self.assertEqual(R.recovery_reports_for_binding(binding, root=self.root), [])

    def test_retargeted_projected_proposal_cannot_be_adopted_even_with_updated_hash(self):
        report = self.recover()
        path = self.root / report['projected_result_ref']['path']
        changed = R.read(path)
        changed['repair_proposals'][0]['target'] = 'reasoning'
        changed['repair_proposals'][0]['action'] = 'reason_again'
        meta_api.write(path, changed)
        report['projected_result_ref'] = N.ref(path, self.root)
        with self.assertRaisesRegex(ValueError, 'only omission'):
            R.verify_report(report, root=self.root)

    def test_new_rationale_or_verdict_in_report_is_rejected(self):
        report = self.recover()
        for field, value in [('verdict', 'no_visible_issue'), ('automatic_repair_safe', True),
                             ('automatic_publication', 0), ('summary', 'Everything works.')]:
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, 'exact projection'):
                R.verify_report(changed, root=self.root)

    def test_foreign_citation_in_discarded_proposal_is_not_hidden_by_omission(self):
        changed = copy.deepcopy(self.result)
        changed['repair_proposals'][0]['sample_ids'] = ['base:999999']
        self.replace_original_result(changed)
        with self.assertRaisesRegex(ValueError, 'foreign'):
            self.recover()
        self.assertFalse((self.root / self.output).exists())

    def test_schema_invalid_raw_result_and_wrong_component_sample_pair_are_rejected(self):
        changed = copy.deepcopy(self.result)
        changed['repair_proposals'][0]['component_ids'] = ['lid']
        self.replace_original_result(changed)
        with self.assertRaisesRegex(ValueError, 'lacks a submitted sample'):
            self.recover()
        changed['unexpected'] = True
        self.replace_original_result(changed)
        with self.assertRaises(Exception):
            self.recover()

    def test_drop_all_and_hidden_verdict_contradiction_are_rejected(self):
        changed = copy.deepcopy(self.result)
        changed['repair_proposals'] = changed['repair_proposals'][:1]
        self.replace_original_result(changed)
        with self.assertRaisesRegex(ValueError, 'remaining valid proposal'):
            self.recover()
        changed = copy.deepcopy(self.result)
        changed['verdict'] = 'no_visible_issue'
        self.replace_original_result(changed)
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            self.recover()

    def test_uncertain_or_missing_accounting_never_becomes_recovered(self):
        receipt = R.read(self.receipt)
        with sqlite3.connect(self.root / '.secrets/meta-budget.sqlite3') as db:
            db.execute("UPDATE requests SET state='uncertain' WHERE request_id=?", (receipt['request_id'],))
        with self.assertRaisesRegex(ValueError, 'settled ledger'):
            self.recover()

    def test_changed_receipt_cost_wire_or_images_is_rejected(self):
        original = R.read(self.receipt)
        for key, value in [('estimated_cost_usd', '0.000000'), ('payload_sha256', 'f' * 64), ('image_inputs', [])]:
            changed = copy.deepcopy(original)
            changed[key] = value
            meta_api.write(self.receipt, changed)
            with self.assertRaises(ValueError):
                self.recover()
        meta_api.write(self.receipt, original)

    def test_pending_output_parser_snapshot_and_source_changes_are_rejected(self):
        pending = self.pending.read_bytes()
        self.pending.write_bytes(pending + b' ')
        with self.assertRaisesRegex(ValueError, 'result bytes differ'):
            self.recover()
        self.pending.write_bytes(pending)
        report = self.recover()
        snapshot = self.root / report['implementation_snapshots']['original-meta-api.py.txt']['path']
        snapshot.write_bytes(snapshot.read_bytes() + b'\n# changed')
        report['implementation_snapshots']['original-meta-api.py.txt'] = N.ref(snapshot, self.root)
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            R.verify_report(report, root=self.root)

    def test_changed_original_source_frame_prevents_derivative(self):
        source = self.root / self.plan['context']['images'][0]['path']
        source.write_bytes(source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.recover()

    def test_noninteger_usage_and_missing_intent_are_rejected(self):
        receipt = R.read(self.receipt)
        receipt['usage']['reasoning_tokens'] = False
        meta_api.write(self.receipt, receipt)
        with self.assertRaisesRegex(ValueError, 'integer accounted token'):
            self.recover()
        receipt['usage']['reasoning_tokens'] = 0
        meta_api.write(self.receipt, receipt)
        intent = self.root / '.private/native-visual-attempts' / self.plan['attempt_namespace'] / 'intent.json'
        intent.unlink()
        with self.assertRaisesRegex(ValueError, 'Bounded visual review|Bounded visual|Bounded existing|Bounded'):
            self.recover()

    def test_a_valid_original_is_not_relabelled_as_recovery(self):
        changed = copy.deepcopy(self.result)
        changed['repair_proposals'] = changed['repair_proposals'][1:]
        self.replace_original_result(changed)
        with self.assertRaisesRegex(ValueError, 'needs no offline recovery'):
            self.recover()


if __name__ == '__main__':
    unittest.main()
