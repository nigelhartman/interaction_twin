"""Actual offline accounting/render/recovery bridges; no provider or GPU contact."""
import copy
import json
import shutil
import unittest
from unittest.mock import patch

from PIL import Image

import admin_machine_context as machine
import admin_visual_audit as admin
from admin_execution_policy import improvement_policy, review_assessment, TASK_PLAN
from agent_memory import FeedbackMemory
import meta_api
import native_visual_recovery_v2 as recovery
import native_visual_review as visual
from object_program import ref
from tests import test_native_scene_audit as audit_fixtures
from tests import test_native_visual_recovery as recovery_fixtures


class PriorRecoveredCriticTests(unittest.TestCase):
    def setUp(self):
        fixture = recovery_fixtures.NativeVisualRecoveryTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root
        self.fixture.output = '.private/native-visual-recoveries-v2/fixture'
        context = fixture.plan['context']
        self.manifest = {'interaction_ref': context['interaction_ref'], 'scene_ref': context['scene_ref'],
                         'source_ref': {'sha256': 'b' * 64}}

    def test_exact_recovered_critic_is_reused_without_completing_or_resending_failed_attempt(self):
        report = recovery.recover(self.fixture.prepared, self.fixture.output, root=self.root)
        before = self.fixture.raw.read_bytes()
        with patch.object(visual, 'run_prepared', side_effect=AssertionError('No paid resend')) as send:
            found, unknown = admin.prior_critic(self.manifest, root=self.root)
        self.assertFalse(unknown); self.assertEqual(found[1], report)
        self.assertEqual(found[0], self.root / self.fixture.output / 'report.json')
        self.assertEqual(self.fixture.raw.read_bytes(), before)
        self.assertFalse((self.fixture.raw.parent / 'report.json').exists())
        self.assertEqual(len(self.fixture.calls), 1); send.assert_not_called()

    def test_proposer_recovery_cannot_replace_the_uncertain_or_failed_critic(self):
        prepared = '.private/native-visual-reviews/proposer'
        visual.prepare(self.fixture.fixture.audit_path, prepared, root=self.root, role='proposer')
        self.fixture.result['role'] = 'proposer'
        with self.assertRaisesRegex(ValueError, recovery.MISMATCH):
            visual.run_prepared(prepared, root=self.root, allow_paid_api=True, client=self.fixture.client,
                authorized_plan_sha256=visual.sha(self.root / prepared / 'plan.json'))
        report = recovery.recover(prepared, '.private/native-visual-recoveries-v2/proposer', root=self.root)
        self.assertEqual(report['role'], 'proposer')
        with patch.object(visual, 'run_prepared', side_effect=AssertionError('No paid resend')) as send:
            found, unknown = admin.prior_critic(self.manifest, root=self.root)
        self.assertIsNone(found); self.assertTrue(unknown)
        self.assertEqual(len(self.fixture.calls), 2); send.assert_not_called()


class RenderRecoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        harness = audit_fixtures.AuditArtifactTests(
            'test_end_to_end_audit_freezes_real_triangle_rasters_code_and_original_bytes')
        harness.setUp(); self.addCleanup(harness.doCleanups)
        self.harness = harness; self.root = harness.root
        for name in admin.CODE:
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(admin.ROOT / name, target)
        budget = self.root / TASK_PLAN; budget.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(admin.ROOT / TASK_PLAN, budget)
        original = harness.folder / 'original.mp4'; original.write_bytes(b'Inert original upload fixture')
        normalized = harness.folder / 'normalized.mp4'; normalized.write_bytes(b'Inert normalized evidence fixture')
        self.source = ref(original, self.root)
        program = json.loads(harness.program.read_text())
        program['source'].update(path=ref(normalized, self.root)['path'], sha256=visual.sha(normalized),
                                 original_sha256=self.source['sha256'])
        for evidence in program['evidence']:
            rgb = self.root / evidence['frame_path']
            with Image.open(rgb) as image: image.resize((32, 32)).save(rgb)
            evidence['frame_sha256'] = visual.sha(rgb)
            for observation in evidence['observations']:
                mask = self.root / observation['mask_path']
                with Image.open(mask) as image: image.resize((32, 32), Image.Resampling.NEAREST).save(mask)
                observation['mask_sha256'] = visual.sha(mask)
        harness.program.write_text(json.dumps(program))
        scene = json.loads(harness.scene.read_text())
        scene['source'].update(sha256=program['source']['sha256'], height=32, width=32)
        scene['evaluation_protocol']['source_sha256'] = program['source']['sha256']
        scene['original_program'] = ref(harness.program, self.root)
        harness.scene.write_text(json.dumps(scene))
        (self.root / admin.ACTIVATION).write_text(json.dumps(admin.activation_template(root=self.root)))
        self.manifest = {'source_ref': self.source, 'interaction_ref': ref(harness.program, self.root),
                         'scene_ref': ref(harness.scene, self.root), 'object_program_ref': None}
        self.job = {'candidate_id': 'candidate_' + 'a' * 32, 'source_ref': self.source,
                    'improvement_policy': improvement_policy(self.root), 'improvement_round': 0}
        self.calls = []; self.mode = 'recoverable'

        def transport(endpoint, payload, timeout):
            self.calls.append(copy.deepcopy(payload))
            if self.mode == 'uncertain': raise TimeoutError('Inert offline transport interruption')
            prompt = payload['messages'][1]['content'][0]['text']
            context = json.loads(prompt.split('CONTEXT (data):\n')[1].split('\nPREVIOUS REVIEW')[0])
            sample = context['samples'][0]
            component = sample['component_id']; citation = sample['sample_id']
            proposal = lambda target, action: {
                'target': target, 'action': action, 'component_ids': [component], 'sample_ids': [citation],
                'rationale': 'Inspect the source-bound development discrepancy.',
                'expected_visible_change': 'A bounded experiment; no physical accuracy promised.'}
            result = {
                'format': visual.FORMAT + '-result', 'version': 1, 'role': 'critic',
                'verdict': 'repair_recommended', 'summary': 'Development discrepancy remains uncertain.',
                'issues': [{'category': 'pose_alignment', 'severity': 'warning', 'component_ids': [component],
                    'sample_ids': [citation], 'visible_observation': 'Compare the observed boundary.',
                    'why_it_matters': 'Alignment is estimated.', 'confidence': 'low'}],
                'repair_proposals': [proposal('tracking', 'reason_again')],
                'limitations': ['No independent physical validation.'], 'visual_observation_performed': True,
                'independent_model_review': False, 'final_test_evidence_used': False,
                'physical_accuracy_verified': False, 'simulation_ready': False,
                'automatic_publication': False, 'training_performed': False}
            if self.mode == 'recoverable':
                result['repair_proposals'] += [proposal('mechanics', 'refit_mechanism'),
                                               proposal('capture_more_evidence', 'request_capture')]
            return {'http_status': 200, 'document': {
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                'usage': {'prompt_tokens': 200, 'completion_tokens': 300}}}

        self.client = meta_api.MetaClient(ledger_path=self.root / '.secrets/meta-budget.sqlite3', transport=transport)

    def review(self):
        original = visual.build_admin_reviewer
        with patch.object(visual, 'build_admin_reviewer', side_effect=
                lambda *args, **kwargs: original(*args, **kwargs, client=self.client)):
            return admin.build_reviewer(self.job, root=self.root)(self.manifest)

    def test_real_accounted_recovery_reaches_measured_assessment_and_typed_machine_guidance(self):
        first = self.review()
        budget_before = self.client.ledger.status()
        second = self.review()
        self.assertEqual(first['status'], 'complete')
        self.assertTrue(first['raw_critic_recovered_offline'])
        self.assertTrue(second['raw_critic_reused'])
        self.assertEqual(first['report_ref'], second['report_ref'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.ledger.status(), budget_before)
        assessment = review_assessment(first, self.manifest, root=self.root)
        self.assertIsNotNone(assessment)
        measured = json.loads((self.root / first['report_ref']['path']).read_text())
        report = json.loads((self.root / measured['raw_critic_ref']['path']).read_text())
        self.assertTrue(recovery.verify_any_report(report, root=self.root))
        self.assertEqual(report['retained_proposal_indices'], [1, 2])
        self.assertEqual([p['target'] for p in measured['repair_proposals']], ['mechanics', 'capture_more_evidence'])
        self.assertFalse(report['automatic_repair_safe']); self.assertTrue(report['human_approval_required'])
        self.assertFalse((self.root / report['model_result_ref']['path']).parent.joinpath('report.json').exists())
        guide_ref = machine.snapshot(first['report_ref'], source_sha256=self.source['sha256'], root=self.root)
        guidance = machine.load(guide_ref, source_sha256=self.source['sha256'], root=self.root)
        self.assertEqual([row['source_index'] for row in guidance['typed_citations']], [1, 3])
        self.assertEqual([row['timestamp_s'] for row in guidance['typed_citations']], [0., .6])
        self.assertFalse(guidance['human_feedback_authored']); self.assertFalse(guidance['observed_facts'])
        self.assertEqual(FeedbackMemory(self.root).entries(), [])
        self.assertFalse(measured['final_test_evidence_used'])

    def test_rejected_omission_stays_unknown_and_cannot_resend(self):
        self.mode = 'drop_all'
        with self.assertRaisesRegex(ValueError, 'remaining valid proposal'): self.review()
        budget_before = self.client.ledger.status()
        second = self.review()
        self.assertEqual(second['status'], 'unavailable')
        self.assertIn('no automatic resend', second['diagnostic'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.ledger.status(), budget_before)
        self.assertFalse(list((self.root / recovery.PREFIX).glob('*/report.json')))

    def test_transport_uncertainty_preserves_hold_and_stops_before_any_second_send(self):
        self.mode = 'uncertain'
        with self.assertRaises(meta_api.MetaError): self.review()
        budget_before = self.client.ledger.status()
        second = self.review()
        self.assertEqual(second['status'], 'unavailable')
        self.assertIn('uncertain', second['diagnostic'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.ledger.status(), budget_before)
        self.assertGreater(float(budget_before['reserved_usd']), 0)


if __name__ == '__main__':
    unittest.main()
