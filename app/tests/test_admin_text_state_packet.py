"""Correct future state packets; immutable historical reviews never resend."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import admin_muse_reviews as current
from tests import test_admin_muse_reviews as fixtures


class CompiledStatePacketTests(unittest.TestCase):
    def setUp(self):
        harness = fixtures.AdminMuseReviewTests()
        harness.setUp(); self.addCleanup(harness.doCleanups)
        self.harness = harness; self.root = harness.root
        self.manifest = harness.manifest; self.memory = harness.memory
        path = current.ROOT / 'output/overnight-pipeline-20261102/text-review-packet-v2-source/admin_muse_reviews.py'
        spec = importlib.util.spec_from_file_location('historical_text_review_v2_fixture', path)
        self.previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.previous)

    def test_future_packet_contains_exact_compiled_variables_assertions_and_source_provenance(self):
        plan = current.prepare('semantics', self.manifest, self.memory, root=self.root)
        doc = self.harness.fixture.doc
        packet = plan['evidence_packet']
        self.assertEqual(plan['version'], 3)
        self.assertEqual(plan['review_protocol'], 'text_roles_v3_compiled_states_8000_total')
        self.assertEqual(plan['summary']['state_variables'], doc['variables'])
        self.assertEqual(plan['summary']['assertions'], doc['assertions'])
        self.assertEqual(packet['state_variables_source_field'], 'variables')
        self.assertEqual(packet['assertions_source_field'], 'assertions')
        self.assertEqual(packet['state_variables_sha256'], current.digest(doc['variables']))
        self.assertEqual(packet['assertions_sha256'], current.digest(doc['assertions']))
        self.assertEqual(packet['state_variable_count'], len(doc['variables']))
        self.assertEqual(packet['assertion_count'], len(doc['assertions']))
        self.assertFalse(packet['definitions_are_observations'])
        self.assertFalse(packet['physical_accuracy_verified'])
        metadata = json.loads(plan['prompt'].split('UNTRUSTED_SAVED_METADATA=')[1].split('\nUNTRUSTED_SCOPED_OWNER_FEEDBACK=')[0])
        self.assertEqual(metadata['state_variables'], doc['variables'])
        self.assertEqual(metadata['assertions'], doc['assertions'])
        self.assertEqual(plan['image_inputs'], [])

    def test_missing_compiled_definitions_never_silently_become_empty(self):
        doc = copy.deepcopy(self.harness.fixture.doc)
        del doc['variables']
        with self.assertRaises(KeyError): current._state_packet(doc)

    def old_review(self, *, fail=False):
        client = fixtures.OfflineClient(fail=fail)
        reviewer = self.previous.build_reviewer(self.memory, root=self.root, client=client, allow_paid_api=True)
        return client, reviewer

    def test_completed_v2_report_remains_unchanged_verifiable_and_reused_without_v3_send(self):
        old_client, old_reviewer = self.old_review()
        report = old_reviewer('semantics', self.manifest)
        folder = self.root / report['plan_ref']['path']; folder = folder.parent
        before = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        self.assertTrue(current.verify_report(report, self.manifest, root=self.root))
        self.assertEqual(report['review_protocol'], current.V2_PROTOCOL)
        new_client = fixtures.OfflineClient()
        reused = current.build_reviewer(self.memory, root=self.root, client=new_client, allow_paid_api=True)(
            'semantics', self.manifest)
        self.assertEqual(reused, report); self.assertEqual(new_client.calls, [])
        self.assertEqual(len(old_client.calls), 1)
        self.assertEqual(before, {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob('*') if p.is_file()})
        namespace = current.review_namespace('semantics', current.artifact_binding(self.manifest),
                                             self.memory['context_sha256'], protocol=current.REVIEW_PROTOCOL)
        self.assertFalse((self.root / '.private/admin-muse-reviews' / namespace).exists())

    def test_uncertain_v2_attempt_cannot_be_reissued_under_new_packet_or_explicit_upgrade_flag(self):
        old_client, old_reviewer = self.old_review(fail=True)
        with self.assertRaises(RuntimeError): old_reviewer('evidence', self.manifest)
        new_client = fixtures.OfflineClient()
        reviewer = current.build_reviewer(self.memory, root=self.root, client=new_client,
                                         allow_paid_api=True, allow_protocol_upgrade=True)
        with self.assertRaisesRegex(ValueError, 'V2 hosted role.*no paid resend'):
            reviewer('evidence', self.manifest)
        self.assertEqual(len(old_client.calls), 1); self.assertEqual(new_client.calls, [])

    def test_current_report_binds_exact_state_packet_even_if_a_plan_hash_is_recomputed(self):
        client = fixtures.OfflineClient()
        report = current.build_reviewer(self.memory, root=self.root, client=client, allow_paid_api=True)(
            'semantics', self.manifest)
        self.assertTrue(current.verify_report(report, self.manifest, root=self.root))
        self.assertEqual(report['evidence_packet']['state_variables_sha256'],
                         current.digest(self.harness.fixture.doc['variables']))
        path = self.root / report['plan_ref']['path']; plan = json.loads(path.read_bytes())
        plan['summary']['state_variables'] = []
        plan['request_key'] = current.digest({k: v for k, v in plan.items() if k != 'request_key'})
        path.write_text(json.dumps(plan))
        changed = copy.deepcopy(report); changed['plan_ref'] = current._ref(path, self.root)
        changed['request_key'] = plan['request_key']
        with self.assertRaisesRegex(ValueError, 'exact compiled state'):
            current.verify_report(changed, self.manifest, root=self.root)
        self.assertEqual(len(client.calls), 1)


if __name__ == '__main__':
    unittest.main()
