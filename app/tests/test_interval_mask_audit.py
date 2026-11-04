"""Frozen source joins and audit versioning; no hosted inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import interval_mask_audit as audit


class AuditBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.base = self.root / 'data/interval-motion-20261031/test-plan'
        self.base.mkdir(parents=True)
        self.image, self.mask = self.base / 'image.png', self.base / 'mask.png'
        Image.new('RGB', (80, 100), '#334455').save(self.image)
        Image.new('L', (80, 100), 255).save(self.mask)
        self.plan = self.base / 'plan.json'; self.plan.write_text('{"synthetic_contract_fixture":true}')
        self.request = {'format': 'short-interval-mask-scope-audit-request', 'track_id': 'part',
                        'plan_path': str(self.plan.relative_to(self.root)), 'plan_sha256': audit.sha(self.plan),
                        'targets': [{'sample_index': 1, 'timestamp_s': 0.,
                                     'source_image_url': '/files/' + str(self.image.relative_to(self.root)),
                                     'source_image_sha256': audit.sha(self.image),
                                     'mask_url': '/files/' + str(self.mask.relative_to(self.root)), 'mask_sha256': audit.sha(self.mask)}]}
        self.path = self.base / 'audit-request.json'; self.path.write_text(json.dumps(self.request))
        self.result = {'summary': 'Explicit synthetic binding fixture', 'limitations': [],
                       'observations': [{'sample_index': 1, 'verdict': 'uncertain', 'single_component_visible': False,
                                         'component_boundary_plausible': False, 'other_components_inside_mask': False,
                                         'notes': 'Synthetic test, not a model result'}]}

    def tearDown(self): self.tmp.cleanup()

    def test_only_exact_complete_observation_set_is_bound_without_changing_judgments(self):
        original = copy.deepcopy(self.result)
        rows = audit.bind_result(self.result, self.request)
        self.assertEqual(self.result, original)
        self.assertEqual(rows[0]['mask_sha256'], self.request['targets'][0]['mask_sha256'])
        self.assertEqual(rows[0]['verdict'], 'uncertain')
        for proposed in [[], self.result['observations'] * 2, [dict(self.result['observations'][0], sample_index=2)]]:
            with self.subTest(proposed=proposed), self.assertRaises(ValueError):
                audit.bind_result(dict(self.result, observations=proposed), self.request)

    def test_scope_clarification_is_a_new_prompt_not_a_manual_verdict_edit(self):
        before = self.path.read_bytes()
        _, _, v1 = audit.prepare(self.path, self.base / 'v1', root=self.root)
        _, _, v2 = audit.prepare(self.path, self.base / 'v2', root=self.root, scope_definition='mask_interior_v2')
        self.assertNotEqual(v1, v2)
        self.assertIn('OUTSIDE the teal mask', v2)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((self.base / 'v1/request.json').read_bytes(), (self.base / 'v2/request.json').read_bytes())
        self.assertEqual(audit.bind_result(self.result, self.request)[0]['single_component_visible'], False)

    def test_changed_source_or_plan_is_rejected_before_any_client_exists(self):
        self.image.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.prepare(self.path, self.base / 'changed', root=self.root)
        self.plan.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            audit.prepare(self.path, self.base / 'changed-plan', root=self.root)

    def test_external_or_private_urls_and_unbounded_requests_are_rejected(self):
        for value in ['https://example.com/image.png', '/files/.secrets/meta.env', '/files/data/../../.secrets/meta.env']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.local(self.root, value, '0' * 64)
        value = dict(self.request, targets=self.request['targets'] * 31)
        self.path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): audit.prepare(self.path, self.base / 'huge', root=self.root)

    def test_both_exact_versioned_experiment_roots_accept_saved_requests(self):
        for directory in ['interval-motion-20261031', 'assembly-feasibility-20261101']:
            with self.subTest(directory=directory):
                base = self.root / 'data' / directory / 'explicit-fixture'
                base.mkdir(parents=True)
                plan = base / 'plan.json'; plan.write_bytes(self.plan.read_bytes())
                request = copy.deepcopy(self.request)
                request.update(plan_path=str(plan.relative_to(self.root)), plan_sha256=audit.sha(plan))
                path = base / 'request.json'; path.write_text(json.dumps(request))
                returned, images, _ = audit.prepare(path, base / 'new-audit', root=self.root)
                self.assertEqual(returned, request)
                self.assertEqual(len(images), 1)
                self.assertFalse((base / 'new-audit/audit.json').exists(), 'Preparation must not impersonate a hosted result')

    def test_allowlist_rejects_similar_prefixes_other_data_and_wrong_output_roots(self):
        for directory in ['assembly-feasibility-20261101-extra', 'interval-motion-20261031-extra',
                          'assembly-feasibility-20261102', 'unrelated']:
            with self.subTest(directory=directory):
                outside = self.root / 'data' / directory; outside.mkdir()
                request = outside / 'request.json'; request.write_bytes(self.path.read_bytes())
                output = outside / 'new-audit'
                with self.assertRaisesRegex(ValueError, 'versioned experiment directory'):
                    audit.prepare(request, self.base / ('reject-request-' + directory), root=self.root)
                with self.assertRaisesRegex(ValueError, 'versioned experiment directory'):
                    audit.prepare(self.path, output, root=self.root)
                self.assertFalse(output.exists())

    def test_valid_hash_does_not_allow_plan_or_symlink_escape_from_exact_roots(self):
        outside = self.root / 'data/unrelated'; outside.mkdir()
        external_plan = outside / 'plan.json'; external_plan.write_bytes(self.plan.read_bytes())
        request = copy.deepcopy(self.request)
        request.update(plan_path=str(external_plan.relative_to(self.root)), plan_sha256=audit.sha(external_plan))
        path = self.base / 'external-plan-request.json'; path.write_text(json.dumps(request))
        output = self.base / 'blocked-external-plan'
        with self.assertRaisesRegex(ValueError, 'plan changed'):
            audit.prepare(path, output, root=self.root)
        self.assertFalse(output.exists())
        escaped = self.base / 'escaped-request.json'; escaped.symlink_to(outside / 'request.json')
        (outside / 'request.json').write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'versioned experiment directory'):
            audit.prepare(escaped, self.base / 'blocked-symlink-request', root=self.root)


if __name__ == '__main__': unittest.main()
