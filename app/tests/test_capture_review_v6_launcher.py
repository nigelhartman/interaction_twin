"""Versioned selection only: old review/model/launcher bytes stay preserved."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from scripts import launch_capture_review as V1
from scripts import launch_capture_review_v2 as V2
from scripts.restore_capture_review_runtime_20261104_v3 import restore_spec

PRIVATE = (V2.ROOT / 'output/new-captures-20261103/private-native-batch-v6/snapshot.json').is_file()

class CaptureReviewV6Tests(unittest.TestCase):
    @unittest.skipUnless(PRIVATE, 'Private capture evidence is not included in this checkout')
    def test_v1_code_configuration_and_three_snapshots_are_preserved(self):
        self.assertEqual(hashlib.sha256(Path(V1.__file__).read_bytes()).hexdigest(), '2d1e911afd05071b645a9cd573477ab50d8b347df513439429fd2f7eb2d5dd51')
        config = V1.load_configuration()
        self.assertEqual([s['port'] for s in config['viewers']], [51108, 65139, 50984])
        for spec in config['viewers']: V1.checked(spec['snapshot_ref'])

    def test_all_http_ownership_and_child_cleanup_boundaries_are_identical_to_v1(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        first, second = functions(V1.__file__), functions(V2.__file__)
        allowed_selection_changes = {'load_configuration', 'source_binding', 'prepare_spec', 'owned_listener', 'reopen'}
        self.assertEqual(set(first), set(second))
        for name in set(first) - allowed_selection_changes:
            with self.subTest(function=name): self.assertEqual(first[name], second[name])

    @unittest.skipUnless(PRIVATE, 'Private capture evidence is not included in this checkout')
    def test_new_exact_configuration_native_source_and_manifest_bindings(self):
        config = V2.load_configuration(); specs = [V2.prepare_spec(restore_spec(s)) for s in config['viewers']]
        self.assertEqual([s['port'] for s in specs], [52175, 65139, 52154])
        self.assertEqual([len(s['routes']) for s in specs], [147, 15, 23])
        self.assertEqual(specs[2]['routes']['/manifest.json'], specs[2]['browser_manifest_ref'])
        foreign = restore_spec(config['viewers'][2]); foreign['browser_manifest_ref'] = V1.load_configuration()['viewers'][2]['snapshot_ref']
        with self.assertRaisesRegex(ValueError, 'browser manifest changed'): V2.prepare_spec(foreign)

    def test_new_listener_identity_cannot_reuse_baseline_or_unrelated_command(self):
        spec = V2.load_configuration()['viewers'][2]
        own = {'pid': 111, 'uid': 44, 'argv': ['python', str(V2.ROOT / 'scripts/launch_capture_review_v2.py'), '--serve', spec['id']]}
        self.assertTrue(V2.owned_listener(spec, own, uid=44))
        own['argv'][1] = str(V2.ROOT / 'scripts/launch_capture_review.py')
        self.assertFalse(V2.owned_listener(spec, own, uid=44))
        old = {'pid': spec['legacy_pid'], 'uid': 44, 'argv': ['python', 'scripts/serve_laptop_hinge_review_20261103_v2.py', '--port', '0', '--server-id', 'preview2']}
        self.assertFalse(V2.owned_listener(spec, old, uid=44))

    @unittest.skipUnless(PRIVATE, 'Private capture evidence is not included in this checkout')
    def test_v6_changes_only_wrapper_style_and_three_experimental_downloads(self):
        config = V2.load_configuration(); new = json.loads(V2.checked(config['viewers'][0]['snapshot_ref']))
        prior = json.loads(V1.checked(V1.load_configuration()['viewers'][0]['snapshot_ref']))
        self.assertEqual(set(new['routes']), set(prior['routes']))
        changed = {route for route in new['routes'] if new['routes'][route] != prior['routes'][route]}
        self.assertEqual(changed, {'index.html', 'owner.css', 'c/laptop/experimental-joint-program.json', 'c/laptop/experimental-scene.json', 'c/laptop/experimental-fk-audit.json'})
        for old, current in zip(prior['cases'], new['cases']):
            for field in ['candidate_id', 'revision_id', 'parts', 'source_ref', 'source_program_ref', 'normalized_source_ref', 'semantic_companion_ref']:
                with self.subTest(case=current['capture_id'], field=field): self.assertEqual(old[field], current[field])
        self.assertFalse(new['public_promotion']); self.assertTrue(new['historical_owner_candidates_and_revisions_unchanged'])

    @unittest.skipUnless(PRIVATE, 'Private capture evidence is not included in this checkout')
    def test_new_wrapper_keeps_candid_partial_research_limits(self):
        snapshot = json.loads(V2.checked(V2.load_configuration()['viewers'][0]['snapshot_ref']))
        page = V2.checked(snapshot['routes']['index.html']).decode()
        for phrase in ['Keyboard surface recovered; screen texture remains imperfect.', '27 supported / 39 unknown', 'no verified closed zero', 'Separate unapproved research supplement']:
            with self.subTest(phrase=phrase): self.assertIn(phrase, page)
        self.assertIn('http://127.0.0.1:52154/', page)
        manifest = snapshot['supplemental_hinge_experiment']['manifest']
        self.assertEqual(snapshot['routes']['c/laptop/experimental-joint-program.json'], manifest['object_program_ref'])
        self.assertEqual(snapshot['routes']['c/laptop/experimental-scene.json'], manifest['scene_ref'])


if __name__ == '__main__': unittest.main()
