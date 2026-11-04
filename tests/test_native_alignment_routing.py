"""Native routes with real receipts, no legacy runtime, transport or model calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mechanism_pipeline as M
import pipeline as P


class NativeAlignmentRoutingTests(unittest.TestCase):
    def setUp(self):
        original_root = P.ROOT
        self.profile = P.load_profile('meta_sam3d_default')
        budget = (original_root / M.BUDGET_PLAN).read_bytes()
        temporary = tempfile.TemporaryDirectory(dir=original_root / 'data')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(P, 'ROOT', self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        self.file('.venv-sam/bin/python', b'INERT_TEST_RUNTIME_MARKER_NEVER_EXECUTED')
        self.file(M.BUDGET_PLAN, budget)
        self.case = self.root / 'data/case'
        self.config = self.file('data/case/annotations.json', {'tracks': []})
        self.file('data/case/manifest.json', {'source': {'sha256': 'a' * 64}})
        self.file('data/case/reasoning/discovery.json', {})
        self.file('data/case/proposal.json', {})
        self.file('data/case/segmentation/native/tracks.json', {})
        self.file('data/case/segmentation/native/frames.json', {})
        self.file('data/case/segmentation/native/model-audit-v1/audit.json', {})
        self.metadata = self.file('data/native/metadata.json', {
            'object_id': 'widget', 'source': {'image_path': 'rgb.png', 'mask_path': 'mask.png'}})
        for name in ['asset.glb', 'rgb.png', 'mask.png']:
            self.file('data/native/' + name, b'INERT_FIXTURE_NOT_MODEL_OUTPUT')
        self.runner = P.Runner('native-routes', run_root=self.root / 'data/runs')
        self.runner.profile = dict(self.profile)
        self.runner.state['provenance'] = {}
        self.calls = []

    def file(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes): path.write_bytes(value)
        else: P.write(path, value)
        return path

    def run_native(self, *, downstream_scene=False, scene_has_gaps=False):
        real_phase = self.runner.phase
        segmentation = self.case / 'segmentation/native'
        def phase(name, dependencies, action, **kwargs):
            if name in {'pose', 'motion'}:
                return real_phase(name, dependencies, action, **kwargs)
            if name == 'prepare': return {'case': P.rel(self.case)}
            if name == 'discover':
                return {'discovery': P.rel(self.case / 'reasoning/discovery.json'),
                        'annotations': P.rel(self.config)}
            if name == 'segment': return {'directory': P.rel(segmentation), 'run': 'native'}
            if name == 'audit': return {'directory': P.rel(segmentation / 'model-audit-v1')}
            if name == 'select': return {'selections': [], 'components': []}
            if name == 'reconstruct':
                return {'models': [P.rel(self.metadata)], 'components': [], 'pending_components': []}
            if name == 'procedure_proposal': return {'proposal': P.rel(self.case / 'proposal.json')}
            if name == 'procedure': return action(1)[0]
            return {}
        def command(argv, timeout):
            script = Path(argv[1]).name
            args = list(map(str, argv[2:])); self.calls.append((script, args))
            if script != 'stage5_compile.py':
                self.fail('Native routing invoked an obsolete or paid adapter: ' + script)
            if args[0] == 'build':
                folder = Path(args[args.index('--out') + 1])
                P.write(folder / 'interaction.json', {'source': {'sha256': 'a' * 64}})
                (folder / 'index.html').write_text('Inert viewer fixture')
        def completed_scene(program, **kwargs):
            manifest_path = self.file('data/scene/scene.json', {'fixture': 'estimated constrained scene'})
            manifest = P.file_receipts([manifest_path])[0]
            self.runner.state['phases'].append({
                'name': 'mechanism_scene', 'status': 'complete_with_gaps' if scene_has_gaps else 'complete',
                'inputs': P.file_receipts([program]),
                'result': {'status': 'estimated_constrained_scene_ready', 'manifest': manifest,
                           'has_gaps': scene_has_gaps}})
            self.runner.state['artifacts']['constrained_scene'] = manifest['path']
        with patch.object(self.runner, 'phase', side_effect=phase), \
             patch.object(self.runner, 'command', side_effect=command), \
             patch.object(P, 'label_generated_meta_viewer'):
            if downstream_scene:
                with patch.object(self.runner, '_mechanism_follow_on', side_effect=completed_scene):
                    self.runner._infer(self.case / 'input.mp4', {'sha256': 'a' * 64})
            else:
                # The actual destination stops before dispatch because no
                # verified ledger exists, rather than inventing an alignment.
                self.runner._infer(self.case / 'input.mp4', {'sha256': 'a' * 64})

    def test_native_pose_motion_route_without_any_legacy_runtime_and_keep_pending_gap(self):
        self.assertFalse((self.root / 'models').exists())
        self.assertFalse((self.root / 'scripts/stage4_generated_pose.py').exists())
        self.assertFalse((self.root / 'requirements-local-geometry.lock.txt').exists())
        self.run_native()
        phases = {row['name']: row for row in self.runner.state['phases']}
        for name in ['pose', 'motion']:
            row = phases[name]
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['result']['status'], 'deferred_to_mechanism_scene')
            self.assertFalse(row['result']['alignment_output_available'])
            self.assertFalse(row['result']['old_poses_reused'])
            self.assertEqual(row['result']['experiments'], [])
            self.assertTrue(P.verify_receipts(row['files']))
            receipt = P.read(self.root / row['result']['routing_receipt'])
            self.assertEqual(receipt['destination_stage'], 'mechanism_scene')
            self.assertTrue(all(not any(word in item['path'].lower() for word in
                ['cotracker', 'triposr', 'stage4_', 'alignment_fit', 'geometry.lock']) for item in row['inputs']))
        build = next(args for script, args in self.calls if args[0] == 'build')
        self.assertNotIn('--motion', build)
        self.assertNotIn('--scene-viewer', build)
        self.assertEqual(self.runner.state['artifacts']['motion'], [])
        self.assertTrue(self.runner.state['native_alignment_has_gaps'])
        self.assertEqual(self.runner.state['native_alignment_status']['destination_status'], 'pending_verified_part_ledger')
        self.assertFalse(self.runner.state['native_alignment_status']['alignment_output_available'])
        self.assertNotIn('constrained_scene', self.runner.state['artifacts'])
        self.assertEqual([script for script, args in self.calls], ['stage5_compile.py', 'stage5_compile.py'])

    def test_route_only_resolves_after_source_bound_real_destination_and_preserves_unknowns(self):
        self.run_native(downstream_scene=True, scene_has_gaps=True)
        result = self.runner.state['native_alignment_status']
        self.assertEqual(result['status'], 'estimated_native_alignment_available')
        self.assertTrue(result['alignment_output_available'])
        self.assertTrue(result['has_gaps'])
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertFalse(result['simulation_ready'])
        self.assertEqual(self.runner.state['artifacts']['motion'], [])
        self.assertFalse(self.runner.state['provenance']['motion_output_available'])
        self.assertTrue(self.runner.state['provenance']['native_scene_alignment_output_available'])

    def test_successful_destination_does_not_leave_a_false_route_gap_and_tampering_reopens_gap(self):
        self.run_native(downstream_scene=True)
        self.assertFalse(self.runner.state['native_alignment_has_gaps'])
        phases = {row['name']: row for row in self.runner.state['phases']}
        procedure = {'interaction': self.runner.state['artifacts']['interaction']}
        original_resolution = self.root / self.runner.state['artifacts']['native_alignment_resolution']
        before = original_resolution.read_bytes()
        (self.root / procedure['interaction']).write_text('{"changed":"new program"}')
        self.runner._resolve_native_alignment_route(phases['pose']['result'], phases['motion']['result'], procedure)
        self.assertTrue(self.runner.state['native_alignment_has_gaps'])
        self.assertFalse(self.runner.state['native_alignment_status']['alignment_output_available'])
        self.assertEqual(original_resolution.read_bytes(), before, 'Earlier resolution receipt stays immutable')
        self.runner._resolve_native_alignment_route(phases['pose']['result'], phases['motion']['result'], {})
        self.assertTrue(self.runner.state['native_alignment_has_gaps'], 'Old saved scene cannot resolve a missing new procedure')

    def test_changed_scene_cannot_stay_available_only_because_its_phase_completed(self):
        self.run_native(downstream_scene=True)
        self.assertFalse(self.runner.state['native_alignment_has_gaps'])
        phases = {row['name']: row for row in self.runner.state['phases']}
        (self.root / self.runner.state['artifacts']['constrained_scene']).write_text('{"changed":true}')
        self.runner._resolve_native_alignment_route(phases['pose']['result'], phases['motion']['result'],
            {'interaction': self.runner.state['artifacts']['interaction']})
        self.assertTrue(self.runner.state['native_alignment_has_gaps'])
        self.assertFalse(self.runner.state['provenance']['native_scene_alignment_output_available'])

    def test_pending_destination_affects_final_run_status_without_a_failed_route(self):
        for gap in [False, True]:
            with self.subTest(gap=gap):
                runner = P.Runner('final-status-' + str(gap).lower(), run_root=self.root / 'data/runs')
                def inference(*args):
                    runner.state['native_alignment_has_gaps'] = gap
                    runner.state['phases'] = [{'name': 'pose', 'status': 'complete'},
                                              {'name': 'motion', 'status': 'complete'}]
                with patch.object(P, 'load_profile', return_value=dict(self.profile)), \
                     patch.object(P, 'validate_source', return_value={'sha256': 'a' * 64}), \
                     patch.object(runner, '_infer', side_effect=inference):
                    result = runner.execute('meta_sam3d_default', input_path=self.case / 'input.mp4')
                self.assertEqual(result['status'], 'complete_with_gaps' if gap else 'complete')
                self.assertNotIn('failed', [row['status'] for row in result['phases']])

    def test_comparison_profile_still_requires_its_real_pose_and_tracker_dependencies(self):
        self.runner.profile.update(reconstruction_provider='triposr_mps')
        self.runner.profile.pop('mechanism_pipeline_mode')
        self.run_native()
        phases = {row['name']: row for row in self.runner.state['phases']}
        self.assertEqual(phases['pose']['status'], 'failed')
        self.assertIn('stage4_generated_pose.py', phases['pose']['error'])
        self.assertEqual(phases['motion']['status'], 'failed')
        self.assertIn('stage4_track_generated.py', phases['motion']['error'])
        self.assertNotIn('native_alignment_status', self.runner.state)
        self.assertEqual(self.runner.state['provenance']['motion'], 'CoTracker3_MPS')


if __name__ == '__main__':
    unittest.main()
