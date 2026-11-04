"""Upload routing and evidence joins without paid inference or cluster contact."""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import pipeline as P
import sam3d_uploads as uploads
import upgrade_bundles as U
from tests import test_initial_sam3d_handoff as fixtures


class Sam3DUploadAdapterTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.InitialHandoffTests('test_prepares_without_baseline_and_reuses_adapter_input_validator')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.component = {'track_id': 'widget_92', 'selection': str(fixture.selection)}
        self.destination = fixture.case / 'new-native-candidate'

    def invoke(self, callback, **kwargs):
        bridge = types.SimpleNamespace(reconstruct=callback)
        with patch.dict(sys.modules, {'sam3d_cluster_bridge': bridge}):
            return uploads.reconstruct_component(self.component, self.fixture.input.parent,
                self.destination, root=self.fixture.root, **kwargs)

    def test_pending_gpu_freezes_exact_source_without_substitute_and_reuses_identity(self):
        seen = []
        def pending(handoff, destination, **kwargs):
            checked = uploads.handoffs.inspect(handoff, root=self.fixture.root)
            seen.append((handoff, checked))
            return {'status': 'awaiting_gpu_worker', 'reason': 'SSH unavailable', 'cluster_contacted': False}
        result = self.invoke(pending)
        self.destination = self.fixture.case / 'different-resume-directory'
        resumed = self.invoke(pending)
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertEqual(result['handoff'], resumed['handoff'])
        self.assertTrue(result['has_gaps'])
        self.assertNotIn('model', result)
        self.assertFalse(result['old_poses_read'])
        self.assertFalse(result['baseline_asset_read'])
        self.assertFalse((self.fixture.root / 'models').exists())

    def test_pending_backend_cannot_supply_a_substitute_mesh(self):
        with self.assertRaisesRegex(ValueError, 'substitute'):
            self.invoke(lambda *args, **kwargs: {'status': 'awaiting_gpu_worker', 'model': 'some-other-mesh'})

    def test_wrong_component_cannot_dispatch(self):
        self.component['track_id'] = 'different_part'
        with self.assertRaisesRegex(ValueError, 'component/handoff'):
            self.invoke(lambda *args, **kwargs: self.fail('Must not reach cluster bridge'))

    def test_cancel_before_dispatch_preserves_input_without_remote_action(self):
        event = threading.Event(); event.set()
        with self.assertRaises(InterruptedError):
            self.invoke(lambda *args, **kwargs: self.fail('Must not dispatch after cancellation'), cancel_event=event)
        self.assertTrue(list((self.fixture.root / 'data/initial-sam3d-handoffs').glob('*/handoff.json')))

    def native_result(self, *, source=None, changed=False, wrong_timestamp=False):
        folder = self.fixture.root / 'data/native-result'; folder.mkdir()
        asset = folder / 'asset.glb'; asset.write_bytes(b'INERT_NATIVE_CONTRACT_FIXTURE_NOT_MODEL_OUTPUT')
        handoff = next((self.fixture.root / 'data/initial-sam3d-handoffs').glob('*/handoff.json'))
        source_record = dict(self.fixture.manifest['source'], video_sha256=source or U.sha(self.fixture.video))
        if wrong_timestamp: source_record['timestamp_s'] += .1
        metadata = {'object_id': 'widget_92', 'sam3d_output': True, 'learned_inference_executed': True,
            'source': source_record,
            'initial_reconstruction_lineage': {'baseline_asset_read': False, 'old_poses_read': False,
                'handoff': uploads.handoffs.reference(handoff, self.fixture.root),
                'source_binding_sha256': U.sha(handoff.parent / 'source-binding.json')},
            'artifact_files': [{'file': 'asset.glb', 'sha256': U.sha(asset)}]}
        path = folder / 'metadata.json'; U.write(path, metadata)
        if changed: asset.write_bytes(b'changed after verified import')
        return {'status': 'native_candidate_ready', 'model': 'data/native-result/metadata.json',
                'native_asset': 'data/native-result/asset.glb'}

    def test_native_join_retains_unknown_motion_and_rejects_changed_asset(self):
        result = self.invoke(lambda *args, **kwargs: self.native_result())
        self.assertEqual(result['native_motion_status'], 'fresh_native_alignment_required')
        self.assertFalse(result['source_video_pose_validated'])
        self.assertTrue(result['has_gaps'])
        self.assertFalse(result['simulation_ready'])
        self.assertEqual(result['provider'], 'sam3d_native')
        asset = self.fixture.root / result['native_asset']; asset.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'asset changed'):
            self.invoke(lambda *args, **kwargs: result)

    def test_native_join_rejects_other_video(self):
        with self.assertRaisesRegex(ValueError, 'upload source/component'):
            self.invoke(lambda *args, **kwargs: self.native_result(source='e' * 64))

    def test_same_video_component_cannot_substitute_a_different_anchor_timestamp(self):
        with self.assertRaisesRegex(ValueError, 'upload source/component'):
            self.invoke(lambda *args, **kwargs: self.native_result(wrong_timestamp=True))


class Sam3DDefaultProfileTests(unittest.TestCase):
    def test_default_profile_uses_three_reviewed_meta_models_without_observed_fallback(self):
        profile = P.load_profile('meta_sam3d_default')
        self.assertEqual(P.validate_providers(profile), ('meta_api', 'meta_sam31_api'))
        self.assertEqual(profile['reconstruction_provider'], 'sam3d_native')
        self.assertEqual(profile['reasoning_model'], 'muse-spark-1.3')
        self.assertNotIn('observed_follow_on', profile)
        self.assertIn('cluster', profile['disclosure'])
        self.assertIn('100', profile['disclosure'])
        self.assertIn('GPU worker', P.provider_disclosure(profile))
        for field, value in [('reconstruction_provider', 'arbitrary-shell'), ('reasoning_provider', 'codex_cli'),
                             ('native_motion_mode', 'reuse_old_poses')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                P.validate_providers(dict(profile, **{field: value}))

    def test_pending_phase_reconciles_but_completed_native_phase_is_cached(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT / 'data') as temporary:
            root = Path(temporary); dependency = root / 'dependency.txt'; dependency.write_text('fixed')
            runner = P.Runner('native-pending-fixture', run_root=root / 'runs')
            runner.profile = P.load_profile('meta_sam3d_default')
            calls = []
            def action(attempt):
                calls.append(attempt)
                receipt = root / ('attempt-' + str(attempt) + '.json')
                P.write(receipt, {'fixture': True})
                return {'pending_components': ['widget'] if attempt == 1 else [], 'has_gaps': True}, [receipt]
            runner.phase('reconstruct', [dependency], action, optional=True)
            runner.phase('reconstruct', [dependency], action, optional=True)
            runner.phase('reconstruct', [dependency], action, optional=True)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(runner.state['phases'][0]['status'], 'reused_verified_with_gaps')

    def test_two_native_workers_preserve_order_isolate_failure_and_never_exceed_two(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT / 'data') as temporary:
            root = Path(temporary); runner = P.Runner('native-parallel-fixture', run_root=root / 'runs')
            barrier = threading.Barrier(2, timeout=5); lock = threading.Lock()
            active, maximum = 0, 0
            components = [{'track_id': value} for value in ['first', 'broken', 'last']]
            def action(component, directory):
                nonlocal active, maximum
                directory.mkdir()
                with lock:
                    active += 1; maximum = max(maximum, active)
                try:
                    if component['track_id'] != 'last': barrier.wait()
                    if component['track_id'] == 'broken': raise RuntimeError('separate model failed')
                    return {'status': 'awaiting_gpu_worker', 'has_gaps': True}
                finally:
                    with lock: active -= 1
            result, artifacts = runner.native_component_batch(1, root, components, action)
            self.assertEqual(maximum, 2)
            self.assertEqual([r['track_id'] for r in result['components']], ['first', 'broken', 'last'])
            self.assertEqual([r['status'] for r in result['components']], ['complete_with_gaps', 'failed', 'complete_with_gaps'])
            self.assertEqual(result['failures'][0]['track_id'], 'broken')
            self.assertEqual(len(artifacts), 4)

    def test_native_parallel_cancellation_retains_partial_receipts_and_stops_queued_part(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT / 'data') as temporary:
            root = Path(temporary); runner = P.Runner('native-parallel-cancel', run_root=root / 'runs')
            barrier = threading.Barrier(2, timeout=5); called = []
            def action(component, directory):
                called.append(component['track_id']); directory.mkdir()
                barrier.wait(); runner.cancel_event.set()
                raise P.Cancelled('cancel this batch')
            with self.assertRaises(P.Cancelled):
                runner.native_component_batch(1, root, [{'track_id': value} for value in ['first', 'second', 'never']], action)
            self.assertEqual(set(called), {'first', 'second'})
            receipt = P.read(root / 'flow-model-receipts-1/components.json')
            self.assertEqual([r['status'] for r in receipt['components']], ['cancelled'] * 3)

    def test_native_pipeline_calls_bridge_and_muse_but_never_comparison_geometry_or_pose(self):
        with tempfile.TemporaryDirectory(dir=P.ROOT / 'data') as temporary:
            root = Path(temporary); case = root / 'case'; case.mkdir()
            config = case / 'annotations.json'; P.write(config, {'tracks': []})
            runner = P.Runner('native-routing-fixture', run_root=root / 'runs')
            runner.profile = P.load_profile('meta_sam3d_default'); runner.state['provenance'] = {}
            calls, phases = [], {}
            component = {'track_id': 'arbitrary_part', 'selection': P.rel(case / 'selection.json')}
            def command(argv, timeout):
                script = Path(argv[1]).name
                if script == 'run_budgeted_stage.py': script = str(argv[2]); argv = argv[1:]
                args = list(map(str, argv[2:])); calls.append((script, args))
                if script == 'discover_rigid_components.py':
                    P.write(case / 'reasoning/flow-discovery-1/metadata.json', {'tracking_config': P.rel(config)})
                elif script == 'stage2_meta_video_segment.py':
                    (case / 'segmentation/flow-components-1').mkdir(parents=True)
                elif script == 'stage2_meta_reason.py' and args[0] == 'audit':
                    (case / 'segmentation/flow-components-1/model-audit-v1').mkdir()
                elif script == 'stage5_meta_propose.py':
                    P.write(case / 'flow-procedure-proposal-1/proposal.json', {'fixture': True})
            def phase(name, dependencies, action, **kwargs):
                phases[name] = list(map(str, dependencies))
                if name == 'prepare': return {'case': P.rel(case)}
                if name == 'select': return {'selections': [component['selection']], 'components': [component],
                                             'bundles': [P.rel(case / 'portable')]}
                if name in {'discover', 'segment', 'audit', 'reconstruct', 'procedure_proposal'}: return action(1)[0]
                if name in {'pose', 'motion'}:
                    try: return action(1)[0]
                    except ValueError as exc: return {'unavailable': str(exc)}
                return {}
            def bridge(component, bundle, destination, **kwargs):
                destination.mkdir()
                return {'status': 'awaiting_gpu_worker', 'has_gaps': True, 'cluster_contacted': False}
            with patch.object(runner, 'phase', side_effect=phase), patch.object(runner, 'command', side_effect=command), \
                 patch.object(uploads, 'reconstruct_component', side_effect=bridge) as native:
                runner._infer(case / 'input.mp4', {'sha256': 'offline-fixture'})
            self.assertEqual(native.call_count, 1)
            self.assertFalse(any('triposr' in name.lower() for name in phases['reconstruct']))
            self.assertFalse(any(script in {'stage3_reconstruct_local.py', 'stage4_generated_pose.py',
                                           'stage4_track_generated.py', 'stage5_compile.py'} for script, _ in calls))
            self.assertTrue(any(script == 'stage5_meta_propose.py' for script, _ in calls))
            self.assertEqual(runner.state['sam3d_reconstruction_status']['pending_component_ids'], ['arbitrary_part'])
            self.assertEqual(runner.state['provenance']['reconstruction'], 'Meta_SAM3D_Objects_CUDA')


if __name__ == '__main__':
    unittest.main()
