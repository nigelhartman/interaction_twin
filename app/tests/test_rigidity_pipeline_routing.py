"""Real default-route withholding/export; transport and CPU fitting are inert."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mechanism_pipeline as M
import object_program as O
import pipeline as P
import rigidity_feasibility as R
from tests.test_rigidity_feasibility import GuardArtifactTests


class RigidityPipelineRoutingTests(unittest.TestCase):
    def route_fixture(self, *, contradiction=True):
        original_root = P.ROOT
        profile = P.load_profile('meta_sam3d_default')
        helper = GuardArtifactTests()
        self.addCleanup(helper.doCleanups)
        temporary_directory = tempfile.TemporaryDirectory
        with patch('tests.test_object_program.tempfile.TemporaryDirectory',
                   side_effect=lambda: temporary_directory(dir=original_root / 'data')):
            fixture = helper.fixture(contradiction=contradiction)
        root = fixture.root.resolve()
        # Retain the real joint's cited evidence identifier while providing the
        # three source timestamps required by the audit/guard fixture.
        program = O.read_json(root / fixture.doc['evidence_program']['path'])
        program['evidence'][0]['id'] = 'source_a'
        program_ref = fixture.file(fixture.doc['evidence_program']['path'], program)
        source = O.read_json(root / fixture.doc['source_scene']['path'])
        source['original_program'] = program_ref
        source_ref = fixture.file(fixture.doc['source_scene']['path'], source)
        proposal = O.read_json(root / fixture.doc['proposal']['path'])
        proposal['interaction_sha256'] = program_ref['sha256']
        proposal_ref = fixture.file(fixture.doc['proposal']['path'], proposal)
        fixture.doc.update(evidence_program=program_ref, source_scene=source_ref, proposal=proposal_ref)
        fixture.mechanics.update(source_scene=source_ref, proposal=proposal_ref)
        fixture.scene['original_program'] = program_ref
        fixture.update_mechanics()
        for name in {*M.CODE, *R.CODE, 'scripts/run_mechanism_pipeline.py', 'scene_replay_publication.py'}:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_root / name, target)
        budget = root / M.BUDGET_PLAN
        budget.parent.mkdir(parents=True, exist_ok=True)
        budget.write_bytes((original_root / M.BUDGET_PLAN).read_bytes())
        P.write(root / 'data/case/manifest.json', {'source': fixture.source})
        (root / 'data/case/frames').mkdir()
        segmentation = root / 'data/case/segmentation/native'
        segmentation.mkdir(parents=True)
        P.write(segmentation / 'tracks.json', {'fixture': 'No inference'})
        ledger = root / 'data/case/part-ledger.json'
        P.write(ledger, {'fixture': 'Exact ledger preflight is separately tested'})
        root_patch = patch.object(P, 'ROOT', root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        runner = P.Runner('rigidity-default-route', run_root=root / 'data/runs')
        runner.profile = {**profile, 'publication_mode': 'admin_review'}
        runner.state['artifacts']['part_ledger'] = P.rel(ledger)
        program = root / fixture.doc['evidence_program']['path']
        original_phase = runner.phase
        calls = []

        def phase(name, dependencies, action, **kwargs):
            if name == 'mechanism_topology':
                return {'status': 'reused_verified_mechanism_topology', 'proposal': fixture.doc['proposal']}
            return original_phase(name, dependencies, action, **kwargs)

        def command(argv, timeout):
            # Run the actual Python CPU guard inside Runner and real exporter.
            # Only expensive fitting/transport is replaced by a saved fixture.
            args = list(map(str, argv))
            self.assertEqual(Path(args[1]).name, 'run_mechanism_pipeline.py')
            kind = args[2]
            report = Path(args[args.index('--report') + 1])
            calls.append(kind)
            if kind == 'scene':
                value = {'status': 'estimated_constrained_scene_ready',
                         'manifest': fixture.doc['constrained_scene'],
                         'mechanics': fixture.doc['mechanics'],
                         'original_program': fixture.doc['evidence_program'],
                         'artifacts': [fixture.doc[key] for key in
                                       ['constrained_scene', 'mechanics', 'proposal', 'evidence_program']],
                         'has_gaps': False, 'new_api_calls': 0, 'new_gpu_calls': 0}
            elif kind == 'export':
                self.assertIn('--private', args)
                scene_ref = json.loads(args[args.index('--scene-ref') + 1])
                value = M.export_scene(scene_ref, root=root, publish=False)
            else:
                self.fail('Unexpected paid or unrelated dispatch: ' + kind)
            P.write(report, value)

        def run():
            with patch.object(runner, 'phase', side_effect=phase), \
                 patch.object(runner, 'command', side_effect=command), \
                 patch('part_ledger.metadata_paths', return_value=[]), \
                 patch.object(P, 'provider_dependencies', return_value=[]), \
                 patch.object(M, 'scene_input_paths', return_value=[]):
                runner._mechanism_follow_on(program, case=root / 'data/case', segmentation=segmentation)
            return runner.state
        return fixture, runner, calls, run

    def test_default_route_withholds_explicit_nonrigid_child_before_real_export(self):
        fixture, runner, calls, run = self.route_fixture()
        originals = {ref['path']: (fixture.root / ref['path']).read_bytes() for ref in
                     [fixture.doc[key] for key in ['constrained_scene', 'mechanics', 'proposal', 'evidence_program']]}
        state = run()
        self.assertEqual(calls, ['scene', 'export'])
        phase = next(row for row in state['phases'] if row['name'] == 'mechanism_scene')
        self.assertEqual(phase['status'], 'complete_with_gaps')
        self.assertIn('rigidity_feasibility.py', {item['path'] for item in phase['inputs']})
        self.assertIn('native_scene_audit.py', {item['path'] for item in phase['inputs']})
        result = phase['result']
        self.assertEqual(result['base_manifest'], fixture.doc['constrained_scene'])
        self.assertNotEqual(result['manifest'], result['base_manifest'])
        self.assertEqual(result['rigidity_feasibility']['withheld_joint_ids'], ['hinge'])
        doc = P.read(fixture.root / state['artifacts']['object_program'])
        self.assertEqual(doc['constrained_scene'], result['manifest'])
        self.assertTrue(O.validate(doc, root=fixture.root))
        self.assertEqual(doc['joints'], [])
        self.assertEqual(doc['unresolved_connections'][0]['reason'],
                         'rigid_motion_withheld_due_to_explicit_source_nonrigidity')
        self.assertEqual(len(doc['component_assets']), 2)
        self.assertFalse(doc['simulation_ready'])
        self.assertEqual(state['mechanism_pipeline_status']['rigidity_feasibility'],
                         'unsupported_rigid_child_motion_withheld')
        self.assertTrue(P.verify_receipts(phase['files']))
        for path, payload in originals.items():
            self.assertEqual((fixture.root / path).read_bytes(), payload)
        self.assertFalse((fixture.root / 'showcase/aligned-scenes.json').exists())

    def test_no_declaration_keeps_original_scene_and_verified_cache_including_receipt(self):
        fixture, runner, calls, run = self.route_fixture(contradiction=False)
        run()
        phase = next(row for row in runner.state['phases'] if row['name'] == 'mechanism_scene')
        self.assertEqual(phase['result']['manifest'], fixture.doc['constrained_scene'])
        self.assertEqual(phase['result']['mechanics'], fixture.doc['mechanics'])
        self.assertIn('object_program', runner.state['artifacts'], runner.state['phases'])
        doc = P.read(fixture.root / runner.state['artifacts']['object_program'])
        self.assertEqual(doc['joints'][0]['id'], 'hinge')
        self.assertEqual(doc['unresolved_connections'], [])
        receipt = fixture.root / runner.state['artifacts']['rigidity_feasibility']
        before = copy.deepcopy(phase['result'])
        run()
        self.assertEqual(calls, ['scene', 'export'])
        reused = next(row for row in runner.state['phases'] if row['name'] == 'mechanism_scene')
        self.assertEqual(reused['status'], 'reused_verified_with_gaps')
        self.assertEqual(reused['result'], before)
        receipt.write_text('{}')
        run()
        self.assertEqual(calls, ['scene', 'export', 'scene'])
        replacement = next(row for row in runner.state['phases'] if row['name'] == 'mechanism_scene')
        self.assertEqual(replacement['attempt'], 2)
        self.assertNotEqual(replacement['result']['rigidity_feasibility']['assessment_ref']['path'], P.rel(receipt))

    def test_guard_failure_never_exports_raw_fit_or_retains_old_public_scene(self):
        fixture, runner, calls, run = self.route_fixture()
        runner.state['artifacts'].update(constrained_scene='data/stale/scene.json',
                                        object_program='data/stale/program.json',
                                        rigidity_feasibility='data/stale/receipt.json')
        with patch.object(R, 'build', side_effect=ValueError('Exact proposal evidence changed')):
            run()
        self.assertEqual(calls, ['scene'])
        phase = next(row for row in runner.state['phases'] if row['name'] == 'mechanism_scene')
        self.assertEqual(phase['status'], 'failed')
        self.assertIn('Exact proposal evidence changed', phase['error'])
        self.assertEqual(runner.state['mechanism_pipeline_status']['object_program'], 'pending_constrained_scene')
        for key in ['constrained_scene', 'object_program', 'rigidity_feasibility']:
            self.assertNotIn(key, runner.state['artifacts'])


if __name__ == '__main__':
    unittest.main()
