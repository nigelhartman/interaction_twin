import copy
import json
from pathlib import Path
import unittest

from guarded_task_metadata_v1 import derive, ROOT


class NonrigidCollectionMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT/'data/rigidity-feasibility/overnight-camera-wallet-20261103-v1/wallet/scene.json').read_text())
        cls.interaction = json.loads((ROOT/cls.scene['original_program']['path']).read_text())
        cls.program = json.loads((ROOT/'data/object-programs/auto-91382bad396ae8ad499b1bb3d639c9f5de4170f1ac37c7cf3b8b417430143457/object-program.json').read_text())

    def test_actual_wallet_is_flexible_and_removable_without_promoting_its_rejected_rigid_hinge(self):
        self.assertEqual(derive(self.interaction, object_program=self.program, scene=self.scene), ['flexible_or_uncertain', 'removable_components'])

    def test_missing_guard_preserves_legacy_typed_classifier(self):
        scene = copy.deepcopy(self.scene)
        del scene['mechanics']['rigidity_feasibility']
        self.assertIn('hinged_assemblies', derive(self.interaction, object_program=self.program, scene=scene))

    def test_unrelated_real_revolute_is_not_removed(self):
        scene, program = copy.deepcopy(self.scene), copy.deepcopy(self.program)
        # Existing unblocked typed capability remains discoverable.
        program['joints'].append({'id': 'another_hinge', 'type': 'revolute'})
        self.assertEqual(derive(self.interaction, object_program=program, scene=scene), ['flexible_or_uncertain', 'hinged_assemblies', 'removable_components'])

    def test_foreign_source_program_proposal_or_non_null_child_reject(self):
        for mutate in [lambda s: s['mechanics']['rigidity_feasibility'].update(source_sha256='a'*64),
                       lambda s: s['mechanics']['rigidity_feasibility'].update(interaction_ref={'path': 'data/foreign.json', 'sha256': 'a'*64, 'bytes': 1}),
                       lambda s: s['mechanics']['rigidity_feasibility'].update(proposal_ref={'path': 'data/foreign.json', 'sha256': 'a'*64, 'bytes': 1}),
                       lambda s: s['components'][1]['poses'][0].update(status='estimated'),
                       lambda s: s['mechanics']['rigidity_feasibility']['findings'][0]['declarations'][0].update(normalized_assertion='may be deformable')]:
            scene = copy.deepcopy(self.scene)
            mutate(scene)
            with self.assertRaises(ValueError):
                derive(self.interaction, object_program=self.program, scene=scene)

    def test_derivation_does_not_mutate_evidence_or_program(self):
        before = json.dumps([self.scene, self.interaction, self.program], sort_keys=True)
        derive(self.interaction, object_program=self.program, scene=self.scene)
        self.assertEqual(before, json.dumps([self.scene, self.interaction, self.program], sort_keys=True))


if __name__ == '__main__':
    unittest.main()
