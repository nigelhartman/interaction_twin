"""Synthetic numerical/security tests, never claims about captured-object physics."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import physics_validation as P
from tests.test_stage5 import fixture as procedure_fixture


class PhysicsValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = procedure_fixture()
        # Declared artifact hashes are real for these small contract fixtures.
        for item in [self.document['source'], *self.document['inputs']]:
            item['path'] = 'data/fixture/' + item['path']
            item['sha256'] = self.write(item['path'], b'SYNTHETIC CONTRACT BYTES')['sha256']
        for frame in self.document['evidence']:
            frame['frame_path'] = 'data/fixture/' + frame['frame_path']
            frame['frame_sha256'] = self.write(frame['frame_path'], b'SYNTHETIC IMAGE BYTES')['sha256']
        self.reference = self.write('data/fixture/interaction.json', self.document)
        self.contract = P.build_contract(self.root, self.reference)

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return {'path': path, 'sha256': P.sha(target)}

    def validate(self, value=None):
        return P.validate_contract(self.root, value or self.contract)

    def test_unknown_values_stay_null_and_no_simulator_is_run(self):
        before = copy.deepcopy(self.contract)
        result = self.validate()
        self.assertEqual(self.contract, before)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['simulator_executed_for_components'])
        self.assertFalse(result['simulation_ready'])
        self.assertFalse(result['adapter']['automatic_parameter_defaults'])
        self.assertFalse(result['adapter']['actual_object_execution_implemented'])
        self.assertTrue(any(gap['field'] == 'inertia_kg_m2' for gap in result['blocking_inputs']))
        self.assertIsNone(self.contract['components'][0]['mass_kg'])
        self.assertIsNone(self.contract['assembly']['joint_definitions'])
        self.assertIsNone(self.contract['trial']['gravity_m_s2'])

    def test_program_asset_source_and_identity_cannot_be_swapped(self):
        for mutation in [lambda c: c.update(case_id='other'),
                         lambda c: c['source_video'].update(sha256='0' * 64),
                         lambda c: c['components'][0].update(identity_status='confirmed'),
                         lambda c: c['components'][0].update(id='made_up'),
                         lambda c: c['components'][0].update(visual_assets=[{'path': 'inferred.glb'}]),
                         lambda c: c.update(proposed_relationships=[])]:
            changed = copy.deepcopy(self.contract)
            mutation(changed)
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_unknown_fields_and_physical_promotions_fail_closed(self):
        for field in ['version', 'simulation_ready', 'physical_accuracy_verified', 'physical_execution_authorized', 'invented_flag']:
            changed = copy.deepcopy(self.contract)
            changed[field] = True
            with self.assertRaises(ValueError):
                self.validate(changed)
        changed = copy.deepcopy(self.contract)
        changed['components'][0]['grasp_frame'] = [[1]]
        with self.assertRaises(ValueError):
            self.validate(changed)
        changed = copy.deepcopy(self.contract)
        changed['source_identity_verification']['declared_artifact_hashes_verified'] += 1
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_physical_numbers_need_correct_units_shapes_and_realizable_inertia(self):
        cases = [('mass_kg', -1), ('mass_kg', True), ('meters_per_native_unit', 0),
                 ('center_of_mass_m', [0, 0]), ('mass_kg', float('nan')),
                 ('inertia_kg_m2', [[1, 0, 0], [0, 1, 0], [0, 0, 3]]),
                 ('inertia_kg_m2', [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 ('inertia_kg_m2', [[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
                 ('inertia_kg_m2', [[1e-18, 0, 0], [0, 1e-18, 0], [0, 0, 3e-18]]),
                 ('inertia_kg_m2', [[1e-18, 1e-19, 0], [0, 1e-18, 0], [0, 0, 1e-18]]),
                 ('contact_material', {'friction': .5}),
                 ('contact_material', {'friction': -.1, 'restitution': 0}),
                 ('contact_material', {'friction': .5, 'restitution': 1.01})]
        for field, value in cases:
            changed = copy.deepcopy(self.contract)
            changed['components'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(changed)

    def test_positive_physical_numbers_do_not_become_verified_without_evidence(self):
        changed = copy.deepcopy(self.contract)
        changed['components'][0].update(mass_kg=.01, inertia_kg_m2=[[.01, 0, 0], [0, .01, 0], [0, 0, .01]])
        result = self.validate(changed)
        self.assertFalse(result['simulation_ready'])
        self.assertIn('mass_kg_provenance', [r['field'] for r in result['blocking_inputs']])
        self.assertFalse(result['physical_accuracy_verified'])

    def test_transform_scale_reflection_and_nonrigid_pose_are_rejected(self):
        for diagonal in [(2, 1, 1), (-1, 1, 1), (0, 1, 1)]:
            matrix = [[diagonal[0], 0, 0, 0], [0, diagonal[1], 0, 0],
                      [0, 0, diagonal[2], 0], [0, 0, 0, 1]]
            changed = copy.deepcopy(self.contract)
            changed['components'][0]['world_from_body_m'] = matrix
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_collision_requires_separate_declared_representation_and_review(self):
        import trimesh
        geometry = self.write('data/fixture/collision-candidate.glb', trimesh.creation.box().export(file_type='glb'))
        changed = copy.deepcopy(self.contract)
        changed['components'][0]['collision_geometry'] = {'artifact': geometry,
            'representation': 'automatic_convex_hull', 'acceptance_evidence': self.reference}
        with self.assertRaisesRegex(ValueError, 'hull'):
            self.validate(changed)

    def test_program_dependencies_are_allowlisted_and_cannot_read_hidden_files(self):
        ref = self.write('annotations/automatic.json', {'origin': 'synthetic_contract'})
        self.assertEqual(P.program_artifact_path(self.root, {**ref, 'kind': 'model_tracklet_proposals'}),
                         (self.root / ref['path']).resolve())
        for kind, path in [('model_tracklet_proposals', '.secrets/never-open'),
                           ('schema', '.secrets/never-open'), ('schema', 'schemas/other.json'),
                           ('deterministic_compiler_code', 'scripts/evil.py')]:
            with self.assertRaises(ValueError):
                P.program_artifact_path(self.root, {'path': path, 'kind': kind})

    def test_bounded_trial_and_unsupported_joint_adapters(self):
        for mutation in [lambda c: c['trial'].update(steps=2001),
                         lambda c: c['trial'].update(steps=True),
                         lambda c: c['trial'].update(timestep_s=1),
                         lambda c: c['trial'].update(gravity_m_s2=[0, 0]),
                         lambda c: c['trial'].update(success_criteria='looks good'),
                         lambda c: c['assembly'].update(joint_definitions=[{'type': 'hinge'}]),
                         lambda c: c['assembly'].update(world_axis_convention='unknown_native')]:
            changed = copy.deepcopy(self.contract)
            mutation(changed)
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_undeclared_io_hash_changes_and_traversal_are_rejected_before_open(self):
        for relative in ['../secret', '/etc/passwd', 'data/../secret', 'data/.secrets/key',
                         'data//image.jpg', 'data/./image.jpg', 'https://site/a', 'data/a?token=x']:
            with self.subTest(path=relative), self.assertRaises(ValueError):
                P.data_path(self.root, relative)
        outside = self.root / 'outside'
        outside.write_bytes(b'PRIVATE SYNTHETIC')
        (self.root / 'data/linked').symlink_to(outside)
        with self.assertRaises(ValueError):
            P.data_path(self.root, 'data/linked')
        (self.root / self.document['source']['path']).write_bytes(b'CHANGED')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.validate()

    def test_duplicate_json_and_nonfinite_constants_are_rejected(self):
        for raw in [b'{"x": 1,"x":2}', b'{"x":NaN}']:
            path = self.root / 'bad.json'
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                P.read_json(path)

    def test_glb_external_paths_and_bad_lengths_never_reach_an_importer(self):
        def glb(value):
            data = json.dumps(value).encode()
            data += b' ' * (-len(data) % 4)
            chunks = struct.pack('<II', len(data), 0x4E4F534A) + data + struct.pack('<II', 4, 0x004E4942) + b'\0' * 4
            return struct.pack('<4sII', b'glTF', 2, len(chunks) + 12) + chunks
        for value in [{'buffers': [{'uri': '../../private', 'byteLength': 4}]},
                      {'images': [{'uri': 'https://example.test/image'}]}]:
            with self.assertRaisesRegex(ValueError, 'URI'):
                P.glb_document(glb(value))
        raw = glb({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}]})
        P.glb_document(raw)
        with self.assertRaises(ValueError):
            P.glb_document(raw[:-1])

    def test_synthetic_box_and_open_patch_topology_are_different_without_repairs(self):
        import trimesh
        box = self.write('data/fixture/synthetic-box.glb', trimesh.creation.box().export(file_type='glb'))
        result = P.inspect_mesh(self.root, box)
        self.assertTrue(result['raw_index_watertight'])
        self.assertEqual(result['triangles'], 12)
        self.assertEqual(result['raw_index_boundary_edge_count'], 0)
        self.assertFalse(result['collision_geometry_accepted'])
        self.assertFalse(result['processing']['repair'])
        patch = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        ref = self.write('data/fixture/synthetic-patch.glb', patch.export(file_type='glb'))
        result = P.inspect_mesh(self.root, ref)
        self.assertEqual(result['raw_index_boundary_edge_count'], 3)
        self.assertFalse(result['raw_index_watertight'])
        self.assertTrue(result['input_unchanged'])

    def test_rotated_box_bottom_uses_corners_rather_than_center_only(self):
        fixture = {'format': 'blender-bullet-synthetic-software-validation', 'fixture_only': True,
                   'real_reconstructed_assets_used': False,
                   'fixture_definition': {'box_side_m': 2, 'floor_top_z_m': 0},
                   'minimum_box_bottom_z_m': .3,
                   'samples': [{'position_m': [0, 0, 1.3],
                                'rotation_quaternion_wxyz': [math.cos(math.pi / 8), math.sin(math.pi / 8), 0, 0]}]}
        result = P.synthetic_box_floor_recheck(fixture)
        self.assertAlmostEqual(result['minimum_corner_clearance_above_floor_m'], 1.3 - math.sqrt(2))
        self.assertFalse(result['floor_penetration_below_5mm'])
        self.assertTrue(result['derived_only_no_solver_rerun'])
        self.assertEqual(fixture['minimum_box_bottom_z_m'], .3)


if __name__ == '__main__':
    unittest.main()
