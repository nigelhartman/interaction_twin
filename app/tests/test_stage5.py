import copy
import tempfile
import unittest
from pathlib import Path

import stage5 as P


def fixture():
    h = '0'*64
    evidence = [{'id': f'source_{i:06d}', 'source_index': i, 'timestamp_s': float(i-1),
                 'frame_path': f'frames/{i}.png', 'frame_sha256': h, 'observations': []} for i in [1, 2, 3]]
    return {'format': 'interaction-twin-evidence-procedure', 'version': '0.1', 'case_id': 'arbitrary_input', 'summary': 'Generic visible component handling.',
            'source': {'path': 'source.mp4', 'sha256': h, 'original_sha256': h, 'duration_s': 3., 'time_domain': 'source_video_presentation_seconds'},
            'inputs': [{'id': 'semantic_proposal', 'path': 'proposal.json', 'sha256': h, 'kind': 'model_semantic_proposal'},
                       {'id': 'motion_1', 'path': 'motion.json', 'sha256': h, 'kind': 'model_conditional_motion_experiment'}],
            'reasoning': {'provider': 'codex_cli', 'inference_location': 'hosted_OpenAI_via_local_CLI', 'model': 'configured_model', 'role': 'temporary_substitute_for_Meta_reasoning',
                          'proposal_input_id': 'semantic_proposal', 'human_reviewed': False, 'manual_semantic_story_used': False},
            'tracklets': [{'id': t, 'label': t, 'candidate_entity_ids': [t+'_candidate'], 'identity_status': 'model_proposed_physical_identity_unverified',
                          'source_sample_range': [1, 3], 'limitations': ['Physical identity is unverified.']} for t in ['part', 'holder']],
            'assets': [], 'evidence': evidence,
            'relationships': [{'id': 'candidate_part', 'subject_track_id': 'part', 'predicate': 'component_of', 'object_track_id': 'holder',
                               'status': 'inferred', 'evidence_ids': ['source_000001', 'source_000002'], 'rationale': 'Model-proposed structural association.', 'limitations': ['Attachment unknown.']}],
            'variables': [{'id': 'part_visual_state', 'subject_track_id': 'part', 'name': 'visual state', 'values': ['unknown', 'closed', 'open']}],
            'assertions': [{'id': 'closed_view', 'variable_id': 'part_visual_state', 'value': 'closed', 'evidence_id': 'source_000001', 'status': 'visual_model_proposal', 'rationale': 'Visible state.'},
                           {'id': 'open_view', 'variable_id': 'part_visual_state', 'value': 'open', 'evidence_id': 'source_000002', 'status': 'visual_model_proposal', 'rationale': 'Visible state.'}],
            'steps': [{'id': 'visible_opening', 'title': 'Visible opening', 'description': 'A part moves relative to its holder.', 'participant_track_ids': ['part', 'holder'],
                       'evidence_ids': ['source_000001', 'source_000002'], 'start_s': 0., 'end_s': 1.,
                       'preconditions': [{'variable_id': 'part_visual_state', 'value': 'closed', 'evidence_id': 'source_000001'}],
                       'postconditions': [{'variable_id': 'part_visual_state', 'value': 'open', 'evidence_id': 'source_000002'}], 'limitations': ['No mechanism is measured.']}],
            'motion': [{'track_id': 'part', 'experiment_input_id': 'motion_1', 'source_sample_index': i, 'timestamp_s': float(i-1),
                        'status': 'supported_conditional_motion_candidate' if i == 2 else 'withheld_insufficient_evidence',
                        'camera_from_native_mesh': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]] if i == 2 else None,
                        'reasons': [] if i == 2 else ['Occluded'], 'physical_pose_verified': False} for i in [2, 3]],
            'links': {'scene_viewer': '', 'dense_viewer': ''},
            'runtime_policy': {'state_between_evidence': 'unknown', 'pose_between_evidence': 'absent', 'pose_interpolation': False, 'simulation_ready': False, 'physical_identity_confirmed': False},
            'limitations': ['Model interpretations, not physical ground truth.']}


class ProcedureTests(unittest.TestCase):
    def setUp(self):
        self.document = fixture()

    def validate(self, value):
        return P.validate_interaction(value, verify_files=False)

    def test_generic_program_validates_and_checks_actual_evidence(self):
        self.assertTrue(self.validate(self.document))
        result = P.replay(self.document)
        self.assertFalse(result['simulation_ready'])
        self.assertEqual(result['steps'][0]['preconditions'][0]['result'], 'matches_model_proposal')
        self.assertFalse(result['steps'][0]['physical_completion_verified'])

    def test_state_never_persists_between_samples_or_uses_future_observation(self):
        self.assertEqual(P.state_at(self.document, 0.)['part_visual_state']['value'], 'closed')
        for time in [.00001, .5, .99999, 1.5, 2.9]:
            self.assertEqual(P.state_at(self.document, time)['part_visual_state']['value'], 'unknown')
        self.assertEqual(P.state_at(self.document, 1.)['part_visual_state']['value'], 'open')

    def test_pose_gaps_and_withheld_frames_never_generate_motion(self):
        self.assertEqual(len(P.poses_at(self.document, 1.)), 1)
        for time in [.5, 1.1, 2.]:
            self.assertEqual(P.poses_at(self.document, time), [])
        changed = copy.deepcopy(self.document)
        changed['motion'][1]['camera_from_native_mesh'] = changed['motion'][0]['camera_from_native_mesh']
        with self.assertRaisesRegex(ValueError, 'Unsupported pose'):
            self.validate(changed)

    def test_unknown_condition_is_not_reported_as_success(self):
        changed = copy.deepcopy(self.document)
        changed['assertions'].pop()
        self.assertTrue(self.validate(changed))
        result = P.replay(changed)['steps'][0]['postconditions'][0]
        self.assertEqual(result['result'], 'unknown')

    def test_invented_references_and_timestamps_are_rejected(self):
        for kind in ['participant', 'evidence', 'condition', 'time', 'pose_time', 'pose_input', 'asset_track', 'state_variable']:
            changed = copy.deepcopy(self.document)
            if kind == 'participant': changed['steps'][0]['participant_track_ids'] = ['invented']
            if kind == 'evidence': changed['steps'][0]['evidence_ids'][0] = 'invented'
            if kind == 'condition': changed['steps'][0]['preconditions'][0]['evidence_id'] = 'invented'
            if kind == 'time': changed['steps'][0]['start_s'] = .5
            if kind == 'pose_time': changed['motion'][0]['timestamp_s'] = 4.
            if kind == 'pose_input': changed['motion'][0]['experiment_input_id'] = 'invented'
            if kind == 'asset_track': changed['relationships'][0]['object_track_id'] = 'invented'
            if kind == 'state_variable': changed['assertions'][0]['variable_id'] = 'invented'
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(changed)

    def test_structural_cycles_rejected_but_functional_reciprocity_allowed(self):
        changed = copy.deepcopy(self.document)
        inverse = copy.deepcopy(changed['relationships'][0])
        inverse.update(id='reverse', subject_track_id='holder', object_track_id='part')
        changed['relationships'].append(inverse)
        with self.assertRaisesRegex(ValueError, 'cycle'): self.validate(changed)
        changed['relationships'][0]['predicate'] = 'attached_to'
        inverse['predicate'] = 'attached_to'
        self.assertTrue(self.validate(changed))

    def test_contradictory_states_conditions_and_duplicate_ids_rejected(self):
        for kind in ['assertion', 'condition', 'duplicate', 'value', 'unknown']:
            changed = copy.deepcopy(self.document)
            if kind == 'assertion':
                a = copy.deepcopy(changed['assertions'][0]); a.update(id='contradictory', value='open'); changed['assertions'].append(a)
            if kind == 'condition': changed['steps'][0]['preconditions'][0]['value'] = 'open'
            if kind == 'duplicate': changed['steps'].append(copy.deepcopy(changed['steps'][0]))
            if kind == 'value': changed['assertions'][0]['value'] = 'seated_successfully'
            if kind == 'unknown': changed['assertions'][0]['status'] = 'unknown'
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(changed)

    def test_physical_identity_and_false_precision_cannot_be_enabled(self):
        for kind in ['identity', 'simulation', 'interpolation', 'nan', 'scale']:
            changed = copy.deepcopy(self.document)
            if kind == 'identity': changed['tracklets'][0]['identity_status'] = 'confirmed'
            if kind == 'simulation': changed['runtime_policy']['simulation_ready'] = True
            if kind == 'interpolation': changed['runtime_policy']['pose_interpolation'] = True
            if kind == 'nan': changed['motion'][0]['camera_from_native_mesh'][0][0] = float('nan')
            if kind == 'scale': changed['motion'][0]['camera_from_native_mesh'][0][0] = 2.
            with self.subTest(kind=kind), self.assertRaises(Exception): self.validate(changed)

    def test_artifact_paths_cannot_escape_project_or_use_absolute_paths(self):
        for path in ['/etc/passwd', '../secret.json']:
            changed = copy.deepcopy(self.document); changed['source']['path'] = path
            with self.subTest(path=path), self.assertRaises(ValueError): self.validate(changed)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'root'; root.mkdir(); (root/'link').symlink_to(Path(temp))
            with self.assertRaises(ValueError): P.safe_path(root, 'link/outside')

    def test_changed_evidence_file_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'source.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed evidence'):
                P.validate_interaction(self.document, root=root, verify_files=True)

    def test_condition_cannot_cite_a_different_step_boundary(self):
        changed = copy.deepcopy(self.document)
        changed['steps'][0]['preconditions'][0]['evidence_id'] = 'source_000002'
        changed['steps'][0]['preconditions'][0]['value'] = 'open'
        with self.assertRaisesRegex(ValueError, 'first/last'): self.validate(changed)

    def test_text_export_and_replay_are_deterministic_and_explicit(self):
        self.assertEqual(P.text_export(self.document), P.text_export(copy.deepcopy(self.document)))
        self.assertEqual(P.replay(self.document), P.replay(copy.deepcopy(self.document)))
        text = P.text_export(self.document)
        self.assertIn('not physical simulation', text)
        self.assertIn('source_000002', text)
        self.assertIn('hosted_OpenAI', text)

    def test_pose_requires_same_track_mesh_hash_and_no_manual_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = {'track_id': 'part', 'path': 'part.glb', 'sha256': '1'*64}
            pack = {'provenance': {'mesh_sha256': '1'*64, 'mesh_file': str(root/'part.glb'), 'manual_geometry_used': False,
                                   'manual_pose_used': False, 'manual_corners_used': False}}
            pose = {'track_id': 'part'}
            self.assertTrue(P.validate_motion_join(pack, pose, [asset], root))
            for kind in ['hash', 'track', 'path', 'manual', 'missing_manual_flag']:
                changed = copy.deepcopy(pack)
                assets = [copy.deepcopy(asset)]
                if kind == 'hash': changed['provenance']['mesh_sha256'] = '2'*64
                if kind == 'track': assets[0]['track_id'] = 'another_part'
                if kind == 'path': changed['provenance']['mesh_file'] = str(root/'other.glb')
                if kind == 'manual': changed['provenance']['manual_pose_used'] = True
                if kind == 'missing_manual_flag': changed['provenance'].pop('manual_corners_used')
                with self.subTest(kind=kind), self.assertRaises(ValueError): P.validate_motion_join(changed, pose, assets, root)

    def test_unordered_procedure_steps_are_rejected(self):
        changed = copy.deepcopy(self.document)
        later = copy.deepcopy(changed['steps'][0])
        later.update(id='later_action', evidence_ids=['source_000002'], start_s=1., end_s=1., preconditions=[], postconditions=[])
        changed['steps'].insert(0, later)
        with self.assertRaisesRegex(ValueError, 'chronological'):
            self.validate(changed)

    def test_step_pose_coverage_counts_only_its_participants(self):
        changed = copy.deepcopy(self.document)
        changed['steps'][0]['participant_track_ids'] = ['part']
        other = copy.deepcopy(changed['motion'][0])
        other['track_id'] = 'holder'
        changed['motion'].append(other)
        self.assertEqual(P.replay(changed)['steps'][0]['supported_pose_samples_in_interval'], 1)


if __name__ == '__main__':
    unittest.main()
