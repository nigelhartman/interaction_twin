"""Explicit window/reference joins; no files, masks or model inference needed."""
import copy
import unittest
import interval_motion as M
from simultaneous_pair_input import validate_window


def fixture():
    frames = [{'sample_index': i, 'timestamp_s': float(i), 'original_program_sample_index': i,
               'source_image_sha256': str(i), 'width': 40, 'height': 30} for i in range(1, 5)]
    episode = {'id': 'episode-fixed', 'reference_sample_index': 1, 'frames': copy.deepcopy(frames),
               'reference_fit': {'candidates': ['immutable-fixture']}, 'intrinsics': {'K': 'fixed'}}
    common = {'source_video_sha256': 'source', 'interaction_sha256': 'program', 'track_id': 'base', 'mesh_sha256': 'base-mesh'}
    plan = common | {'episodes': [copy.deepcopy(episode)]}
    prior = common | {'episodes': [copy.deepcopy(episode)], 'policy': copy.deepcopy(M.POLICY)}
    peer = common | {'track_id': 'lid', 'frames': copy.deepcopy(frames), 'fixed_intrinsics': {'K': 'fixed'}}
    peer_result = common | {'component_id': 'lid'}
    return plan, prior, peer, peer_result


class SimultaneousPairInputTests(unittest.TestCase):
    def test_existing_first_frame_reference_is_reused_without_search(self):
        values = fixture()
        episode, reference, matched = validate_window(*values, 'episode-fixed')
        self.assertEqual(reference['sample_index'], 1)
        self.assertEqual(reference['source_image_sha256'], matched['source_image_sha256'])
        self.assertEqual(episode['reference_fit'], {'candidates': ['immutable-fixture']})

    def test_missing_reference_or_different_pixel_bytes_rejected(self):
        values = fixture(); values[2]['frames'][0]['original_program_sample_index'] = None
        with self.assertRaisesRegex(ValueError, 'reference'): validate_window(*values, 'episode-fixed')
        values = fixture(); values[2]['frames'][0]['source_image_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'reference'): validate_window(*values, 'episode-fixed')

    def test_source_intrinsics_geometry_or_gates_cannot_be_retargeted(self):
        for target, key, replacement in [(2, 'source_video_sha256', 'other'), (1, 'mesh_sha256', 'other'),
                                         (2, 'fixed_intrinsics', {'K': 'new'}), (1, 'policy', {})]:
            values = fixture(); values[target][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError): validate_window(*values, 'episode-fixed')

    def test_nonchronological_duplicate_and_unbounded_frames_rejected(self):
        for change in ['order', 'duplicate', 'too_many']:
            values = fixture()
            if change == 'order': values[2]['frames'].reverse()
            elif change == 'duplicate': values[2]['frames'][1]['sample_index'] = 1
            else: values[2]['frames'] *= 8
            with self.subTest(change=change), self.assertRaises(ValueError): validate_window(*values, 'episode-fixed')


if __name__ == '__main__': unittest.main()
