import copy
from decimal import Decimal
import unittest

from scripts import compare_perception_runs as C


def fixture(times=(0., 1., 2.), track_id='part'):
    doc = {'case_id': 'example', 'source': {'path': 'data/example/source/video.mp4', 'sha256': 'a'*64,
            'original_sha256': 'b'*64, 'time_domain': C.TIME_DOMAIN}, 'reasoning': {'model': 'saved_model'},
           'tracklets': [{'id': track_id, 'label': 'A component observation', 'identity_status': 'unverified_candidate'}],
           'inputs': [], 'assets': [], 'links': {}, 'evidence': [], 'motion': []}
    frames = []
    for i, t in enumerate(times, 1):
        doc['evidence'].append({'id': f'e{i}', 'source_index': i, 'timestamp_s': t, 'frame_path': f'data/example/frames/{i}.jpg',
            'observations': [{'track_id': track_id, 'mask_path': f'data/example/masks/{i}.png', 'mask_sha256': 'c'*64,
                              'audit_status': 'plausible_visible_mask'}]})
        frames.append({'sample_index': i, 'timestamp_s': t, 'raw_mask_sha256': 'c'*64, 'raw_area_pixels': 15,
                       'native_returned_instance_preserved': True, 'zero_placeholder_for_missing_or_withheld': False})
        doc['motion'].append({'track_id': track_id, 'timestamp_s': t,
                             'status': C.REFERENCE if i == 1 else C.SUPPORTED})
    return doc, {'tracks': [{'track_id': track_id, 'frames': frames}]}


class PerceptionComparisonTests(unittest.TestCase):
    def test_different_original_source_hashes_reject_cross_video_comparison(self):
        a, am = fixture(); b, bm = fixture()
        b['source']['original_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'Source content hash'):
            C.compare_documents(a, b, am, bm)

    def test_different_normalized_resolutions_can_share_verified_original_lineage(self):
        a, am = fixture(); b, bm = fixture()
        b['source']['sha256'] = 'd'*64
        result = C.compare_documents(a, b, am, bm)
        self.assertTrue(result['source_identity']['same_original_source'])
        self.assertEqual(result['pts_join']['exact_common_samples'], 3)

    def test_original_and_derived_lineage_kinds_are_not_equated(self):
        a, am = fixture(); b, bm = fixture()
        del b['source']['original_sha256']
        b['source']['sha256'] = a['source']['original_sha256']
        with self.assertRaisesRegex(ValueError, 'lineage kind'):
            C.compare_documents(a, b, am, bm)

    def test_strict_pts_intersection_does_not_join_nearby_frames(self):
        a, am = fixture((0., 1., 2.)); b, bm = fixture((0., 1.00000001, 2.))
        result = C.compare_documents(a, b, am, bm)
        self.assertEqual(result['pts_join']['exact_common_source_pts_decimal'], ['0.0', '2.0'])
        self.assertEqual(result['pts_join']['rounding_tolerance_s'], 0)
        self.assertEqual(result['baseline']['exact_common_source_pts']['conditional_nonreference_pose_samples'], 1)
        self.assertFalse(result['pts_join']['nearest_frame_matching'])
        self.assertFalse(result['pts_join']['interpolation'])

    def test_denser_cadence_is_reported_separately_from_matching_pts(self):
        a, am = fixture((0., 1., 2.)); b, bm = fixture((0., .5, 1., 1.5, 2.))
        result = C.compare_documents(a, b, am, bm)
        self.assertEqual(result['candidate']['all_source_pts']['conditional_nonreference_pose_samples'], 4)
        self.assertEqual(result['candidate']['exact_common_source_pts']['conditional_nonreference_pose_samples'], 2)
        self.assertEqual(result['candidate_minus_baseline_record_counts']['all_source_pts']['conditional_nonreference_pose_samples'], 2)
        self.assertEqual(result['candidate_minus_baseline_record_counts']['exact_common_source_pts']['conditional_nonreference_pose_samples'], 0)
        self.assertFalse(result['interpretation']['denser_sampling_implies_improved_quality'])

    def test_reference_is_not_counted_as_conditional_motion_success(self):
        a, am = fixture(); b, bm = fixture()
        result = C.compare_documents(a, b, am, bm)
        summary = result['baseline']['all_source_pts']
        self.assertEqual(summary['conditional_nonreference_pose_samples'], 2)
        self.assertEqual(summary['unverified_reference_pose_samples'], 1)
        self.assertIsNone(summary['physical_pose_accuracy'])
        self.assertIsNone(summary['model_audit_accuracy'])

    def test_placeholder_and_unknown_native_provenance_are_distinct(self):
        a, am = fixture(); b, bm = fixture()
        frame = bm['tracks'][0]['frames'][1]
        frame.update(raw_area_pixels=0, native_returned_instance_preserved=False, zero_placeholder_for_missing_or_withheld=True)
        del bm['tracks'][0]['frames'][2]['native_returned_instance_preserved']
        b['evidence'][1]['observations'][0]['audit_status'] = 'reject_wrong_or_lost_mask'
        result = C.compare_documents(a, b, am, bm)['candidate']['all_source_pts']
        self.assertEqual(result['cited_mask_observations'], 3)
        self.assertEqual(result['recorded_nonempty_masks'], 2)
        self.assertEqual(result['native_preserved_mask_observations'], 1)
        self.assertEqual(result['native_classification_unknown'], 1)
        self.assertEqual(result['zero_placeholders_for_missing_or_withheld'], 1)
        self.assertEqual(result['model_audit_verdicts']['reject_wrong_or_lost_mask'], 1)

    def test_different_tracklet_ids_do_not_create_a_physical_identity_mapping(self):
        a, am = fixture(track_id='before_gap'); b, bm = fixture(track_id='after_gap')
        result = C.compare_documents(a, b, am, bm)
        self.assertEqual(result['baseline']['tracklets'][0]['track_id'], 'before_gap')
        self.assertEqual(result['candidate']['tracklets'][0]['track_id'], 'after_gap')
        self.assertFalse(result['interpretation']['track_ids_matched_as_physical_identities'])

    def test_unsupported_motion_is_counted_without_inventing_pose_coverage(self):
        a, am = fixture(); b, bm = fixture()
        b['motion'][1]['status'] = 'unsupported'
        result = C.compare_documents(a, b, am, bm)['candidate']['all_source_pts']
        self.assertEqual(result['unsupported_or_withheld_motion_samples'], 1)
        self.assertEqual(result['conditional_nonreference_pose_samples'], 1)

    def test_duplicate_pts_or_mask_join_changes_are_rejected(self):
        for mode in ['duplicate_source_pts', 'mask_pts', 'mask_index', 'mask_hash']:
            a, am = fixture(); b, bm = fixture()
            if mode == 'duplicate_source_pts': b['evidence'][1]['timestamp_s'] = 0.
            if mode == 'mask_pts': bm['tracks'][0]['frames'][1]['timestamp_s'] += .000001
            if mode == 'mask_index': bm['tracks'][0]['frames'][1]['sample_index'] = 99
            if mode == 'mask_hash': bm['tracks'][0]['frames'][1]['raw_mask_sha256'] = 'e'*64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                C.compare_documents(a, b, am, bm)

    def test_encoded_time_domain_and_nonfinite_pts_rejected(self):
        a, am = fixture(); b, bm = fixture()
        b['source']['time_domain'] = 'sampled_video_encoded_seconds'
        with self.assertRaisesRegex(ValueError, 'original source presentation'):
            C.compare_documents(a, b, am, bm)
        for value in [True, float('nan'), float('inf'), -1., '1.0']:
            with self.subTest(value=value), self.assertRaises(ValueError): C.exact_time(value)
        self.assertEqual(C.exact_time(1.00000001), Decimal('1.00000001'))

    def test_source_and_mask_documents_are_not_modified(self):
        a, am = fixture(); b, bm = fixture((0., .5, 1., 2.))
        before = copy.deepcopy((a, am, b, bm))
        C.compare_documents(a, b, am, bm)
        self.assertEqual((a, am, b, bm), before)

    def test_hidden_credential_paths_rejected_before_file_verification(self):
        doc, _ = fixture()
        doc['source']['path'] = '.secrets/meta.env'
        with self.assertRaisesRegex(ValueError, 'visible project-relative'):
            C._guard_paths(doc, C.ROOT)


if __name__ == '__main__':
    unittest.main()
