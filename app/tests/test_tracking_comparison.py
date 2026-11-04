import unittest
from copy import deepcopy
from tracking_comparison import compare


def pack(times, supported):
    return {'object_id': 'part', 'provenance': {'source_video_sha256': 'same-source', 'mesh_sha256': 'same-mesh'},
            'frames': [{'sample_index': i+1, 'timestamp_s': t,
                        'status': 'supported_conditional_motion_candidate' if t in supported else 'unsupported'}
                       for i, t in enumerate(times)]}


class TrackingComparisonTests(unittest.TestCase):
    def test_density_alone_does_not_change_held_label_time_or_matched_results(self):
        old = pack([0, 1, 2, 3, 4], {0, 1})
        new = pack([i*.2 for i in range(21)], {i*.2 for i in range(10)})
        r = compare(old, new)
        self.assertEqual(r['baseline']['held_label_seconds_diagnostic_only'], 2.)
        self.assertAlmostEqual(r['candidate']['held_label_seconds_diagnostic_only'], 2.)
        self.assertEqual(r['nonreference_matched_time_transitions']['improved_at_matched_time'], 0)
        self.assertGreater(r['candidate']['counts']['supported'], r['baseline']['counts']['supported'])

    def test_changed_shape_or_source_rejected(self):
        a = pack([0, 1, 2], {0, 1})
        for key in ['source_video_sha256', 'mesh_sha256']:
            b = deepcopy(a)
            b['provenance'][key] = 'different'
            with self.assertRaises(ValueError): compare(a, b)

    def test_actual_matched_time_improvement_and_regression_are_separate(self):
        a, b = pack([0, 1, 2, 3], {0, 1}), pack([0, .5, 1, 1.5, 2, 2.5, 3], {0, 2})
        result = compare(a, b)['nonreference_matched_time_transitions']
        self.assertEqual(result['improved_at_matched_time'], 1)
        self.assertEqual(result['regressed_at_matched_time'], 1)

    def test_invalid_clock_or_no_overlap_rejected(self):
        a = pack([0, 1, 2], {0})
        for b in [pack([4, 5], {4}), pack([0, 0, 1], {0}), pack([0, float('nan')], {0})]:
            with self.assertRaises(ValueError): compare(a, b)
