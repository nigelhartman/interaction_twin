import copy
import unittest
from tests.test_assembly_scale_feasibility import documents
from simultaneous_pair_motion import same_source_rows, compatibility_from_named_formats


class SimultaneousPairMotionTests(unittest.TestCase):
    def test_explicit_format_adapter_preserves_pinned_fields(self):
        a, b = documents(); a['format'] = 'short-interval-pair-motion-run'; b['format'] = 'short-interval-density-run'
        old = copy.deepcopy([a, b])
        result = compatibility_from_named_formats(a, b)
        self.assertEqual(result['accepted_conditional_hypotheses'], 1)
        self.assertEqual([a, b], old)
        self.assertFalse(result['physical_assembly_established'])

    def test_missing_context_cannot_be_promoted_to_assembly(self):
        a, b = documents(); a['format'] = 'short-interval-pair-motion-run'; b['format'] = 'short-interval-density-run'
        a['status'] = 'unsupported'
        result = compatibility_from_named_formats(a, b)
        self.assertEqual(result['accepted_conditional_hypotheses'], 0)
        self.assertFalse(result['physical_assembly_established'])

    def test_source_join_rejects_interpolation_wrong_bytes_or_time_split(self):
        rows = [{'sample_index': 701, 'timestamp_s': 23.33, 'native_pts_exact_s': '23.330000',
                 'native_decoded_frame_index_zero_based': 700, 'source_image_sha256': 'a',
                 'width': 540, 'height': 960, 'temporal_split': 'reference_selection'}]
        self.assertTrue(same_source_rows(rows, copy.deepcopy(rows)))
        for key, value in [('timestamp_s', 23.34), ('source_image_sha256', 'b'), ('temporal_split', 'evaluation')]:
            other = copy.deepcopy(rows); other[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): same_source_rows(rows, other)
        with self.assertRaises(ValueError): same_source_rows(rows, [])


if __name__ == '__main__': unittest.main()
